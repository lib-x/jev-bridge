//! The bridge: one resident upstream model behind a Jev-style scoring API.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::{
    body::Bytes,
    extract::State,
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::prompt::{direct_messages, prompt_sha256, softmax, ChatTemplate, LocalComponents, RuntimeClient, LETTERS, PROMPT_VERSION};
use crate::strategy::{
    build_body, candidates_restricted_to, input_tokens, parse_candidates, parse_logprobs,
    request_url, truncated, CompletionResponse, Probe,
};
use crate::wire::{
    request_rows, response_from_results, OrderedMap, Row, ScoredAnswer, PROBABILITY_STATUS,
};

/// One configured bridge over a single upstream model.
pub struct Bridge {
    client: reqwest::Client,
    template: ChatTemplate,
    probe: Probe,
    upstream_model: String,
    upstream_key: Option<String>,
    openai_base: String,
    native_base: String,
    slots: Vec<u32>,
    served_model: String,
    description: String,
    release_date: String,
    max_input_tokens: Option<usize>,
    /// Serialize upstream work so one resident model is not driven concurrently.
    lock: Mutex<()>,
}

/// Everything [`Bridge::connect`] needs to reach one upstream model.
pub struct BridgeConfig {
    /// OpenAI-compatible root, for example `http://127.0.0.1:8080/v1`.
    pub openai_base: String,
    /// Native root for llama.cpp's `/completion`.
    pub native_base: String,
    /// Model name sent upstream.
    pub upstream_model: String,
    /// Bearer token for the upstream service, when it needs one.
    pub upstream_key: Option<String>,
    /// Model id this bridge accepts from clients.
    pub served_model: String,
    /// Description returned by `GET /v1/models`.
    pub description: String,
    /// ISO release date returned by `GET /v1/models`.
    pub release_date: String,
    /// Template variables, for example `{"enable_thinking": false}`.
    pub chat_template_kwargs: Value,
    /// Reject rows above this token count instead of truncating them.
    pub max_input_tokens: Option<usize>,
    /// Render the chat template and/or tokenize in-process instead of asking
    /// the runtime.
    pub local: LocalComponents,
}

/// A scored row plus the evidence needed to compare a bridged run against a
/// reference run row by row.
#[derive(Debug, Clone)]
pub struct DetailedScore {
    /// The normalized answer handed to the wire layer.
    pub answer: ScoredAnswer,
    /// Raw option log probabilities, before the softmax.
    pub option_logprobs: Vec<f64>,
    /// SHA-256 of the rendered prompt, for comparing runs row by row.
    pub prompt_sha256: String,
}

impl Bridge {
    /// Resolve the prompt contract and pick a transport, or fail with reasons.
    pub async fn connect(client: reqwest::Client, mut config: BridgeConfig) -> Result<Self> {
        let runtime = RuntimeClient::new(
            client.clone(),
            config.native_base.clone(),
            config.upstream_model.clone(),
            config.upstream_key.clone(),
        );
        let template = ChatTemplate::new(
            runtime,
            std::mem::take(&mut config.local),
            config.chat_template_kwargs.clone(),
        );

        let slots = template
            .resolve_slots()
            .await
            .context("resolving the single-token answer slots failed")?;

        // A sixteen-option probe exercises the widest contract this bridge
        // serves, so a transport that passes it will hold for real requests.
        let probe_options: Vec<crate::wire::RowOption> = LETTERS
            .chars()
            .map(|letter| crate::wire::RowOption {
                id: letter.to_string(),
                description: format!("Probe option {letter}"),
            })
            .collect();
        let messages = direct_messages(&json!("probe"), "probe", &probe_options);
        let prompt = template
            .render(&messages)
            .await
            .context("rendering the probe prompt failed")?;
        template
            .verify_boundary(&prompt, &slots)
            .await
            .context("verifying the answer-token boundary failed")?;

        let letters: Vec<char> = LETTERS.chars().collect();
        let mut chosen = None;
        let mut failures = Vec::new();
        for probe in crate::strategy::CANDIDATES {
            match try_probe(
                &client,
                probe,
                &config,
                &prompt,
                &slots,
                &letters,
            )
            .await
            {
                Ok(()) => {
                    chosen = Some(probe);
                    break;
                }
                Err(error) => failures.push(format!("{}: {error:#}", probe.name())),
            }
        }
        let probe = chosen.ok_or_else(|| {
            anyhow::anyhow!(
                "no supported transport answered the probe:\n  {}",
                failures.join("\n  ")
            )
        })?;

        Ok(Self {
            client,
            template,
            probe,
            upstream_model: config.upstream_model,
            upstream_key: config.upstream_key,
            openai_base: config.openai_base,
            native_base: config.native_base,
            slots,
            served_model: config.served_model,
            description: config.description,
            release_date: config.release_date,
            max_input_tokens: config.max_input_tokens,
            lock: Mutex::new(()),
        })
    }

    /// The transport negotiated at startup.
    pub fn probe(&self) -> Probe {
        self.probe
    }

    /// Resolved token id for each of the sixteen answer letters.
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    /// The model name this bridge sends upstream.
    pub fn upstream_model(&self) -> &str {
        &self.upstream_model
    }

    /// Template variables forwarded on every render.
    pub fn chat_template_kwargs(&self) -> Value {
        self.template.chat_template_kwargs()
    }

    /// Whether the prompt is rendered in-process rather than by the runtime.
    pub fn renders_locally(&self) -> bool {
        self.template.renders_locally()
    }

    /// Whether tokenization happens in-process rather than by the runtime.
    pub fn tokenizes_locally(&self) -> bool {
        self.template.tokenizes_locally()
    }

    /// Score every row of one request against the same upstream model.
    pub async fn score_rows(&self, rows: &[Row]) -> Result<Vec<DetailedScore>> {
        let _guard = self.lock.lock().await;
        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push(self.score_row(row).await?);
        }
        Ok(results)
    }

    async fn score_row(&self, row: &Row) -> Result<DetailedScore> {
        let count = row.options.len();
        let messages = direct_messages(&row.state, &row.question, &row.options);
        let prompt = self
            .template
            .render(&messages)
            .await
            .with_context(|| format!("row {:?}", row.id))?;

        let slots = &self.slots[..count];
        let letters: Vec<char> = LETTERS.chars().take(count).collect();
        let body = build_body(self.probe, &self.upstream_model, &prompt, slots);
        let url = request_url(self.probe, &self.openai_base, &self.native_base);
        let response: CompletionResponse =
            post_json(&self.client, &url, &body, self.upstream_key.as_deref()).await?;

        if truncated(self.probe, &response) {
            bail!(
                "row {:?}: the upstream runtime truncated the prompt; shorten the state or raise \
                 its context size",
                row.id
            );
        }
        let input_tokens = input_tokens(self.probe, &response);
        if let Some(limit) = self.max_input_tokens
            && input_tokens > limit as u64
        {
            bail!(
                "row {:?}: {input_tokens} input tokens exceed limit {limit}; no truncation allowed",
                row.id
            );
        }

        let logprobs = parse_logprobs(self.probe, &response, slots, &letters)
            .with_context(|| format!("row {:?}", row.id))?;
        let probabilities = softmax(&logprobs);

        Ok(DetailedScore {
            answer: ScoredAnswer {
                id: row.id.clone(),
                option_ids: row.options.iter().map(|option| option.id.clone()).collect(),
                probabilities,
                input_tokens,
                prompt_version: Some(PROMPT_VERSION.to_string()),
            },
            option_logprobs: logprobs,
            prompt_sha256: prompt_sha256(&prompt),
        })
    }
}

async fn try_probe(
    client: &reqwest::Client,
    probe: Probe,
    config: &BridgeConfig,
    prompt: &str,
    slots: &[u32],
    letters: &[char],
) -> Result<()> {
    let body = build_body(probe, &config.upstream_model, prompt, slots);
    let url = request_url(probe, &config.openai_base, &config.native_base);
    let response: CompletionResponse = post_json(client, &url, &body, config.upstream_key.as_deref()).await?;
    let logprobs = parse_logprobs(probe, &response, slots, letters)?;
    crate::strategy::validate_probe(&logprobs)?;
    if probe.restricts_distribution() {
        let candidates = parse_candidates(probe, &response)?;
        if !candidates_restricted_to(&candidates, slots, letters) {
            bail!(
                "the server ignored the distribution restriction and returned foreign tokens, so \
                 this transport would not be what it claims"
            );
        }
    }
    Ok(())
}

/// POST a typed request body and decode a typed response.
///
/// The bridge always knows which shape it expects, so the response is decoded
/// into a struct instead of being poked at as a generic document; a server that
/// answers with something else fails here rather than later.
pub async fn post_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    body: &(impl serde::Serialize + ?Sized),
    api_key: Option<&str>,
) -> Result<T> {
    let mut request = client.post(url).json(body);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("POST {url} failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("reading the {url} response failed"))?;
    if !status.is_success() {
        bail!(
            "POST {url} returned HTTP {status}: {}",
            text.chars().take(400).collect::<String>()
        );
    }
    serde_json::from_str(&text).with_context(|| {
        format!(
            "POST {url} returned an unexpected body: {}",
            text.chars().take(400).collect::<String>()
        )
    })
}

/// Shared HTTP state.
pub struct AppState {
    /// The configured bridge, shared by every handler.
    pub bridge: Bridge,
    /// When set, clients must present this bearer token.
    pub api_key: Option<String>,
}

/// The HTTP surface: `/v1/systemone`, `/v1/models` and `/health`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state)
}

async fn systemone(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let payload: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("body: must be a JSON object ({error})"),
            )
        }
    };
    let (specs, rows) = match request_rows(&payload, &state.bridge.served_model) {
        Ok(parsed) => parsed,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.0),
    };
    let results = match state.bridge.score_rows(&rows).await {
        Ok(results) => results,
        Err(error) => {
            return error_response(StatusCode::BAD_GATEWAY, format!("{error:#}"));
        }
    };
    let answers: Vec<ScoredAnswer> = results.into_iter().map(|scored| scored.answer).collect();
    match response_from_results(&state.bridge.served_model, &specs, &answers) {
        Ok(response) => Json(response).into_response(),
        Err(error) => error_response(StatusCode::INTERNAL_SERVER_ERROR, error.0),
    }
}

/// One entry in `GET /v1/models`.
#[derive(Debug, Serialize)]
struct ModelEntry {
    name: String,
    description: String,
    release_date: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    models: Vec<ModelEntry>,
}

/// What the bridge negotiated, exposed so an operator can see it rather than
/// guess which transport is in use.
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    served_model: String,
    upstream_model: String,
    probe: &'static str,
    probe_note: &'static str,
    endpoint: &'static str,
    prompt_version: &'static str,
    /// The resolved token id of every answer letter.
    answer_slots: OrderedMap<u32>,
    chat_template_kwargs: Value,
    renders_locally: bool,
    tokenizes_locally: bool,
    probability_status: &'static str,
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    Json(ModelsResponse {
        models: vec![ModelEntry {
            name: state.bridge.served_model.clone(),
            description: state.bridge.description.clone(),
            release_date: state.bridge.release_date.clone(),
        }],
    })
    .into_response()
}

async fn health(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let mut answer_slots = OrderedMap::new();
    for (letter, id) in LETTERS.chars().zip(state.bridge.slots()) {
        answer_slots.insert(letter.to_string(), *id);
    }
    Json(HealthResponse {
        status: "ok",
        served_model: state.bridge.served_model.clone(),
        upstream_model: state.bridge.upstream_model().to_string(),
        probe: state.bridge.probe().name(),
        probe_note: state.bridge.probe().note(),
        endpoint: state.bridge.probe().endpoint(),
        prompt_version: PROMPT_VERSION,
        answer_slots,
        chat_template_kwargs: state.bridge.chat_template_kwargs(),
        renders_locally: state.bridge.renders_locally(),
        tokenizes_locally: state.bridge.tokenizes_locally(),
        probability_status: PROBABILITY_STATUS,
    })
    .into_response()
}

fn error_response(status: StatusCode, message: String) -> Response {
    (status, Json(json!({"error": message}))).into_response()
}

/// Rejecting a request returns a full HTTP response, which is larger than the
/// success value; boxing it would only add an allocation on the error path.
#[allow(clippy::result_large_err)]
fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = &state.api_key else {
        return Ok(());
    };
    let presented = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .unwrap_or_default();
    if constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
        Ok(())
    } else {
        Err(error_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token".to_string(),
        ))
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |accumulator, (a, b)| accumulator | (a ^ b))
        == 0
}
