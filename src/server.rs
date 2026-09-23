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

use crate::prompt::{
    direct_messages, direct_messages_reusing, evidence_json, prompt_sha256, softmax, ChatTemplate,
    LocalComponents, RuntimeClient, LETTERS, PROMPT_VERSION,
};
use crate::strategy::{
    build_body_with_candidates, candidate_ladder, candidates_restricted_to, input_tokens,
    parse_batch_candidates, parse_candidates, parse_logprobs, request_url,
    slot_logprobs_from_candidates, truncated, CompletionResponse, Probe,
};
use crate::wire::{
    request_batch, response_from_results, OrderedMap, Question, ReadoutStatus, Row, ScoredAnswer,
    PROBABILITY_STATUS,
};
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// Accept any non-empty `model` name instead of requiring the served model.
    accept_any_model: bool,
    /// What the deployment declared about the readout channel on this model.
    readout: ReadoutStatus,
    /// How many batched readouts were served by the sequential fallback
    /// because the endpoint rejected the array-`prompt` shape. Non-zero means
    /// the shared prefill did not happen; `GET /health` reports it.
    batch_fallbacks: AtomicU64,
    /// Serialize upstream work so one resident model is not driven concurrently.
    lock: Mutex<()>,
}

/// One question's readout from one upstream call.
struct Readout {
    /// Raw option log probabilities, before the softmax.
    logprobs: Vec<f64>,
    /// Prompt tokens attributed to this question. For the tail of a batched
    /// readout this is 0: the endpoint reports one number for the whole batch
    /// and it is attached to the first readout, so summing the vector
    /// reproduces the endpoint's own count instead of a faked per-prompt split.
    input_tokens: u64,
}

/// A non-success HTTP response from the upstream service.
///
/// The status code is kept because callers have to tell a *request-shape*
/// rejection — the endpoint refuses this shape at all, where a fallback can
/// help — from a call-level failure. A 429, 401/403 or 5xx must propagate:
/// falling back there would turn a throttled or broken endpoint into a
/// "successful" run.
#[derive(Debug)]
pub struct UpstreamStatus {
    /// The URL that was posted to.
    pub url: String,
    /// The HTTP status the upstream returned.
    pub status: u16,
    /// The first 400 characters of the response body.
    pub body: String,
}

impl UpstreamStatus {
    /// Whether the endpoint rejected the request *shape* rather than failing
    /// on this particular call.
    pub fn is_shape_rejection(&self) -> bool {
        matches!(self.status, 400 | 404 | 405 | 415 | 422)
    }
}

impl std::fmt::Display for UpstreamStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "POST {} returned HTTP {}: {}", self.url, self.status, self.body)
    }
}

impl std::error::Error for UpstreamStatus {}

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
    /// Accept any non-empty `model` name in a request instead of requiring the
    /// served model.
    ///
    /// Off by default: the bridge does not serve model ids it is not (that is
    /// what the `jev` prefix check on `--served-model` protects). Third-party
    /// TypeSafe clients hard-code an id like `jev-latest`, so a deployment
    /// that wants to serve them turns this on explicitly. Responses still
    /// report the served model.
    pub accept_any_model: bool,
    /// What the deployment knows about the readout channel on this model.
    ///
    /// Defaults to `unvalidated`: the bridge cannot prove readout quality at
    /// runtime, so a deployment that checked it (for example with
    /// [`crate::evaluate`] over an alignment run) states so explicitly.
    pub readout: ReadoutStatus,
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
            accept_any_model: config.accept_any_model,
            readout: config.readout,
            batch_fallbacks: AtomicU64::new(0),
            lock: Mutex::new(()),
        })
    }

    /// The transport negotiated at startup.
    pub fn probe(&self) -> Probe {
        self.probe
    }

    /// What the deployment declared about the readout channel on this model.
    pub fn readout(&self) -> &ReadoutStatus {
        &self.readout
    }

    /// Whether any non-empty `model` name is accepted, instead of only the
    /// served model.
    pub fn accepts_any_model(&self) -> bool {
        self.accept_any_model
    }

    /// How many batched readouts fell back to sequential requests because the
    /// endpoint rejected the array-`prompt` shape.
    pub fn batch_fallbacks(&self) -> u64 {
        self.batch_fallbacks.load(Ordering::Relaxed)
    }

    /// Run the readout self-check: a few questions whose answers are obvious,
    /// read through the exact slot channel every real request uses.
    ///
    /// It cannot prove the model is good at decisions — that needs a
    /// gold-labelled workload, which is [`crate::evaluate`]'s job — but it
    /// catches the model that is not answering the question at all, the
    /// failure mode that would otherwise be silently confident.
    pub async fn run_readout_check(&self) -> Result<crate::readout::ReadoutCheck> {
        let questions = crate::readout::questions();
        let scored = self
            .score_questions(&json!(crate::readout::PROBE_STATE), &questions)
            .await
            .context("the readout self-check could not be scored")?;
        let results = crate::readout::PROBES
            .iter()
            .zip(&scored)
            .map(|(probe, score)| {
                crate::readout::outcome(probe, &score.answer.option_ids, &score.answer.probabilities)
            })
            .collect();
        Ok(crate::readout::ReadoutCheck::from_results(results))
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

    /// Score one decision that carries its own state (the `--score` shape).
    pub async fn score_row(&self, row: &Row) -> Result<DetailedScore> {
        let question = Question {
            id: row.id.clone(),
            question: row.question.clone(),
            options: row.options.clone(),
        };
        let mut scored = self.score_questions(&row.state, &[question]).await?;
        Ok(scored.remove(0))
    }

    /// Score several questions that share one state (the System One shape).
    ///
    /// The state is serialised once and reused by every question. When the
    /// negotiated transport accepts an array `prompt` and more than one
    /// question is asked, the whole batch is read in **one** request, so a
    /// prefix-caching server prefills the shared state once. A transport that
    /// rejects the batched shape falls back to sequential single-prompt calls;
    /// the fallback is counted (`GET /health` reports it) because a run served
    /// that way did not get the shared prefill.
    pub async fn score_questions(
        &self,
        state: &Value,
        questions: &[Question],
    ) -> Result<Vec<DetailedScore>> {
        let _guard = self.lock.lock().await;

        // Render up front: the evidence text is serialised once for the whole
        // batch, and every question reuses it.
        let evidence = evidence_json(state);
        let mut prompts = Vec::with_capacity(questions.len());
        for question in questions {
            let messages =
                direct_messages_reusing(&evidence, &question.question, &question.options);
            let prompt = self
                .template
                .render(&messages)
                .await
                .with_context(|| format!("question {:?}", question.id))?;
            prompts.push(prompt);
        }

        let readouts = if questions.len() > 1 && self.probe.supports_batch() {
            match self.read_batch(questions, &prompts).await {
                Ok(readouts) => readouts,
                Err(error) => {
                    // Only a *request-shape* rejection means "this endpoint
                    // cannot batch". Anything else is about the call itself
                    // and must propagate.
                    if error
                        .downcast_ref::<UpstreamStatus>()
                        .is_some_and(UpstreamStatus::is_shape_rejection)
                    {
                        self.batch_fallbacks.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "the upstream rejected the batched readout shape ({error}); falling \
                             back to {} sequential requests (no shared prefill on this endpoint)",
                            questions.len()
                        );
                        self.read_one_by_one(questions, &prompts).await?
                    } else {
                        return Err(error);
                    }
                }
            }
        } else {
            self.read_one_by_one(questions, &prompts).await?
        };

        // Assemble: every answer is derived from its own readout only.
        Ok(readouts
            .into_iter()
            .zip(questions)
            .zip(&prompts)
            .map(|((readout, question), prompt)| DetailedScore {
                answer: ScoredAnswer {
                    id: question.id.clone(),
                    option_ids: question
                        .options
                        .iter()
                        .map(|option| option.id.clone())
                        .collect(),
                    probabilities: softmax(&readout.logprobs),
                    input_tokens: readout.input_tokens,
                    prompt_version: Some(PROMPT_VERSION.to_string()),
                },
                option_logprobs: readout.logprobs,
                prompt_sha256: prompt_sha256(prompt),
            })
            .collect())
    }

    /// One batched readout for every question, in request order.
    ///
    /// A missing option token deepens the request instead of failing it: the
    /// first attempt sizes top-k from the option count, and a large prompt can
    /// push an answer letter past it (measured: the tetris demo's sixteen
    /// options missed `F` at k=64). A retry reuses the server's prefix cache,
    /// so only the readout is recomputed.
    async fn read_batch(&self, questions: &[Question], prompts: &[String]) -> Result<Vec<Readout>> {
        // One `logprobs` value covers every prompt of the request, so it is
        // sized for the largest declared option set; each question then reads
        // only its own slots from its own candidate list.
        let max_options = questions
            .iter()
            .map(|question| question.options.len())
            .max()
            .unwrap_or(0);
        let slots = &self.slots[..max_options];
        let url = request_url(self.probe, &self.openai_base, &self.native_base);

        let mut last_error = None;
        for candidates_requested in candidate_ladder(self.probe, max_options) {
            let body = build_body_with_candidates(
                self.probe,
                &self.upstream_model,
                prompts,
                slots,
                candidates_requested,
            );
            let response: CompletionResponse =
                post_json(&self.client, &url, &body, self.upstream_key.as_deref()).await?;

            // The endpoint reports one usage block for the whole batch, and
            // endpoints disagree on what it counts (llama.cpp b11096 counts the
            // shared prefix once). It is attached to the first readout so summing
            // the vector reproduces the endpoint's own number.
            let batch_tokens = input_tokens(self.probe, &response);

            // A batched prompt cannot be checked against `max_input_tokens` per
            // prompt: the only number the endpoint gives is the batch total. The
            // loose upper bound below still catches a request that is over budget
            // even if every prompt shared one prefix.
            if let Some(limit) = self.max_input_tokens
                && batch_tokens > limit as u64 * questions.len() as u64
            {
                bail!(
                    "the batched readout reports {batch_tokens} input tokens, over {} x {limit}; \
                     no truncation allowed",
                    questions.len()
                );
            }

            let candidates = parse_batch_candidates(self.probe, &response, questions.len())?;
            let mut readouts = Vec::with_capacity(questions.len());
            let mut missing = None;
            for (index, (question, candidates)) in questions.iter().zip(&candidates).enumerate() {
                let count = question.options.len();
                let slots = &self.slots[..count];
                let letters: Vec<char> = LETTERS.chars().take(count).collect();
                match slot_logprobs_from_candidates(candidates, slots, &letters) {
                    Ok(logprobs) => readouts.push(Readout {
                        logprobs,
                        input_tokens: if index == 0 { batch_tokens } else { 0 },
                    }),
                    Err(error) => {
                        missing = Some(error.context(format!("question {:?}", question.id)));
                        break;
                    }
                }
            }
            match missing {
                None => return Ok(readouts),
                // A restricted distribution is already told which tokens to
                // return; a missing one there is not fixed by asking deeper.
                Some(error) if !self.probe.restricts_distribution() => {
                    eprintln!(
                        "{error:#}; deepening the batched readout past {candidates_requested} \
                         candidates"
                    );
                    last_error = Some(error);
                }
                Some(error) => return Err(error),
            }
        }
        Err(last_error.expect("a failed ladder records its last error"))
    }

    /// One readout per question, one upstream request each.
    ///
    /// A missing option token deepens the per-question request the same way
    /// [`Self::read_batch`] does.
    async fn read_one_by_one(
        &self,
        questions: &[Question],
        prompts: &[String],
    ) -> Result<Vec<Readout>> {
        let mut readouts = Vec::with_capacity(questions.len());
        for (question, prompt) in questions.iter().zip(prompts) {
            let count = question.options.len();
            let slots = &self.slots[..count];
            let letters: Vec<char> = LETTERS.chars().take(count).collect();
            let url = request_url(self.probe, &self.openai_base, &self.native_base);

            let mut last_error = None;
            let mut readout = None;
            for candidates_requested in candidate_ladder(self.probe, count) {
                let body = build_body_with_candidates(
                    self.probe,
                    &self.upstream_model,
                    std::slice::from_ref(prompt),
                    slots,
                    candidates_requested,
                );
                let response: CompletionResponse =
                    post_json(&self.client, &url, &body, self.upstream_key.as_deref()).await?;

                if truncated(self.probe, &response) {
                    bail!(
                        "question {:?}: the upstream runtime truncated the prompt; shorten the state \
                         or raise its context size",
                        question.id
                    );
                }
                let tokens = input_tokens(self.probe, &response);
                if let Some(limit) = self.max_input_tokens
                    && tokens > limit as u64
                {
                    bail!(
                        "question {:?}: {tokens} input tokens exceed limit {limit}; no truncation \
                         allowed",
                        question.id
                    );
                }

                match parse_logprobs(self.probe, &response, slots, &letters) {
                    Ok(logprobs) => {
                        readout = Some(Readout {
                            logprobs,
                            input_tokens: tokens,
                        });
                        break;
                    }
                    Err(error) if !self.probe.restricts_distribution() => {
                        eprintln!(
                            "question {:?}: {error:#}; deepening the readout past \
                             {candidates_requested} candidates",
                            question.id
                        );
                        last_error = Some(error.context(format!("question {:?}", question.id)));
                    }
                    Err(error) => {
                        return Err(error).with_context(|| format!("question {:?}", question.id));
                    }
                }
            }
            readouts.push(
                readout.ok_or_else(|| last_error.expect("a failed ladder records its last error"))?,
            );
        }
        Ok(readouts)
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
    let url = request_url(probe, &config.openai_base, &config.native_base);
    // A probe can miss an answer token for the same reason a readout can, so it
    // walks the ladder too: a transport must not be declared unsupported just
    // because its first candidate list was too shallow.
    let mut last_error = None;
    for candidates_requested in candidate_ladder(probe, slots.len()) {
        let body = build_body_with_candidates(
            probe,
            &config.upstream_model,
            &[prompt.to_string()],
            slots,
            candidates_requested,
        );
        let response: CompletionResponse =
            post_json(client, &url, &body, config.upstream_key.as_deref()).await?;
        match parse_logprobs(probe, &response, slots, letters) {
            Ok(logprobs) => {
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
                return Ok(());
            }
            // Only an unrestricted distribution can be fixed by asking deeper;
            // a restricted one that misses a slot is misreporting itself.
            Err(error) if !probe.restricts_distribution() => {
                last_error = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_error.expect("a failed ladder records its last error"))
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
        return Err(anyhow::Error::new(UpstreamStatus {
            url: url.to_string(),
            status: status.as_u16(),
            body: text.chars().take(400).collect(),
        }));
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
    /// The startup readout self-check, when one ran.
    pub readout_check: Option<crate::readout::ReadoutCheck>,
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
    let batch = match request_batch(
        &payload,
        &state.bridge.served_model,
        state.bridge.accepts_any_model(),
    ) {
        Ok(parsed) => parsed,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.0),
    };
    let results = match state
        .bridge
        .score_questions(&batch.state, &batch.questions)
        .await
    {
        Ok(results) => results,
        Err(error) => {
            return error_response(StatusCode::BAD_GATEWAY, format!("{error:#}"));
        }
    };
    let answers: Vec<ScoredAnswer> = results.into_iter().map(|scored| scored.answer).collect();
    match response_from_results(
        &state.bridge.served_model,
        &batch.specs,
        &answers,
        state.bridge.readout(),
    ) {
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
    /// Whether the readout channel was validated on the served model.
    readout: ReadoutStatus,
    /// Batched readouts served by the sequential fallback (0 = every batch got
    /// the shared prefill).
    batch_fallbacks: u64,
    /// The startup readout self-check, when one ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    readout_check: Option<crate::readout::ReadoutCheck>,
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
        readout: state.bridge.readout().clone(),
        batch_fallbacks: state.bridge.batch_fallbacks(),
        readout_check: state.readout_check.clone(),
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
