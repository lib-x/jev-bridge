//! Shared helpers for the Rust contract suite.
//!
//! `contract.rs` drives the HTTP surface the same way `tests/test_contract.py`
//! does from outside the process; `bridge.rs` uses the same mock upstream but
//! calls the library directly.

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{extract::State, routing::post, Json, Router};
use serde_json::{json, Value};

use jev_bridge::server::{router, AppState, Bridge, BridgeConfig};

pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";
pub const MOCK_PROMPT: &str = "<|im_start|>system\nmock<|im_end|>\n<|im_start|>assistant\n";
pub const MOCK_PROMPT_IDS: [u32; 3] = [1, 2, 3];
pub const RELEASE_DATE: &str = "2026-09-22";

/// Answer-slot token ids, mirroring the ids measured on a real tokenizer.
pub fn slot_for(letter: char) -> u32 {
    54 + (letter as u32 - 'A' as u32)
}

/// Behaviour switches for the mock upstream.
#[derive(Clone, Copy)]
pub struct UpstreamConfig {
    pub letters: usize,
    pub single_token: bool,
    pub foreign_tokens: bool,
    /// When false, `/tokenize` answers 404, which proves a bridge configured
    /// with a local tokenizer never calls it.
    pub tokenize: bool,
    /// How the mock answers a batched (array-`prompt`) request.
    pub batch: BatchBehaviour,
    /// When nonzero, a request that asks for fewer than this many candidates
    /// drops the last answer letter, the way an unrestricted distribution can
    /// rank a letter below the requested top-k. A deepened retry then sees
    /// every letter, which is how the bridge's retry ladder gets exercised.
    pub deep_enough: usize,
    /// Score binary (yes/no) prompts instead of answer letters.
    pub binary: bool,
}

/// How the mock answers a request that carries an array `prompt`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BatchBehaviour {
    /// One choice per prompt, `index` in order: the happy path.
    Supported,
    /// Reject the array shape with 400, the way llama.cpp build b9590 did.
    ShapeRejected,
    /// Answer with one choice fewer than asked; the all-or-nothing check must
    /// trip rather than mis-align prompts and probabilities.
    ShortByOne,
    /// Answer with every `index` set to 0; the permutation check must trip.
    DuplicateIndex,
    /// Answer 429; the bridge must propagate rather than fall back, or a
    /// throttled endpoint would be laundered into a "successful" run.
    Throttled,
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            letters: 16,
            single_token: true,
            foreign_tokens: true,
            tokenize: true,
            batch: BatchBehaviour::Supported,
            deep_enough: 0,
            binary: false,
        }
    }
}

/// A tiny but valid `tokenizer.json` whose answer letters are single tokens.
///
/// The ids match the ones the mock upstream publishes (A=54 … P=69) so the two
/// halves of a test agree without pretending to be a real model vocabulary.
pub fn minimal_tokenizer_json() -> Vec<u8> {
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".to_string(), json!(0));
    for (index, letter) in LETTERS.chars().enumerate() {
        vocab.insert(letter.to_string(), json!(54 + index));
    }
    for (index, byte) in (32u8..127).enumerate() {
        let character = (byte as char).to_string();
        if !vocab.contains_key(&character) {
            vocab.insert(character, json!(200 + index));
        }
    }
    serde_json::to_vec(&json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        // Split on every character so appending a letter cannot change the
        // tokenization of what precedes it — the property the bridge checks.
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"Regex": "[\\s\\S]"},
            "behavior": "Isolated",
            "invert": false
        },        "post_processor": null,
        // Any decoder makes tokenizers join tokens without a separator; the
        // replacement itself is a no-op for single-character tokens.
        "decoder": {"type": "Replace", "pattern": {"String": " "}, "content": ""},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    }))
    .unwrap()
}

async fn apply_template(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    if state.config.binary {
        // Echo the messages so a binary prompt stays recognisable: the fixed
        // prompt below would hide which candidate the request asks about.
        let text = body
            .get("messages")
            .and_then(Value::as_array)
            .map(|messages| {
                messages
                    .iter()
                    .filter_map(|message| message.get("content").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        return Json(json!({"prompt": text}));
    }
    Json(json!({"prompt": MOCK_PROMPT}))
}

async fn tokenize(
    State(state): State<MockState>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let config = state.config;
    if !config.tokenize {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({"error": "this mock has no tokenize endpoint"})),
        )
            .into_response();
    }
    if !config.single_token {
        return Json(json!({"tokens": [1, 2]})).into_response();
    }
    let content = body.get("content").and_then(Value::as_str).unwrap_or_default();
    if config.binary && matches!(content, "yes" | " yes" | "no" | " no") {
        let id = if content.trim_start() == "yes" { BINARY_YES_ID } else { BINARY_NO_ID };
        return Json(json!({"tokens": [id]})).into_response();
    }
    let tokens: Vec<u32> = match content.strip_prefix(MOCK_PROMPT) {
        Some("") => MOCK_PROMPT_IDS.to_vec(),
        Some(tail) => {
            let mut ids = MOCK_PROMPT_IDS.to_vec();
            ids.push(slot_for(tail.chars().next().unwrap()));
            ids
        }
        None => content
            .chars()
            .map(|character| {
                if LETTERS.contains(character) {
                    slot_for(character)
                } else {
                    // Any stable id will do; these tests never assert on it.
                    300 + (character as u32 % 100)
                }
            })
            .collect(),
    };
    Json(json!({"tokens": tokens})).into_response()
}

async fn detokenize() -> Json<Value> {
    Json(json!({"content": LETTERS}))
}

/// Answer-letter candidates, ranked so A wins and P loses.
///
/// Foreign tokens are included by default because a real unrestricted
/// distribution always carries ordinary vocabulary; without them the bridge's
/// restriction check could not tell a restricted response from an unrestricted
/// one.
/// Token id the mock scores `yes` with.
pub const BINARY_YES_ID: u32 = 100;
/// Token id the mock scores `no` with.
pub const BINARY_NO_ID: u32 = 101;

fn candidates(config: UpstreamConfig, requested: usize, prompt: &str) -> Value {
    // The transport probe still asks with answer letters, so only a binary
    // prompt — the one that asks whether a candidate matches — is scored with
    // yes/no.
    if config.binary && prompt.contains("Does this candidate match") {
        // The prompt names one candidate; the one it names leans yes and any
        // other leans no, so a test can tell an independent readout from a
        // positional one.
        let (yes, no) = if prompt.contains("alpha") {
            (-0.2, -3.0)
        } else {
            (-3.0, -0.2)
        };
        return Value::Array(vec![
            json!({"id": BINARY_YES_ID, "token": "yes", "logprob": yes}),
            json!({"id": BINARY_NO_ID, "token": "no", "logprob": no}),
            json!({"id": 608, "token": "The", "logprob": -20.0}),
        ]);
    }
    // A shallow request can miss the last letter, so a test can prove that a
    // missing slot deepens the retry instead of failing the decision.
    let letters = if config.deep_enough > 0 && requested < config.deep_enough {
        config.letters.saturating_sub(1)
    } else {
        config.letters
    };
    let mut entries: Vec<Value> = (0..letters)
        .map(|index| {
            let letter = LETTERS.chars().nth(index).unwrap();
            json!({
                "id": slot_for(letter),
                "token": letter.to_string(),
                "logprob": -(index as f64 + 1.0),
            })
        })
        .collect();
    if config.foreign_tokens {
        entries.push(json!({"id": 608, "token": "The", "logprob": -20.0}));
        entries.push(json!({"id": 8, "token": "<think>", "logprob": -21.0}));
    }
    Value::Array(entries)
}

/// The mock's shared state: the behaviour switches plus a call counter, so a
/// test can assert how many upstream calls a request actually made.
#[derive(Clone)]
struct MockState {
    config: UpstreamConfig,
    completion_calls: Arc<AtomicUsize>,
}

async fn completions(
    State(state): State<MockState>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    state.completion_calls.fetch_add(1, Ordering::Relaxed);
    let config = state.config;
    // How many candidates the caller asked for; a shallow request may miss an
    // answer letter, which is what the retry ladder exists for.
    let requested = body
        .get("logprobs")
        .or_else(|| body.get("n_probs"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;

    // A single prompt arrives as a string; a batched readout as an array.
    let prompts: Vec<String> = match body.get("prompt") {
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .expect("a batched prompt must hold strings")
                    .to_string()
            })
            .collect(),
        other => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": format!("prompt must be a string or an array, got {other:?}")}})),
            )
                .into_response()
        }
    };
    let batched = prompts.len() > 1;

    if batched {
        match config.batch {
            BatchBehaviour::ShapeRejected => {
                // The shape llama.cpp build b9590 answered to an array prompt.
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    Json(json!({"error": {"message": "type must be string, but is an array"}})),
                )
                    .into_response();
            }
            BatchBehaviour::Throttled => {
                return (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    Json(json!({"error": {"message": "rate limited"}})),
                )
                    .into_response();
            }
            BatchBehaviour::Supported | BatchBehaviour::ShortByOne | BatchBehaviour::DuplicateIndex => {}
        }
    }

    let mut choices: Vec<Value> = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let reported_index = if batched && config.batch == BatchBehaviour::DuplicateIndex {
                0
            } else {
                index
            };
            json!({
                "index": reported_index,
                "finish_reason": "length",
                "logprobs": {"content": [{
                    "id": slot_for('A'),
                    "token": "A",
                    "logprob": -1.0,
                    "top_logprobs": candidates(config, requested, prompt),
                }]},
            })
        })
        .collect();
    if batched && config.batch == BatchBehaviour::ShortByOne {
        choices.pop();
    }

    // One usage block for the whole request, counted per prompt the way the
    // reference endpoint counts it. The bridge attaches it to the first
    // readout, so the vector sums to exactly this number.
    let prompt_tokens = 42 * prompts.len() as u64;
    Json(json!({
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": prompts.len(),
            "total_tokens": prompt_tokens + prompts.len() as u64,
        },
    }))
    .into_response()
}

/// Run a mock llama.cpp-shaped upstream, returning its base URL.
pub async fn spawn_upstream(config: UpstreamConfig) -> String {
    spawn_upstream_counting(config).await.0
}

/// Like [`spawn_upstream`], and also hands back the number of
/// `/v1/completions` calls the mock has served, so a test can prove a request
/// was read in one batch (1) or fell back to sequential calls (N).
pub async fn spawn_upstream_counting(config: UpstreamConfig) -> (String, Arc<AtomicUsize>) {
    let completion_calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route("/apply-template", post(apply_template))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/completions", post(completions))
        .route("/completion", post(completions))
        .with_state(MockState {
            config,
            completion_calls: completion_calls.clone(),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), completion_calls)
}

pub fn config_for(upstream: &str) -> BridgeConfig {
    BridgeConfig {
        openai_base: format!("{upstream}/v1"),
        native_base: upstream.to_string(),
        upstream_model: "mock-model".to_string(),
        upstream_key: None,
        served_model: "bridge-mock".to_string(),
        description: "mock bridge".to_string(),
        release_date: RELEASE_DATE.to_string(),
        chat_template_kwargs: json!({"enable_thinking": false}),
        max_input_tokens: None,
        local: Default::default(),
        accept_any_model: false,
        scoring: Default::default(),
        confidence: Default::default(),
        readout: Default::default(),
    }
}

pub async fn connect(upstream: &str) -> anyhow::Result<Bridge> {
    Bridge::connect(reqwest::Client::new(), config_for(upstream)).await
}

/// Connect with the binary candidate contract.
pub async fn connect_binary(upstream: &str) -> anyhow::Result<Bridge> {
    let mut config = config_for(upstream);
    config.scoring = jev_bridge::server::Scoring::Binary;
    Bridge::connect(reqwest::Client::new(), config).await
}

/// Connect with in-process components, for the local-rendering and
/// local-tokenization paths.
pub async fn connect_with(
    upstream: &str,
    local: jev_bridge::prompt::LocalComponents,
) -> anyhow::Result<Bridge> {
    let mut config = config_for(upstream);
    config.local = local;
    Bridge::connect(reqwest::Client::new(), config).await
}

/// Connect with an explicitly declared readout status.
pub async fn connect_with_readout(
    upstream: &str,
    readout: jev_bridge::wire::ReadoutStatus,
) -> anyhow::Result<Bridge> {
    let mut config = config_for(upstream);
    config.readout = readout;
    Bridge::connect(reqwest::Client::new(), config).await
}

/// Connect with `accept_any_model` on, for third-party clients that hard-code
/// a model id the bridge is not.
pub async fn connect_accepting_any_model(upstream: &str) -> anyhow::Result<Bridge> {
    let mut config = config_for(upstream);
    config.accept_any_model = true;
    Bridge::connect(reqwest::Client::new(), config).await
}

/// A bridge served over HTTP, for contract tests that speak the wire format.
pub struct RunningBridge {
    pub url: String,
    client: reqwest::Client,
}

impl RunningBridge {
    pub async fn start(upstream: &str, api_key: Option<&str>) -> Self {
        let bridge = connect(upstream).await.expect("the bridge should connect");
        Self::serve(bridge, api_key).await
    }

    /// Serve a bridge running the binary candidate contract.
    pub async fn start_binary(upstream: &str, api_key: Option<&str>) -> Self {
        let bridge = connect_binary(upstream).await.expect("the bridge should connect");
        Self::serve(bridge, api_key).await
    }

    pub async fn serve(bridge: Bridge, api_key: Option<&str>) -> Self {
        Self::serve_with_check(bridge, api_key, None).await
    }

    /// Serve with a startup readout self-check attached, as `main` does.
    pub async fn serve_with_check(
        bridge: Bridge,
        api_key: Option<&str>,
        readout_check: Option<jev_bridge::readout::ReadoutCheck>,
    ) -> Self {
        let state = Arc::new(AppState {
            bridge,
            api_key: api_key.map(str::to_string),
            readout_check,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router(state)).await.unwrap();
        });
        Self {
            url: format!("http://{address}"),
            client: reqwest::Client::new(),
        }
    }

    pub async fn get(&self, path: &str, token: Option<&str>) -> (u16, Value) {
        let mut request = self.client.get(format!("{}{path}", self.url));
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        read(request.send().await.expect("request should complete")).await
    }

    pub async fn post(&self, path: &str, payload: &Value, token: Option<&str>) -> (u16, Value) {
        let mut request = self.client.post(format!("{}{path}", self.url)).json(payload);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        read(request.send().await.expect("request should complete")).await
    }
}

async fn read(response: reqwest::Response) -> (u16, Value) {
    let status = response.status().as_u16();
    let body = response.json().await.unwrap_or(Value::Null);
    (status, body)
}

// --------------------------------------------------------------------------
// request payloads
// --------------------------------------------------------------------------

pub fn systemone_payload(state: &str, questions: Value, model: &str) -> Value {
    json!({"model": model, "state": state, "questions": questions})
}

pub fn two_question_payload() -> Value {
    json!({
        "department": {
            "type": "choice",
            "instructions": "Which queue?",
            "criteria": {"billing": "Refunds.", "sales": "Pricing."},
        },
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    })
}

pub fn three_kind_payload() -> Value {
    json!({
        "department": {
            "type": "choice",
            "instructions": "Which queue?",
            "criteria": {"billing": "Refunds.", "sales": "Pricing."},
        },
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "severity": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["Minor", "Degraded", "Blocking"],
        },
    })
}

pub fn sixteen_option_payload() -> Value {
    let criteria: serde_json::Map<String, Value> = (0..16)
        .map(|index| (format!("opt{index:02}"), json!(format!("Bucket {index}"))))
        .collect();
    json!({
        "route": {
            "type": "choice",
            "instructions": "Which bucket fits best?",
            "criteria": criteria,
        }
    })
}
