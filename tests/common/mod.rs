//! Shared helpers for the Rust contract suite.
//!
//! `contract.rs` drives the HTTP surface the same way `tests/test_contract.py`
//! does from outside the process; `bridge.rs` uses the same mock upstream but
//! calls the library directly.

#![allow(dead_code)]

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
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            letters: 16,
            single_token: true,
            foreign_tokens: true,
            tokenize: true,
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

async fn apply_template() -> Json<Value> {
    Json(json!({"prompt": MOCK_PROMPT}))
}

async fn tokenize(
    State(config): State<UpstreamConfig>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
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
fn candidates(config: UpstreamConfig) -> Value {
    let mut entries: Vec<Value> = (0..config.letters)
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

async fn completions(State(config): State<UpstreamConfig>, Json(body): Json<Value>) -> Json<Value> {
    assert!(
        body.get("prompt").and_then(Value::as_str).is_some(),
        "scoring needs a prompt"
    );
    Json(json!({
        "choices": [{
            "finish_reason": "length",
            "logprobs": {"content": [{
                "id": slot_for('A'),
                "token": "A",
                "logprob": -1.0,
                "top_logprobs": candidates(config),
            }]},
        }],
        "usage": {"prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43},
    }))
}

/// Run a mock llama.cpp-shaped upstream, returning its base URL.
pub async fn spawn_upstream(config: UpstreamConfig) -> String {
    let app = Router::new()
        .route("/apply-template", post(apply_template))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/completions", post(completions))
        .route("/completion", post(completions))
        .with_state(config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{address}")
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
    }
}

pub async fn connect(upstream: &str) -> anyhow::Result<Bridge> {
    Bridge::connect(reqwest::Client::new(), config_for(upstream)).await
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

    pub async fn serve(bridge: Bridge, api_key: Option<&str>) -> Self {
        let state = Arc::new(AppState {
            bridge,
            api_key: api_key.map(str::to_string),
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
