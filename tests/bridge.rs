//! End-to-end tests against a mock upstream that speaks the llama.cpp dialect.
//!
//! The mock reproduces the shapes measured on llama.cpp b11096: `/apply-template`
//! renders the prompt, `/tokenize` reports single-token letters, and
//! `/v1/completions` answers with chat-shaped `logprobs.content[0].top_logprobs`.

use axum::{extract::Json, routing::post, Router};
use serde_json::{json, Value};

use jev_bridge::prompt::LETTERS;
use jev_bridge::server::{Bridge, BridgeConfig};
use jev_bridge::strategy::Probe;
use jev_bridge::wire::{request_rows, response_from_results, Row, RowOption};

const MOCK_PROMPT: &str = "<|im_start|>system\nmock<|im_end|>\n<|im_start|>assistant\n";
const MOCK_PROMPT_IDS: [u32; 3] = [1, 2, 3];

fn slot_for(letter: char) -> u32 {
    54 + (letter as u32 - 'A' as u32)
}

async fn mock_apply_template(Json(body): Json<Value>) -> Json<Value> {
    assert!(body["messages"].is_array(), "apply-template needs messages");
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    Json(json!({"prompt": MOCK_PROMPT}))
}

async fn mock_tokenize(Json(body): Json<Value>) -> Json<Value> {
    let content = body["content"].as_str().unwrap_or_default();
    let mut tokens: Vec<u32> = MOCK_PROMPT_IDS.to_vec();
    match content.strip_prefix(MOCK_PROMPT) {
        Some("") => {}
        Some(tail) => tokens.push(slot_for(tail.chars().next().unwrap())),
        None => tokens = content.chars().map(slot_for).collect(),
    }
    Json(json!({"tokens": tokens}))
}

/// A tokenizer that never yields single-token letters.
async fn mock_tokenize_always_two() -> Json<Value> {
    Json(json!({"tokens": [1, 2]}))
}

async fn mock_detokenize() -> Json<Value> {
    Json(json!({"content": LETTERS}))
}

/// Every letter candidate, ranked so A wins and P loses.
///
/// Two foreign tokens are included because a real unrestricted distribution
/// always carries ordinary vocabulary; without them the restriction check
/// could not tell a restricted response from an unrestricted one.
fn letter_candidates() -> Value {
    let mut candidates: Vec<Value> = LETTERS
        .chars()
        .enumerate()
        .map(|(index, letter)| {
            json!({
                "id": slot_for(letter),
                "token": letter.to_string(),
                "logprob": -(index as f64 + 1.0),
            })
        })
        .collect();
    candidates.push(json!({"id": 608, "token": "The", "logprob": -20.0}));
    candidates.push(json!({"id": 8, "token": "<think>", "logprob": -21.0}));
    Value::Array(candidates)
}

fn completions_response(candidates: Value) -> Value {
    json!({
        "choices": [{
            "finish_reason": "length",
            "logprobs": {"content": [{
                "id": 54,
                "token": "A",
                "logprob": -1.0,
                "top_logprobs": candidates,
            }]},
        }],
        "usage": {"prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43},
    })
}

async fn mock_completions(Json(body): Json<Value>) -> Json<Value> {
    assert!(body["prompt"].is_string(), "completions needs a prompt");
    Json(completions_response(letter_candidates()))
}

/// Only the first four letters, so wider decisions cannot be scored.
async fn mock_completions_truncated() -> Json<Value> {
    let candidates = Value::Array(
        letter_candidates()
            .as_array()
            .unwrap()
            .iter()
            .take(4)
            .cloned()
            .collect(),
    );
    Json(completions_response(candidates))
}

async fn serve(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{address}")
}

fn config_for(base: &str) -> BridgeConfig {
    BridgeConfig {
        openai_base: format!("{base}/v1"),
        native_base: base.to_string(),
        upstream_model: "mock-model".to_string(),
        upstream_key: None,
        served_model: "bridge-mock".to_string(),
        description: "mock".to_string(),
        release_date: "2026-09-22".to_string(),
        chat_template_kwargs: json!({"enable_thinking": false}),
        max_input_tokens: None,
    }
}

async fn connect_mock() -> Bridge {
    let base = serve(
        Router::new()
            .route("/apply-template", post(mock_apply_template))
            .route("/tokenize", post(mock_tokenize))
            .route("/detokenize", post(mock_detokenize))
            .route("/v1/completions", post(mock_completions))
            .route("/completion", post(mock_completions)),
    )
    .await;
    Bridge::connect(reqwest::Client::new(), config_for(&base))
        .await
        .expect("the bridge should connect to the mock upstream")
}

fn row_with_options(count: usize) -> Row {
    Row {
        id: "row".to_string(),
        state: json!("I was charged twice."),
        question: "Which queue?".to_string(),
        options: (0..count)
            .map(|index| RowOption {
                id: format!("opt{index}"),
                description: format!("candidate bucket {index}"),
            })
            .collect(),
    }
}

#[tokio::test]
async fn probe_picks_the_completions_path_and_resolves_all_sixteen_slots() {
    let bridge = connect_mock().await;
    assert_eq!(bridge.probe(), Probe::CompletionsTopK);
    let expected: Vec<u32> = (54..70).collect();
    assert_eq!(bridge.slots(), expected.as_slice());
}

#[tokio::test]
async fn systemone_request_scores_without_generating_text() {
    let bridge = connect_mock().await;
    let payload = json!({
        "model": "bridge-mock",
        "state": "I was charged twice.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"billing": "Refunds.", "sales": "Pricing."},
            },
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        },
    });
    let (specs, rows) = request_rows(&payload, "bridge-mock").unwrap();
    let results = bridge.score_rows(&rows).await.unwrap();
    let answers: Vec<_> = results.into_iter().map(|scored| scored.answer).collect();
    let response = response_from_results("bridge-mock", &specs, &answers).unwrap();

    // Option A always wins in the mock distribution, so the first criterion wins.
    // The mock ranks logprobs -1, -2, ... so A holds roughly 0.63 of the mass.
    assert_eq!(response["answers"]["department"]["choice"], "billing");
    let billing = response["answers"]["department"]["probabilities"]["billing"]
        .as_f64()
        .unwrap();
    let sales = response["answers"]["department"]["probabilities"]["sales"]
        .as_f64()
        .unwrap();
    assert!(billing > 0.6 && billing > sales, "{billing} vs {sales}");
    assert!(response["answers"]["urgent"]["noul"].as_f64().unwrap() > 0.6);
    assert_eq!(response["usage"]["output_tokens"], 0);
    assert_eq!(response["usage"]["input_tokens"], 84);
    assert_eq!(response["fastjev"]["prompt_versions"][0], "direct-options-v1");
}

#[tokio::test]
async fn a_sixteen_option_decision_is_fully_scored() {
    let bridge = connect_mock().await;
    let results = bridge.score_rows(&[row_with_options(16)]).await.unwrap();
    let probabilities = &results[0].answer.probabilities;
    assert_eq!(probabilities.len(), 16);
    let total: f64 = probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-9);
    assert!(probabilities[0] > probabilities[15]);
}

#[tokio::test]
async fn a_truncated_distribution_is_rejected_at_connect_time() {
    let base = serve(
        Router::new()
            .route("/apply-template", post(mock_apply_template))
            .route("/tokenize", post(mock_tokenize))
            .route("/detokenize", post(mock_detokenize))
            .route("/v1/completions", post(mock_completions_truncated)),
    )
    .await;
    // The sixteen-option probe cannot be scored from four published letters, so
    // the bridge refuses to start rather than serving decisions it cannot read.
    let error = match Bridge::connect(reqwest::Client::new(), config_for(&base)).await {
        Ok(_) => panic!("a truncated distribution must not connect"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("absent from the returned distribution"),
        "{message}"
    );
}

#[tokio::test]
async fn a_non_single_token_alphabet_is_rejected_at_connect_time() {
    let base = serve(
        Router::new()
            .route("/apply-template", post(mock_apply_template))
            .route("/tokenize", post(mock_tokenize_always_two))
            .route("/detokenize", post(mock_detokenize))
            .route("/v1/completions", post(mock_completions)),
    )
    .await;
    let error = match Bridge::connect(reqwest::Client::new(), config_for(&base)).await {
        Ok(_) => panic!("a two-token alphabet must not connect"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("not a single token"), "{error:#}");
}
