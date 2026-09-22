//! The Rust half of the contract suite.
//!
//! Same case list as `tests/test_contract.py`, but driven over HTTP from inside
//! the process instead of through the binary. `tests/common/mod.rs` provides the
//! mock upstream both suites use.

mod common;

use serde_json::json;

use common::{
    connect_with, minimal_tokenizer_json,
    connect, sixteen_option_payload, slot_for, spawn_upstream, systemone_payload,
    three_kind_payload, two_question_payload, RunningBridge, UpstreamConfig, LETTERS,
};

// --------------------------------------------------------------------------
// happy paths
// --------------------------------------------------------------------------

#[tokio::test]
async fn health_reports_the_selected_transport() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let (status, health) = bridge.get("/health", None).await;

    assert_eq!(status, 200);
    assert_eq!(health["probe"], "openai-completions-top-k");
    assert_eq!(health["endpoint"], "/completions");
    assert_eq!(health["prompt_version"], "direct-options-v1");
    assert_eq!(health["chat_template_kwargs"]["enable_thinking"], false);
    let slots = health["answer_slots"].as_object().unwrap();
    for letter in LETTERS.chars() {
        assert_eq!(slots[&letter.to_string()], json!(slot_for(letter)));
    }
    assert!(
        health["probability_status"]
            .as_str()
            .unwrap()
            .starts_with("conditional option scores"),
        "{health}"
    );
}

#[tokio::test]
async fn models_endpoint_returns_configured_metadata() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let (status, models) = bridge.get("/v1/models", None).await;

    assert_eq!(status, 200);
    assert_eq!(models["models"][0]["name"], "bridge-mock");
    assert_eq!(models["models"][0]["release_date"], "2026-09-22");
}

#[tokio::test]
async fn choice_noul_and_score_round_trip() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let payload = systemone_payload("I was charged twice.", three_kind_payload(), "bridge-mock");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;

    assert_eq!(status, 200);
    assert_eq!(body["model"], "bridge-mock");
    let answers = &body["answers"];
    // The mock ranks A highest, so the first criterion wins every time.
    assert_eq!(answers["department"]["choice"], "billing");
    assert!(answers["department"]["probabilities"]["billing"].as_f64().unwrap() > 0.6);
    assert!(answers["urgent"]["noul"].as_f64().unwrap() > 0.6);
    assert_eq!(answers["severity"]["legend"]["2"], "Blocking");
    let score = answers["severity"]["score"].as_f64().unwrap();
    assert!((0.0..=2.0).contains(&score), "{score}");
    assert_eq!(body["fastjev"]["confidence_method"], "one-minus-normalized-entropy");
}

#[tokio::test]
async fn usage_reports_zero_output_tokens() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let payload = systemone_payload("I was charged twice.", two_question_payload(), "bridge-mock");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;

    assert_eq!(status, 200);
    // Two questions, each costing one 42-token prompt and no generated token.
    assert_eq!(body["usage"]["input_tokens"], 84);
    assert_eq!(body["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn sixteen_options_are_fully_scored() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let payload = systemone_payload("The customer cannot log in.", sixteen_option_payload(), "bridge-mock");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;

    assert_eq!(status, 200);
    let answer = &body["answers"]["route"];
    let probabilities = answer["probabilities"].as_object().unwrap();
    assert_eq!(probabilities.len(), 16);
    let total: f64 = probabilities.values().map(|value| value.as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-9, "{total}");
    assert_eq!(answer["choice"], "opt00");
}

// --------------------------------------------------------------------------
// request validation
// --------------------------------------------------------------------------

#[tokio::test]
async fn model_mismatch_is_rejected_with_400() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let payload = systemone_payload("x", two_question_payload(), "jev-latest");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;

    assert_eq!(status, 400);
    assert!(
        body["error"].as_str().unwrap().contains("does not serve Jev aliases"),
        "{body}"
    );
}

#[tokio::test]
async fn empty_state_is_rejected_with_400() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let payload = systemone_payload("", two_question_payload(), "bridge-mock");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;

    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().starts_with("state: "), "{body}");
}

#[tokio::test]
async fn bearer_auth_is_enforced_when_configured() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, Some("secret-token")).await;

    assert_eq!(bridge.get("/health", None).await.0, 401);
    assert_eq!(bridge.get("/health", Some("wrong-token")).await.0, 401);
    assert_eq!(bridge.get("/health", Some("secret-token")).await.0, 200);
}

// --------------------------------------------------------------------------
// in-process components
// --------------------------------------------------------------------------

#[cfg(feature = "local-tokenizer")]
#[tokio::test]
async fn a_local_tokenizer_replaces_the_runtime_endpoint() {
    use jev_bridge::prompt::LocalComponents;
    use jev_bridge::tokenizer::LocalTokenizer;

    // This mock publishes no /tokenize at all, so a bridge that still called it
    // could not even resolve its answer slots.
    let upstream = spawn_upstream(UpstreamConfig {
        tokenize: false,
        ..Default::default()
    })
    .await;
    let tokenizer = LocalTokenizer::from_bytes(&minimal_tokenizer_json())
        .expect("the minimal tokenizer should load");
    let bridge = connect_with(
        &upstream,
        LocalComponents {
            tokenizer: Some(tokenizer),
            ..Default::default()
        },
    )
    .await
    .expect("a local tokenizer should make the bridge independent of /tokenize");

    assert!(bridge.tokenizes_locally());
    assert!(!bridge.renders_locally());

    let bridge = RunningBridge::serve(bridge, None).await;
    let payload = systemone_payload("I was charged twice.", two_question_payload(), "bridge-mock");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;

    assert_eq!(status, 200, "{body}");
    assert_eq!(body["answers"]["department"]["choice"], "billing");
    assert_eq!(body["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn a_local_renderer_replaces_the_runtime_template_endpoint() {
    use jev_bridge::prompt::LocalComponents;
    use jev_bridge::render::LocalRenderer;

    // The mock's /apply-template is still registered, but the local renderer
    // must win: its output is what the mock upstream would not have produced.
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let renderer = LocalRenderer::new(
        format!("LOCAL{}", "{{ messages[0].content }}"),
        serde_json::Map::new(),
    )
    .unwrap();
    let bridge = connect_with(
        &upstream,
        LocalComponents {
            renderer: Some(renderer),
            ..Default::default()
        },
    )
    .await
    .expect("a local renderer should connect against the mock");

    assert!(bridge.renders_locally());

    let bridge = RunningBridge::serve(bridge, None).await;
    let (status, health) = bridge.get("/health", None).await;
    assert_eq!(status, 200);
    assert_eq!(health["renders_locally"], true);
    assert_eq!(health["tokenizes_locally"], false);
}

// --------------------------------------------------------------------------
// startup contracts
// --------------------------------------------------------------------------

#[tokio::test]
async fn non_single_token_alphabet_refuses_to_start() {
    let upstream = spawn_upstream(UpstreamConfig {
        single_token: false,
        ..Default::default()
    })
    .await;
    let error = match connect(&upstream).await {
        Ok(_) => panic!("a two-token alphabet must not connect"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("not a single token"), "{error:#}");
}

#[tokio::test]
async fn truncated_distribution_refuses_to_start() {
    let upstream = spawn_upstream(UpstreamConfig {
        letters: 4,
        ..Default::default()
    })
    .await;
    let error = match connect(&upstream).await {
        Ok(_) => panic!("a truncated distribution must not connect"),
        Err(error) => error,
    };
    assert!(
        format!("{error:#}").contains("absent from the returned distribution"),
        "{error:#}"
    );
}
