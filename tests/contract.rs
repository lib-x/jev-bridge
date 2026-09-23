//! The Rust half of the contract suite.
//!
//! Same case list as `tests/test_contract.py`, but driven over HTTP from inside
//! the process instead of through the binary. `tests/common/mod.rs` provides the
//! mock upstream both suites use.

mod common;

use serde_json::json;

use common::{
    connect, connect_accepting_any_model, connect_with, connect_with_readout,
    minimal_tokenizer_json, sixteen_option_payload, slot_for, spawn_upstream,
    spawn_upstream_counting, systemone_payload, three_kind_payload, two_question_payload,
    BatchBehaviour, RunningBridge, UpstreamConfig, LETTERS,
};
use std::sync::atomic::Ordering;

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
async fn accept_any_model_admits_third_party_clients() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;

    // The default refusal names the switch that relaxes it.
    let bridge = RunningBridge::start(&upstream, None).await;
    let payload = systemone_payload("x", two_question_payload(), "jev-latest");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 400);
    assert!(
        body["error"].as_str().unwrap().contains("--accept-any-model"),
        "{body}"
    );

    // With the switch on, a client that hard-codes `jev-latest` (the AI SDK,
    // the djev-run demos) is served — and the response still reports the
    // served model, so the bridge never claims to be the alias.
    let bridge = connect_accepting_any_model(&upstream).await.unwrap();
    let bridge = RunningBridge::serve(bridge, None).await;
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], "bridge-mock");
    assert_eq!(body["answers"]["department"]["choice"], "billing");

    // An empty name is still refused: it names nothing.
    let payload = systemone_payload("x", two_question_payload(), "  ");
    let (status, _body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 400);
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
// readout declaration
// --------------------------------------------------------------------------

#[tokio::test]
async fn the_readout_status_defaults_to_unvalidated() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;

    // Health states the default, with no evidence attached...
    let (status, health) = bridge.get("/health", None).await;
    assert_eq!(status, 200);
    assert_eq!(health["readout"]["status"], "unvalidated");
    assert!(health["readout"].get("evidence").is_none(), "{health}");

    // ...and every answer carries the same declaration.
    let payload = systemone_payload("I was charged twice.", two_question_payload(), "bridge-mock");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200);
    assert_eq!(body["fastjev"]["readout"]["status"], "unvalidated");
    assert!(body["fastjev"]["readout"].get("evidence").is_none(), "{body}");
}

#[tokio::test]
async fn a_declared_readout_validation_travels_with_every_answer() {
    use jev_bridge::wire::ReadoutStatus;

    let evidence = "argmax agreement 139/144 on fastjev authored144";
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = connect_with_readout(
        &upstream,
        ReadoutStatus {
            status: "validated".to_string(),
            evidence: Some(evidence.to_string()),
        },
    )
    .await
    .expect("a declared readout status must not affect connecting");

    let bridge = RunningBridge::serve(bridge, None).await;
    let (status, health) = bridge.get("/health", None).await;
    assert_eq!(status, 200);
    assert_eq!(health["readout"]["status"], "validated");
    assert_eq!(health["readout"]["evidence"], evidence);

    let payload = systemone_payload("I was charged twice.", two_question_payload(), "bridge-mock");
    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200);
    assert_eq!(body["fastjev"]["readout"]["status"], "validated");
    assert_eq!(body["fastjev"]["readout"]["evidence"], evidence);
}

// --------------------------------------------------------------------------
// batched readout
// --------------------------------------------------------------------------

#[tokio::test]
async fn a_multi_question_request_is_read_in_one_batch() {
    let (upstream, calls) = spawn_upstream_counting(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;
    // Connecting probes transports, which spends completions calls of its own.
    let baseline = calls.load(Ordering::Relaxed);
    let payload = systemone_payload("I was charged twice.", three_kind_payload(), "bridge-mock");

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200, "{body}");
    // One batched readout for all three questions, not three sequential calls.
    assert_eq!(calls.load(Ordering::Relaxed) - baseline, 1);
    assert_eq!(body["answers"]["department"]["choice"], "billing");
    assert_eq!(body["answers"]["severity"]["legend"]["2"], "Blocking");
    // The endpoint's single usage block is attached once, so the vector sums
    // to the endpoint's own count.
    assert_eq!(body["usage"]["input_tokens"], 42 * 3);
    let (_, health) = bridge.get("/health", None).await;
    assert_eq!(health["batch_fallbacks"], 0);
}

#[tokio::test]
async fn a_missing_slot_deepens_the_readout_instead_of_failing() {
    // The mock hides the last answer letter until a request asks for 1024
    // candidates: the first attempt (256 for sixteen options) misses it, the
    // deepened retry sees it, and the decision is never turned into an error.
    let (upstream, calls) = spawn_upstream_counting(UpstreamConfig {
        deep_enough: 1024,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let baseline = calls.load(Ordering::Relaxed);
    let payload = systemone_payload("The board is full.", sixteen_option_payload(), "bridge-mock");

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200, "{body}");
    // A shallow readout, then exactly one deepened retry.
    assert_eq!(calls.load(Ordering::Relaxed) - baseline, 2);
    // The letter the shallow attempt hid is scored, never dropped or zeroed.
    assert!(
        body["answers"]["route"]["probabilities"]["opt15"].is_number(),
        "{body}"
    );
}

#[tokio::test]
async fn a_batch_rejected_by_shape_falls_back_to_sequential() {
    let (upstream, calls) = spawn_upstream_counting(UpstreamConfig {
        batch: BatchBehaviour::ShapeRejected,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let baseline = calls.load(Ordering::Relaxed);
    let payload = systemone_payload("I was charged twice.", three_kind_payload(), "bridge-mock");

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200, "{body}");
    // The batched shape was rejected once, then three sequential reads.
    assert_eq!(calls.load(Ordering::Relaxed) - baseline, 4);
    assert_eq!(body["answers"]["department"]["choice"], "billing");
    // Each sequential read reports its own prompt tokens again.
    assert_eq!(body["usage"]["input_tokens"], 42 * 3);

    // The fallback is counted and visible, never hidden: a run served this way
    // did not get the shared prefill.
    let (status, health) = bridge.get("/health", None).await;
    assert_eq!(status, 200);
    assert_eq!(health["batch_fallbacks"], 1);
}

#[tokio::test]
async fn a_throttled_batch_propagates_instead_of_falling_back() {
    let (upstream, calls) = spawn_upstream_counting(UpstreamConfig {
        batch: BatchBehaviour::Throttled,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let baseline = calls.load(Ordering::Relaxed);
    let payload = systemone_payload("I was charged twice.", three_kind_payload(), "bridge-mock");

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    // 429 is about the call, not the shape: falling back would turn a
    // throttled endpoint into a "successful" run.
    assert_eq!(status, 502, "{body}");
    assert_eq!(calls.load(Ordering::Relaxed) - baseline, 1);
    assert!(body["error"].as_str().unwrap().contains("429"), "{body}");
}

#[tokio::test]
async fn a_short_batch_fails_loudly() {
    let (upstream, calls) = spawn_upstream_counting(UpstreamConfig {
        batch: BatchBehaviour::ShortByOne,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let baseline = calls.load(Ordering::Relaxed);
    let payload = systemone_payload("I was charged twice.", three_kind_payload(), "bridge-mock");

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 502, "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("all-or-nothing"),
        "{body}"
    );
    // A short answer is a data-integrity failure, not a shape rejection:
    // re-asking one prompt at a time cannot fix it.
    assert_eq!(calls.load(Ordering::Relaxed) - baseline, 1);
}

#[tokio::test]
async fn a_duplicate_index_batch_fails_loudly() {
    let (upstream, _calls) = spawn_upstream_counting(UpstreamConfig {
        batch: BatchBehaviour::DuplicateIndex,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let payload = systemone_payload("I was charged twice.", three_kind_payload(), "bridge-mock");

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 502, "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("permutation"),
        "{body}"
    );
}

#[tokio::test]
async fn a_single_question_request_never_uses_the_array_shape() {
    // One question is one prompt, so even an endpoint that rejects arrays
    // answers it — and no fallback is counted.
    let (upstream, calls) = spawn_upstream_counting(UpstreamConfig {
        batch: BatchBehaviour::ShapeRejected,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start(&upstream, None).await;
    let baseline = calls.load(Ordering::Relaxed);
    let payload = systemone_payload(
        "I was charged twice.",
        json!({
            "department": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"billing": "Refunds.", "sales": "Pricing."},
            }
        }),
        "bridge-mock",
    );

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(calls.load(Ordering::Relaxed) - baseline, 1);
    let (_, health) = bridge.get("/health", None).await;
    assert_eq!(health["batch_fallbacks"], 0);
}

// --------------------------------------------------------------------------
// readout self-check
// --------------------------------------------------------------------------

#[tokio::test]
async fn the_readout_self_check_reports_what_the_model_named() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = connect(&upstream).await.unwrap();

    let check = bridge
        .run_readout_check()
        .await
        .expect("the self-check must run against the mock");

    // The mock always ranks A highest, so exactly the two probes whose
    // expected option is the first one pass — and the report says which two
    // did not, rather than collapsing to a single number.
    assert_eq!(check.probes, 4);
    assert_eq!(check.passed, 2);
    assert!(!check.is_ok());
    let failed: Vec<&str> = check
        .results
        .iter()
        .filter(|result| !result.passed)
        .map(|result| result.id.as_str())
        .collect();
    assert_eq!(failed, vec!["ice-not-hot", "summer-day"]);

    // The report travels to /health, so an operator can see it.
    let bridge = RunningBridge::serve_with_check(bridge, None, Some(check)).await;
    let (status, health) = bridge.get("/health", None).await;
    assert_eq!(status, 200);
    assert_eq!(health["readout_check"]["probes"], 4);
    assert_eq!(health["readout_check"]["passed"], 2);
    assert_eq!(
        health["readout_check"]["results"].as_array().unwrap().len(),
        4
    );
    assert_eq!(health["readout_check"]["results"][0]["id"], "sky-is-blue");
    assert_eq!(health["readout_check"]["results"][0]["passed"], true);
}

#[tokio::test]
async fn health_omits_the_self_check_when_none_ran() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = RunningBridge::start(&upstream, None).await;

    let (status, health) = bridge.get("/health", None).await;
    assert_eq!(status, 200);
    assert!(health.get("readout_check").is_none(), "{health}");
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

#[tokio::test]
async fn the_binary_contract_judges_every_candidate_in_its_own_prompt() {
    let upstream = spawn_upstream(UpstreamConfig {
        binary: true,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start_binary(&upstream, None).await;
    let payload = systemone_payload(
        "The customer was charged twice.",
        json!({
            "route": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": {"alpha": "charges and refunds", "beta": "new purchases"},
            }
        }),
        "bridge-mock",
    );

    let (status, body) = bridge.post("/v1/systemone", &payload, None).await;
    assert_eq!(status, 200, "{body}");
    // The prompt that names alpha leans yes, so alpha wins on its own merits.
    assert_eq!(body["answers"]["route"]["choice"], "alpha");
    let probabilities = &body["answers"]["route"]["probabilities"];
    assert!(
        probabilities["alpha"].as_f64().unwrap() > probabilities["beta"].as_f64().unwrap(),
        "{body}"
    );
    // The response declares the binary contract it was scored under.
    assert_eq!(body["fastjev"]["prompt_versions"][0], "binary-candidates-v1");
}

#[tokio::test]
async fn reordering_options_does_not_move_the_binary_answer() {
    let upstream = spawn_upstream(UpstreamConfig {
        binary: true,
        ..Default::default()
    })
    .await;
    let bridge = RunningBridge::start_binary(&upstream, None).await;
    let question = |criteria: serde_json::Value| {
        systemone_payload(
            "The customer was charged twice.",
            json!({
                "route": {
                    "type": "choice",
                    "instructions": "Which department should handle this request?",
                    "criteria": criteria,
                }
            }),
            "bridge-mock",
        )
    };

    let (_, forward) = bridge
        .post(
            "/v1/systemone",
            &question(json!({"alpha": "charges and refunds", "beta": "new purchases"})),
            None,
        )
        .await;
    let (_, reversed) = bridge
        .post(
            "/v1/systemone",
            &question(json!({"beta": "new purchases", "alpha": "charges and refunds"})),
            None,
        )
        .await;

    // No candidate sees the others, so its probability cannot move when the
    // other candidate changes position. With the letter contract the same swap
    // flips the answer (measured on the reference endpoint).
    for option in ["alpha", "beta"] {
        assert_eq!(
            forward["answers"]["route"]["probabilities"][option],
            reversed["answers"]["route"]["probabilities"][option],
            "{option} moved when the option order changed"
        );
    }
    assert_eq!(
        forward["answers"]["route"]["choice"],
        reversed["answers"]["route"]["choice"]
    );
}
