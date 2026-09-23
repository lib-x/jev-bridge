//! Latency measurements against a real upstream, driven through the bridge
//! library.
//!
//! Ignored by default: these need a reachable upstream and they take minutes.
//! Everything (endpoint, model, key) comes from the environment; no credential
//! is ever written to disk.
//!
//! ```bash
//! JEV_BRIDGE_UPSTREAM_URL=https://host/v1 \
//! JEV_BRIDGE_MODEL=my-model \
//! JEV_BRIDGE_UPSTREAM_KEY=... \
//! cargo test --release --test latency -- --ignored --nocapture
//! ```
//!
//! The measurement separates two costs that a single average hides:
//!
//! * **cache-warm** — the same prompt again, which a prefix-caching server
//!   answers without a fresh prefill; and
//! * **fresh prompt** — every decision carries a different prompt, so the
//!   prefill is paid again.
//!
//! It also measures what a multi-question System One request costs end to end,
//! since the bridge scores its rows one after another (the upstream call is
//! serialized by design), and reports the marginal cost of each added question.

use std::time::Instant;

use jev_bridge::server::{Bridge, BridgeConfig};
use jev_bridge::wire::{Question, Row, RowOption};
use serde_json::json;

const STATE: &str = "My payouts have been failing for 3 days. I was charged twice and need a refund today.";

/// How many repeats each arm runs. Enough for a p95 to mean something, small
/// enough that a shared GPU is not held for long.
const REPEATS: usize = 6;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn required_env(name: &str) -> String {
    env(name).unwrap_or_else(|| panic!("{name} must be set to run this measurement"))
}

/// The `p`-th percentile of already-sorted milliseconds (nearest rank).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[index]
}

fn report(label: &str, samples: &[f64]) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    println!(
        "{label:<34} n={:<2} min={:>6.0}  p50={:>6.0}  p95={:>6.0}  max={:>6.0}  (ms)",
        sorted.len(),
        sorted[0],
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.95),
        sorted[sorted.len() - 1],
    );
}

fn queue_options() -> Vec<RowOption> {
    vec![
        RowOption { id: "billing".into(), description: "Billing, payments, and refunds.".into() },
        RowOption { id: "technical".into(), description: "Bugs, outages, and integrations.".into() },
        RowOption { id: "sales".into(), description: "Pricing, plans, and contracts.".into() },
    ]
}

fn level_options() -> Vec<RowOption> {
    ["Calm", "Mildly annoyed", "Frustrated", "Very angry"]
        .iter()
        .enumerate()
        .map(|(index, text)| RowOption { id: index.to_string(), description: (*text).into() })
        .collect()
}

fn urgent_options() -> Vec<RowOption> {
    vec![
        RowOption { id: "true".into(), description: "The answer is yes.".into() },
        RowOption { id: "false".into(), description: "The answer is no.".into() },
    ]
}

fn row(id: &str, question: &str, options: Vec<RowOption>) -> Row {
    Row {
        id: id.to_string(),
        state: json!(STATE),
        question: question.to_string(),
        options,
    }
}

const QUEUE_QUESTION: &str = "Which queue should handle this request?";
const URGENT_QUESTION: &str = "Does this convey urgency?";
const LEVEL_QUESTION: &str = "How frustrated is the customer?";
const CHURN_QUESTION: &str = "Might this customer churn?";

/// One distinct row per index, cycling through the four questions.
fn distinct_rows(count: usize) -> Vec<Row> {
    let questions = [
        (QUEUE_QUESTION, queue_options()),
        (URGENT_QUESTION, urgent_options()),
        (LEVEL_QUESTION, level_options()),
        (CHURN_QUESTION, urgent_options()),
    ];
    (0..count)
        .map(|index| {
            let (question, options) = &questions[index % questions.len()];
            row(&format!("q{index}"), question, options.clone())
        })
        .collect()
}

/// `count` rows that render to the *same* prompt (the id never enters it).
fn identical_rows(count: usize) -> Vec<Row> {
    (0..count)
        .map(|index| row(&format!("same{index}"), QUEUE_QUESTION, queue_options()))
        .collect()
}

/// Score one request and return (elapsed ms, total prompt tokens).
///
/// Every row in this suite shares one state (the System One shape), so the
/// rows are scored as **one batch** — which is what the measurement is about.
/// Every answer is checked for a probability vector that sums to one: a
/// latency number for a wrong answer would be worthless.
async fn timed_score(bridge: &Bridge, rows: &[Row]) -> (f64, u64) {
    let state = rows[0].state.clone();
    let questions: Vec<Question> = rows
        .iter()
        .map(|row| Question {
            id: row.id.clone(),
            question: row.question.clone(),
            options: row.options.clone(),
        })
        .collect();
    let started = Instant::now();
    let scored = bridge
        .score_questions(&state, &questions)
        .await
        .expect("scoring must succeed against a reachable upstream");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(scored.len(), rows.len(), "one score per row");

    let mut tokens = 0;
    for score in &scored {
        let sum: f64 = score.answer.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "probabilities must sum to 1, got {sum}");
        tokens += score.answer.input_tokens;
    }
    (elapsed_ms, tokens)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a reachable upstream (JEV_BRIDGE_UPSTREAM_URL / JEV_BRIDGE_MODEL) and takes minutes"]
async fn measure_decision_latency() {
    let base_url = required_env("JEV_BRIDGE_UPSTREAM_URL");
    let model = required_env("JEV_BRIDGE_MODEL");
    let upstream_key = env("JEV_BRIDGE_UPSTREAM_KEY");
    let native_base = base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .to_string();

    let bridge = Bridge::connect(
        reqwest::Client::new(),
        BridgeConfig {
            openai_base: base_url.clone(),
            native_base,
            upstream_model: model.clone(),
            upstream_key,
            served_model: "latency-probe".to_string(),
            description: "latency measurement".to_string(),
            release_date: "2026-09-23".to_string(),
            chat_template_kwargs: json!({"enable_thinking": false}),
            max_input_tokens: None,
            local: Default::default(),
            accept_any_model: false,
            scoring: Default::default(),
            confidence: Default::default(),
            readout: Default::default(),
        },
    )
    .await
    .expect("the bridge must negotiate a transport with the upstream");

    println!();
    println!("upstream: {base_url}");
    println!("model:    {model}");
    println!("transport: {} via {}", bridge.probe().name(), bridge.probe().endpoint());

    // Warm-up: first call pays whatever cold path the upstream has.
    let (warm_ms, warm_tokens) = timed_score(&bridge, &identical_rows(1)).await;
    println!("warm-up: {warm_ms:.0} ms ({warm_tokens} prompt tokens)");

    // 1) The same prompt again and again: a prefix-caching server can answer
    //    from cache, so this is the floor, not the typical case.
    let mut warm = Vec::new();
    for _ in 0..REPEATS {
        let (ms, _) = timed_score(&bridge, &identical_rows(1)).await;
        warm.push(ms);
    }
    report("1 question, same prompt (warm)", &warm);

    // 2) A fresh prompt each time: every decision pays its own prefill.
    let mut fresh = Vec::new();
    for index in 0..REPEATS {
        // Cycle the question so consecutive prompts differ.
        let rows = vec![distinct_rows(index + 1).remove(index % 4)];
        let (ms, _) = timed_score(&bridge, &rows).await;
        fresh.push(ms);
    }
    report("1 question, fresh prompt each time", &fresh);

    // 3) Three *identical* prompts inside one System One call: three upstream
    //    calls, all cache-warm after the first.
    let mut identical_batch = Vec::new();
    for _ in 0..REPEATS {
        let (ms, _) = timed_score(&bridge, &identical_rows(3)).await;
        identical_batch.push(ms);
    }
    report("3 questions, identical prompts", &identical_batch);

    // 4) Three *distinct* prompts inside one call: the shape a real System One
    //    request has (one state, several questions).
    let mut distinct_batch = Vec::new();
    for _ in 0..REPEATS {
        let (ms, _) = timed_score(&bridge, &distinct_rows(3)).await;
        distinct_batch.push(ms);
    }
    report("3 questions, distinct prompts", &distinct_batch);

    // 5) Marginal cost: what the 2nd, 3rd and 4th question add.
    println!();
    println!("marginal cost of each added question (one call per size):");
    let mut previous = 0.0;
    for count in 1..=4 {
        let (ms, tokens) = timed_score(&bridge, &distinct_rows(count)).await;
        println!(
            "  {count} question(s): {ms:>7.0} ms total, {tokens:>4} prompt tokens, \
             marginal {:>+7.0} ms",
            if count == 1 { 0.0 } else { ms - previous }
        );
        previous = ms;
    }

    // Sanity floors, not performance claims: a shared or CPU-bound upstream is
    // legitimately slow, and this suite must not fail for that reason. The
    // numbers above are the evidence; these assertions only catch a hang.
    for (label, samples) in [
        ("same prompt", &warm),
        ("fresh prompt", &fresh),
        ("identical batch", &identical_batch),
        ("distinct batch", &distinct_batch),
    ] {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p50 = percentile(&sorted, 0.5);
        assert!(
            p50 < 120_000.0,
            "{label}: p50 {p50:.0} ms exceeds the 120 s sanity floor"
        );
    }
}
