//! Internal-API integration tests.
//!
//! `contract.rs` covers the HTTP surface the same way `test_contract.py` does
//! from outside. This file calls the library directly, which is the only way to
//! assert on `DetailedScore` fields such as the prompt hash that the wire
//! format deliberately does not expose.

mod common;

use serde_json::json;

use common::{connect, slot_for, spawn_upstream, UpstreamConfig, LETTERS};
use jev_bridge::strategy::Probe;
use jev_bridge::wire::{Row, RowOption};

fn two_option_row() -> Row {
    Row {
        id: "row".to_string(),
        state: json!("I was charged twice."),
        question: "Which queue?".to_string(),
        options: vec![
            RowOption {
                id: "billing".to_string(),
                description: "Refunds.".to_string(),
            },
            RowOption {
                id: "sales".to_string(),
                description: "Pricing.".to_string(),
            },
        ],
    }
}

fn sixteen_option_row() -> Row {
    Row {
        id: "row".to_string(),
        state: json!("The customer cannot log in."),
        question: "Which bucket fits best?".to_string(),
        options: (0..16)
            .map(|index| RowOption {
                id: format!("opt{index:02}"),
                description: format!("Bucket {index}"),
            })
            .collect(),
    }
}

#[tokio::test]
async fn probe_picks_the_completions_path_and_resolves_all_sixteen_slots() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = connect(&upstream).await.unwrap();

    assert_eq!(bridge.probe(), Probe::CompletionsTopK);
    let expected: Vec<u32> = LETTERS.chars().map(slot_for).collect();
    assert_eq!(bridge.slots(), expected.as_slice());
}

#[tokio::test]
async fn detailed_scores_expose_logprobs_and_prompt_hash() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = connect(&upstream).await.unwrap();
    let scored = bridge.score_rows(&[two_option_row()]).await.unwrap();
    let first = &scored[0];

    assert_eq!(first.answer.id, "row");
    assert_eq!(first.answer.option_ids, vec!["billing", "sales"]);
    // The mock ranks A above B, so the first option leads.
    assert_eq!(first.option_logprobs, vec![-1.0, -2.0]);
    assert!(first.answer.probabilities[0] > first.answer.probabilities[1]);
    assert_eq!(first.answer.input_tokens, 42);
    assert_eq!(first.prompt_sha256.len(), 64);
    assert_eq!(
        first.answer.prompt_version.as_deref(),
        Some("direct-options-v1")
    );
}

#[tokio::test]
async fn sixteen_option_decision_is_scored_from_one_prompt() {
    let upstream = spawn_upstream(UpstreamConfig::default()).await;
    let bridge = connect(&upstream).await.unwrap();
    let scored = bridge.score_rows(&[sixteen_option_row()]).await.unwrap();

    let probabilities = &scored[0].answer.probabilities;
    assert_eq!(probabilities.len(), 16);
    let total: f64 = probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "{total}");
    assert!(probabilities[0] > probabilities[15]);
}
