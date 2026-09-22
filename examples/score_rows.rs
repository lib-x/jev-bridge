//! Score a handful of rows against a running inference server.
//!
//! The upstream is any OpenAI-compatible service that returns next-token log
//! probabilities — llama.cpp, vLLM, or another runtime the probe recognises.
//!
//! ```bash
//! export JEV_BRIDGE_UPSTREAM_URL=http://127.0.0.1:8080/v1
//! export JEV_BRIDGE_UPSTREAM_KEY=...          # only if the server requires it
//! cargo run --example score_rows
//! ```
//!
//! Every decision costs one prompt evaluation and zero output tokens.

use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::json;

use jev_bridge::server::{Bridge, BridgeConfig};
use jev_bridge::wire::{Row, RowOption};

fn option(id: &str, description: &str) -> RowOption {
    RowOption {
        id: id.to_string(),
        description: description.to_string(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let openai_base = std::env::var("JEV_BRIDGE_UPSTREAM_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
    let native_base = std::env::var("JEV_BRIDGE_NATIVE_URL")
        .unwrap_or_else(|_| openai_base.trim_end_matches("/v1").to_string());
    let upstream_key = std::env::var("JEV_BRIDGE_UPSTREAM_KEY").ok();
    let upstream_model =
        std::env::var("JEV_BRIDGE_UPSTREAM_MODEL").unwrap_or_else(|_| "local-model".to_string());

    let started = Instant::now();
    let bridge = Bridge::connect(
        reqwest::Client::new(),
        BridgeConfig {
            openai_base,
            native_base,
            upstream_model,
            upstream_key,
            served_model: "example-bridge".to_string(),
            description: "example bridge".to_string(),
            release_date: "2026-09-22".to_string(),
            chat_template_kwargs: json!({"enable_thinking": false}),
            max_input_tokens: None,
            local: Default::default(),
        },
    )
    .await
    .context("connecting to the upstream service failed")?;

    println!(
        "connected in {:.2?}: transport={} endpoint={} renders_locally={} tokenizes_locally={}",
        started.elapsed(),
        bridge.probe().name(),
        bridge.probe().endpoint(),
        bridge.renders_locally(),
        bridge.tokenizes_locally(),
    );

    let rows = vec![
        Row {
            id: "refund".to_string(),
            state: json!("I was charged twice and need a refund today."),
            question: "Which queue should handle this request?".to_string(),
            options: vec![
                option("access", "Account access and authentication."),
                option("billing", "Billing, payments, and refunds."),
                option("sales", "Pricing and new contracts."),
            ],
        },
        Row {
            id: "outage".to_string(),
            state: json!("The API returns 502 for every request since this morning."),
            question: "Which queue should handle this request?".to_string(),
            options: vec![
                option("access", "Account access and authentication."),
                option("billing", "Billing, payments, and refunds."),
                option("sales", "Pricing and new contracts."),
            ],
        },
    ];

    let started = Instant::now();
    let scored = bridge.score_rows(&rows).await?;
    println!("scored {} rows in {:.2?}\n", scored.len(), started.elapsed());

    for row in &scored {
        let winner = row
            .answer
            .probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| row.answer.option_ids[index].as_str())
            .unwrap();
        println!("{} -> {winner}", row.answer.id);
        for (id, probability) in row.answer.option_ids.iter().zip(&row.answer.probabilities) {
            println!("    {id:<8} {probability:.4}");
        }
        println!(
            "    input_tokens={} prompt_sha256={}",
            row.answer.input_tokens,
            &row.prompt_sha256[..16]
        );
    }

    // Probabilities are conditional on the supplied options and uncalibrated.
    println!("\nThese scores are conditional option probabilities, not calibrated confidence.");
    Ok(())
}
