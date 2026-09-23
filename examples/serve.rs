//! Serve the bridge over HTTP with the System One compatible surface.
//!
//! ```bash
//! export JEV_BRIDGE_UPSTREAM_URL=http://127.0.0.1:8080/v1
//! export JEV_BRIDGE_UPSTREAM_KEY=...      # only if the server requires it
//! export JEV_BRIDGE_API_KEY=...           # optional, for clients of this service
//! cargo run --example serve
//! ```
//!
//! Then:
//!
//! ```bash
//! curl http://127.0.0.1:8100/health
//! curl http://127.0.0.1:8100/v1/systemone \
//!   -H 'Content-Type: application/json' \
//!   -d '{"model":"example-bridge","state":"I was charged twice.",
//!        "questions":{"queue":{"type":"choice","criteria":{"billing":"Refunds","sales":"Pricing"}}}}'
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::json;

use jev_bridge::server::{router, AppState, Bridge, BridgeConfig};

const ADDRESS: &str = "127.0.0.1:8100";

#[tokio::main]
async fn main() -> Result<()> {
    let openai_base = std::env::var("JEV_BRIDGE_UPSTREAM_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
    let native_base = std::env::var("JEV_BRIDGE_NATIVE_URL")
        .unwrap_or_else(|_| openai_base.trim_end_matches("/v1").to_string());
    let upstream_model =
        std::env::var("JEV_BRIDGE_UPSTREAM_MODEL").unwrap_or_else(|_| "local-model".to_string());

    let bridge = Bridge::connect(
        reqwest::Client::new(),
        BridgeConfig {
            openai_base,
            native_base,
            upstream_model,
            upstream_key: std::env::var("JEV_BRIDGE_UPSTREAM_KEY").ok(),
            served_model: "example-bridge".to_string(),
            description: "example bridge over a generic inference API".to_string(),
            release_date: "2026-09-22".to_string(),
            chat_template_kwargs: json!({"enable_thinking": false}),
            max_input_tokens: None,
            local: Default::default(),
            readout: Default::default(),
        },
    )
    .await
    .context("connecting to the upstream service failed")?;

    println!("transport: {} via {}", bridge.probe().name(), bridge.probe().endpoint());

    let state = Arc::new(AppState {
        bridge,
        api_key: std::env::var("JEV_BRIDGE_API_KEY").ok(),
    });
    let listener = tokio::net::TcpListener::bind(ADDRESS)
        .await
        .with_context(|| format!("binding {ADDRESS} failed"))?;
    println!("listening on http://{ADDRESS}");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("the HTTP server stopped unexpectedly")
}
