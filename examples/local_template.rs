//! Render the chat template in-process instead of asking the runtime.
//!
//! Runtimes that expose `/apply-template` (llama.cpp) render the prompt for
//! you, which guarantees the prompt is exactly what that runtime feeds the
//! model. Runtimes that do not — vLLM, for instance — need the template here.
//!
//! Take the model's own Jinja template, not a hand-written one. For a llama.cpp
//! server it is served by `/props`:
//!
//! ```bash
//! curl -s "$JEV_BRIDGE_NATIVE_URL/props?model=$JEV_BRIDGE_UPSTREAM_MODEL" \
//!   | jq -r .chat_template > template.jinja
//! ```
//!
//! Then:
//!
//! ```bash
//! export JEV_BRIDGE_CHAT_TEMPLATE=template.jinja
//! export JEV_BRIDGE_CHAT_CONTEXT='{"bos_token": "<s>", "eos_token": "</s>"}'
//! cargo run --example local_template
//! ```
//!
//! The context carries the variables transformers supplies besides the
//! messages. A template that begins with `{{- bos_token }}` needs `bos_token`
//! here, or the rendered prompt will silently lack its BOS token.

use anyhow::{Context, Result};
use serde_json::{json, Value};

use jev_bridge::prompt::direct_messages;
use jev_bridge::render::LocalRenderer;
use jev_bridge::wire::RowOption;

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::var("JEV_BRIDGE_CHAT_TEMPLATE").unwrap_or_else(|_| "template.jinja".to_string());
    let source = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {path} failed; see the module docs for how to obtain it"))?;

    let context: Value = serde_json::from_str(
        &std::env::var("JEV_BRIDGE_CHAT_CONTEXT").unwrap_or_else(|_| "{}".to_string()),
    )
    .context("JEV_BRIDGE_CHAT_CONTEXT must be a JSON object")?;
    let context = context
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("JEV_BRIDGE_CHAT_CONTEXT must be a JSON object"))?;

    let renderer = LocalRenderer::new(source, context).context("compiling the chat template failed")?;

    let messages = direct_messages(
        &json!("I was charged twice and need a refund today."),
        "Which queue should handle this request?",
        &[
            RowOption {
                id: "access".to_string(),
                description: "Account access and authentication.".to_string(),
            },
            RowOption {
                id: "billing".to_string(),
                description: "Billing, payments, and refunds.".to_string(),
            },
            RowOption {
                id: "sales".to_string(),
                description: "Pricing and new contracts.".to_string(),
            },
        ],
    );

    let prompt = renderer.render(&messages, &json!({"enable_thinking": false}))?;
    println!("rendered {} bytes:\n", prompt.len());
    println!("{prompt}");
    println!("\n--- debug ---");
    println!("{:?}", prompt);
    println!(
        "\nThis is the exact text the upstream model must be scored on. Point the bridge at it \
         with --chat-template-file so the prompt contract stays byte-identical."
    );
    Ok(())
}
