//! Bridge a generic OpenAI-compatible inference API into a Jev-style scoring service.
//!
//! The upstream model keeps its own runtime; this process only adds the
//! decision readout: one rendered prompt, a softmax over the log
//! probabilities of the fixed answer letters, and a System One compatible
//! response. No answer text is generated and no model weights are copied.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::Value;

use jev_bridge::prompt;
use jev_bridge::server::{router, AppState, Bridge, BridgeConfig};
use jev_bridge::wire::Row;

#[derive(Parser, Debug)]
#[command(
    name = "jev-bridge",
    about = "Expose a Jev-style decision API over a generic OpenAI-compatible inference service",
    long_about = None,
)]
struct Args {
    /// OpenAI-compatible base URL of the upstream service, for example http://127.0.0.1:8080/v1
    #[arg(long)]
    base_url: String,

    /// Native base URL used for llama.cpp /completion; defaults to --base-url without a trailing /v1
    #[arg(long)]
    native_base_url: Option<String>,

    /// Model name sent to the upstream service
    #[arg(long)]
    model: String,

    /// Bearer token for the upstream service; prefer the environment variable
    #[arg(long, env = "JEV_BRIDGE_UPSTREAM_KEY", hide_env_values = true)]
    upstream_key: Option<String>,

    /// Model ID this bridge accepts from clients; defaults to --model
    #[arg(long)]
    served_model: Option<String>,

    /// Human-readable description returned by GET /v1/models
    #[arg(
        long,
        default_value = "jev-bridge direct option-logit readout over a generic inference API"
    )]
    served_model_description: String,

    /// ISO release date returned by GET /v1/models
    #[arg(long)]
    served_model_release_date: String,

    /// JSON object forwarded as chat_template_kwargs when rendering the template
    #[arg(long, default_value = "{\"enable_thinking\": false}")]
    chat_template_kwargs: String,

    /// Reject rows whose upstream prompt exceeds this many tokens
    #[arg(long)]
    max_input_tokens: Option<usize>,

    /// Address to bind
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind
    #[arg(long, default_value_t = 8100)]
    port: u16,

    /// Bearer token clients must present; prefer the environment variable
    #[arg(long, env = "JEV_BRIDGE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Probe the upstream service, print the chosen transport, and exit
    #[arg(long)]
    probe_only: bool,

    /// Batch scoring mode: read fastjev-format rows from --input and write one
    /// JSONL prediction per row to --output, then exit
    #[arg(long)]
    score: bool,

    /// Input JSONL for --score, one row per line
    #[arg(long)]
    input: Option<PathBuf>,

    /// Output JSONL for --score, created fresh
    #[arg(long)]
    output: Option<PathBuf>,
}

/// One prediction line, shaped for row-by-row comparison with a reference run.
#[derive(serde::Serialize)]
struct ScoreLine {
    id: String,
    option_ids: Vec<String>,
    probabilities: Vec<f64>,
    option_logprobs: Vec<f64>,
    input_tokens: u64,
    prompt_sha256: String,
    prompt_version: String,
    probe: String,
}

async fn run_score(bridge: &Bridge, input: &Path, output: &Path) -> Result<()> {
    let reader = BufReader::new(
        File::open(input).with_context(|| format!("opening {} failed", input.display()))?,
    );
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("creating {} failed", output.display()))?,
    );
    let mut count = 0usize;
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(&line)
            .with_context(|| format!("{}:{} is not a fastjev row", input.display(), index + 1))?;
        let id = row.id.clone();
        let mut scored = bridge
            .score_rows(&[row])
            .await
            .with_context(|| format!("scoring row {id:?} failed"))?;
        let scored = scored.remove(0);
        let record = serde_json::to_string(&ScoreLine {
            id: scored.answer.id,
            option_ids: scored.answer.option_ids,
            probabilities: scored.answer.probabilities,
            option_logprobs: scored.option_logprobs,
            input_tokens: scored.answer.input_tokens,
            prompt_sha256: scored.prompt_sha256,
            prompt_version: scored.answer.prompt_version.unwrap_or_default(),
            probe: bridge.probe().name().to_string(),
        })?;
        writeln!(writer, "{record}")?;
        count += 1;
        if count.is_multiple_of(20) {
            eprintln!("scored {count} rows");
        }
    }
    writer.flush()?;
    eprintln!("wrote {count} rows to {}", output.display());
    Ok(())
}

fn default_native_base(base_url: &str) -> String {    let trimmed = base_url.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_string()
}

fn is_iso_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if args.base_url.trim().is_empty() {
        bail!("--base-url must not be empty");
    }
    if !is_iso_date(&args.served_model_release_date) {
        bail!("--served-model-release-date must be an ISO date such as 2026-09-18");
    }
    let served_model = args.served_model.clone().unwrap_or_else(|| args.model.clone());
    if served_model.to_lowercase().starts_with("jev") {
        bail!(
            "--served-model must identify this bridge and must not impersonate a Jev model or alias"
        );
    }
    let chat_template_kwargs: Value = serde_json::from_str(&args.chat_template_kwargs)
        .context("--chat-template-kwargs must be a JSON object")?;
    if !chat_template_kwargs.is_object() {
        bail!("--chat-template-kwargs must be a JSON object");
    }
    let native_base = args
        .native_base_url
        .clone()
        .unwrap_or_else(|| default_native_base(&args.base_url));

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("building the HTTP client failed")?;

    let config = BridgeConfig {
        openai_base: args.base_url.clone(),
        native_base: native_base.clone(),
        upstream_model: args.model.clone(),
        upstream_key: args.upstream_key.clone(),
        served_model: served_model.clone(),
        description: args.served_model_description.clone(),
        release_date: args.served_model_release_date.clone(),
        chat_template_kwargs,
        max_input_tokens: args.max_input_tokens,
    };

    eprintln!(
        "probing upstream {} (native {}) for model {:?}",
        args.base_url, native_base, args.model
    );
    let bridge = Bridge::connect(client, config)
        .await
        .context("connecting to the upstream service failed")?;
    eprintln!(
        "transport: {} via {} ({})",
        bridge.probe().name(),
        bridge.probe().endpoint(),
        bridge.probe().note()
    );

    if args.probe_only {
        let slots: Vec<Value> = prompt::LETTERS
            .chars()
            .zip(bridge.slots())
            .map(|(letter, id)| serde_json::json!({"letter": letter.to_string(), "token_id": id}))
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "served_model": served_model,
                "upstream_model": bridge.upstream_model(),
                "probe": bridge.probe().name(),
                "probe_note": bridge.probe().note(),
                "endpoint": bridge.probe().endpoint(),
                "answer_slots": slots,
            }))?
        );
        return Ok(());
    }

    if args.score {
        let input = args.input.clone().context("--score requires --input")?;
        let output = args.output.clone().context("--score requires --output")?;
        if output.exists() {
            bail!(
                "{} already exists; benchmark outputs are create-only, pass a new path",
                output.display()
            );
        }
        return run_score(&bridge, &input, &output).await;
    }

    let state = Arc::new(AppState {
        bridge,
        api_key: args.api_key.clone(),
    });
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port))
        .await
        .with_context(|| format!("binding {}:{} failed", args.host, args.port))?;
    eprintln!(
        "jev-bridge serving {:?} on http://{}:{} (POST /v1/systemone, GET /v1/models, GET /health)",
        served_model, args.host, args.port
    );
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("the HTTP server stopped unexpectedly")?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("shutting down");
}
