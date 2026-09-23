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

use jev_bridge::evaluate;
use jev_bridge::prompt::{self, LocalComponents};
use jev_bridge::render::LocalRenderer;
use jev_bridge::server::{router, AppState, Bridge, BridgeConfig};
#[cfg(feature = "local-tokenizer")]
use jev_bridge::tokenizer::LocalTokenizer;
use jev_bridge::wire::{ReadoutStatus, Row};

#[derive(Parser, Debug)]
#[command(
    name = "jev-bridge",
    about = "Expose a Jev-style decision API over a generic OpenAI-compatible inference service",
    long_about = None,
)]
struct Args {
    /// OpenAI-compatible base URL of the upstream service, for example http://127.0.0.1:8080/v1
    #[arg(long, required_unless_present = "evaluate")]
    base_url: Option<String>,

    /// Native base URL used for llama.cpp /completion; defaults to --base-url without a trailing /v1
    #[arg(long)]
    native_base_url: Option<String>,

    /// Model name sent to the upstream service
    #[arg(long, required_unless_present = "evaluate")]
    model: Option<String>,

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
    #[arg(long, required_unless_present = "evaluate")]
    served_model_release_date: Option<String>,

    /// JSON object forwarded as chat_template_kwargs when rendering the template
    #[arg(long, default_value = "{\"enable_thinking\": false}")]
    chat_template_kwargs: String,

    /// Render the chat template locally from this Jinja file instead of asking
    /// the runtime; needed for runtimes without /apply-template, such as vLLM
    #[arg(long)]
    chat_template_file: Option<PathBuf>,

    /// JSON object added to every local render, for example {"bos_token": "<s>"}
    #[arg(long, default_value = "{}")]
    chat_template_context: String,

    /// Tokenize with this local tokenizer.json instead of the runtime's
    /// /tokenize endpoint; must be the model's own vocabulary
    #[arg(long)]
    tokenizer_json: Option<PathBuf>,

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

    /// Evaluate calibration metrics offline from --predictions and --gold, then
    /// exit; no upstream connection is made
    #[arg(long)]
    evaluate: bool,

    /// Predictions JSONL for --evaluate, as written by --score
    #[arg(long, requires = "evaluate")]
    predictions: Option<PathBuf>,

    /// Gold JSONL for --evaluate: one {"id", "gold", "family"?, "positive"?}
    /// object per line
    #[arg(long, requires = "evaluate")]
    gold: Option<PathBuf>,

    /// Equal-width confidence bins for ECE and the reliability curve
    #[arg(long, default_value_t = jev_bridge::evaluate::DEFAULT_BINS, requires = "evaluate")]
    bins: usize,

    /// Write the --evaluate report JSON here
    #[arg(long, requires = "evaluate")]
    out: Option<PathBuf>,

    /// Recompute the report from --predictions and --gold and compare it
    /// against this file; exits non-zero on any mismatch
    #[arg(long, requires = "evaluate")]
    verify: Option<PathBuf>,

    /// Numeric tolerance for --verify comparisons
    #[arg(long, default_value_t = 1e-12, requires = "evaluate")]
    tol: f64,

    /// Readout-channel validation status exposed to clients; conventions are
    /// `unvalidated` (nothing was checked on this model) or `validated`
    #[arg(long, default_value = "unvalidated")]
    readout_status: String,

    /// Free-form evidence for --readout-status, for example an alignment summary
    #[arg(long)]
    readout_evidence: Option<String>,

    /// Require the startup readout self-check to pass before serving; the
    /// default reports it on /health without blocking startup
    #[arg(long)]
    require_readout_check: bool,

    /// Accept any non-empty `model` name in a request instead of requiring
    /// --served-model; for third-party clients that hard-code a model id
    #[arg(long)]
    accept_any_model: bool,

    /// Timeout for one upstream request, in seconds
    #[arg(long, default_value_t = 600)]
    upstream_timeout_secs: u64,
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
        let scored = bridge
            .score_row(&row)
            .await
            .with_context(|| format!("scoring row {id:?} failed"))?;
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

/// The offline `--evaluate` path: metrics from two files, plus an optional
/// verification of a written report against a fresh recomputation.
fn run_evaluate(args: &Args) -> Result<()> {
    let predictions_path = args
        .predictions
        .clone()
        .context("--evaluate requires --predictions")?;
    let gold_path = args.gold.clone().context("--evaluate requires --gold")?;
    let predictions_text = std::fs::read_to_string(&predictions_path)
        .with_context(|| format!("reading {} failed", predictions_path.display()))?;
    let gold_text = std::fs::read_to_string(&gold_path)
        .with_context(|| format!("reading {} failed", gold_path.display()))?;

    let report = evaluate::evaluate(&predictions_text, &gold_text, args.bins)?;

    if let Some(verify_path) = &args.verify {
        let claimed_text = std::fs::read_to_string(verify_path)
            .with_context(|| format!("reading {} failed", verify_path.display()))?;
        let claimed: evaluate::Report = serde_json::from_str(&claimed_text)
            .with_context(|| format!("{} is not an evaluation report", verify_path.display()))?;
        let outcome = evaluate::verify(&claimed, &report, args.tol);
        print!("{}", outcome.report());
        if !outcome.is_ok() {
            bail!(
                "{} does not reproduce from its inputs (tolerance {})",
                verify_path.display(),
                args.tol
            );
        }
    }

    if let Some(out) = &args.out {
        let text = serde_json::to_string_pretty(&report)?;
        std::fs::write(out, format!("{text}\n"))
            .with_context(|| format!("writing {} failed", out.display()))?;
        eprintln!("wrote {}", out.display());
    }

    print!("{}", evaluate::render_table(&report));
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

    // `--evaluate` is fully offline: no upstream connection, no model. It runs
    // before any of the serving configuration is validated so that an
    // evaluation needs nothing but the two files.
    if args.evaluate {
        return run_evaluate(&args);
    }

    // clap enforces both via `required_unless_present = "evaluate"`.
    let base_url = args.base_url.clone().expect("clap requires --base-url");
    let model = args.model.clone().expect("clap requires --model");
    let release_date = args
        .served_model_release_date
        .clone()
        .expect("clap requires --served-model-release-date");

    if model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if base_url.trim().is_empty() {
        bail!("--base-url must not be empty");
    }
    if !is_iso_date(&release_date) {
        bail!("--served-model-release-date must be an ISO date such as 2026-09-18");
    }
    let served_model = args.served_model.clone().unwrap_or_else(|| model.clone());
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
        .unwrap_or_else(|| default_native_base(&base_url));

    let local_renderer = match &args.chat_template_file {
        None => None,
        Some(path) => {
            let source = std::fs::read_to_string(path)
                .with_context(|| format!("reading {} failed", path.display()))?;
            let context: Value = serde_json::from_str(&args.chat_template_context)
                .context("--chat-template-context must be a JSON object")?;
            let context = context
                .as_object()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("--chat-template-context must be a JSON object"))?;
            Some(
                LocalRenderer::new(source, context)
                    .with_context(|| format!("compiling {} failed", path.display()))?,
            )
        }
    };
    let local_tokenizer = match &args.tokenizer_json {
        None => None,
        Some(path) => Some(
            LocalTokenizer::from_file(path)
                .with_context(|| format!("loading {} failed", path.display()))?,
        ),
    };

    // A connect timeout bounds the half-open case: the request timeout alone
    // would let a dead upstream hold the bridge's serialized scoring path for
    // the whole window.
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(args.upstream_timeout_secs))
        .build()
        .context("building the HTTP client failed")?;

    let config = BridgeConfig {
        openai_base: base_url.clone(),
        native_base: native_base.clone(),
        upstream_model: model.clone(),
        upstream_key: args.upstream_key.clone(),
        served_model: served_model.clone(),
        description: args.served_model_description.clone(),
        release_date: release_date.clone(),
        chat_template_kwargs,
        max_input_tokens: args.max_input_tokens,
        local: LocalComponents {
            renderer: local_renderer,
            tokenizer: local_tokenizer,
        },
        accept_any_model: args.accept_any_model,
        readout: ReadoutStatus {
            status: args.readout_status.clone(),
            evidence: args.readout_evidence.clone(),
        },
    };

    eprintln!(
        "probing upstream {} (native {}) for model {:?}",
        base_url, native_base, model
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
        let readout_check = bridge
            .run_readout_check()
            .await
            .context("the readout self-check could not be run")?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "served_model": served_model,
                "upstream_model": bridge.upstream_model(),
                "probe": bridge.probe().name(),
                "probe_note": bridge.probe().note(),
                "endpoint": bridge.probe().endpoint(),
                "answer_slots": slots,
                "readout_self_check": readout_check,
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

    // The readout self-check: a few questions whose answers are obvious, read
    // through the exact channel every request uses. It reports; it does not
    // block startup unless --require-readout-check says so.
    eprintln!("running the readout self-check ...");
    let readout_check = match bridge.run_readout_check().await {
        Ok(check) => {
            eprintln!(
                "readout self-check: {}/{} probes named the expected option",
                check.passed, check.probes
            );
            for result in check.results.iter().filter(|result| !result.passed) {
                eprintln!(
                    "  {}: expected {:?}, named {:?} (expected-option probability {:.3})",
                    result.id, result.expected, result.argmax, result.expected_probability
                );
            }
            Some(check)
        }
        Err(error) => {
            eprintln!("the readout self-check could not run: {error:#}");
            None
        }
    };
    if args.require_readout_check
        && !readout_check
            .as_ref()
            .is_some_and(jev_bridge::readout::ReadoutCheck::is_ok)
    {
        bail!(
            "--require-readout-check: the readout self-check did not pass; the model may not \
             answer with slot letters at all (the per-probe detail is on GET /health)"
        );
    }

    let state = Arc::new(AppState {
        bridge,
        api_key: args.api_key.clone(),
        readout_check,
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
