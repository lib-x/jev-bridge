//! Serve the djev-run demo pages next to a running jev-bridge.
//!
//! [taeold/djev-run](https://github.com/taeold/djev-run) ships three standalone
//! HTML games (snake / dino / tetris) that call `POST /v1/systemone` from the
//! browser, expecting it on the same origin. This example serves them from
//! `--demo-dir` and proxies `/v1/systemone` to a running bridge, so the games
//! play against whatever model the bridge fronts.
//!
//! The demo HTML is **not** bundled here: it lives in the djev-run repository
//! (which ships no license), so clone that repository and point `--demo-dir`
//! at it. The demos hard-code `model: "jev-latest"`, so start the bridge with
//! `--accept-any-model` — its default refuses model ids it is not.
//!
//! ```bash
//! # terminal 1: the bridge
//! jev-bridge \
//!   --base-url http://127.0.0.1:8080/v1 \
//!   --model my-model \
//!   --served-model bridge-my-model \
//!   --served-model-release-date 2026-09-23 \
//!   --accept-any-model
//!
//! # terminal 2: the playground
//! cargo run --example playground -- --demo-dir /path/to/djev-run
//!
//! # browser
//! open http://127.0.0.1:8000/snake
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{bail, Result};
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;

/// The demo pages this playground serves.
const PAGES: [&str; 3] = ["snake", "dino", "tetris"];

/// A small script injected into every demo page.
///
/// It lets the page be configured from the URL and auto-plays the game,
/// without touching the upstream HTML (which ships no license):
///
/// * `model=<id>` rewrites the `model` field of every `/v1/systemone` request,
///   so the demos' hard-coded `jev-latest` becomes whatever the bridge serves;
/// * `api=<url>` re-points the request at another bridge (default: same origin,
///   i.e. this playground's proxy);
/// * `auto=1` (the default) presses Start whenever the game is idle — on load
///   and again after a game over.
const INJECTED_SCRIPT: &str = r#"
<script>
(() => {
  const params = new URLSearchParams(location.search);
  const model = params.get("model") || "";
  const api = params.get("api") || "";
  const auto = params.get("auto") !== "0";

  if (model || api) {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      let url = typeof input === "string" ? input : (input && input.url) || "";
      if (url.endsWith("/v1/systemone") && init && typeof init.body === "string") {
        try {
          const body = JSON.parse(init.body);
          if (model) body.model = model;
          init = Object.assign({}, init, { body: JSON.stringify(body) });
          if (api) url = api;
        } catch (error) {
          // Not our request shape; pass it through untouched.
        }
      }
      return originalFetch(url, init);
    };
  }

  if (auto) {
    setInterval(() => {
      const start = [...document.querySelectorAll("button")]
        .find((button) => /^\s*start\s*$/i.test(button.textContent || ""));
      if (start) start.click();
    }, 1500);
  }
})();
</script>
"#;

#[derive(Parser, Debug)]
#[command(
    name = "playground",
    about = "Serve the djev-run demo games against a running jev-bridge"
)]
struct Args {
    /// Directory holding the djev-run demo HTML (snake.html, dino.html,
    /// tetris.html); clone https://github.com/taeold/djev-run and point this
    /// at the repository root
    #[arg(long)]
    demo_dir: PathBuf,

    /// Base URL of the running jev-bridge
    #[arg(long, default_value = "http://127.0.0.1:8199")]
    bridge: String,

    /// Address to bind
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind
    #[arg(long, default_value_t = 8000)]
    port: u16,
}

#[derive(Clone)]
struct Playground {
    demo_dir: PathBuf,
    bridge: String,
    client: reqwest::Client,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    for page in PAGES {
        let path = args.demo_dir.join(format!("{page}.html"));
        if !path.is_file() {
            bail!(
                "{} not found; clone https://github.com/taeold/djev-run and point --demo-dir at \
                 the repository root",
                path.display()
            );
        }
    }

    let bridge = args.bridge.trim_end_matches('/').to_string();
    let playground = Playground {
        demo_dir: args.demo_dir.clone(),
        bridge: bridge.clone(),
        client: reqwest::Client::new(),
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/{page}", get(page))
        .route("/v1/systemone", post(systemone))
        .with_state(playground);

    let address: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!(
        "playground: http://{address}/ (snake / dino / tetris) from {}",
        args.demo_dir.display()
    );
    println!("proxying POST /v1/systemone to {bridge}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Response {
    let body = r#"<!doctype html>
<html>
<meta charset="utf-8">
<title>jev-bridge playground</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 46rem; }
  #status { font-family: ui-monospace, monospace; }
  label { display: block; margin: .5rem 0; }
  input[type=text] { width: 22rem; font-family: ui-monospace, monospace; }
  li { margin: .4rem 0; font-size: 1.1rem; }
  .muted { color: #666; font-size: .9rem; }
</style>
<h1>jev-bridge playground</h1>
<p>Bridge: <span id="status">checking…</span></p>
<p class="muted">Games from <a href="https://github.com/taeold/djev-run">djev-run</a>,
played against this bridge. They auto-start; press Pause to take over.</p>
<label>Model id sent by the games: <input type="text" id="model" placeholder="(loading from /health…)"></label>
<label><input type="checkbox" id="auto" checked> auto-play (start, and restart after a game over)</label>
<ul id="games"></ul>
<script>
const games = ["snake", "dino", "tetris"];
const modelInput = document.getElementById("model");
const autoInput = document.getElementById("auto");
const statusEl = document.getElementById("status");
const listEl = document.getElementById("games");

function rebuild() {
  const params = new URLSearchParams();
  if (modelInput.value.trim()) params.set("model", modelInput.value.trim());
  params.set("auto", autoInput.checked ? "1" : "0");
  listEl.innerHTML = games.map(
    (game) => `<li><a href="/${game}?${params}">${game}</a> — auto-plays against this bridge</li>`
  ).join("");
}
modelInput.addEventListener("input", rebuild);
autoInput.addEventListener("change", rebuild);
rebuild();

fetch("/health")
  .then((response) => response.json())
  .then((health) => {
    const readout = health.readout_check
      ? `, readout check ${health.readout_check.passed}/${health.readout_check.probes}`
      : "";
    statusEl.textContent =
      `${health.served_model} via ${health.probe}` +
      ` (batch fallbacks ${health.batch_fallbacks}${readout})`;
    // Default to the model the bridge actually serves, so the games work
    // without --accept-any-model.
    if (!modelInput.value.trim()) {
      modelInput.value = health.served_model || "";
      rebuild();
    }
  })
  .catch((error) => {
    statusEl.textContent = "bridge unreachable: " + error;
  });
</script>
</html>
"#;
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        body,
    )
        .into_response()
}

async fn page(State(playground): State<Playground>, Path(name): Path<String>) -> Response {
    if !PAGES.contains(&name.as_str()) {
        return (StatusCode::NOT_FOUND, "unknown page").into_response();
    }
    let path = playground.demo_dir.join(format!("{name}.html"));
    // Synchronous read on purpose: the demo pages are a few tens of KB and
    // this example is not production code, so no `tokio` fs feature is worth
    // adding to the crate for it.
    match std::fs::read_to_string(&path) {
        Ok(body) => (
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            format!("{body}\n{INJECTED_SCRIPT}"),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("reading {} failed: {error}", path.display()),
        )
            .into_response(),
    }
}

/// Forward the bridge's health payload, so the index page can show what is
/// actually being played against.
async fn health(State(playground): State<Playground>) -> Response {
    let url = format!("{}/health", playground.bridge);
    match playground.client.get(&url).send().await {
        Ok(response) => {
            let status = response.status();
            match response.bytes().await {
                Ok(body) => (
                    status,
                    [(header::CONTENT_TYPE, "application/json")],
                    body,
                )
                    .into_response(),
                Err(error) => (
                    StatusCode::BAD_GATEWAY,
                    format!("reading {url} failed: {error}"),
                )
                    .into_response(),
            }
        }
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            format!("GET {url} failed: {error} (is the bridge running?)"),
        )
            .into_response(),
    }
}

/// Forward one decision request to the bridge, verbatim.
///
/// The demos' `fetch()` expects a JSON answer on the same origin; the body and
/// status are passed through so a bridge-side error stays visible in the game
/// UI instead of being replaced by a generic proxy failure.
async fn systemone(State(playground): State<Playground>, body: Bytes) -> Response {
    let url = format!("{}/v1/systemone", playground.bridge);
    let response = playground
        .client
        .post(&url)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await;
    match response {
        Ok(response) => {
            let status = response.status();
            match response.bytes().await {
                Ok(body) => (
                    status,
                    [(header::CONTENT_TYPE, "application/json")],
                    body,
                )
                    .into_response(),
                Err(error) => (
                    StatusCode::BAD_GATEWAY,
                    format!("reading the bridge response failed: {error}"),
                )
                    .into_response(),
            }
        }
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            format!("POST {url} failed: {error} (is the bridge running?)"),
        )
            .into_response(),
    }
}
