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

use anyhow::{bail, Context, Result};
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
/// It shows the raw System One exchange, lets the page be configured from the
/// URL, auto-plays the game and rebrands it — all without touching the
/// upstream HTML (which ships no license):
///
/// * a fixed panel reports the decision channel's state: `requesting…` with a
///   running timer while a call is in flight, then `returned · HTTP 200 · …`
///   (green) or `failed · …` (red), with the full response JSON below it;
/// * `model=<id>` rewrites the `model` field of every `/v1/systemone` request,
///   so the demos' hard-coded `jev-latest` becomes whatever the bridge serves;
/// * `api=<url>` re-points the request at another bridge (default: same origin,
///   i.e. this playground's proxy);
/// * `auto=1` (the default) presses Start whenever the game is idle — on load
///   and again after a game over;
/// * the upstream product titles (`djev / snake`, `djev (DiffusionGemma-Jev)`)
///   are rewritten to `jev-bridge`, because that is what is actually serving
///   the page. Attribution links (`mmastrac/djev-spark`) are left alone.
const INJECTED_SCRIPT: &str = r##"
<script>
(() => {
  const params = new URLSearchParams(location.search);
  const model = params.get("model") || "";
  const api = params.get("api") || "";
  const auto = params.get("auto") !== "0";

  // ---- the decision-channel panel: pending / returned / failed -------------
  const PANEL = "jev-bridge-panel";
  const COLORS = { idle: "#38bdf8", pending: "#facc15", ok: "#4ade80", error: "#f87171" };
  let pendingTimer = null;
  let pendingStarted = 0;

  const ensurePanel = () => {
    let panel = document.getElementById(PANEL);
    if (panel) return panel;
    panel = document.createElement("div");
    panel.id = PANEL;
    panel.style.cssText = [
      "position:fixed", "left:0", "right:0", "bottom:0", "z-index:2147483647",
      "background:rgba(15,23,42,.96)", "color:#e2e8f0",
      "font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace",
      "border-top:2px solid " + COLORS.idle, "box-shadow:0 -4px 16px rgba(0,0,0,.35)",
    ].join(";");
    panel.innerHTML =
      '<div style="display:flex;gap:10px;align-items:center;padding:6px 12px;border-bottom:1px solid #1e293b">' +
        '<b style="color:#38bdf8">jev-bridge</b>' +
        '<span>POST /v1/systemone</span>' +
        '<span id="' + PANEL + '-meta" style="color:#94a3b8">waiting for the first request…</span>' +
        '<span style="flex:1"></span>' +
        '<button id="' + PANEL + '-toggle" style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:2px 8px;cursor:pointer">collapse</button>' +
      '</div>' +
      '<pre id="' + PANEL + '-body" style="margin:0;padding:8px 12px;max-height:32vh;overflow:auto;white-space:pre-wrap">The raw System One response appears here; the status line above shows whether a call is in flight.</pre>';
    document.body.appendChild(panel);
    const body = panel.querySelector("#" + PANEL + "-body");
    panel.querySelector("#" + PANEL + "-toggle").addEventListener("click", (event) => {
      const hidden = body.style.display === "none";
      body.style.display = hidden ? "" : "none";
      event.target.textContent = hidden ? "collapse" : "expand";
    });
    return panel;
  };
  const setStatus = (kind, text) => {
    const panel = ensurePanel();
    panel.style.borderTopColor = COLORS[kind] || COLORS.idle;
    panel.querySelector("#" + PANEL + "-meta").textContent = text;
  };
  const setBody = (text) => {
    ensurePanel().querySelector("#" + PANEL + "-body").textContent = text;
  };
  ensurePanel();

  const beginRequest = () => {
    pendingStarted = performance.now();
    clearInterval(pendingTimer);
    setStatus("pending", "requesting… 0.0 s");
    pendingTimer = setInterval(() => {
      setStatus("pending", "requesting… " + ((performance.now() - pendingStarted) / 1000).toFixed(1) + " s");
    }, 200);
  };
  const finishRequest = (ok, status, payload) => {
    clearInterval(pendingTimer);
    const latencyMs = Math.round(performance.now() - pendingStarted);
    const answers = payload && payload.answers ? Object.keys(payload.answers).length : 0;
    setStatus(
      ok ? "ok" : "error",
      (ok ? "returned · " : "failed · ") + "HTTP " + status + " · " + latencyMs + " ms" +
        (answers ? " · " + answers + " answer(s)" : "") +
        (payload && payload.model ? " · " + payload.model : "")
    );
    setBody(JSON.stringify(payload, null, 2));
  };

  // ---- request rewriting + status capture ---------------------------------
  const originalFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    let url = typeof input === "string" ? input : (input && input.url) || "";
    const isDecision = url.endsWith("/v1/systemone");
    if (isDecision && init && typeof init.body === "string" && (model || api)) {
      try {
        const payload = JSON.parse(init.body);
        if (model) payload.model = model;
        init = Object.assign({}, init, { body: JSON.stringify(payload) });
        if (api) url = api;
      } catch (error) {
        // Not our request shape; pass it through untouched.
      }
    }
    if (isDecision) beginRequest();
    return originalFetch(url, init).then(
      (response) => {
        if (isDecision) {
          response.clone().json()
            .then((payload) => finishRequest(response.ok, response.status, payload))
            .catch(() => finishRequest(response.ok, response.status, { error: "response body was not JSON" }));
        }
        return response;
      },
      (error) => {
        if (isDecision) finishRequest(false, 0, { error: String(error) });
        throw error;
      }
    );
  };

  if (auto) {
    setInterval(() => {
      const start = [...document.querySelectorAll("button")]
        .find((button) => /start/i.test(button.textContent || ""));
      if (start) start.click();
    }, 1500);
  }

  // ---- rebrand ------------------------------------------------------------
  const rebrandText = (text) =>
    text
      .replace(/djev\s*\(DiffusionGemma-Jev\)/gi, "jev-bridge")
      .replace(/\bdjev\s*\/\s*(?=[A-Za-z])/g, "jev-bridge / ");
  const insidePanel = (node) => {
    let element = node.parentNode;
    while (element) {
      if (element.id === PANEL) return true;
      element = element.parentNode;
    }
    return false;
  };
  const rebrand = () => {
    document.title = rebrandText(document.title);
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) =>
        node.parentNode &&
        (/^(SCRIPT|STYLE)$/.test(node.parentNode.nodeName) || insidePanel(node))
          ? NodeFilter.FILTER_REJECT
          : NodeFilter.FILTER_ACCEPT,
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      if (/djev/i.test(node.nodeValue || "")) {
        node.nodeValue = rebrandText(node.nodeValue);
      }
    }
  };
  rebrand();
  // Some titles are built after load; give them a second pass.
  setTimeout(rebrand, 600);
})();
</script>
"##;

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

    /// Timeout for one proxied decision request, in seconds. The games' prompts
    /// can be large (tetris sends a whole board plus sixteen placements), and a
    /// slow upstream can take minutes; reqwest's 30 s default is far too short.
    #[arg(long, default_value_t = 600)]
    timeout_secs: u64,
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
        client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(args.timeout_secs))
            .build()
            .context("building the HTTP client failed")?,
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
