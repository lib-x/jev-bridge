<div align="center">

# jev-bridge

**Turn any generic OpenAI-compatible inference service into a Jev-style decision scoring service.**

A single 7 MB binary with no Python dependency.

[![crates.io](https://img.shields.io/crates/v/jev-bridge.svg)](https://crates.io/crates/jev-bridge)
[![docs.rs](https://docs.rs/jev-bridge/badge.svg)](https://docs.rs/jev-bridge)
[![license](https://img.shields.io/crates/l/jev-bridge.svg)](https://github.com/lib-x/jev-bridge/blob/main/LICENSE)
[![downloads](https://img.shields.io/crates/d/jev-bridge.svg)](https://crates.io/crates/jev-bridge)
[![rust](https://img.shields.io/badge/rust-2024%20edition-blue.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)

[English](https://github.com/lib-x/jev-bridge/blob/main/README.md) | [简体中文](https://github.com/lib-x/jev-bridge/blob/main/README.zh-CN.md)

</div>

## What it does

Jev-style decisions are not a model format; they are a readout. This bridge
implements the [`direct-options-v1`](https://github.com/chengyongru/fastjev)
contract on top of any runtime that can return next-token log probabilities:

1. render a fixed decision prompt through the **serving runtime's own** chat
   template (system instruction plus a `{"evidence", "criterion", "options"}`
   JSON payload, thinking disabled);
2. take the **final position** of one forward pass;
3. softmax **only** over the 2–16 fixed uppercase answer letters `A`–`P`;
4. return a typed System One result without generating any answer text.

The upstream model, its weights and its quantization stay exactly where they
are. This process copies no weights and modifies no model.

## What it does not do

- It is not a reimplementation of Jev and does not impersonate it:
  `--served-model` refuses names starting with `jev`.
- Returned probabilities are **conditional scores over the supplied options**
  with `calibrated=False`. They are not operational confidence.
- It performs no calibration and recommends no thresholds. Validate on your own
  workload before automating anything consequential.

## Supported types

### Question types (`POST /v1/systemone`)

| Type | Shape | Returns |
|---|---|---|
| `choice` | `criteria` object with 2–16 options | winning option id, full distribution, `confidence` |
| `noul` | optional `criteria` with `true` / `false` | probability assigned to `true` |
| `score` | ordered `criteria` array of 2–10 levels | probability-weighted value, `legend`, `confidence` |

`instructions` is optional for all three; a type-specific generic question is
substituted when it is omitted. Criteria values may be strings, objects, arrays
or `null`; structured values are rendered as JSON.

### Upstream runtimes

| Runtime | Status | How it is reached |
|---|---|---|
| **llama.cpp** (`llama-server`) | fully supported | `/apply-template` + `/tokenize` for the contract, `/v1/completions` or `/completion` for scoring |
| Generic OpenAI-compatible | supported when the runtime exposes logprobs | `/v1/completions` with `logprobs` |
| vLLM | supported; supply the template | `allowed_token_ids` + `logprob_token_ids`, or `allowed_token_ids` + top-k; render with `--chat-template-file` |

The bridge picks one of these transports at startup by probing, and refuses to
start rather than guessing. See [Transport probing](#transport-probing).

## Quality alignment

144 authored decisions from fastjev, scored row by row. The reference is the
committed **BF16 Torch** run; this bridge ran **llama.cpp Q8_0** over the same
fixture with the same metric definition (`mean_family_balanced_accuracy`).

| Metric | Reference (BF16 Torch) | Bridge (llama.cpp Q8_0) |
|---|---|---|
| `prompt_sha256` equal | — | **144 / 144** |
| `input_tokens` equal | — | **144 / 144** |
| `mean_family_balanced_accuracy` | 0.6863 | 0.6810 |
| `global_balanced_accuracy` | 0.6998 | 0.6930 |
| argmax agreement | — | 139 / 144 (96.5%) |

**`prompt_sha256` and `input_tokens` match on every one of the 144 rows.** That
is the strongest available evidence that the bridge renders the prompt
byte-identically to the reference implementation and tokenizes identically.
The 0.0053 quality delta is therefore attributable to Q8_0 quantization, not to
the bridge.

All five argmax disagreements fall on rows where the top two options are close
(top-2 margins between 0.01 and 0.24). The bridge flips two rows in the right
direction and three in the wrong one, which matches the observed net delta.
Per family, `candidate_selection` and `rule_application` are identical and only
`evidence_interpretation` moves (0.7675 → 0.7516).

Row-level evidence:

- [results/alignment-minicpm5-2b-authored144.json](https://github.com/lib-x/jev-bridge/blob/main/results/alignment-minicpm5-2b-authored144.json)
- [results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl](https://github.com/lib-x/jev-bridge/blob/main/results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl)

## Quick start

> Build tip: if the crates.io index is slow on your network, configure a mirror
> in a local `.cargo/config.toml`. That file is gitignored.

```bash
cargo build --release

export JEV_BRIDGE_UPSTREAM_KEY='<your upstream token>'

./target/release/jev-bridge \
  --base-url https://your-llama-server.example/v1 \
  --model MiniCPM5-2B-Q8_0 \
  --served-model bridge-minicpm5-2b \
  --served-model-release-date 2026-09-22 \
  --port 8100
```

Probe only, without serving:

```bash
./target/release/jev-bridge --probe-only \
  --base-url https://your-llama-server.example/v1 \
  --model MiniCPM5-2B-Q8_0 \
  --served-model-release-date 2026-09-22
```

The probe prints the chosen transport, its endpoint, and the sixteen answer-slot
token ids it resolved.

```bash
curl http://127.0.0.1:8100/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bridge-minicpm5-2b",
    "state": "I was charged twice and need a refund today.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which queue should handle this request?",
        "criteria": {
          "billing": "Billing, payments, and refunds.",
          "technical": "Bugs, outages, and integrations."
        }
      }
    }
  }'
```

### Batch scoring

`--score` reads fastjev-format rows and writes JSONL predictions, which is what
the alignment run above used:

```bash
./target/release/jev-bridge --score \
  --input authored144.jsonl \
  --output bridge-predictions.jsonl \
  --base-url https://your-llama-server.example/v1 \
  --model MiniCPM5-2B-Q8_0 \
  --served-model bridge-minicpm5-2b \
  --served-model-release-date 2026-09-22
```

Input rows are `{"id", "state", "question", "options": [{"id", "description"}]}`;
extra fields are ignored. Output rows carry `probabilities`, `option_logprobs`,
`input_tokens`, `prompt_sha256` and `prompt_version`. The output path must not
already exist.

This path **bypasses the System One criteria rendering** (which prefixes each
option description with `id: `), so it matches the reference prompt contract
exactly — that is why the alignment check can compare prompt hashes byte for
byte.

### Calibration evaluation

`--evaluate` is fully offline — no upstream connection, no `--base-url`, no
`--model`. It turns `--score` predictions plus a gold file into accuracy,
balanced accuracy, NLL, Brier (both flavours), ECE and the reliability curve,
stratified by family:

```bash
./target/release/jev-bridge --evaluate \
  --predictions bridge-predictions.jsonl \
  --gold gold.jsonl \
  --out report.json
```

Gold rows are `{"id", "gold", "family"?, "positive"?}`. `family` is the
stratification key (rows without one are reported under `unlabelled`, never
silently pooled); `positive` names the slot used for `brier_binary` and defaults
to `true` on a two-slot `true`/`false` row. Every definition is frozen in
[`src/evaluate.rs`](src/evaluate.rs).

A report is a *claim*; the prediction and gold files are the *evidence*.
`--verify` recomputes the report and compares it field by field, exiting
non-zero on any mismatch:

```bash
./target/release/jev-bridge --evaluate \
  --predictions bridge-predictions.jsonl \
  --gold gold.jsonl \
  --verify report.json        # -> "84 checks; 0 mismatches"
```

The comparison tolerance belongs to the caller (`--tol`, default `1e-12`) and is
never read out of the report, so a rewritten number cannot be laundered by
declaring a looser tolerance. This is the standard the alignment JSON in
`results/` is meant to meet: a number that cannot be recomputed is not evidence.

The bridge itself performs no calibration and recommends no thresholds — this
measures whether the probabilities are usable on your workload, and nothing
more.

### Batched readout

Several questions in one System One request share one `state`, so the bridge
reads them in **one** upstream request (an array `prompt`), letting a
prefix-caching server prefill the shared state once. Measured on llama.cpp
b11096 with a 9B Q8_0 model, 3 questions per arm, fresh prompts every round,
arms interleaved:

| Path | 3-question request (median) | Per question |
|---|---|---|
| batched (1 request) | **6.94 s** | 2312 ms |
| sequential (3 requests) | 8.64 s | 2880 ms |
| speedup | **1.25x** | — |

The win is bounded by what the server can share — here the prefill of a
~100-token prefix — so it is a 1.25x, not an order of magnitude. When the
endpoint rejects the array shape (llama.cpp builds before b11065 answered
`400`), the bridge falls back to sequential single-prompt calls; the fallback
is **counted** and reported as `batch_fallbacks` on `GET /health`, because a
run served that way did not get the shared prefill. A 429, 401/403 or 5xx is
never treated as a shape rejection — it propagates, so a throttled endpoint
cannot be laundered into a "successful" run.

Batched responses are read **all-or-nothing**: a different number of choices
than prompts, `index` fields that are not a permutation of `0..N`, or any
choice without a usable distribution fails the whole request. A partial
success would silently mis-align prompts and probabilities. The endpoint's one
`usage` block is attached to the first answer (the rest report 0), so summing
`input_tokens` reproduces the endpoint's own count instead of a faked
per-prompt split.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/systemone` | System One compatible scoring for `choice`, `noul` and `score` |
| GET | `/v1/models` | Configured model metadata |
| GET | `/health` | Chosen transport, endpoint, answer-slot ids, template kwargs |

`GET /health` is the first place to look when something is off: it states what
the bridge actually negotiated instead of leaving you to guess.

Every response's `fastjev` block also carries a `readout` declaration:
`{"status": "unvalidated"}` by default, or whatever the deployment states with
`--readout-status` / `--readout-evidence`. The readout channel — one token per
answer letter, log probabilities at the final position — assumes the served
model can actually use single-token slot letters. A general instruct model that
was never trained for it does not fail loudly; it degrades silently (a
catch-all option acquires a probability floor). The bridge cannot prove readout
quality at runtime without gold data, so the deployment has to say whether it
checked — `--evaluate` over an alignment run is exactly that check. `GET /health`
reports the same declaration.

`GET /health` also carries two runtime facts:

- `readout_check` — the startup self-check, when one ran: a few questions whose
  answers are obvious (`Is ice hotter than boiling water?`), read through the
  exact slot channel every request uses. It cannot prove the model is *good* at
  decisions (that needs a gold-labelled workload); it catches the model that is
  not answering the question at all. `--require-readout-check` turns a failed
  check into a refusal to start.
- `batch_fallbacks` — how many batched readouts fell back to sequential calls
  because the endpoint rejected the array-`prompt` shape (0 = every batch got
  the shared prefill).

## Using it as a library

```toml
[dependencies]
jev-bridge = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
reqwest = { version = "0.12", features = ["json"] }
serde_json = "1"
```

```rust
use jev_bridge::server::{Bridge, BridgeConfig};
use jev_bridge::wire::{Question, Row, RowOption};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bridge = Bridge::connect(
        reqwest::Client::new(),
        BridgeConfig {
            openai_base: "http://127.0.0.1:8080/v1".into(),
            native_base: "http://127.0.0.1:8080".into(),
            upstream_model: "MiniCPM5-2B-Q8_0".into(),
            upstream_key: None,
            served_model: "bridge-minicpm5-2b".into(),
            description: "local bridge".into(),
            release_date: "2026-09-22".into(),
            chat_template_kwargs: serde_json::json!({"enable_thinking": false}),
            max_input_tokens: None,
            local: Default::default(),
            readout: Default::default(),
        },
    )
    .await?;

    // One decision that carries its own state (the `--score` shape).
    let row = Row {
        id: "ticket".into(),
        state: serde_json::json!("I was charged twice."),
        question: "Which queue should handle this?".into(),
        options: vec![
            RowOption { id: "billing".into(), description: "Refunds and payments.".into() },
            RowOption { id: "sales".into(), description: "Pricing.".into() },
        ],
    };
    let score = bridge.score_row(&row).await?;
    println!("{} {:?}", score.answer.id, score.answer.probabilities);

    // Several questions about one state (the System One shape): the state is
    // serialised once, and the whole batch is read in one upstream request.
    let questions = vec![
        Question {
            id: "queue".into(),
            question: "Which queue should handle this?".into(),
            options: row.options.clone(),
        },
        Question {
            id: "urgent".into(),
            question: "Is this urgent?".into(),
            options: vec![
                RowOption { id: "true".into(), description: "Yes".into() },
                RowOption { id: "false".into(), description: "No".into() },
            ],
        },
    ];
    let state = serde_json::json!("I was charged twice.");
    for score in bridge.score_questions(&state, &questions).await? {
        println!("{} {:?}", score.answer.id, score.answer.probabilities);
    }
    Ok(())
}
```

Serve the same bridge over HTTP with `jev_bridge::server::router`:

```rust
use std::sync::Arc;
use jev_bridge::server::{router, AppState};

let state = Arc::new(AppState { bridge, api_key: None, readout_check: None });
let listener = tokio::net::TcpListener::bind("127.0.0.1:8100").await?;
axum::serve(listener, router(state)).await?;
```

Public modules:

| Module | Contents |
|---|---|
| `server` | `Bridge`, `BridgeConfig`, `DetailedScore`, `UpstreamStatus`, `router`, `AppState` |
| `strategy` | `Probe`, typed request bodies, `CompletionResponse`, parsing, batch alignment |
| `prompt` | prompt contract, `ChatMessage`, `RuntimeClient`, `LocalComponents`, `ChatTemplate` |
| `render` | `LocalRenderer` — minijinja rendering with CPython string methods |
| `tokenizer` | `LocalTokenizer` — the `tokenizers` crate behind the same checks |
| `wire` | `SystemOneResponse`, `Answer`, `Row`, `Question`, `RequestBatch`, `ReadoutStatus` |
| `evaluate` | offline calibration metrics and report verification |
| `readout` | the startup readout self-check (`Probe`, `ReadoutCheck`) |

Bodies, responses and answers are typed structs rather than loose JSON
documents: the scoring request/response pairs, the System One response with its
`choice`/`noul`/`score` enum, and the health payload are all `Serialize`/
`Deserialize` types. `serde_json::Value` remains only where the payload really
is dynamic — the caller's `state`, the values inside `criteria`, and the Python
JSON layout that the prompt contract pins byte for byte.

## Running without runtime endpoints

Runtimes differ in which helper endpoints they expose. The bridge uses the
runtime for both rendering and tokenization by default, and each half can be
moved in-process independently:

| Flag | Replaces | Needed for |
|---|---|---|
| `--chat-template-file` + `--chat-template-context` | `/apply-template` | vLLM and anything else without that endpoint |
| `--tokenizer-json` | `/tokenize` and `/detokenize` | runtimes with no tokenize endpoint, or fully offline startup |

```bash
# Render locally, tokenize through the runtime
./target/release/jev-bridge \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-4B \
  --served-model bridge-qwen3.5-4b \
  --served-model-release-date 2026-09-22 \
  --chat-template-file qwen3.5.chat_template.jinja \
  --chat-template-context '{"bos_token": "", "eos_token": "<|im_end|>"}'

# Both halves local: no /apply-template and no /tokenize needed
./target/release/jev-bridge \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-4B \
  --served-model bridge-qwen3.5-4b \
  --served-model-release-date 2026-09-22 \
  --chat-template-file qwen3.5.chat_template.jinja \
  --chat-template-context '{"bos_token": "", "eos_token": "<|im_end|>"}' \
  --tokenizer-json /models/Qwen3.5-4B/tokenizer.json
```

Take the **model's own** template and tokenizer, never a hand-written or
foreign pair: a different vocabulary silently scores the wrong token ids, and a
different template silently changes the prompt. For a llama.cpp server both can
be read from `/props`:

```bash
curl -s "http://127.0.0.1:8080/props?model=$MODEL" | jq -r .chat_template > template.jinja
```

The template context carries the variables transformers supplies besides the
messages. A template that begins with `{{- bos_token }}` needs `bos_token`
there, or the rendered prompt silently lacks its BOS token.

`GET /health` reports `renders_locally` and `tokenizes_locally`, so you can
confirm what the bridge actually negotiated.

Templates written for transformers use Python string methods (`split`,
`replace`, `startswith`, `strip`, …) that the Rust minijinja lacks. The bridge
supplies them with CPython semantics, including treating the argument of
`strip`/`lstrip`/`rstrip` as a character set rather than a prefix. Local
rendering was checked byte for byte against llama.cpp's own `/apply-template`
output on a real model template.

## Examples

Three runnable examples live in `examples/`. Each reads its configuration from
the environment, so no credentials are baked in:

| Example | Shows |
|---|---|
| `cargo run --example score_rows` | connecting a `Bridge`, scoring rows, reading `DetailedScore` |
| `cargo run --example serve` | exposing the same bridge over HTTP with `router` |
| `cargo run --example local_template` | rendering a template in-process and inspecting the prompt |

```bash
export JEV_BRIDGE_UPSTREAM_URL=http://127.0.0.1:8080/v1
export JEV_BRIDGE_UPSTREAM_KEY=...        # only if the server requires it
cargo run --example score_rows
```

## Transport probing

At startup the bridge tries each transport below with a real sixteen-option
probe request and adopts the first one that works:

| Name | Endpoint | Fields relied on | Typical runtime |
|---|---|---|---|
| `vllm-logprob-token-ids` | `/v1/completions` | `allowed_token_ids` + `logprob_token_ids` | vLLM |
| `vllm-allowed-token-ids-top-k` | `/v1/completions` | `allowed_token_ids` + top-k | vLLM (fallback) |
| `openai-completions-top-k` | `/v1/completions` | top-k | generic OpenAI-compatible, llama.cpp |
| `llamacpp-native-n-probs` | `/completion` | `n_probs` | llama.cpp native |

The two vLLM transports claim the distribution is restricted to the option
tokens. If a server ignores those fields and returns foreign tokens, the
transport is **rejected** rather than silently adopted — otherwise `/health`
would report a contract the service is not actually honouring.

Unrestricted distributions need a generous top-k: on a 2B quantized model,
`k=20` missed four of the sixteen answer letters and `k=50` covered all of
them. Unrestricted transports therefore request `4 × options` (floor 32, cap
128). A missing option is always an error, never a zero.

## Startup contracts

Two checks run before serving; failing either exits instead of scoring wrong
numbers:

1. **Single-token contract** — each of `A`–`P` must be exactly one token, and
   `/detokenize` must return them as `ABCDEFGHIJKLMNOP`.
2. **Boundary contract** — appending a letter to the rendered prompt must
   produce exactly `prompt tokens + [that letter's token]`. Otherwise the
   prompt's last token merges with the letter and the scored logit belongs to a
   different token than the chosen option.

## CLI reference

| Flag | Description |
|---|---|
| `--base-url` | Upstream OpenAI-compatible root, e.g. `http://127.0.0.1:8080/v1` |
| `--native-base-url` | Native root for llama.cpp `/completion`; defaults to `--base-url` without `/v1` |
| `--model` | Model name sent upstream |
| `--upstream-key` | Upstream bearer token; prefer `JEV_BRIDGE_UPSTREAM_KEY` |
| `--served-model` | Model id this bridge accepts; defaults to `--model`; must not start with `jev` |
| `--served-model-description` | Description returned by `GET /v1/models` |
| `--served-model-release-date` | ISO date returned by `GET /v1/models` (required) |
| `--chat-template-kwargs` | JSON passed when rendering the template; default `{"enable_thinking": false}` |
| `--chat-template-file` | Render the template locally from this Jinja file instead of calling `/apply-template` |
| `--chat-template-context` | JSON added to every local render, e.g. `{"bos_token": "<s>"}` |
| `--tokenizer-json` | Tokenize locally with this `tokenizer.json` instead of calling `/tokenize` |
| `--max-input-tokens` | Reject rows above this token count (no truncation) |
| `--host` / `--port` | Bind address, default `127.0.0.1:8100` |
| `--api-key` | Bearer token clients must present; prefer `JEV_BRIDGE_API_KEY` |
| `--probe-only` | Probe, print the result, exit |
| `--score` / `--input` / `--output` | Batch scoring mode |
| `--evaluate` / `--predictions` / `--gold` / `--out` | Offline calibration metrics from predictions and a gold file |
| `--verify` / `--tol` | Recompute an evaluation report and compare it against the file |
| `--bins` | Equal-width confidence bins for ECE and the reliability curve (default 10) |
| `--readout-status` / `--readout-evidence` | Readout-channel declaration on `/health` and every answer (default `unvalidated`) |
| `--require-readout-check` | Refuse to start when the startup readout self-check does not pass (default: report only) |
| `--upstream-timeout-secs` | Timeout for one upstream request (default 600; connect timeout is 10) |

`--base-url`, `--model` and `--served-model-release-date` are required unless
`--evaluate` is given, which needs neither.

Credentials are read from environment variables or the command line and are
never written to disk.

## Alignment with fastjev

- The payload uses Python's `json.dumps(..., ensure_ascii=False)` separator
  layout (`, ` and `: `); a unit test pins this against real CPython output, and
  the alignment run confirms it with 144 matching prompt hashes.
- Softmax matches fastjev's helper (subtract the maximum first).
- `fastjev.confidence_method` is `one-minus-normalized-entropy`, matching
  fastjev's System One adapter; it is not TypeSafe's private statistic.
- Template rendering is delegated to the serving runtime (`/apply-template`), so
  the prompt is exactly what that runtime feeds the model. This is also why the
  **chat completions endpoint must not be used**: there the prompt is decided by
  the server's template and cannot be guaranteed to match this contract.

## Related projects

- [fastjev](https://github.com/chengyongru/fastjev) (MIT) — the
  `direct-options-v1` contract this bridge implements, and the source of the 144
  authored decisions used for the alignment run.
- [jev-clone](https://github.com/alitrack/jev-clone) (Apache-2.0) — an
  independent, contract-compatible System One server focused on *measurement*.
  The frozen metric definitions in [`src/evaluate.rs`](src/evaluate.rs) (ECE,
  both Brier flavours, NLL, the reliability curve, the first-max argmax rule)
  and the hand-computed four-row test fixture follow its published
  definitions; the `readout` declaration on `/health` and in every answer
  (`--readout-status`) follows its finding that the readout channel is
  model-dependent — a model never trained for single-token slot readout does
  not fail loudly, it degrades silently. The implementation here is independent
  Rust; see the upstream repository for the original work and its evidence.
- [TypeSafe System One](https://docs.typesafe.ai) — the public HTTP contract the
  wire format is field-compatible with.

## Testing

```bash
cargo test
```

103 tests: 73 unit tests (wire contract, CPython JSON layout, the three
response shapes, softmax stability, missing options must error,
CPython-semantics template methods, the frozen evaluation-metric definitions,
and the readout self-check's probes) plus 30 integration tests across
`tests/bridge.rs` (library API, including `DetailedScore` fields),
`tests/contract.rs` (HTTP surface, local rendering, local tokenization, the
readout declaration, the readout self-check, and the batched-readout paths:
one request for several questions, the sequential fallback and its counter,
and the all-or-nothing failures), and `tests/evaluate.rs` (the `--evaluate` /
`--verify` CLI round trip and its failure modes: a tampered report, a report
from other inputs, out-of-sync files).

`tests/latency.rs` is `#[ignore]`d: it measures real decision latency against
a reachable upstream (endpoint, model and key come from
`JEV_BRIDGE_UPSTREAM_URL` / `JEV_BRIDGE_MODEL` / `JEV_BRIDGE_UPSTREAM_KEY`,
nothing is written to disk):

```bash
JEV_BRIDGE_UPSTREAM_URL=... JEV_BRIDGE_MODEL=... JEV_BRIDGE_UPSTREAM_KEY=... \
  cargo test --release --test latency -- --ignored --nocapture
```

## Known limitations

- A batched readout only wins what the server can share: on llama.cpp b11096
  with a 9B Q8_0 model it is 1.25x (the prefill of the shared state), not an
  order of magnitude. On an endpoint whose own prefix cache already serves the
  sequential path, it can be ~1x.
- The upstream revision cannot be pinned by this process. Reproducibility
  depends on the upstream's own version management.
- The alignment reference is BF16 while the bridge run is Q8_0, so the 0.0053
  delta is a quantization delta, not a bridge delta. An equal-precision
  comparison needs a BF16 upstream.
- With a local tokenizer the bridge trusts the file it is given. A foreign
  `tokenizer.json` produces plausible-looking scores from the wrong token ids,
  so `verify_special_tokens` checks the context's special tokens against the
  vocabulary at startup but cannot check that the vocabulary is the model's own.
