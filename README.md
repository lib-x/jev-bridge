# jev-bridge

[English](README.md) | [简体中文](README.zh-CN.md)

Turn any generic OpenAI-compatible inference service into a Jev-style decision
scoring service, in a single 7 MB binary with no Python dependency.

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
| vLLM | probe paths implemented, local template rendering not yet | `allowed_token_ids` + `logprob_token_ids`, or `allowed_token_ids` + top-k |

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

- [results/alignment-minicpm5-2b-authored144.json](results/alignment-minicpm5-2b-authored144.json)
- [results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl](results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl)

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

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/systemone` | System One compatible scoring for `choice`, `noul` and `score` |
| GET | `/v1/models` | Configured model metadata |
| GET | `/health` | Chosen transport, endpoint, answer-slot ids, template kwargs |

`GET /health` is the first place to look when something is off: it states what
the bridge actually negotiated instead of leaving you to guess.

## Using it as a library

```toml
[dependencies]
jev-bridge = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
reqwest = { version = "0.12", features = ["json"] }
serde_json = "1"
```

```rust
use jev_bridge::server::{Bridge, BridgeConfig};
use jev_bridge::wire::{Row, RowOption};

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
        },
    )
    .await?;

    let rows = vec![Row {
        id: "ticket".into(),
        state: serde_json::json!("I was charged twice."),
        question: "Which queue should handle this?".into(),
        options: vec![
            RowOption { id: "billing".into(), description: "Refunds and payments.".into() },
            RowOption { id: "sales".into(), description: "Pricing.".into() },
        ],
    }];

    for score in bridge.score_rows(&rows).await? {
        println!("{} {:?}", score.answer.id, score.answer.probabilities);
    }
    Ok(())
}
```

Serve the same bridge over HTTP with `jev_bridge::server::router`:

```rust
use std::sync::Arc;
use jev_bridge::server::{router, AppState};

let state = Arc::new(AppState { bridge, api_key: None });
let listener = tokio::net::TcpListener::bind("127.0.0.1:8100").await?;
axum::serve(listener, router(state)).await?;
```

Public modules:

| Module | Contents |
|---|---|
| `server` | `Bridge`, `BridgeConfig`, `DetailedScore`, `router`, `AppState` |
| `strategy` | `Probe`, transport bodies, response parsing |
| `prompt` | prompt contract, single-token check, softmax, `ServerTemplate` |
| `wire` | System One request validation and response construction |

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
| `--max-input-tokens` | Reject rows above this token count (no truncation) |
| `--host` / `--port` | Bind address, default `127.0.0.1:8100` |
| `--api-key` | Bearer token clients must present; prefer `JEV_BRIDGE_API_KEY` |
| `--probe-only` | Probe, print the result, exit |
| `--score` / `--input` / `--output` | Batch scoring mode |

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

## Testing

```bash
cargo test
```

39 tests: 26 unit tests (wire contract, CPython JSON layout, the three response
shapes, softmax stability, missing options must error) plus 13 integration tests
split between `tests/bridge.rs` (library API, including `DetailedScore` fields)
and `tests/contract.rs` (HTTP surface).

## Known limitations

- Only runtimes that render templates server-side are supported today
  (llama.cpp `/apply-template` + `/tokenize`). Reaching vLLM needs a local
  tokenizer plus minijinja rendering path, which is not implemented.
- Decisions are serialized: several questions in one request are scored one
  after another, and upstream calls are not batched.
- The upstream revision cannot be pinned by this process. Reproducibility
  depends on the upstream's own version management.
- The alignment reference is BF16 while the bridge run is Q8_0, so the 0.0053
  delta is a quantization delta, not a bridge delta. An equal-precision
  comparison needs a BF16 upstream.
