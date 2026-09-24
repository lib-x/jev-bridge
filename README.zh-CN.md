<div align="center">

# jev-bridge

**把任意通用的 OpenAI 兼容推理服务，变成 Jev 风格的决策打分服务。**

单个 7 MB 二进制，不依赖 Python。

[![crates.io](https://img.shields.io/crates/v/jev-bridge.svg)](https://crates.io/crates/jev-bridge)
[![docs.rs](https://docs.rs/jev-bridge/badge.svg)](https://docs.rs/jev-bridge)
[![license](https://img.shields.io/crates/l/jev-bridge.svg)](https://github.com/lib-x/jev-bridge/blob/main/LICENSE)
[![downloads](https://img.shields.io/crates/d/jev-bridge.svg)](https://crates.io/crates/jev-bridge)
[![rust](https://img.shields.io/badge/rust-2024%20edition-blue.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)

[English](https://github.com/lib-x/jev-bridge/blob/main/README.md) | [简体中文](https://github.com/lib-x/jev-bridge/blob/main/README.zh-CN.md)

</div>

## 它做什么

Jev 的能力不是权重里的东西，而是一层读法协议。这个桥接层在
[`direct-options-v1`](https://github.com/chengyongru/fastjev) 契约上实现了这层读法，
只要上游能返回 next-token 的对数概率就能接：

1. 用**服务端自己的** chat template 渲染一条固定的决策 prompt
   （system 指令 + `{"evidence", "criterion", "options"}` JSON payload，关闭 thinking）；
2. 只做一次前向，取**最后一个位置**；
3. 只在 `A`–`P` 这 2–16 个固定答案字母的 token 上做 softmax；
4. 不生成任何答案文本，返回 System One 兼容的 typed 结果。

上游模型、权重、量化格式都保持原样。本进程不复制权重，也不修改模型。

## 它不做什么

- 不是 Jev 的复刻，也不冒充 Jev：`--served-model` 拒绝以 `jev` 开头的名字。
- 返回的概率是**给定选项下的条件分数**，`calibrated=False`，不是可直接用于自动化的置信度。
- 不做校准，不推荐阈值。上线前需要在自己的工作负载上验证。

## 支持的类型

### 问题类型（`POST /v1/systemone`）

| 类型 | 形状 | 返回 |
|---|---|---|
| `choice` | `criteria` 对象，2–16 个选项 | 获胜选项 id、完整分布、`confidence` |
| `noul` | 可选 `criteria`，键为 `true` / `false` | 分配给 `true` 的概率 |
| `score` | 有序 `criteria` 数组，2–10 个等级 | 概率加权值、`legend`、`confidence` |

三种类型的 `instructions` 都可以省略，省略时会补一个类型相关的通用问题。criteria 的值
可以是字符串、对象、数组或 `null`，结构化值会渲染成 JSON。

### 上游运行时

| 运行时 | 状态 | 接入方式 |
|---|---|---|
| **llama.cpp**（`llama-server`） | 完整支持 | 契约走 `/apply-template` + `/tokenize`，打分走 `/v1/completions` 或 `/completion` |
| 通用 OpenAI 兼容 | 在暴露 logprobs 时可支持 | `/v1/completions` + `logprobs` |
| vLLM | 支持，需自行提供模板 | `allowed_token_ids` + `logprob_token_ids`，或 `allowed_token_ids` + top-k；用 `--chat-template-file` 本地渲染 |

启动时通过探测在下面几种传输里选一个，选不出来就拒绝启动，不猜。见
[传输探测](#传输探测)。

## 质量对齐

用 fastjev 的 144 条 authored 决策逐行打分。参考侧是已提交的 **BF16 Torch** 预测，
本桥接跑的是 **llama.cpp Q8_0**，同一份 fixture、同一套指标定义
（`mean_family_balanced_accuracy`，按 family 分组后取 balanced accuracy 均值）。

| 指标 | 参考实现 (BF16 Torch) | 本桥接 (llama.cpp Q8_0) |
|---|---|---|
| `prompt_sha256` 一致 | — | **144 / 144** |
| `input_tokens` 一致 | — | **144 / 144** |
| `mean_family_balanced_accuracy` | 0.6863 | 0.6810 |
| `global_balanced_accuracy` | 0.6998 | 0.6930 |
| argmax 逐行一致 | — | 139 / 144 (96.5%) |

**`prompt_sha256` 与 `input_tokens` 在全部 144 行上完全相同**，这是现有手段里最强的
一条证据：桥接层渲染出的 prompt 与参考实现逐字节一致，分词也完全对齐。因此
0.0053 的质量差只能归因于 Q8_0 量化，而不是桥接实现本身。

5 个 argmax 分歧全部落在 top-2 概率接近的行上（分歧行的 top-2 间距在 0.01–0.24），
其中桥接侧改判对了 2 行、改判错了 3 行，净差与观测到的质量差一致。逐 family 看，
`candidate_selection` 与 `rule_application` 完全一致，差异只在
`evidence_interpretation`（0.7675 → 0.7516）。

行级证据：

- [results/alignment-minicpm5-2b-authored144.json](https://github.com/lib-x/jev-bridge/blob/main/results/alignment-minicpm5-2b-authored144.json)
- [results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl](https://github.com/lib-x/jev-bridge/blob/main/results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl)

## 快速开始

> 构建提示：如果访问 crates.io 索引很慢，可以在本地 `.cargo/config.toml` 里配置镜像源。
> 该文件已在 `.gitignore` 中，不会进入版本库。

```bash
cargo build --release

export JEV_BRIDGE_UPSTREAM_KEY='<你的上游 token>'

./target/release/jev-bridge \
  --base-url https://your-llama-server.example/v1 \
  --model MiniCPM5-2B-Q8_0 \
  --served-model bridge-minicpm5-2b \
  --served-model-release-date 2026-09-22 \
  --port 8100
```

先只探测、不起服务：

```bash
./target/release/jev-bridge --probe-only \
  --base-url https://your-llama-server.example/v1 \
  --model MiniCPM5-2B-Q8_0 \
  --served-model-release-date 2026-09-22
```

探测会打印选中的传输方式、端点，以及解析出的 16 个答案字母 token id。

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

### 批量打分

`--score` 直接读 fastjev 格式的 rows，写 JSONL 预测，上面的对齐验证就是这么跑的：

```bash
./target/release/jev-bridge --score \
  --input authored144.jsonl \
  --output bridge-predictions.jsonl \
  --base-url https://your-llama-server.example/v1 \
  --model MiniCPM5-2B-Q8_0 \
  --served-model bridge-minicpm5-2b \
  --served-model-release-date 2026-09-22
```

输入行是 `{"id", "state", "question", "options": [{"id", "description"}]}`，多余字段会被忽略。
输出行带 `probabilities`、`option_logprobs`、`input_tokens`、`prompt_sha256`、`prompt_version`。
输出路径必须是不存在的文件。

这条路径**绕过 System One 的 criteria 渲染**（后者会给每个选项描述加上 `id: ` 前缀），
因此与参考实现的 prompt 契约完全一致——这正是对齐验证能逐字节比较 prompt 哈希的原因。

### 校准评估

`--evaluate` 完全离线——不连上游，不需要 `--base-url`，也不需要 `--model`。
它把 `--score` 的预测加一个 gold 文件算成 accuracy、balanced accuracy、NLL、
Brier（两种口径）、ECE 与可靠性曲线，并按 family 分层：

```bash
./target/release/jev-bridge --evaluate \
  --predictions bridge-predictions.jsonl \
  --gold gold.jsonl \
  --out report.json
```

gold 每行是 `{"id", "gold", "family"?, "positive"?}`。`family` 是分层键（缺省的行
记为 `unlabelled`，绝不悄悄并进总分）；`positive` 指定 `brier_binary` 的正类，
两槽 `true`/`false` 行默认 `true`。全部口径冻结在 [`src/evaluate.rs`](src/evaluate.rs)。

报告是**主张**，预测与 gold 文件是**证据**。`--verify` 重算报告并逐字段比对，
任何不一致都以非零退出：

```bash
./target/release/jev-bridge --evaluate \
  --predictions bridge-predictions.jsonl \
  --gold gold.jsonl \
  --verify report.json        # -> "84 checks; 0 mismatches"
```

比对容差属于调用方（`--tol`，默认 `1e-12`），绝不从报告里读取——改写过的数字
不能靠自报一个更宽的容差蒙混过关。`results/` 里的对齐 JSON 同样按这个标准要求：
重算不出来的数字不算证据。

桥接层自身不做校准、不推荐阈值——这里只测量概率在你的负载上是否可用，仅此而已。

### 批量读

一个 System One 请求里的多个问题共享同一个 `state`，所以桥接层用**一次**上游请求
（数组 `prompt`）读完它们，让带前缀缓存的服务端只 prefill 一次共享前缀。在 llama.cpp
b11096 + 9B Q8_0 模型上实测（每轮 3 个问题、每轮全新 prompt、两种路径交替）：

| 路径 | 3 问题请求（中位数） | 每问题 |
|---|---|---|
| 批量（1 次请求） | **6.94 s** | 2312 ms |
| 逐条（3 次请求） | 8.64 s | 2880 ms |
| 加速比 | **1.25x** | — |

收益的上限就是服务端能共享的部分（这里是一个 ~100 token 前缀的 prefill），所以是
1.25x，不是数量级。端点拒绝数组形状时（b11065 之前的 llama.cpp 构建返回 `400`），
桥接层回退为逐条单 prompt 请求；回退会被**计数**并出现在 `GET /health` 的
`batch_fallbacks` 上——这样跑出来的结果没有拿到共享 prefill，报告里必须能看出来。
429、401/403 或 5xx 绝不当作形状拒绝：它们直接传播，限流不会被洗成"成功"。

批量响应按**全有或全无**读取：choices 数与 prompts 数不符、`index` 字段不是 `0..N`
的排列、或任一 choice 没有可用分布，整个请求失败——部分成功会让 prompt 与概率
静默错位。端点的那一个 `usage` 块附在第一个答案上（其余报 0），所以
`input_tokens` 求和等于端点自己报的数，而不是伪造的按题拆分。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/systemone` | System One 兼容打分，支持 `choice` / `noul` / `score` |
| GET | `/v1/models` | 返回配置的模型元数据 |
| GET | `/health` | 返回选中的传输、端点、答案 token id、模板参数 |

出问题时先看 `GET /health`：它把实际协商到的结果显式暴露出来，而不是让你猜服务端支持什么。

每个响应的 `fastjev` 块还带 `readout` 声明：默认 `{"status": "unvalidated"}`，
或用 `--readout-status` / `--readout-evidence` 声明部署方验证过的状态。
readout 通道（每个答案字母一个 token、读末位 log 概率）假设被服务的模型真的会用
单 token 槽位字母。未经此训练的通用 instruct 模型不会报错，而是静默退化（某个
兜底选项会获得概率地板）。桥接层无法在运行时自证 readout 质量，所以必须由部署方
说明是否验证过——拿对齐运行跑一遍 `--evaluate` 就是那次验证。`GET /health`
报告同一份声明。

`GET /health` 还带两项运行时事实：

- `readout_check` —— 启动自检的结果（跑过才有）：几条答案显而易见的问题
  （`Is ice hotter than boiling water?`），走每个真实请求都在用的同一条槽位通道。
  它不能证明模型**擅长**决策（那需要带 gold 的负载）；它抓的是"根本没在回答问题"
  的模型。`--require-readout-check` 会把自检不通过变成拒绝启动。
- `batch_fallbacks` —— 有多少批量读因为端点拒绝数组 `prompt` 形状而回退成逐条请求
  （0 = 每个批次都拿到了共享 prefill）。

## 作为库使用

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

    // 单条决策，自带 state（`--score` 的形状）。
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

    // 多个问题共享一个 state（System One 的形状）：state 只序列化一次，
    // 整批用一次上游请求读完。
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

用 `jev_bridge::server::router` 把同一个 bridge 挂成 HTTP 服务：

```rust
use std::sync::Arc;
use jev_bridge::server::{router, AppState};

let state = Arc::new(AppState { bridge, api_key: None, readout_check: None });
let listener = tokio::net::TcpListener::bind("127.0.0.1:8100").await?;
axum::serve(listener, router(state)).await?;
```

公开模块：

| 模块 | 内容 |
|---|---|
| `server` | `Bridge`、`BridgeConfig`、`DetailedScore`、`UpstreamStatus`、`router`、`AppState` |
| `strategy` | `Probe`、类型化的请求体、`CompletionResponse`、解析、批量对齐 |
| `prompt` | prompt 契约、`ChatMessage`、`RuntimeClient`、`LocalComponents`、`ChatTemplate` |
| `render` | `LocalRenderer` —— minijinja 渲染 + CPython 字符串方法 |
| `tokenizer` | `LocalTokenizer` —— `tokenizers` crate，接入同一套校验 |
| `wire` | `SystemOneResponse`、`Answer`、`Row`、`Question`、`RequestBatch`、`ReadoutStatus` |
| `evaluate` | 离线校准指标与报告复算 |
| `readout` | 启动 readout 自检（`Probe`、`ReadoutCheck`） |

请求体、响应与答案都是类型化 struct，而不是松散的 JSON 文档：打分请求/响应对、带
`choice`/`noul`/`score` 枚举的 System One 响应、health 负载，全都是 `Serialize`/`Deserialize`
类型。`serde_json::Value` 只保留在真正动态的地方——调用方的 `state`、`criteria` 里的值，
以及 prompt 契约要求逐字节固定的 Python JSON 布局。

## 不依赖运行时端点

不同运行时暴露的辅助端点不一样。桥接层默认把渲染和分词都交给运行时，但这两半可以各自独立地搬到本进程：

| 参数 | 替代 | 用途 |
|---|---|---|
| `--chat-template-file` + `--chat-template-context` | `/apply-template` | vLLM 以及没有该端点的运行时 |
| `--tokenizer-json` | `/tokenize` 与 `/detokenize` | 没有分词端点，或需要完全离线启动 |

```bash
# 本地渲染，分词仍走运行时
./target/release/jev-bridge \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-4B \
  --served-model bridge-qwen3.5-4b \
  --served-model-release-date 2026-09-22 \
  --chat-template-file qwen3.5.chat_template.jinja \
  --chat-template-context '{"bos_token": "", "eos_token": "<|im_end|>"}'

# 两半都本地：不再需要 /apply-template 和 /tokenize
./target/release/jev-bridge \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-4B \
  --served-model bridge-qwen3.5-4b \
  --served-model-release-date 2026-09-22 \
  --chat-template-file qwen3.5.chat_template.jinja \
  --chat-template-context '{"bos_token": "", "eos_token": "<|im_end|>"}' \
  --tokenizer-json /models/Qwen3.5-4B/tokenizer.json
```

务必使用**模型自己的**模板与分词器，不要手写、也不要混用别的模型：词表不对会静默地给错误的
token id 打分，模板不对会静默地改变 prompt。llama.cpp 服务上两者都能从 `/props` 取到：

```bash
curl -s "http://127.0.0.1:8080/props?model=$MODEL" | jq -r .chat_template > template.jinja
```

模板上下文承载 transformers 除 messages 之外提供的变量。以 `{{- bos_token }}` 开头的模板
必须在这里给出 `bos_token`，否则渲染出的 prompt 会静默地缺少 BOS。

`GET /health` 会报告 `renders_locally` 与 `tokenizes_locally`，可以直接确认桥接层实际协商到的形态。

为 transformers 编写的模板会用到 Python 风格字符串方法（`split`、`replace`、`startswith`、
`strip` 等），Rust 的 minijinja 并不提供。桥接层按 CPython 语义补齐了它们，包括把
`strip`/`lstrip`/`rstrip` 的参数当作字符集合而不是前缀。本地渲染已用真实模型模板与
llama.cpp 自己的 `/apply-template` 输出做过逐字节比对。

## Examples

`examples/` 下有四个可直接运行的示例，配置全部从环境变量（或参数）读取，没有硬编码任何凭据：

| 示例 | 展示内容 |
|---|---|
| `cargo run --example score_rows` | 连接 `Bridge`、给 rows 打分、读取 `DetailedScore` |
| `cargo run --example serve` | 用 `router` 把同一个 bridge 挂成 HTTP 服务 |
| `cargo run --example local_template` | 在本地渲染模板并检查渲染出的 prompt |
| `cargo run --example playground` | 把 [djev-run](https://github.com/taeold/djev-run) 的浏览器游戏（snake / dino / tetris）接到运行中的 bridge 上 |

```bash
export JEV_BRIDGE_UPSTREAM_URL=http://127.0.0.1:8080/v1
export JEV_BRIDGE_UPSTREAM_KEY=...        # 仅当服务端需要时
cargo run --example score_rows
```

playground 需要一个接受 demo 硬编码 model id 的 bridge（`--accept-any-model`），
以及一份 demo 页面的 checkout（它们没有 license，所以不随本仓库分发）：

```bash
# 终端 1：bridge
jev-bridge --base-url http://127.0.0.1:8080/v1 --model my-model \
  --served-model bridge-my-model --served-model-release-date 2026-09-23 \
  --accept-any-model

# 终端 2：playground
git clone https://github.com/taeold/djev-run
cargo run --example playground -- --demo-dir ./djev-run

# 浏览器
open http://127.0.0.1:8000/
```

playground 首页展示 bridge 的 `/health` 并预填实际服务的模型名；同时向每个页面注入
一小段脚本（**上游文件不被修改**）：

- **替换产品标题**：页面原本写着 `djev / snake`、`djev (DiffusionGemma-Jev)`，
  在浏览器里被改写为 `jev-bridge`——因为真正在服务它们的就是 jev-bridge。
  署名链接（`mmastrac/djev-spark`、`trungdq88/jev-tetris`）保持原样。
- **从 URL 配置 API**：`?model=<id>` 改写每个请求的 `model` 字段（默认用 bridge
  实际服务的模型，所以不需要 `--accept-any-model`）；`?api=<url>` 把页面指向别的 bridge。
- **自动玩**：`?auto=1`（默认）在游戏空闲时自动点开始按钮——打开即玩，结束后自动重开。

注意 `--timeout-secs`（默认 600）：各游戏请求大小差别很大——snake 约 200 token，
tetris 要发整块棋盘加十六个落点，在慢上游上一次请求可能要好几分钟。
reqwest 默认的 30 秒会把它变成 502，游戏就静默降级到内置沙盒了。

## 传输探测

启动时按顺序用真实的 16 选项探针请求尝试下面的传输，第一个通过的会被采用：

| 名称 | 端点 | 依赖的字段 | 典型运行时 |
|---|---|---|---|
| `vllm-logprob-token-ids` | `/v1/completions` | `allowed_token_ids` + `logprob_token_ids` | vLLM |
| `vllm-allowed-token-ids-top-k` | `/v1/completions` | `allowed_token_ids` + top-k | vLLM（降级） |
| `openai-completions-top-k` | `/v1/completions` | top-k | 通用 OpenAI 兼容、llama.cpp |
| `llamacpp-native-n-probs` | `/completion` | `n_probs` | llama.cpp 原生 |

两条 vLLM 传输声明"分布被限制到选项 token"。如果服务端忽略了这些字段、返回了选项之外的
token，该传输会被**拒绝**而不是默默采用——否则 `/health` 报告的就是服务端并没有兑现的契约。

不受限分布需要足够大的 top-k，而且缺了选项时会**自动加深重试**而不是直接失败：
实测 2B 量化模型上 `k=20` 漏掉 16 个答案字母里的 4 个；9B 模型在约 1000 token
的 prompt 下 `k=64` 也会漏掉一个。因此不受限传输从 `16 × 选项数` 起步（下限 32，
上限 256），并沿阶梯加深——4×、16×、再到整个词表——每一步都复用服务端的 prefix
cache，重试的代价只比一次读取略高。只有最深的一步仍然缺失才报错，绝不会补 0。
同一套阶梯也保护传输探测：一次浅请求不会让可用的传输被误判为不支持。

### 选项顺序：字母契约与二元契约

字母契约把全部选项列在一个 prompt 里、读答案字母，模型因此按顺序看到选项——而它有位置
偏好。在参考端点上用两个接近的选项实测：同一问题，`frustrated` 排第一时得 0.692，
顺序反过来后 `furious` 得 0.759——**答案翻转**，三轮全部如此。

`--scoring binary` 消除了这种依赖：每个候选单独成一个 prompt（`Candidate: billing` /
`Does this candidate match the context?`），模型只回答 `yes` 或 `no`，候选彼此不可见。
同一组实测在两个顺序下给出**完全相同**的概率，候选概率就是 yes 概率做线性归一化的结果。
代价是每候选一次上游请求；运行时的 prefix cache 会复用共享证据（实测 345 个 token 中
命中 341 个），只有问题尾部需要重新 prefill。

`--confidence` 选择随响应记录的置信度口径：`entropy`（默认，`1 - H/ln(K)`）或
`max-probability`（`(K·max(p) - 1)/(K - 1)`，二元候选参考实现使用的形状）。两者都不是
TypeSafe 的私有公式；响应会声明实际用了哪一个。

### 测量选项顺序翻转

`--perturb --input rows.jsonl` 会先按原样评一遍每行，再把选项顺序反转评一遍，报告胜出
选项发生变化的比例。它不需要 gold 文件：基线本身就是对照，报告里带基线胜出选项在改动
前后的概率。

在参考端点上用三行实测：字母契约翻转了其中一行——同一问题先答 `frustrated`（0.634），
反转后基线胜出者的概率跌到 0.141、答案变成 `furious`——而 `--scoring binary` 报告零翻转，
两个顺序下概率完全相同。

## 启动契约

起服务前跑两项检查，任一失败就退出，而不是算出一个错的分数：

1. **单 token 契约**：`A`–`P` 每个字母必须恰好是一个 token，且 `/detokenize` 能把它们
   解回 `ABCDEFGHIJKLMNOP`。
2. **边界契约**：把字母直接拼在渲染后的 prompt 之后，token 序列必须恰好等于
   `prompt tokens + [该字母的 token]`。否则 prompt 末尾会和字母合并，读到的 logit
   就不是"模型选择该选项"的概率。

## 参数

| 参数 | 说明 |
|---|---|
| `--base-url` | 上游 OpenAI 兼容根地址，如 `http://127.0.0.1:8080/v1` |
| `--native-base-url` | 原生端点根地址（llama.cpp `/completion`），默认取 `--base-url` 去掉 `/v1` |
| `--model` | 发给上游的模型名 |
| `--upstream-key` | 上游 bearer token；建议用环境变量 `JEV_BRIDGE_UPSTREAM_KEY` |
| `--served-model` | 本服务接受的模型 ID，默认等于 `--model`，不得以 `jev` 开头 |
| `--served-model-description` | `GET /v1/models` 的描述 |
| `--served-model-release-date` | `GET /v1/models` 的 ISO 日期（必填） |
| `--chat-template-kwargs` | 渲染模板时透传的参数，默认 `{"enable_thinking": false}` |
| `--chat-template-file` | 从该 Jinja 文件本地渲染模板，不再调用 `/apply-template` |
| `--chat-template-context` | 每次本地渲染附加的 JSON，例如 `{"bos_token": "<s>"}` |
| `--tokenizer-json` | 用该 `tokenizer.json` 本地分词，不再调用 `/tokenize` |
| `--max-input-tokens` | 超过该 token 数的行直接报错（不做截断） |
| `--host` / `--port` | 监听地址，默认 `127.0.0.1:8100` |
| `--api-key` | 客户端必须携带的 bearer token；建议用环境变量 `JEV_BRIDGE_API_KEY` |
| `--probe-only` | 只探测并打印结果，然后退出 |
| `--score` / `--input` / `--output` | 批量打分模式 |
| `--evaluate` / `--predictions` / `--gold` / `--out` | 离线校准指标：从预测与 gold 文件计算 |
| `--verify` / `--tol` | 重算评估报告并与文件逐项比对 |
| `--bins` | ECE 与可靠性曲线的等宽箱数，默认 10 |
| `--readout-status` / `--readout-evidence` | readout 通道声明，出现在 `/health` 与每个答案上（默认 `unvalidated`） |
| `--require-readout-check` | 启动自检不通过时拒绝启动（默认只报告，不阻断） |
| `--accept-any-model` | 接受任意非空 `model` 名，不再要求等于 `--served-model`（给硬编码模型 id 的第三方客户端用） |
| `--upstream-timeout-secs` | 单次上游请求超时秒数（默认 600；连接超时固定 10 秒） |

`--base-url`、`--model` 与 `--served-model-release-date` 仅在未给 `--evaluate` 时必填；
`--evaluate` 三者都不需要。

凭据只从环境变量或命令行读取，不写入任何文件。

## 与 fastjev 的对齐程度

- payload 用 Python `json.dumps(..., ensure_ascii=False)` 的分隔符布局（`, ` 与 `: `）；
  单元测试用 CPython 的真实输出固定这个布局，对齐验证再用 144 行全等的 prompt 哈希确认。
- softmax 实现与 fastjev 的 `softmax` 相同（先减最大值）。
- `fastjev.confidence_method` 是 `one-minus-normalized-entropy`，与 fastjev 的 System One
  适配层一致；它不是 TypeSafe 的私有统计量。
- 模板渲染交给服务端（`/apply-template`），所以拿到的就是该运行时真正喂给模型的 prompt。
  这也意味着**不要**用 chat completions 端点：那条路径的 prompt 由服务端模板决定，
  无法保证与本契约一致。

## 参考项目

- [fastjev](https://github.com/chengyongru/fastjev)（MIT）——本项目实现的
  `direct-options-v1` 契约，也是对齐验证所用 144 条 authored 决策的来源。
- [jev-clone](https://github.com/alitrack/jev-clone)（Apache-2.0）——独立的、
  契约兼容的 System One 服务，专注**测量**。[`src/evaluate.rs`](src/evaluate.rs)
  的冻结指标口径（ECE、两种 Brier、NLL、可靠性曲线、first-max argmax 规则）
  与手算四行测试用例沿用其公开发布的定义；`/health` 与每个答案里的 `readout`
  声明（`--readout-status`）来自它的发现——readout 通道是模型依赖的，未受
  单 token 槽位训练的模型不会报错，而是静默退化。本项目实现为独立 Rust 代码，
  原始工作与证据见上游仓库。
- [TypeSafe System One](https://docs.typesafe.ai)——wire 格式保持字段级兼容的
  公开 HTTP 契约。
- [taeold/djev-run](https://github.com/taeold/djev-run)——把 DiffusionGemma-Jev
  挂在 TypeSafe 兼容 API 上，并附带三个独立的浏览器游戏（snake / dino /
  tetris），直接在浏览器里 `POST /v1/systemone`。它们通过
  `cargo run --example playground` 接到本 bridge 上（见 Examples）；因为它们
  硬编码 `jev-latest`，bridge 需要 `--accept-any-model`。

## 测试

```bash
cargo test
```

103 个测试：73 个单元测试（wire 契约、CPython JSON 布局、三种响应形状、softmax 稳定性、
缺失选项必须报错、CPython 语义的模板方法、冻结的评估指标口径，以及 readout 自检的探针），
加 30 个集成测试，分在 `tests/bridge.rs`（库 API，含 `DetailedScore` 字段）、
`tests/contract.rs`（HTTP 层、本地渲染、本地分词、readout 声明、readout 自检，
以及批量读的各条路径：一次请求读多个问题、逐条回退及其计数、全有或全无的失败）
和 `tests/evaluate.rs`（`--evaluate` / `--verify` 的 CLI 往返及其失败模式：
被篡改的报告、来自其他输入的报告、两边不同步的文件）。

`tests/latency.rs` 是 `#[ignore]` 的：它对着可达的上游测真实决策延迟
（端点、模型、密钥来自 `JEV_BRIDGE_UPSTREAM_URL` / `JEV_BRIDGE_MODEL` /
`JEV_BRIDGE_UPSTREAM_KEY`，不落盘）：

```bash
JEV_BRIDGE_UPSTREAM_URL=... JEV_BRIDGE_MODEL=... JEV_BRIDGE_UPSTREAM_KEY=... \
  cargo test --release --test latency -- --ignored --nocapture
```

## 已知限制

- 批量读只赢服务端能共享的部分：在 llama.cpp b11096 + 9B Q8_0 上是 1.25x
  （共享 state 的那次 prefill），不是数量级。若端点自带的前缀缓存已经让逐条路径
  同样受益，收益可能接近 1x。
- 上游 revision 无法由本进程钉住，结果的可复现性取决于上游自身的版本管理。
- 对齐验证的参考侧是 BF16，桥接侧是 Q8_0，因此 0.0053 的质量差是量化差，不是桥接差；
  要得到同精度对比需要把上游换成 BF16 服务。
- 使用本地分词器时，桥接层信任你给的文件。词表不对会从错误的 token id 上算出看似合理的分数。
  `verify_special_tokens` 会在启动时校验上下文里的特殊 token 是否存在于该词表，但无法判断
  这个词表是否就是该模型自己的。
