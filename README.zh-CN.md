# jev-bridge

[English](README.md) | [简体中文](README.zh-CN.md)

把任意通用的 OpenAI 兼容推理服务，变成 Jev 风格的决策打分服务。单个 7 MB 二进制，不依赖 Python。

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
| vLLM | 探测路径已实现，本地模板渲染未实现 | `allowed_token_ids` + `logprob_token_ids`，或 `allowed_token_ids` + top-k |

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

- [results/alignment-minicpm5-2b-authored144.json](results/alignment-minicpm5-2b-authored144.json)
- [results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl](results/bridge-minicpm5-2b-q8-authored144.predictions.jsonl)

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

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/systemone` | System One 兼容打分，支持 `choice` / `noul` / `score` |
| GET | `/v1/models` | 返回配置的模型元数据 |
| GET | `/health` | 返回选中的传输、端点、答案 token id、模板参数 |

出问题时先看 `GET /health`：它把实际协商到的结果显式暴露出来，而不是让你猜服务端支持什么。

## 作为库使用

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

用 `jev_bridge::server::router` 把同一个 bridge 挂成 HTTP 服务：

```rust
use std::sync::Arc;
use jev_bridge::server::{router, AppState};

let state = Arc::new(AppState { bridge, api_key: None });
let listener = tokio::net::TcpListener::bind("127.0.0.1:8100").await?;
axum::serve(listener, router(state)).await?;
```

公开模块：

| 模块 | 内容 |
|---|---|
| `server` | `Bridge`、`BridgeConfig`、`DetailedScore`、`router`、`AppState` |
| `strategy` | `Probe`、各传输的请求体、响应解析 |
| `prompt` | prompt 契约、单 token 校验、softmax、`ServerTemplate` |
| `wire` | System One 请求校验与响应构造 |

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

不受限分布需要足够大的 top-k：实测在 2B 量化模型上，`k=20` 会漏掉 16 个答案字母里的
4 个，`k=50` 才全覆盖。因此不受限传输按 `4 × 选项数` 请求（下限 32，上限 128）。
缺失任何一个选项都会直接报错，不会补 0。

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
| `--max-input-tokens` | 超过该 token 数的行直接报错（不做截断） |
| `--host` / `--port` | 监听地址，默认 `127.0.0.1:8100` |
| `--api-key` | 客户端必须携带的 bearer token；建议用环境变量 `JEV_BRIDGE_API_KEY` |
| `--probe-only` | 只探测并打印结果，然后退出 |
| `--score` / `--input` / `--output` | 批量打分模式 |

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

## 测试

```bash
cargo test
```

39 个测试：26 个单元测试（wire 契约、CPython JSON 布局、三种响应形状、softmax 稳定性、
缺失选项必须报错），加 13 个集成测试，分在 `tests/bridge.rs`（库 API，含 `DetailedScore`
字段）和 `tests/contract.rs`（HTTP 层）。

## 已知限制

- 目前只支持服务端渲染模板的运行时（llama.cpp `/apply-template` + `/tokenize`）。
  对接 vLLM 需要补一条本地 tokenizer + minijinja 渲染路径，尚未实现。
- 决策串行执行：一次请求内的多个问题逐条打分，上游调用不批量并发。
- 上游 revision 无法由本进程钉住，结果的可复现性取决于上游自身的版本管理。
- 对齐验证的参考侧是 BF16，桥接侧是 Q8_0，因此 0.0053 的质量差是量化差，不是桥接差；
  要得到同精度对比需要把上游换成 BF16 服务。
