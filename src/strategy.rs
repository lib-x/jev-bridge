//! Runtime probing of a generic OpenAI-compatible inference service.
//!
//! The bridge never guesses: every candidate transport is exercised with a
//! real probe request before it is adopted, and the chosen probe is reported
//! by `GET /health` so an operator can see exactly which contract is in use.
//!
//! Probe results on llama.cpp (b11096) shaped this module:
//!   * `/v1/completions` answers with chat-shaped `logprobs.content[0].top_logprobs`
//!     entries that carry an explicit `id`, and ignores `allowed_token_ids`.
//!   * `/completion` exposes `completion_probabilities[0].top_logprobs` and
//!     leaves `top_probs` null.
//!   * An unrestricted top-k needs k >= 50 before all sixteen answer letters
//!     appear, so unrestricted probes request a generous k.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How the target service exposes option-token log probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// vLLM: restrict the distribution to the option tokens and ask for their
    /// exact token ids. This is the same contract fastjev's own vLLM backend
    /// uses through `SamplingParams`.
    LogprobTokenIds,
    /// vLLM without `logprob_token_ids`: restrict the distribution and read
    /// the top-k, which must then contain every option token.
    AllowedTokenIdsTopK,
    /// OpenAI-compatible completions: read the top-k of the unrestricted
    /// next-token distribution.
    CompletionsTopK,
    /// llama.cpp native `/completion` with `n_probs`.
    LlamaCppNProbs,
}

/// The transports to try, in the order they are probed.
pub const CANDIDATES: [Probe; 4] = [
    Probe::LogprobTokenIds,
    Probe::AllowedTokenIdsTopK,
    Probe::CompletionsTopK,
    Probe::LlamaCppNProbs,
];

impl Probe {
    /// Stable identifier reported by `GET /health`.
    pub fn name(self) -> &'static str {
        match self {
            Probe::LogprobTokenIds => "vllm-logprob-token-ids",
            Probe::AllowedTokenIdsTopK => "vllm-allowed-token-ids-top-k",
            Probe::CompletionsTopK => "openai-completions-top-k",
            Probe::LlamaCppNProbs => "llamacpp-native-n-probs",
        }
    }

    /// One line describing what this transport relies on.
    pub fn note(self) -> &'static str {
        match self {
            Probe::LogprobTokenIds => {
                "option distribution is restricted to the declared answer tokens; exact token ids requested"
            }
            Probe::AllowedTokenIdsTopK => {
                "option distribution is restricted to the declared answer tokens; top-k must cover them"
            }
            Probe::CompletionsTopK => {
                "unrestricted next-token distribution; every answer token must fall inside top-k"
            }
            Probe::LlamaCppNProbs => {
                "llama.cpp native completion probabilities; every answer token must fall inside n_probs"
            }
        }
    }

    /// Whether the request is expected to restrict the distribution to the
    /// answer tokens. A probe that claims restriction but returns foreign
    /// candidates is rejected rather than silently trusted.
    pub fn restricts_distribution(self) -> bool {
        matches!(self, Probe::LogprobTokenIds | Probe::AllowedTokenIdsTopK)
    }

    /// The path this transport posts to.
    pub fn endpoint(self) -> &'static str {
        match self {
            Probe::LlamaCppNProbs => "/completion",
            _ => "/completions",
        }
    }

    fn url(self, openai_base: &str, native_base: &str) -> String {
        match self {
            Probe::LlamaCppNProbs => format!("{}/completion", native_base.trim_end_matches('/')),
            _ => format!("{}/completions", openai_base.trim_end_matches('/')),
        }
    }
}

/// Candidate count requested from a top-k endpoint.
///
/// A restricted distribution only ever contains the answer tokens, so k equal
/// to the option count suffices. An unrestricted distribution also contains
/// ordinary vocabulary, and measurement on a 2B llama.cpp model showed four of
/// sixteen answer letters falling outside k=20, so unrestricted probes ask for
/// a wide margin instead.
fn top_k(probe: Probe, slot_count: usize) -> usize {
    match probe {
        Probe::LogprobTokenIds => 1,
        Probe::AllowedTokenIdsTopK => slot_count.max(4),
        Probe::CompletionsTopK | Probe::LlamaCppNProbs => (slot_count * 4).clamp(32, 128),
    }
}

/// The request body for an OpenAI-compatible `/v1/completions` call.
///
/// Optional fields are omitted rather than sent as `null`: a runtime that does
/// not know `allowed_token_ids` is likelier to answer a request without it, and
/// the probe already checks whether the field had any effect.
#[derive(Debug, Serialize)]
pub struct OpenAiCompletionRequest {
    model: String,
    prompt: String,
    /// One token is enough: only the first position's distribution is read.
    max_tokens: u32,
    /// Left at 1.0 so the returned log probabilities are the model's own.
    temperature: f32,
    /// How many candidates to return from the head of the distribution.
    logprobs: usize,
    /// Ask vLLM to label candidates by token id instead of token text, which
    /// removes any ambiguity about leading spaces.
    #[serde(skip_serializing_if = "Option::is_none")]
    return_tokens_as_token_ids: Option<bool>,
    /// vLLM extension: restrict sampling to these token ids, which turns the
    /// returned log probabilities into the conditional option distribution.
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_token_ids: Option<Vec<u32>>,
    /// vLLM extension: report these exact token ids even when they fall
    /// outside the top-k.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprob_token_ids: Option<Vec<u32>>,
}

/// The request body for llama.cpp's native `/completion`.
///
/// Every sampler knob is set explicitly and neutrally. llama.cpp reports the
/// distribution *after* the sampler chain, so inheriting a server preset such
/// as `repeat-penalty = 1.05` or `top-p = 0.85` would return a distorted
/// distribution instead of the model's own.
#[derive(Debug, Serialize)]
pub struct LlamaCppCompletionRequest {
    model: String,
    prompt: String,
    n_predict: u32,
    /// How many candidates to return, matching `logprobs` above.
    n_probs: usize,
    cache_prompt: bool,
    temperature: f32,
    /// 0 disables top-k truncation.
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    typical_p: f32,
}

/// A scoring request in whichever dialect the chosen probe speaks.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum RequestBody {
    /// `/v1/completions` in the OpenAI dialect.
    OpenAi(OpenAiCompletionRequest),
    /// llama.cpp's native `/completion`.
    LlamaCpp(LlamaCppCompletionRequest),
}

/// Build the scoring request for one probe.
pub fn build_body(probe: Probe, model: &str, prompt: &str, slots: &[u32]) -> RequestBody {
    let candidates = top_k(probe, slots.len());
    match probe {
        Probe::LogprobTokenIds => RequestBody::OpenAi(OpenAiCompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens: 1,
            temperature: 1.0,
            logprobs: candidates,
            return_tokens_as_token_ids: Some(true),
            allowed_token_ids: Some(slots.to_vec()),
            logprob_token_ids: Some(slots.to_vec()),
        }),
        Probe::AllowedTokenIdsTopK => RequestBody::OpenAi(OpenAiCompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens: 1,
            temperature: 1.0,
            logprobs: candidates,
            return_tokens_as_token_ids: Some(true),
            allowed_token_ids: Some(slots.to_vec()),
            logprob_token_ids: None,
        }),
        Probe::CompletionsTopK => RequestBody::OpenAi(OpenAiCompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens: 1,
            temperature: 1.0,
            logprobs: candidates,
            return_tokens_as_token_ids: Some(true),
            allowed_token_ids: None,
            logprob_token_ids: None,
        }),
        Probe::LlamaCppNProbs => RequestBody::LlamaCpp(LlamaCppCompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            n_predict: 1,
            n_probs: candidates,
            cache_prompt: true,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            typical_p: 1.0,
        }),
    }
}

/// The full URL a probe posts to.
pub fn request_url(probe: Probe, openai_base: &str, native_base: &str) -> String {
    probe.url(openai_base, native_base)
}

/// One candidate token in a returned distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Token id, when the runtime reported one.
    pub id: Option<u32>,
    /// Token text, or `token_id:<n>` in the OpenAI map shape.
    pub token: String,
    /// Natural-log probability.
    pub logprob: f64,
}

/// One candidate as a runtime serialised it.
///
/// llama.cpp sends an array of these on both endpoints, with an explicit `id`
/// and a `logprob`; older builds send `prob` instead. Everything is optional
/// because the bridge must report a *missing* candidate rather than default one.
#[derive(Debug, Clone, Deserialize)]
struct CandidateEntry {
    #[serde(default)]
    id: Option<u32>,
    #[serde(default)]
    token: String,
    #[serde(default)]
    logprob: Option<f64>,
    #[serde(default)]
    prob: Option<f64>,
}

/// A position's candidates, in whichever shape the runtime uses.
///
/// OpenAI and vLLM return a map from token text — or `token_id:<n>` when
/// `return_tokens_as_token_ids` is honoured — to a log probability. llama.cpp
/// returns an array of entries. Both are accepted, and the map's values stay
/// dynamic because they may be a bare number or an object with a `logprob`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CandidateList {
    Entries(Vec<CandidateEntry>),
    Map(serde_json::Map<String, Value>),
}

/// One generated position on llama.cpp's native `/completion`.
#[derive(Debug, Deserialize)]
struct NativePosition {
    #[serde(default)]
    top_logprobs: Option<CandidateList>,
    /// Only consulted when `top_logprobs` is absent: a probability carries less
    /// precision than a log probability.
    #[serde(default)]
    top_probs: Option<CandidateList>,
}

/// One generated position in an OpenAI-shaped choice.
#[derive(Debug, Deserialize)]
struct ChoiceLogprobs {
    /// OpenAI completions shape: one entry per generated position.
    #[serde(default)]
    top_logprobs: Vec<CandidateList>,
    /// llama.cpp answers the completions endpoint with chat-shaped content.
    #[serde(default)]
    content: Vec<NativePosition>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
}

/// Every field the bridge reads from a scoring response.
///
/// One struct covers both endpoints: a runtime fills the fields its dialect
/// uses and leaves the rest absent, and the probe that was chosen already
/// decided which fields to read.
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
    /// llama.cpp native `/completion`.
    #[serde(default)]
    completion_probabilities: Vec<NativePosition>,
    #[serde(default)]
    tokens_evaluated: Option<u64>,
    #[serde(default)]
    truncated: bool,
}

fn candidate_logprob(entry: &CandidateEntry) -> Option<f64> {
    entry
        .logprob
        .or_else(|| entry.prob.map(f64::ln))
        .filter(|value| value.is_finite())
}

fn map_logprob(value: &Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        return Some(number);
    }
    value
        .get("logprob")
        .and_then(Value::as_f64)
        .or_else(|| value.get("prob").and_then(Value::as_f64).map(f64::ln))
}

/// Normalize the several shapes servers use for one position's candidates.
fn collect_candidates(list: &CandidateList) -> Vec<Candidate> {
    match list {
        CandidateList::Entries(entries) => entries
            .iter()
            .filter_map(|entry| {
                Some(Candidate {
                    id: entry.id,
                    token: entry.token.clone(),
                    logprob: candidate_logprob(entry)?,
                })
            })
            .collect(),
        CandidateList::Map(map) => map
            .iter()
            .filter_map(|(key, value)| {
                let logprob = map_logprob(value)?;
                let id = key
                    .strip_prefix("token_id:")
                    .and_then(|text| text.parse::<u32>().ok());
                Some(Candidate {
                    id,
                    token: key.clone(),
                    logprob,
                })
            })
            .collect(),
    }
}

/// Locate the candidate list for the first generated position.
///
/// The order within each probe is deliberate: the shape that probe is expected
/// to receive is tried first, and the alternative is only a fallback for
/// runtimes that answer in the other dialect.
fn candidate_position(
    probe: Probe,
    response: &CompletionResponse,
) -> Result<&CandidateList> {
    let choice = response.choices.first();
    let logprobs = choice.and_then(|choice| choice.logprobs.as_ref());
    match probe {
        // llama.cpp answers `/v1/completions` with chat-shaped logprobs.
        Probe::CompletionsTopK => logprobs
            .and_then(|logprobs| logprobs.content.first())
            .and_then(|position| position.top_logprobs.as_ref())
            .or_else(|| logprobs.and_then(|logprobs| logprobs.top_logprobs.first()))
            .ok_or_else(|| anyhow!("response has no candidates under choices[0].logprobs")),
        Probe::LogprobTokenIds | Probe::AllowedTokenIdsTopK => logprobs
            .and_then(|logprobs| logprobs.top_logprobs.first())
            .or_else(|| {
                logprobs
                    .and_then(|logprobs| logprobs.content.first())
                    .and_then(|position| position.top_logprobs.as_ref())
            })
            .ok_or_else(|| anyhow!("response has no candidates under choices[0].logprobs")),
        Probe::LlamaCppNProbs => response
            .completion_probabilities
            .first()
            .and_then(|position| {
                position
                    .top_logprobs
                    .as_ref()
                    .or(position.top_probs.as_ref())
            })
            .ok_or_else(|| anyhow!("response has no candidates under completion_probabilities[0]")),
    }
}

fn find_candidate(candidates: &[Candidate], slot: u32, letter: char) -> Option<f64> {
    let by_id = candidates.iter().find(|candidate| candidate.id == Some(slot));
    let text = letter.to_string();
    by_id
        .or_else(|| candidates.iter().find(|candidate| candidate.token == text))
        .map(|candidate| candidate.logprob)
}

/// Decode a scoring response into the struct the bridge reads.
pub fn parse_response(body: &Value) -> Result<CompletionResponse> {
    serde_json::from_value(body.clone())
        .context("the scoring response did not match any known shape")
}

/// Extract one log probability per requested slot, in slot order.
///
/// A missing slot is an error rather than a zero: silently dropping an option
/// would change the meaning of the returned distribution.
pub fn parse_logprobs(
    probe: Probe,
    response: &CompletionResponse,
    slots: &[u32],
    letters: &[char],
) -> Result<Vec<f64>> {
    if let Some(finish) = response
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_deref())
        && finish == "error"
    {
        bail!("service reported finish_reason=error");
    }
    let candidates = collect_candidates(candidate_position(probe, response)?);
    if candidates.is_empty() {
        bail!("returned distribution contains no usable candidates");
    }
    let mut result = Vec::with_capacity(slots.len());
    for (slot, letter) in slots.iter().zip(letters) {
        let logprob = find_candidate(&candidates, *slot, *letter).ok_or_else(|| {
            anyhow!(
                "option token {letter:?} (id {slot}) is absent from the returned distribution; \
                 raise the requested top-k or restrict the distribution to the answer tokens"
            )
        })?;
        if !logprob.is_finite() {
            bail!("logprob for option token {letter:?} is not finite");
        }
        result.push(logprob);
    }
    Ok(result)
}

/// Whether every returned candidate belongs to the declared answer tokens.
///
/// A probe that asks for a restricted distribution but receives foreign
/// candidates has been silently ignored by the server, which would make the
/// reported transport a lie.
pub fn candidates_restricted_to(candidates: &[Candidate], slots: &[u32], letters: &[char]) -> bool {
    candidates.iter().all(|candidate| {
        candidate
            .id
            .map(|id| slots.contains(&id))
            .unwrap_or_else(|| candidate.token.chars().count() == 1 && letters.contains(&candidate.token.chars().next().unwrap()))
    })
}

/// Decode the returned candidates without requiring any slot to be present.
pub fn parse_candidates(probe: Probe, response: &CompletionResponse) -> Result<Vec<Candidate>> {
    Ok(collect_candidates(candidate_position(probe, response)?))
}

/// The prompt length the runtime reported, or zero when it reported none.
pub fn input_tokens(probe: Probe, response: &CompletionResponse) -> u64 {
    match probe {
        Probe::LlamaCppNProbs => response.tokens_evaluated.unwrap_or(0),
        _ => response
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens)
            .unwrap_or(0),
    }
}

/// Whether the runtime silently shortened the prompt.
///
/// A truncated prompt would score a decision the caller never asked for, so the
/// bridge turns this into an error instead of a slightly wrong answer.
pub fn truncated(probe: Probe, response: &CompletionResponse) -> bool {
    match probe {
        Probe::LlamaCppNProbs => response.truncated,
        _ => false,
    }
}

/// Validate that a probe response is usable before adopting its transport.
pub fn validate_probe(logprobs: &[f64]) -> Result<()> {
    if logprobs.len() < 2 {
        bail!("probe returned fewer than two option scores");
    }
    if logprobs.iter().any(|value| !value.is_finite()) {
        bail!("probe returned a non-finite option score");
    }
    if logprobs.iter().any(|value| *value > 1e-6) {
        bail!("probe returned a positive log probability, which cannot be a log softmax value");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn logprob_token_ids_body_restricts_and_selects() {
        let RequestBody::OpenAi(body) = build_body(Probe::LogprobTokenIds, "m", "prompt", &[32, 33])
        else {
            panic!("this probe speaks the OpenAI dialect");
        };
        assert_eq!(body.allowed_token_ids.as_deref(), Some(&[32u32, 33][..]));
        assert_eq!(body.logprob_token_ids.as_deref(), Some(&[32u32, 33][..]));
        assert_eq!(body.max_tokens, 1);
        assert_eq!(body.logprobs, 1, "only the requested ids need reporting");
    }

    #[test]
    fn unrestricted_probes_request_a_wide_top_k() {
        assert_eq!(top_k(Probe::CompletionsTopK, 16), 64);
        assert_eq!(top_k(Probe::LlamaCppNProbs, 16), 64);
        assert_eq!(top_k(Probe::CompletionsTopK, 2), 32);
        assert_eq!(top_k(Probe::AllowedTokenIdsTopK, 16), 16);
        assert_eq!(top_k(Probe::LogprobTokenIds, 16), 1);
    }

    #[test]
    fn llamacpp_body_neutralizes_the_sampler() {
        let RequestBody::LlamaCpp(body) = build_body(Probe::LlamaCppNProbs, "m", "prompt", &[54, 55])
        else {
            panic!("this probe speaks the llama.cpp dialect");
        };
        assert_eq!(body.n_predict, 1);
        assert_eq!(body.n_probs, 32);
        assert_eq!(body.temperature, 1.0);
        assert_eq!(body.top_k, 0, "top-k truncation would hide low-ranked options");
        assert_eq!(body.repeat_penalty, 1.0, "a penalty would distort the distribution");
    }

    #[test]
    fn an_unrestricted_probe_omits_the_vllm_only_fields() {
        let RequestBody::OpenAi(body) = build_body(Probe::CompletionsTopK, "m", "prompt", &[54, 55])
        else {
            panic!("this probe speaks the OpenAI dialect");
        };
        assert!(body.allowed_token_ids.is_none());
        assert!(body.logprob_token_ids.is_none());
        assert_eq!(body.logprobs, 32);
    }

    #[test]
    fn vllm_map_response_is_parsed_by_token_id() {
        let body = json!({"choices": [{"logprobs": {"top_logprobs": [{
            "token_id:32": -0.51, "token_id:33": -1.2, "token_id:99": -4.0
        }]}}]});
        let values = parse_logprobs(Probe::LogprobTokenIds, &parse_response(&body).unwrap(), &[32, 33], &['A', 'B']).unwrap();
        assert_eq!(values, vec![-0.51, -1.2]);
    }

    #[test]
    fn vllm_map_response_falls_back_to_token_text() {
        let body = json!({"choices": [{"logprobs": {"top_logprobs": [{"A": -0.1, "B": -2.0}]}}]});
        let values = parse_logprobs(Probe::CompletionsTopK, &parse_response(&body).unwrap(), &[32, 33], &['A', 'B']).unwrap();
        assert_eq!(values, vec![-0.1, -2.0]);
    }

    #[test]
    fn llamacpp_completions_response_is_parsed_by_id() {
        // Shape observed on llama.cpp b11096: chat-style logprobs on the
        // completions endpoint, with an explicit token id.
        let body = json!({"choices": [{"finish_reason": "length", "logprobs": {"content": [{
            "id": 55, "token": "B", "logprob": -2.1e-05,
            "top_logprobs": [
                {"id": 55, "token": "B", "logprob": -2.1e-05},
                {"id": 54, "token": "A", "logprob": -12.32},
                {"id": 56, "token": "C", "logprob": -15.61}
            ]}]}}]});
        let values =
            parse_logprobs(Probe::CompletionsTopK, &parse_response(&body).unwrap(), &[54, 55, 56], &['A', 'B', 'C']).unwrap();
        assert_eq!(values, vec![-12.32, -2.1e-05, -15.61]);
    }

    #[test]
    fn llamacpp_native_response_prefers_top_logprobs() {
        let body = json!({"completion_probabilities": [{"id": 32, "token": "A", "prob": 0.6,
            "top_probs": [{"id": 32, "token": "A", "prob": 0.6}],
            "top_logprobs": [{"id": 32, "token": "A", "logprob": -0.51},
                             {"id": 33, "token": "B", "logprob": -0.92}]}]});
        let values = parse_logprobs(Probe::LlamaCppNProbs, &parse_response(&body).unwrap(), &[32, 33], &['A', 'B']).unwrap();
        assert_eq!(values, vec![-0.51, -0.92]);
    }

    #[test]
    fn llamacpp_native_response_falls_back_to_probabilities() {
        let body = json!({"completion_probabilities": [{"id": 32, "token": "A",
            "top_probs": [{"id": 32, "token": "A", "prob": 0.5}, {"id": 33, "token": "B", "prob": 0.5}]}]});
        let values = parse_logprobs(Probe::LlamaCppNProbs, &parse_response(&body).unwrap(), &[32, 33], &['A', 'B']).unwrap();
        assert!((values[0] - 0.5f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn missing_option_is_an_error_not_a_zero() {
        let body = json!({"choices": [{"logprobs": {"top_logprobs": [{"token_id:32": -0.1}]}}]});
        let error =
            parse_logprobs(Probe::CompletionsTopK, &parse_response(&body).unwrap(), &[32, 33], &['A', 'B']).unwrap_err();
        assert!(error.to_string().contains("absent from the returned distribution"));
    }

    #[test]
    fn foreign_candidates_reveal_an_unrestricted_distribution() {
        let restricted = vec![Candidate { id: Some(54), token: "A".into(), logprob: -0.1 }];
        let unrestricted = vec![
            Candidate { id: Some(54), token: "A".into(), logprob: -0.1 },
            Candidate { id: Some(608), token: "The".into(), logprob: -9.0 },
        ];
        assert!(candidates_restricted_to(&restricted, &[54, 55], &['A', 'B']));
        assert!(!candidates_restricted_to(&unrestricted, &[54, 55], &['A', 'B']));
    }

    #[test]
    fn probe_validation_rejects_positive_logprobs() {
        assert!(validate_probe(&[-0.5, -1.5]).is_ok());
        assert!(validate_probe(&[0.5, -1.5]).is_err());
    }

    #[test]
    fn urls_split_between_openai_and_native_bases() {
        assert_eq!(
            request_url(Probe::CompletionsTopK, "http://host:8000/v1/", "http://host:8000"),
            "http://host:8000/v1/completions"
        );
        assert_eq!(
            request_url(Probe::LlamaCppNProbs, "http://host:8080/v1", "http://host:8080/"),
            "http://host:8080/completion"
        );
    }
}
