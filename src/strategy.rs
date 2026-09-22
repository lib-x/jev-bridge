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

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

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

pub const CANDIDATES: [Probe; 4] = [
    Probe::LogprobTokenIds,
    Probe::AllowedTokenIdsTopK,
    Probe::CompletionsTopK,
    Probe::LlamaCppNProbs,
];

impl Probe {
    pub fn name(self) -> &'static str {
        match self {
            Probe::LogprobTokenIds => "vllm-logprob-token-ids",
            Probe::AllowedTokenIdsTopK => "vllm-allowed-token-ids-top-k",
            Probe::CompletionsTopK => "openai-completions-top-k",
            Probe::LlamaCppNProbs => "llamacpp-native-n-probs",
        }
    }

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

pub fn build_body(probe: Probe, model: &str, prompt: &str, slots: &[u32]) -> Value {
    match probe {
        Probe::LogprobTokenIds => json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": 1,
            "return_tokens_as_token_ids": true,
            "allowed_token_ids": slots,
            "logprob_token_ids": slots,
        }),
        Probe::AllowedTokenIdsTopK => json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": top_k(probe, slots.len()),
            "return_tokens_as_token_ids": true,
            "allowed_token_ids": slots,
        }),
        Probe::CompletionsTopK => json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": top_k(probe, slots.len()),
            "return_tokens_as_token_ids": true,
        }),
        Probe::LlamaCppNProbs => json!({
            "model": model,
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": top_k(probe, slots.len()),
            "cache_prompt": true,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "typical_p": 1.0,
        }),
    }
}

pub fn request_url(probe: Probe, openai_base: &str, native_base: &str) -> String {
    probe.url(openai_base, native_base)
}

/// One candidate token in a returned distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: Option<u32>,
    pub token: String,
    pub logprob: f64,
}

fn as_logprob(value: &Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        return Some(number);
    }
    value
        .get("logprob")
        .and_then(Value::as_f64)
        .or_else(|| value.get("prob").and_then(Value::as_f64).map(f64::ln))
}

fn candidate_from_entry(entry: &Value) -> Option<Candidate> {
    let id = entry
        .get("id")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let token = entry
        .get("token")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let logprob = entry.get("logprob").and_then(Value::as_f64).or_else(|| {
        entry
            .get("prob")
            .and_then(Value::as_f64)
            .map(f64::ln)
    })?;
    Some(Candidate { id, token, logprob })
}

/// Normalize the several shapes servers use for one position's candidates.
fn collect_candidates(value: &Value) -> Vec<Candidate> {
    match value {
        Value::Array(items) => items.iter().filter_map(candidate_from_entry).collect(),
        Value::Object(map) => map
            .iter()
            .filter_map(|(key, value)| {
                let logprob = as_logprob(value)?;
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
        _ => Vec::new(),
    }
}

/// Locate the candidate list for the first generated position.
fn candidate_position<'a>(probe: Probe, body: &'a Value) -> Result<&'a Value> {
    let first = |path: &str| -> Result<&'a Value> {
        body.pointer(path)
            .ok_or_else(|| anyhow!("response has no {path}"))
    };
    match probe {
        // llama.cpp answers `/v1/completions` with chat-shaped logprobs.
        Probe::CompletionsTopK => first("/choices/0/logprobs/content/0/top_logprobs")
            .or_else(|_| first("/choices/0/logprobs/top_logprobs/0")),
        Probe::LogprobTokenIds | Probe::AllowedTokenIdsTopK => {
            first("/choices/0/logprobs/top_logprobs/0")
                .or_else(|_| first("/choices/0/logprobs/content/0/top_logprobs"))
        }
        // `top_probs` is only consulted when `top_logprobs` is absent, because
        // a probability loses precision against a log probability.
        Probe::LlamaCppNProbs => first("/completion_probabilities/0/top_logprobs")
            .or_else(|_| first("/completion_probabilities/0/top_probs")),
    }
}

fn find_candidate(candidates: &[Candidate], slot: u32, letter: char) -> Option<f64> {
    let by_id = candidates.iter().find(|candidate| candidate.id == Some(slot));
    let text = letter.to_string();
    by_id
        .or_else(|| candidates.iter().find(|candidate| candidate.token == text))
        .map(|candidate| candidate.logprob)
}

/// Extract one log probability per requested slot, in slot order.
///
/// A missing slot is an error rather than a zero: silently dropping an option
/// would change the meaning of the returned distribution.
pub fn parse_logprobs(
    probe: Probe,
    body: &Value,
    slots: &[u32],
    letters: &[char],
) -> Result<Vec<f64>> {
    if let Some(finish) = body.pointer("/choices/0/finish_reason").and_then(Value::as_str)
        && finish == "error"
    {
        bail!("service reported finish_reason=error");
    }
    let candidates = collect_candidates(candidate_position(probe, body)?);
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

pub fn parse_candidates(probe: Probe, body: &Value) -> Result<Vec<Candidate>> {
    Ok(collect_candidates(candidate_position(probe, body)?))
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

    #[test]
    fn logprob_token_ids_body_restricts_and_selects() {
        let body = build_body(Probe::LogprobTokenIds, "m", "prompt", &[32, 33]);
        assert_eq!(body["allowed_token_ids"], json!([32, 33]));
        assert_eq!(body["logprob_token_ids"], json!([32, 33]));
        assert_eq!(body["max_tokens"], 1);
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
        let body = build_body(Probe::LlamaCppNProbs, "m", "prompt", &[54, 55]);
        assert_eq!(body["n_predict"], 1);
        assert_eq!(body["n_probs"], 32);
        assert_eq!(body["temperature"], 1.0);
        assert_eq!(body["top_k"], 0);
        assert_eq!(body["repeat_penalty"], 1.0);
    }

    #[test]
    fn vllm_map_response_is_parsed_by_token_id() {
        let body = json!({"choices": [{"logprobs": {"top_logprobs": [{
            "token_id:32": -0.51, "token_id:33": -1.2, "token_id:99": -4.0
        }]}}]});
        let values = parse_logprobs(Probe::LogprobTokenIds, &body, &[32, 33], &['A', 'B']).unwrap();
        assert_eq!(values, vec![-0.51, -1.2]);
    }

    #[test]
    fn vllm_map_response_falls_back_to_token_text() {
        let body = json!({"choices": [{"logprobs": {"top_logprobs": [{"A": -0.1, "B": -2.0}]}}]});
        let values = parse_logprobs(Probe::CompletionsTopK, &body, &[32, 33], &['A', 'B']).unwrap();
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
            parse_logprobs(Probe::CompletionsTopK, &body, &[54, 55, 56], &['A', 'B', 'C']).unwrap();
        assert_eq!(values, vec![-12.32, -2.1e-05, -15.61]);
    }

    #[test]
    fn llamacpp_native_response_prefers_top_logprobs() {
        let body = json!({"completion_probabilities": [{"id": 32, "token": "A", "prob": 0.6,
            "top_probs": [{"id": 32, "token": "A", "prob": 0.6}],
            "top_logprobs": [{"id": 32, "token": "A", "logprob": -0.51},
                             {"id": 33, "token": "B", "logprob": -0.92}]}]});
        let values = parse_logprobs(Probe::LlamaCppNProbs, &body, &[32, 33], &['A', 'B']).unwrap();
        assert_eq!(values, vec![-0.51, -0.92]);
    }

    #[test]
    fn llamacpp_native_response_falls_back_to_probabilities() {
        let body = json!({"completion_probabilities": [{"id": 32, "token": "A",
            "top_probs": [{"id": 32, "token": "A", "prob": 0.5}, {"id": 33, "token": "B", "prob": 0.5}]}]});
        let values = parse_logprobs(Probe::LlamaCppNProbs, &body, &[32, 33], &['A', 'B']).unwrap();
        assert!((values[0] - 0.5f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn missing_option_is_an_error_not_a_zero() {
        let body = json!({"choices": [{"logprobs": {"top_logprobs": [{"token_id:32": -0.1}]}}]});
        let error =
            parse_logprobs(Probe::CompletionsTopK, &body, &[32, 33], &['A', 'B']).unwrap_err();
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
