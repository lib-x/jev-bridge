//! The direct decision prompt contract and the single-token answer-slot check.
//!
//! This mirrors fastjev's `direct-options-v1` contract: one system instruction,
//! one JSON payload of evidence / criterion / lettered options, rendered
//! through the model's own chat template with thinking disabled, then scored
//! by reading the next-token logits of the fixed uppercase answer letters.
//!
//! Rendering is delegated to the serving runtime (`/apply-template` on
//! llama.cpp) rather than reproduced locally. That keeps the prompt byte
//! identical to what the runtime itself would feed the model, which is the
//! property the whole readout depends on.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::wire::{to_python_json, RowOption};

pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";
pub const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
pub const PROMPT_VERSION: &str = "direct-options-v1";

/// Build the chat messages for one decision.
///
/// The payload uses Python's JSON layout so the text matches the reference
/// implementation byte for byte.
pub fn direct_messages(state: &Value, question: &str, options: &[RowOption]) -> Value {
    let entries: Vec<Value> = options
        .iter()
        .enumerate()
        .map(|(index, option)| {
            json!({
                "letter": &LETTERS[index..index + 1],
                "description": option.description,
            })
        })
        .collect();
    let payload = json!({
        "evidence": state,
        "criterion": question,
        "options": entries,
    });
    json!([
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": to_python_json(&payload)},
    ])
}

/// SHA-256 of the rendered prompt.
///
/// This is the strongest available check that a bridged run used exactly the
/// same prompt as a reference run: equal hashes mean equal bytes.
pub fn prompt_sha256(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Softmax over the selected option logits, matching fastjev's helper.
pub fn softmax(values: &[f64]) -> Vec<f64> {    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

/// Chat-template rendering and tokenization delegated to the serving runtime.
pub struct ServerTemplate {
    client: reqwest::Client,
    base: String,
    model: String,
    api_key: Option<String>,
    chat_template_kwargs: Value,
}

impl ServerTemplate {
    pub fn new(
        client: reqwest::Client,
        base: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        chat_template_kwargs: Value,
    ) -> Self {
        Self {
            client,
            base: base.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key,
            chat_template_kwargs,
        }
    }

    pub fn chat_template_kwargs(&self) -> Value {
        self.chat_template_kwargs.clone()
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value> {
        let url = format!("{}{path}", self.base);
        let mut request = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("POST {url} failed"))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .with_context(|| format!("reading {url} response failed"))?;
        if !status.is_success() {
            bail!("POST {url} returned HTTP {status}: {}", truncate(&text));
        }
        serde_json::from_str(&text)
            .with_context(|| format!("POST {url} returned a non-JSON body: {}", truncate(&text)))
    }

    /// Render messages with the runtime's own chat template.
    pub async fn render(&self, messages: &Value) -> Result<String> {
        let body = json!({
            "model": self.model,
            "messages": messages,
            "chat_template_kwargs": self.chat_template_kwargs,
        });
        let response = self
            .post("/apply-template", body)
            .await
            .context("rendering the chat template on the serving runtime failed")?;
        response
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("/apply-template response has no prompt string"))
    }

    pub async fn tokenize(&self, content: &str) -> Result<Vec<u32>> {
        let body = json!({
            "model": self.model,
            "content": content,
            "add_special": false,
        });
        let response = self.post("/tokenize", body).await?;
        let tokens = response
            .get("tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("/tokenize response has no tokens array"))?;
        tokens
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| anyhow::anyhow!("/tokenize returned a non-integer token id"))
            })
            .collect()
    }

    /// Resolve the sixteen answer slots and prove they are exact single tokens.
    ///
    /// A slot that is not one round-tripping token would make the logit readout
    /// meaningless, so this fails loudly instead of scoring the wrong ids.
    pub async fn resolve_slots(&self) -> Result<Vec<u32>> {
        let mut slots = Vec::with_capacity(LETTERS.len());
        for letter in LETTERS.chars() {
            let ids = self.tokenize(&letter.to_string()).await?;
            if ids.len() != 1 {
                bail!(
                    "answer slot {letter:?} is not a single token ({} tokens); this model cannot \
                     serve the direct option-logit contract",
                    ids.len()
                );
            }
            slots.push(ids[0]);
        }
        let mut unique = slots.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != slots.len() {
            bail!("answer-slot tokens collide");
        }
        let decoded = self
            .post("/detokenize", json!({"model": self.model, "tokens": slots}))
            .await
            .context("verifying answer-slot round trip failed")?;
        let text = decoded
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("/detokenize response has no content string"))?;
        if text != LETTERS {
            bail!("answer slots decode to {text:?} instead of {LETTERS:?}");
        }
        Ok(slots)
    }

    /// Prove that appending an answer letter cannot change tokenization.
    ///
    /// Without this, the last prompt token could merge with the letter and the
    /// scored logit would belong to a different token than the chosen option.
    pub async fn verify_boundary(&self, prompt: &str, slots: &[u32]) -> Result<()> {
        let base = self.tokenize(prompt).await?;
        if base.is_empty() {
            bail!("rendered prompt is empty");
        }
        for (index, letter) in LETTERS.chars().enumerate() {
            let combined = self.tokenize(&format!("{prompt}{letter}")).await?;
            let mut expected = base.clone();
            expected.push(slots[index]);
            if combined != expected {
                bail!(
                    "answer boundary changes tokenization for slot {letter:?}: the prompt does \
                     not end in a position where {letter:?} stays one token"
                );
            }
        }
        Ok(())
    }
}

fn truncate(text: &str) -> String {
    let mut owned = text.chars().take(400).collect::<String>();
    if text.chars().count() > 400 {
        owned.push('…');
    }
    owned
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> Vec<RowOption> {
        vec![
            RowOption { id: "billing".into(), description: "Billing and refunds.".into() },
            RowOption { id: "sales".into(), description: "Pricing and contracts.".into() },
        ]
    }

    #[test]
    fn payload_matches_python_json_layout() {
        let messages = direct_messages(&json!("charged twice"), "Which queue?", &options());
        let user = messages[1]["content"].as_str().unwrap();
        assert_eq!(
            user,
            r#"{"evidence": "charged twice", "criterion": "Which queue?", "options": [{"letter": "A", "description": "Billing and refunds."}, {"letter": "B", "description": "Pricing and contracts."}]}"#
        );
        assert_eq!(messages[0]["content"], DIRECT_SYSTEM);
    }

    #[test]
    fn structured_state_is_embedded_as_json() {
        let messages = direct_messages(&json!({"message": "hi", "count": 2}), "q", &options());
        let user = messages[1]["content"].as_str().unwrap();
        assert!(user.contains(r#""evidence": {"message": "hi", "count": 2}"#), "{user}");
    }

    #[test]
    fn softmax_is_stable_for_large_magnitudes() {
        let probabilities = softmax(&[-1000.0, -1001.0]);
        assert!((probabilities[0] - 0.7310585786300049).abs() < 1e-12);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn prompt_hash_is_stable_and_distinguishing() {
        assert_eq!(prompt_sha256("abc").len(), 64);
        assert_ne!(prompt_sha256("abc"), prompt_sha256("abd"));
    }

    #[test]
    fn softmax_matches_the_measured_llamacpp_distribution() {
        // Measured on MiniCPM5-2B-Q8_0 for a billing question: the option
        // logits were -12.32 (A), -2.1e-05 (B), -15.61 (C).
        let probabilities = softmax(&[-12.320333, -0.0000213, -15.617624]);
        assert!(probabilities[1] > 0.9999, "{probabilities:?}");
        assert!(probabilities[0] < 1e-5);
    }
}
