//! Local tokenization through the Hugging Face `tokenizers` crate.
//!
//! The serving runtime's `/tokenize` endpoint is enough when it exists, and the
//! bridge prefers it because it cannot drift from the runtime's own vocabulary.
//! A `tokenizer.json` file makes the whole contract independent of the runtime:
//! the answer-slot check, the boundary check and the prompt length all run
//! in-process, which matters for runtimes without a tokenize endpoint and for
//! deployments that must not depend on extra HTTP round trips.
//!
//! The file must be the model's own `tokenizer.json`. A different vocabulary
//! would silently score the wrong token ids.

use std::path::Path;

use anyhow::{bail, Result};
use tokenizers::Tokenizer;

/// A tokenizer loaded from a local `tokenizer.json`.
pub struct LocalTokenizer {
    tokenizer: Tokenizer,
}

impl LocalTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            anyhow::anyhow!("loading {} failed: {error}", path.display())
        })?;
        Ok(Self { tokenizer })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| anyhow::anyhow!("parsing the tokenizer failed: {error}"))?;
        Ok(Self { tokenizer })
    }

    /// Encode without special tokens, matching fastjev's `encode_prompt`.
    pub fn tokenize(&self, content: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(content, false)
            .map_err(|error| anyhow::anyhow!("tokenizing failed: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode skipping special tokens, matching fastjev's round-trip check.
    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|error| anyhow::anyhow!("detokenizing failed: {error}"))
    }

    /// The vocabulary size, for startup diagnostics.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Refuse a tokenizer whose special tokens disagree with the template.
    ///
    /// A template that emits `{{ bos_token }}` and a tokenizer whose BOS token
    /// is something else would produce a prompt the model never saw in
    /// training, so this is checked rather than assumed.
    pub fn verify_special_tokens(&self, expected: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
        for (name, value) in expected {
            let Some(text) = value.as_str() else {
                continue;
            };
            if !name.ends_with("_token") {
                continue;
            }
            let known = self.tokenizer.token_to_id(text).is_some();
            if !known {
                bail!(
                    "the chat template context sets {name} = {text:?}, but this tokenizer has no \
                     such token; the rendered prompt would not match the model"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_fails_with_a_clear_message() {
        let error = LocalTokenizer::from_file(Path::new("/nonexistent/tokenizer.json"))
            .err()
            .expect("a missing file must not load");
        assert!(
            format!("{error:#}").contains("/nonexistent/tokenizer.json"),
            "{error:#}"
        );
    }

    #[test]
    fn garbage_bytes_are_rejected() {
        let error = LocalTokenizer::from_bytes(b"not a tokenizer")
            .err()
            .expect("garbage must not parse");
        assert!(
            format!("{error:#}").contains("parsing the tokenizer failed"),
            "{error:#}"
        );
    }
}
