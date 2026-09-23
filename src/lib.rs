//! Bridge a generic OpenAI-compatible inference API into a Jev-style decision
//! scoring service.
//! Jev-style decisions are not a model format; they are a readout. This crate
//! implements that readout on top of any runtime that can return next-token log
//! probabilities:
//!
//! 1. render the `direct-options-v1` prompt through the serving runtime's own
//!    chat template, with thinking disabled;
//! 2. read the log probabilities of the fixed uppercase answer letters at the
//!    final position;
//! 3. softmax only over the declared options, so no other vocabulary can leak
//!    into the distribution;
//! 4. answer in the System One wire format with `choice`, `noul` and `score`
//!    questions.
//!
//! No model weights are copied and no answer text is generated: a decision
//! costs one prompt evaluation and zero output tokens.
//!
//! # Layout
//!
//! * [`strategy`] probes the upstream service and parses the response shapes
//!   different runtimes use for the same distribution.
//! * [`prompt`] owns the prompt contract, the single-token answer-slot check
//!   and the softmax.
//! * [`server`] holds [`server::Bridge`] and the HTTP surface.
//! * [`wire`] validates System One requests and builds their responses.
//! * [`evaluate`] turns `--score` predictions plus a gold file into offline
//!   calibration metrics (accuracy, NLL, Brier, ECE, reliability curve) and
//!   recomputes a written report for verification.
//!
//! The `jev-bridge` binary is a thin CLI over [`server::Bridge`].
//!
//! # Documentation
//!
//! Every public item is documented, and the crate builds with
//! `#![warn(missing_docs)]` so that stays true.

#![warn(missing_docs)]

pub mod evaluate;
pub mod prompt;
pub mod render;
pub mod server;
pub mod strategy;
#[cfg(feature = "local-tokenizer")]
pub mod tokenizer;
pub mod wire;
