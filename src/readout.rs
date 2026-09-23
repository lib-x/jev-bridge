//! The readout self-check: does this model actually answer with slot letters?
//!
//! The startup checks in [`crate::prompt`] prove the *mechanism*: the answer
//! letters are single tokens, appending one cannot re-tokenize the prompt, and
//! a transport can read the distribution. They say nothing about whether the
//! model was trained to answer that way. A general instruct model that never
//! saw single-token slot readout does not fail loudly — it degrades silently,
//! typically by acquiring a probability floor on one option — and the bridge
//! would return confident-looking numbers about nothing.
//!
//! So this module asks a few questions whose answers are obvious, through the
//! exact readout path every real request uses, and reports what came back.
//! It cannot prove the model is good at decisions (that needs a real,
//! gold-labelled workload — see [`crate::evaluate`]); it catches the model
//! that is not answering the question at all.

use crate::wire::{Question, RowOption};
use serde::Serialize;

/// The state every probe is asked about.
pub const PROBE_STATE: &str = "This is a self-check of the decision channel. Answer each question with the option that is obviously correct.";

/// One probe: a question whose answer is obvious, and the option that names it.
#[derive(Debug, Clone, Copy)]
pub struct Probe {
    /// Stable id for the report.
    pub id: &'static str,
    /// The criterion to apply to [`PROBE_STATE`].
    pub question: &'static str,
    /// `(option id, description)` pairs, in slot order.
    pub options: &'static [(&'static str, &'static str)],
    /// The option id an answer that understood the question would name.
    pub expected: &'static str,
}

/// The probes, covering all three question shapes.
///
/// Deliberately trivial: a model that cannot name the right option here is not
/// making decisions, whatever its probabilities look like.
pub const PROBES: &[Probe] = &[
    Probe {
        id: "sky-is-blue",
        question: "On a clear day, is the sky blue?",
        options: &[("true", "Yes"), ("false", "No")],
        expected: "true",
    },
    Probe {
        id: "ice-not-hot",
        question: "Is ice hotter than boiling water?",
        options: &[("true", "Yes"), ("false", "No")],
        expected: "false",
    },
    Probe {
        id: "which-is-fruit",
        question: "Which of these is a fruit?",
        options: &[
            ("apple", "An apple"),
            ("granite", "A piece of granite"),
            ("iron", "An iron bar"),
        ],
        expected: "apple",
    },
    Probe {
        id: "summer-day",
        question: "How hot is a summer day in the sun?",
        options: &[
            ("0", "Freezing"),
            ("1", "Cold"),
            ("2", "Warm"),
            ("3", "Hot"),
        ],
        expected: "3",
    },
];

/// The probes as scorer questions, all asked about [`PROBE_STATE`].
pub fn questions() -> Vec<Question> {
    PROBES
        .iter()
        .map(|probe| Question {
            id: probe.id.to_string(),
            question: probe.question.to_string(),
            options: probe
                .options
                .iter()
                .map(|(id, description)| RowOption {
                    id: (*id).to_string(),
                    description: (*description).to_string(),
                })
                .collect(),
        })
        .collect()
}

/// One probe's outcome.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeResult {
    /// Which probe this is.
    pub id: String,
    /// The option that is obviously right.
    pub expected: String,
    /// The option the readout named (first maximum, ties to the earlier slot).
    pub argmax: String,
    /// The probability the readout gave the expected option.
    pub expected_probability: f64,
    /// Whether the readout named the expected option.
    pub passed: bool,
}

/// What the self-check found, as reported by `GET /health`.
#[derive(Debug, Clone, Serialize)]
pub struct ReadoutCheck {
    /// How many probes ran.
    pub probes: usize,
    /// How many named the expected option.
    pub passed: usize,
    /// Every probe's outcome.
    pub results: Vec<ProbeResult>,
}

impl ReadoutCheck {
    /// Whether every probe named the expected option.
    pub fn is_ok(&self) -> bool {
        self.probes > 0 && self.passed == self.probes
    }

    /// Build the report from the per-probe outcomes.
    pub fn from_results(results: Vec<ProbeResult>) -> Self {
        let passed = results.iter().filter(|result| result.passed).count();
        Self {
            probes: results.len(),
            passed,
            results,
        }
    }
}

/// Score one probe's outcome from its answer, using the same first-maximum
/// rule the wire layer reports its answers with.
pub fn outcome(probe: &Probe, option_ids: &[String], probabilities: &[f64]) -> ProbeResult {
    let argmax_index = probabilities
        .iter()
        .enumerate()
        .fold(0usize, |best, (index, value)| {
            if *value > probabilities[best] {
                index
            } else {
                best
            }
        });
    let argmax = option_ids.get(argmax_index).cloned().unwrap_or_default();
    let expected_probability = option_ids
        .iter()
        .position(|id| *id == probe.expected)
        .and_then(|index| probabilities.get(index))
        .copied()
        .unwrap_or(0.0);
    ProbeResult {
        id: probe.id.to_string(),
        expected: probe.expected.to_string(),
        passed: argmax == probe.expected,
        argmax,
        expected_probability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_probe_declares_its_expected_option() {
        for probe in PROBES {
            assert!(
                probe.options.iter().any(|(id, _)| *id == probe.expected),
                "probe {:?}: expected {:?} is not one of its options",
                probe.id,
                probe.expected
            );
            assert!(
                probe.options.len() >= 2,
                "probe {:?} needs at least two options",
                probe.id
            );
        }
        let ids: Vec<&str> = PROBES.iter().map(|probe| probe.id).collect();
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "probe ids must be unique");
    }

    #[test]
    fn questions_carry_the_probes_verbatim() {
        let questions = questions();
        assert_eq!(questions.len(), PROBES.len());
        assert_eq!(questions[0].id, "sky-is-blue");
        assert_eq!(questions[0].options[0].id, "true");
        assert_eq!(questions[0].options[1].description, "No");
    }

    #[test]
    fn an_answer_that_names_the_expected_option_passes() {
        let probe = &PROBES[0];
        let ids = vec!["true".to_string(), "false".to_string()];
        let result = outcome(probe, &ids, &[0.9, 0.1]);
        assert!(result.passed);
        assert_eq!(result.argmax, "true");
        assert!((result.expected_probability - 0.9).abs() < 1e-12);
    }

    #[test]
    fn an_answer_that_names_another_option_fails_and_reports_what_it_named() {
        let probe = &PROBES[1];
        let ids = vec!["true".to_string(), "false".to_string()];
        let result = outcome(probe, &ids, &[0.8, 0.2]);
        assert!(!result.passed);
        assert_eq!(result.argmax, "true");
        assert!((result.expected_probability - 0.2).abs() < 1e-12);
    }

    #[test]
    fn ties_resolve_to_the_earlier_slot() {
        let probe = &PROBES[0];
        let ids = vec!["true".to_string(), "false".to_string()];
        // A flat distribution resolves to the first declared option, exactly
        // as the wire layer's first-max rule does.
        assert_eq!(outcome(probe, &ids, &[0.5, 0.5]).argmax, "true");
    }

    #[test]
    fn a_report_counts_its_passes() {
        let ids = vec!["true".to_string(), "false".to_string()];
        let results = vec![
            outcome(&PROBES[0], &ids, &[0.9, 0.1]),
            outcome(&PROBES[1], &ids, &[0.8, 0.2]),
        ];
        let check = ReadoutCheck::from_results(results);
        assert_eq!(check.probes, 2);
        assert_eq!(check.passed, 1);
        assert!(!check.is_ok());

        let all_pass = ReadoutCheck::from_results(vec![outcome(&PROBES[0], &ids, &[0.9, 0.1])]);
        assert!(all_pass.is_ok());

        // An empty check is not a pass: nothing was verified.
        assert!(!ReadoutCheck::from_results(Vec::new()).is_ok());
    }
}
