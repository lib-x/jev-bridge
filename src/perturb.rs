//! Perturbation flips: a decision that changes when nothing relevant changed
//! is not a decision about the evidence.
//!
//! The method follows the published prompt-lab practice of the local Jev
//! implementations: score every row, then score it again with a change that
//! must not matter, and count how often the winning option moves. Reversing
//! the option order is the sharpest of those changes for this bridge, because
//! the letter contract lets the model see the options in order and it has a
//! position preference (measured on the reference endpoint: the same question
//! answered `frustrated` at 0.692 when that option came first and `furious` at
//! 0.759 when the order was reversed). `--scoring binary` judges each
//! candidate on its own and is expected to report zero flips.

use anyhow::Result;
use serde::Serialize;

use crate::server::Bridge;
use crate::wire::{Row, ScoredAnswer};

/// A change to a row that must not move the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Perturbation {
    /// The declared options in reverse order.
    Reversed,
}

impl Perturbation {
    /// Every perturbation a report covers.
    pub const ALL: [Perturbation; 1] = [Perturbation::Reversed];

    /// The name recorded in the report.
    pub fn name(self) -> &'static str {
        match self {
            Perturbation::Reversed => "reversed-options",
        }
    }

    /// Apply the change to one row.
    pub fn apply(self, row: &Row) -> Row {
        let mut changed = row.clone();
        match self {
            Perturbation::Reversed => changed.options.reverse(),
        }
        changed
    }
}

/// One row's answer under one perturbation.
#[derive(Debug, Serialize)]
pub struct FlipRow {
    /// Row id, copied from the fixture.
    pub id: String,
    /// Which perturbation was applied.
    pub perturbation: String,
    /// The option that won without the change.
    pub baseline: String,
    /// The option that won with the change.
    pub perturbed: String,
    /// Whether the winning option moved.
    pub flipped: bool,
    /// Probability the baseline winner held before the change.
    pub baseline_probability: f64,
    /// Probability the same option held after the change.
    pub perturbed_probability: f64,
}

/// The flip report for one fixture.
#[derive(Debug, Serialize)]
pub struct FlipReport {
    /// Comparisons made: rows times perturbations.
    pub rows: usize,
    /// Comparisons whose winning option moved.
    pub flips: usize,
    /// `flips / rows`.
    pub flip_rate: f64,
    /// Per-comparison detail, in fixture order.
    pub results: Vec<FlipRow>,
}

/// Score every row, then score it again under every perturbation.
pub async fn run(bridge: &Bridge, rows: &[Row]) -> Result<FlipReport> {
    let mut results = Vec::new();
    for row in rows {
        let baseline = bridge.score_row(row).await?;
        let winner = argmax(&baseline.answer)?;
        for perturbation in Perturbation::ALL {
            let perturbed = bridge.score_row(&perturbation.apply(row)).await?;
            let perturbed_winner = argmax(&perturbed.answer)?;
            results.push(FlipRow {
                id: row.id.clone(),
                perturbation: perturbation.name().to_string(),
                baseline: winner.clone(),
                perturbed: perturbed_winner.clone(),
                flipped: winner != perturbed_winner,
                baseline_probability: probability_of(&baseline.answer, &winner)?,
                perturbed_probability: probability_of(&perturbed.answer, &winner)?,
            });
        }
    }
    let flips = results.iter().filter(|row| row.flipped).count();
    Ok(FlipReport {
        rows: results.len(),
        flips,
        flip_rate: if results.is_empty() {
            0.0
        } else {
            flips as f64 / results.len() as f64
        },
        results,
    })
}

/// The option id with the largest probability.
fn argmax(answer: &ScoredAnswer) -> Result<String> {
    let index = answer
        .probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow::anyhow!("a scored answer carries no probabilities"))?;
    Ok(answer.option_ids[index].clone())
}

/// The probability of one option id, wherever it sits in the answer.
fn probability_of(answer: &ScoredAnswer, option: &str) -> Result<f64> {
    let index = answer
        .option_ids
        .iter()
        .position(|id| id == option)
        .ok_or_else(|| anyhow::anyhow!("option {option:?} is missing from the scored answer"))?;
    Ok(answer.probabilities[index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row() -> Row {
        serde_json::from_value(json!({
            "id": "r1",
            "state": "The customer was charged twice.",
            "question": "Which department?",
            "options": [
                {"id": "billing", "description": "charges"},
                {"id": "sales", "description": "purchases"},
            ],
        }))
        .expect("a well-formed row")
    }

    #[test]
    fn reversing_keeps_the_same_options_in_the_opposite_order() {
        let original = row();
        let reversed = Perturbation::Reversed.apply(&original);
        assert_eq!(
            reversed
                .options
                .iter()
                .map(|option| option.id.as_str())
                .collect::<Vec<_>>(),
            vec!["sales", "billing"]
        );
        // The original row is untouched: the baseline run must use it unchanged.
        assert_eq!(original.options[0].id, "billing");
    }

    #[test]
    fn the_winner_is_the_largest_probability_wherever_it_sits() {
        let answer = ScoredAnswer {
            id: "r1".to_string(),
            option_ids: vec!["sales".to_string(), "billing".to_string()],
            probabilities: vec![0.2, 0.8],
            input_tokens: 0,
            prompt_version: None,
        };
        assert_eq!(argmax(&answer).unwrap(), "billing");
        assert_eq!(probability_of(&answer, "billing").unwrap(), 0.8);
        assert!(probability_of(&answer, "missing").is_err());
    }
}
