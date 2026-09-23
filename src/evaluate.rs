//! Offline calibration evaluation for `--score` predictions.
//!
//! `--score` already writes one JSONL line per row with the full probability
//! vector. This module turns those files (plus a gold file) into the numbers
//! that say whether the probabilities are usable — not only whether the answers
//! are right: accuracy, balanced accuracy, NLL, Brier (both flavours), ECE and
//! the reliability curve.
//!
//! The bridge itself performs no calibration and recommends no thresholds; this
//! module only *measures*. A deployment that wants to act on the probabilities
//! can validate them on its own workload first.
//!
//! # Definitions (frozen)
//!
//! * **predicted slot** = `argmax` of the probability vector, ties broken by
//!   the first slot in the row's declared option order (the same first-max rule
//!   the wire layer reports its answers with). Rows without a vector fall back
//!   to their `label`.
//! * **accuracy** = correct / predicted rows.
//! * **balanced accuracy** = mean of per-class recall over the classes that
//!   occur as gold (macro-recall; classes with no gold row are excluded).
//! * **NLL** = mean of `-ln(max(p_gold, 1e-12))` over scored rows.
//! * **brier_multiclass** = mean of `Σ_k (p_k - 1[gold == slot_k])²` over
//!   scored rows; defined for any number of slots.
//! * **brier_binary** = mean of `(p_positive - 1[gold == positive])²` over the
//!   scored rows that declare a positive class. `n_binary` says how many rows
//!   that was. A two-slot `true`/`false` row defaults `positive` to `true`
//!   (the `noul` contract), so the binarisation is never arbitrary.
//! * **confidence** = `max(p)`, the probability mass behind the predicted slot.
//! * **ECE** = `Σ_b (n_b / N) · |acc_b − conf_b|` over `bins` equal-width bins
//!   in `[0,1]`; bin index = `min(bins-1, floor(conf · bins))`, so `conf == 1.0`
//!   lands in the last bin rather than overflowing. `N` is the number of scored
//!   rows.
//! * **reliability curve** = the per-bin `(lower, upper, n, mean_confidence,
//!   accuracy)` tuples ECE is summed from. Empty bins are omitted.
//!
//! Rows that carry no probability vector are counted in `n_items`/`n_missing`
//! and contribute to accuracy, but they are excluded from NLL / Brier / ECE —
//! `n_scored` names exactly how many rows did contribute, so coverage is never
//! hidden.
//!
//! Every metric is computed per stratum (the gold file's `family`, with rows
//! that declare none reported under `unlabelled`), and the report also carries
//! the `overall` stratum. Strata are never silently pooled into one number by
//! this module; a caller that wants an aggregate has to say so itself.
//!
//! # Verification
//!
//! [`verify`] recomputes a written report from the same inputs and compares it
//! field by field: the report is a claim, the prediction and gold files are the
//! evidence. The tolerance belongs to the caller (`--tol`) and is never read
//! out of the report, so a rewritten number cannot be laundered by declaring a
//! looser tolerance.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Probability floor for NLL: keeps `-ln(0)` from becoming `+inf`.
pub const NLL_PROBABILITY_FLOOR: f64 = 1e-12;

/// Default number of equal-width confidence bins for ECE and the reliability
/// curve.
pub const DEFAULT_BINS: usize = 10;

/// The schema identifier every report carries.
pub const REPORT_SCHEMA: &str = "jev-bridge-evaluation-v1";

/// Stratification key for gold rows that declare no `family`.
pub const UNLABELLED_FAMILY: &str = "unlabelled";

/// One prediction line, as `--score` writes it.
///
/// Only `id` and `option_ids` are required. A line with `probabilities` is
/// *scored* and feeds every metric; a line with only a `label` still counts
/// towards accuracy. Extra fields (logprobs, hashes, probe name) are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct PredictionLine {
    /// Row id, matched against the gold file.
    pub id: String,
    /// Option ids in the order the probabilities follow.
    pub option_ids: Vec<String>,
    /// Probability vector aligned to `option_ids`.
    #[serde(default)]
    pub probabilities: Option<Vec<f64>>,
    /// A bare predicted option id, used when no probability vector is present.
    #[serde(default)]
    pub label: Option<String>,
}

/// One gold line: the right answer for one row.
#[derive(Debug, Clone, Deserialize)]
pub struct GoldLine {
    /// Row id, matched against the predictions file.
    pub id: String,
    /// The correct option id; must be one of the row's declared options.
    pub gold: String,
    /// Stratification key; rows without one are reported under `unlabelled`.
    #[serde(default)]
    pub family: Option<String>,
    /// Slot used as the positive class for `brier_binary`.
    ///
    /// Optional: a two-slot `true`/`false` row defaults to `true`; other rows
    /// without a positive declaration are excluded from `brier_binary` only.
    #[serde(default)]
    pub positive: Option<String>,
}

/// One joined item reduced to exactly the quantities the metrics need.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Row id (diagnostics only; the metrics never read it).
    pub id: String,
    /// Stratification key, already resolved (`None` becomes `unlabelled`).
    pub family: String,
    /// Gold slot key.
    pub gold: String,
    /// The item's declared slot keys, in order.
    pub slots: Vec<String>,
    /// `argmax(probs)` when a vector is present, else the row's `label`.
    pub predicted: Option<String>,
    /// Probability vector aligned to `slots`. `None` ⇒ row is unscored.
    pub probs: Option<Vec<f64>>,
    /// Slot key used as the positive class for `brier_binary`.
    pub positive: Option<String>,
}

impl Sample {
    /// The index of `gold` within the declared slots, when it is present.
    fn gold_index(&self) -> Option<usize> {
        self.slots.iter().position(|slot| *slot == self.gold)
    }
}

/// One bin of the reliability curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReliabilityBin {
    /// Inclusive lower edge (`index / bins`).
    pub lower: f64,
    /// Exclusive upper edge (`(index + 1) / bins`).
    pub upper: f64,
    /// Scored rows that landed in this bin.
    pub n: usize,
    /// Mean `max(p)` of those rows.
    pub mean_confidence: f64,
    /// Fraction of those rows whose argmax was the gold slot.
    pub accuracy: f64,
}

/// Every number computed for one stratum.
///
/// `deny_unknown_fields`: a fabricated extra metric must fail to load rather
/// than ride along inside a report that [`verify`] then passes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    /// Rows in the stratum.
    pub n_items: usize,
    /// Rows with neither a probability vector nor a label.
    pub n_missing: usize,
    /// Rows that produced a predicted slot.
    pub n_predicted: usize,
    /// Rows that produced a full probability vector.
    pub n_scored: usize,
    /// `correct / n_predicted`; `None` when nothing was predicted.
    pub accuracy: Option<f64>,
    /// Mean per-class recall over the classes present as gold.
    pub balanced_accuracy: Option<f64>,
    /// The classes balanced accuracy averaged over (the gold classes present).
    pub classes_in_gold: Vec<String>,
    /// Per-class recall for every gold class present.
    pub per_class_recall: BTreeMap<String, f64>,
    /// Mean negative log-likelihood of the gold slot.
    pub nll: Option<f64>,
    /// Mean multiclass Brier score.
    pub brier_multiclass: Option<f64>,
    /// Scored rows that declared a positive class (the `brier_binary` denominator).
    pub n_binary: usize,
    /// Mean binary Brier score over the rows that declared a positive class.
    pub brier_binary: Option<f64>,
    /// Expected calibration error over `bins` equal-width confidence bins.
    pub ece: Option<f64>,
    /// The per-bin tuples ECE is summed from.
    pub reliability: Vec<ReliabilityBin>,
}

/// A written evaluation report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    /// Always [`REPORT_SCHEMA`]; lets a reader detect an incompatible file.
    pub schema: String,
    /// Equal-width bin count used for ECE and the reliability curve.
    pub bins: usize,
    /// SHA-256 of the predictions file's bytes.
    pub predictions_sha256: String,
    /// SHA-256 of the gold file's bytes.
    pub gold_sha256: String,
    /// Metrics over every row.
    pub overall: Metrics,
    /// Metrics per `family`, never pooled across strata.
    pub by_family: BTreeMap<String, Metrics>,
}

/// One disagreement between a report and a fresh recomputation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diff {
    /// JSON-ish path of the field, e.g. `overall.ece`.
    pub path: String,
    /// What the report claims.
    pub claimed: String,
    /// What the evidence recomputes to.
    pub recomputed: String,
}

/// The result of a verification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyOutcome {
    /// How many individual comparisons were made (so "0 diffs" is meaningful).
    pub checks: usize,
    /// Every mismatch, in traversal order.
    pub diffs: Vec<Diff>,
}

impl VerifyOutcome {
    /// Whether the report reproduced exactly (within tolerance).
    pub fn is_ok(&self) -> bool {
        self.diffs.is_empty()
    }

    /// One line per mismatch, for printing.
    pub fn report(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{} checks; {} mismatch{}",
            self.checks,
            self.diffs.len(),
            if self.diffs.len() == 1 { "" } else { "es" }
        );
        for diff in &self.diffs {
            let _ = writeln!(
                out,
                "  {}: claimed {} vs recomputed {}",
                diff.path, diff.claimed, diff.recomputed
            );
        }
        out
    }
}

/// Parse a predictions JSONL document (one [`PredictionLine`] per line).
///
/// Blank lines are skipped; a line that is not a JSON object, or that carries
/// no `id`, is an error that names the line number.
pub fn load_predictions(text: &str) -> Result<Vec<PredictionLine>> {
    let mut lines = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: PredictionLine = serde_json::from_str(line)
            .with_context(|| format!("predictions: line {} is not a prediction row", index + 1))?;
        lines.push(parsed);
    }
    Ok(lines)
}

/// Parse a gold JSONL document (one [`GoldLine`] per line).
pub fn load_gold(text: &str) -> Result<Vec<GoldLine>> {
    let mut lines = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: GoldLine = serde_json::from_str(line)
            .with_context(|| format!("gold: line {} is not a gold row", index + 1))?;
        lines.push(parsed);
    }
    Ok(lines)
}

/// The first-maximum rule: ties resolve to the earliest slot, so the result is
/// a pure function of the vector (the same rule the wire layer reports with).
fn argmax_first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Join predictions with gold by id, validating both sides.
///
/// Failures are loud and specific, because every one of them would otherwise
/// turn into a quietly wrong number:
///
/// * a duplicate id on either side;
/// * a prediction row whose id is not in the gold file (the two files are out
///   of sync — evaluating the overlap would hide exactly the rows that differ);
/// * a probability vector whose length disagrees with `option_ids`;
/// * a non-finite or negative probability;
/// * a gold slot that is not among the row's declared options.
pub fn join(predictions: &[PredictionLine], gold: &[GoldLine]) -> Result<Vec<Sample>> {
    let mut by_id: BTreeMap<&str, &PredictionLine> = BTreeMap::new();
    for line in predictions {
        if by_id.insert(line.id.as_str(), line).is_some() {
            bail!("predictions: duplicate id {:?}", line.id);
        }
    }

    let mut seen_gold: BTreeMap<&str, ()> = BTreeMap::new();
    let mut samples = Vec::with_capacity(gold.len());
    for line in gold {
        if seen_gold.insert(line.id.as_str(), ()).is_some() {
            bail!("gold: duplicate id {:?}", line.id);
        }
        let Some(prediction) = by_id.remove(line.id.as_str()) else {
            // The gold file asked about a row the predictions file never
            // scored. Counted as missing, never silently dropped.
            samples.push(Sample {
                id: line.id.clone(),
                family: line
                    .family
                    .clone()
                    .unwrap_or_else(|| UNLABELLED_FAMILY.to_string()),
                gold: line.gold.clone(),
                slots: Vec::new(),
                predicted: None,
                probs: None,
                positive: line.positive.clone(),
            });
            continue;
        };

        if !prediction.option_ids.contains(&line.gold) {
            bail!(
                "gold row {:?}: gold {:?} is not among the row's declared options {:?}",
                line.id,
                line.gold,
                prediction.option_ids
            );
        }

        let probs = match &prediction.probabilities {
            None => None,
            Some(probs) => {
                if probs.len() != prediction.option_ids.len() {
                    bail!(
                        "predictions row {:?}: {} probabilities for {} options",
                        line.id,
                        probs.len(),
                        prediction.option_ids.len()
                    );
                }
                if probs.iter().any(|value| !value.is_finite() || *value < 0.0) {
                    bail!(
                        "predictions row {:?}: non-finite or negative probability",
                        line.id
                    );
                }
                Some(probs.clone())
            }
        };

        let predicted = match &probs {
            Some(probs) => {
                let winner = argmax_first_max(probs);
                Some(prediction.option_ids[winner].clone())
            }
            None => prediction.label.clone(),
        };

        // A two-slot `true`/`false` row is the `noul` contract, whose positive
        // class is fixed to `true`; anything else has no default.
        let positive = line
            .positive
            .clone()
            .or_else(|| (prediction.option_ids == ["true", "false"]).then(|| "true".to_string()));
        if let Some(positive) = &positive
            && !prediction.option_ids.iter().any(|slot| slot == positive)
        {
            bail!(
                "gold row {:?}: positive {:?} is not among the row's declared options {:?}",
                line.id,
                positive,
                prediction.option_ids
            );
        }

        samples.push(Sample {
            id: line.id.clone(),
            family: line
                .family
                .clone()
                .unwrap_or_else(|| UNLABELLED_FAMILY.to_string()),
            gold: line.gold.clone(),
            slots: prediction.option_ids.clone(),
            predicted,
            probs,
            positive,
        });
    }

    if !by_id.is_empty() {
        let mut extra: Vec<&str> = by_id.keys().copied().collect();
        extra.sort_unstable();
        extra.truncate(5);
        bail!(
            "predictions: {} row(s) have no gold counterpart (first: {:?})",
            by_id.len(),
            extra
        );
    }

    Ok(samples)
}

/// Compute the bin index for a confidence, clamping the top edge into the last
/// bin and absorbing float error at the bottom edge.
fn bin_of(confidence: f64, bins: usize) -> usize {
    let raw = (confidence * bins as f64).floor();
    if raw.is_nan() || raw < 0.0 {
        0
    } else {
        (raw as usize).min(bins - 1)
    }
}

/// Reduce one stratum's samples to [`Metrics`].
///
/// Deterministic: same samples in, bit-identical numbers out. That is what lets
/// [`verify`] re-derive a published report and compare it exactly.
///
/// Panics only if `bins == 0`, which [`evaluate`] rejects first.
pub fn summarize(samples: &[Sample], bins: usize) -> Metrics {
    assert!(bins > 0, "bin count must be >= 1");

    let n_items = samples.len();
    let n_missing = samples
        .iter()
        .filter(|sample| sample.predicted.is_none() && sample.probs.is_none())
        .count();
    let predicted: Vec<&Sample> = samples
        .iter()
        .filter(|sample| sample.predicted.is_some())
        .collect();
    let n_predicted = predicted.len();

    let accuracy = if n_predicted == 0 {
        None
    } else {
        let hits = predicted
            .iter()
            .filter(|sample| sample.predicted.as_deref() == Some(sample.gold.as_str()))
            .count();
        Some(hits as f64 / n_predicted as f64)
    };

    // Per-class recall over the classes that actually occur as gold.
    let mut tallies: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for sample in &predicted {
        let entry = tallies.entry(sample.gold.as_str()).or_insert((0, 0));
        entry.0 += 1;
        if sample.predicted.as_deref() == Some(sample.gold.as_str()) {
            entry.1 += 1;
        }
    }
    let per_class_recall: BTreeMap<String, f64> = tallies
        .iter()
        .map(|(class, (n, ok))| ((*class).to_string(), *ok as f64 / *n as f64))
        .collect();
    let classes_in_gold: Vec<String> = tallies.keys().map(|key| (*key).to_string()).collect();
    let balanced_accuracy = if per_class_recall.is_empty() {
        None
    } else {
        Some(per_class_recall.values().sum::<f64>() / per_class_recall.len() as f64)
    };

    let scored: Vec<(&Sample, &[f64])> = samples
        .iter()
        .filter_map(|sample| {
            sample
                .probs
                .as_deref()
                .filter(|probs| probs.len() == sample.slots.len())
                .map(|probs| (sample, probs))
        })
        .collect();
    let n_scored = scored.len();

    let mut nll_sum = 0.0;
    let mut brier_multiclass_sum = 0.0;
    let mut brier_binary_sum = 0.0;
    let mut n_binary = 0usize;
    let mut bin_counts = vec![(0.0f64, 0usize, 0usize); bins];

    for (sample, probs) in scored.iter().copied() {
        // `gold_index` is `Some` for every sample that came through `join`;
        // a `None` here would mean a slot set that does not contain its own
        // gold, which the joiner rejects.
        let Some(gold_index) = sample.gold_index() else {
            continue;
        };

        nll_sum += -probs[gold_index].max(NLL_PROBABILITY_FLOOR).ln();

        let mut squared = 0.0;
        for (index, probability) in probs.iter().enumerate() {
            let target = if index == gold_index { 1.0 } else { 0.0 };
            squared += (probability - target) * (probability - target);
        }
        brier_multiclass_sum += squared;

        if let Some(positive) = sample
            .positive
            .as_ref()
            .and_then(|key| sample.slots.iter().position(|slot| slot == key))
        {
            let target = if positive == gold_index { 1.0 } else { 0.0 };
            brier_binary_sum += (probs[positive] - target) * (probs[positive] - target);
            n_binary += 1;
        }

        let top = argmax_first_max(probs);
        let confidence = probs[top];
        let bin = bin_of(confidence, bins);
        bin_counts[bin].0 += confidence;
        bin_counts[bin].1 += 1;
        if top == gold_index {
            bin_counts[bin].2 += 1;
        }
    }

    let mut reliability = Vec::new();
    let mut ece = 0.0;
    for (index, (confidence_sum, n, correct)) in bin_counts.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        let mean_confidence = confidence_sum / *n as f64;
        let bin_accuracy = *correct as f64 / *n as f64;
        ece += (*n as f64 / n_scored as f64) * (bin_accuracy - mean_confidence).abs();
        reliability.push(ReliabilityBin {
            lower: index as f64 / bins as f64,
            upper: (index + 1) as f64 / bins as f64,
            n: *n,
            mean_confidence,
            accuracy: bin_accuracy,
        });
    }

    let divide = |sum: f64| -> Option<f64> {
        if n_scored == 0 {
            None
        } else {
            Some(sum / n_scored as f64)
        }
    };

    Metrics {
        n_items,
        n_missing,
        n_predicted,
        n_scored,
        accuracy,
        balanced_accuracy,
        classes_in_gold,
        per_class_recall,
        nll: divide(nll_sum),
        brier_multiclass: divide(brier_multiclass_sum),
        n_binary,
        brier_binary: if n_binary == 0 {
            None
        } else {
            Some(brier_binary_sum / n_binary as f64)
        },
        // NOTE: `ece` above is already the item-weighted sum
        // `Σ_bins (n_bin / n_scored) · |acc_bin − conf_bin|` — it is already
        // normalised and must NOT go through `divide()` again, which would
        // divide it by the item count a second time.
        ece: if n_scored == 0 { None } else { Some(ece) },
        reliability,
    }
}

/// SHA-256 of a document's bytes, as lowercase hex.
fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Evaluate predictions against gold: join, validate, and summarise every
/// stratum plus the overall one.
///
/// Fails when `bins` is zero (there would be no reliability curve to speak of).
pub fn evaluate(predictions_text: &str, gold_text: &str, bins: usize) -> Result<Report> {
    if bins == 0 {
        bail!("bins must be >= 1");
    }
    let predictions = load_predictions(predictions_text)?;
    let gold = load_gold(gold_text)?;
    let samples = join(&predictions, &gold)?;

    let mut by_family: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
    for sample in &samples {
        by_family
            .entry(sample.family.clone())
            .or_default()
            .push(sample.clone());
    }

    Ok(Report {
        schema: REPORT_SCHEMA.to_string(),
        bins,
        predictions_sha256: sha256_hex(predictions_text),
        gold_sha256: sha256_hex(gold_text),
        overall: summarize(&samples, bins),
        by_family: by_family
            .iter()
            .map(|(family, rows)| (family.clone(), summarize(rows, bins)))
            .collect(),
    })
}

/// Recompute nothing: compare a claimed report against a freshly computed one,
/// field by field.
///
/// Numbers match when `|claimed - recomputed| <= tol`; strings, booleans and
/// nulls must match exactly. `tol` is the caller's audit threshold — it is
/// never read out of the report itself.
pub fn verify(claimed: &Report, recomputed: &Report, tol: f64) -> VerifyOutcome {
    let mut outcome = VerifyOutcome {
        checks: 0,
        diffs: Vec::new(),
    };
    let claimed_value = serde_json::to_value(claimed).expect("a Report is always serialisable");
    let recomputed_value =
        serde_json::to_value(recomputed).expect("a Report is always serialisable");
    compare_values(
        "report",
        &claimed_value,
        &recomputed_value,
        tol,
        &mut outcome,
    );
    outcome
}

/// Recursively compare two JSON documents, counting every leaf comparison.
fn compare_values(
    path: &str,
    claimed: &Value,
    recomputed: &Value,
    tol: f64,
    outcome: &mut VerifyOutcome,
) {
    match (claimed, recomputed) {
        (Value::Number(claimed), Value::Number(recomputed)) => {
            outcome.checks += 1;
            let matches = match (claimed.as_f64(), recomputed.as_f64()) {
                (Some(claimed), Some(recomputed)) => {
                    (claimed - recomputed).abs() <= tol || claimed == recomputed
                }
                _ => false,
            };
            if !matches {
                outcome.diffs.push(Diff {
                    path: path.to_string(),
                    claimed: claimed.to_string(),
                    recomputed: recomputed.to_string(),
                });
            }
        }
        (Value::Array(claimed), Value::Array(recomputed)) => {
            if claimed.len() != recomputed.len() {
                outcome.checks += 1;
                outcome.diffs.push(Diff {
                    path: path.to_string(),
                    claimed: format!("{} entries", claimed.len()),
                    recomputed: format!("{} entries", recomputed.len()),
                });
                return;
            }
            for (index, (claimed, recomputed)) in claimed.iter().zip(recomputed).enumerate() {
                compare_values(
                    &format!("{path}[{index}]"),
                    claimed,
                    recomputed,
                    tol,
                    outcome,
                );
            }
        }
        (Value::Object(claimed), Value::Object(recomputed)) => {
            // The key sets must agree exactly: a missing stratum is a mismatch
            // even when every shared field matches.
            let same_keys = claimed.len() == recomputed.len()
                && claimed.keys().all(|key| recomputed.contains_key(key));
            if !same_keys {
                outcome.checks += 1;
                outcome.diffs.push(Diff {
                    path: path.to_string(),
                    claimed: format!("keys {:?}", claimed.keys().collect::<Vec<_>>()),
                    recomputed: format!("keys {:?}", recomputed.keys().collect::<Vec<_>>()),
                });
                return;
            }
            for (key, claimed) in claimed {
                compare_values(
                    &format!("{path}.{key}"),
                    claimed,
                    &recomputed[key],
                    tol,
                    outcome,
                );
            }
        }
        (claimed, recomputed) => {
            outcome.checks += 1;
            if claimed != recomputed {
                outcome.diffs.push(Diff {
                    path: path.to_string(),
                    claimed: claimed.to_string(),
                    recomputed: recomputed.to_string(),
                });
            }
        }
    }
}

/// A short human-readable summary, for the CLI.
pub fn render_table(report: &Report) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} rows ({} scored, {} missing)  bins={}",
        report.overall.n_items, report.overall.n_scored, report.overall.n_missing, report.bins
    );
    let _ = write_stratum(&mut out, "overall", &report.overall);
    if !report.by_family.is_empty() {
        let _ = writeln!(out, "by family:");
        for (family, metrics) in &report.by_family {
            let _ = write_stratum(&mut out, &format!("  {family}"), metrics);
        }
    }
    out
}

fn write_stratum(out: &mut String, name: &str, metrics: &Metrics) -> std::fmt::Result {
    use std::fmt::Write as _;
    let show = |value: Option<f64>| match value {
        Some(value) => format!("{value:.4}"),
        None => "-".to_string(),
    };
    let _ = writeln!(
        out,
        "{name}: acc {}  balanced {}  ece {}  nll {}  brier {}  n={}",
        show(metrics.accuracy),
        show(metrics.balanced_accuracy),
        show(metrics.ece),
        show(metrics.nll),
        show(metrics.brier_multiclass),
        metrics.n_scored,
    );
    if metrics.n_binary > 0 {
        let _ = writeln!(
            out,
            "  brier_binary {}  (n_binary {})",
            show(metrics.brier_binary),
            metrics.n_binary
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(gold: &str, probs: Option<[f64; 2]>, positive: Option<&str>) -> Sample {
        let slots = vec!["A".to_string(), "B".to_string()];
        let vector = probs.map(|probs| vec![probs[0], probs[1]]);
        let predicted = vector
            .as_ref()
            .map(|vector| slots[argmax_first_max(vector)].clone());
        Sample {
            id: format!("s-{gold}-{}", vector.is_some()),
            family: UNLABELLED_FAMILY.to_string(),
            gold: gold.to_string(),
            slots,
            predicted,
            probs: vector,
            positive: positive.map(str::to_owned),
        }
    }

    /// The four-row example whose metrics are hand-computable, with
    /// probabilities chosen to be exact in binary floating point.
    ///
    /// Row 1: p(top)=0.5, correct  → bin 5
    /// Row 2: p(top)=0.75, correct → bin 7
    /// Row 3: p(top)=0.75, wrong   → bin 7
    /// Row 4: p(top)=1.0, correct  → bin 9
    fn known_four() -> Vec<Sample> {
        vec![
            sample("A", Some([0.5, 0.5]), Some("A")),
            sample("A", Some([0.25, 0.75]), Some("A")),
            sample("B", Some([0.75, 0.25]), Some("A")),
            sample("A", Some([1.0, 0.0]), Some("A")),
        ]
    }

    #[test]
    fn bin_assignment_clamps_the_top_edge() {
        assert_eq!(bin_of(0.0, 10), 0);
        assert_eq!(bin_of(0.05, 10), 0);
        assert_eq!(bin_of(0.75, 10), 7);
        assert_eq!(bin_of(0.999, 10), 9);
        assert_eq!(bin_of(1.0, 10), 9);
        assert_eq!(bin_of(1.0, 7), 6);
        assert_eq!(bin_of(-0.25, 10), 0);
        assert_eq!(bin_of(f64::NAN, 10), 0);
    }

    #[test]
    fn argmax_breaks_ties_towards_the_earlier_slot() {
        assert_eq!(argmax_first_max(&[0.5, 0.5]), 0);
        assert_eq!(argmax_first_max(&[0.2, 0.8]), 1);
        assert_eq!(argmax_first_max(&[0.3]), 0);
    }

    #[test]
    fn empty_stratum_has_no_numbers() {
        let metrics = summarize(&[], 10);
        assert_eq!(metrics.n_items, 0);
        assert_eq!(metrics.n_scored, 0);
        assert_eq!(metrics.n_predicted, 0);
        assert_eq!(metrics.n_missing, 0);
        assert!(metrics.accuracy.is_none());
        assert!(metrics.balanced_accuracy.is_none());
        assert!(metrics.ece.is_none());
        assert!(metrics.nll.is_none());
        assert!(metrics.brier_multiclass.is_none());
        assert!(metrics.brier_binary.is_none());
        assert!(metrics.reliability.is_empty());
    }

    #[test]
    fn the_known_four_row_example_reproduces_hand_computed_values() {
        let metrics = summarize(&known_four(), 10);
        assert_eq!(metrics.n_items, 4);
        assert_eq!(metrics.n_predicted, 4);
        assert_eq!(metrics.n_scored, 4);
        assert_eq!(metrics.n_missing, 0);
        assert_eq!(metrics.accuracy, Some(0.5));
        // Gold `A` appears 3x (rows 1, 2, 4) with 2 hits; gold `B` once with 0.
        assert_eq!(
            metrics.classes_in_gold,
            vec!["A".to_string(), "B".to_string()]
        );
        assert_eq!(metrics.per_class_recall["A"], 2.0 / 3.0);
        assert_eq!(metrics.per_class_recall["B"], 0.0);
        assert_eq!(metrics.balanced_accuracy, Some(1.0 / 3.0));
        // Binary Brier (positive = A): 0.25 + 0.5625 + 0.5625 + 0.0 = 1.375 / 4.
        assert_eq!(metrics.brier_binary, Some(0.34375));
        assert_eq!(metrics.n_binary, 4);
        // For two slots the multiclass score is exactly twice the binary one.
        assert_eq!(metrics.brier_multiclass, Some(0.6875));
        // ECE = 1/4*|1-0.5| + 2/4*|0-0.75| + 1/4*|1-1| = 0.5, all exact in binary.
        assert_eq!(metrics.ece, Some(0.5));
        assert_eq!(metrics.reliability.len(), 3);
        assert_eq!(
            metrics.reliability[0],
            ReliabilityBin {
                lower: 0.5,
                upper: 0.6,
                n: 1,
                mean_confidence: 0.5,
                accuracy: 1.0,
            }
        );
        assert_eq!(
            metrics.reliability[1],
            ReliabilityBin {
                lower: 0.7,
                upper: 0.8,
                n: 2,
                mean_confidence: 0.75,
                accuracy: 0.0,
            }
        );
        assert_eq!(
            metrics.reliability[2],
            ReliabilityBin {
                lower: 0.9,
                upper: 1.0,
                n: 1,
                mean_confidence: 1.0,
                accuracy: 1.0,
            }
        );
        // NLL = (-ln 0.5 - ln 0.25 - ln 0.25 - ln 1.0) / 4.
        let expected = (0.5f64.ln().abs() + 0.25f64.ln().abs() * 2.0) / 4.0;
        assert!((metrics.nll.unwrap() - expected).abs() < 1e-15);
    }

    #[test]
    fn a_perfect_forecast_has_zero_nll_and_zero_ece() {
        let rows = vec![
            sample("A", Some([1.0, 0.0]), Some("A")),
            sample("B", Some([0.0, 1.0]), Some("A")),
        ];
        let metrics = summarize(&rows, 10);
        assert_eq!(metrics.accuracy, Some(1.0));
        assert_eq!(metrics.balanced_accuracy, Some(1.0));
        assert_eq!(metrics.ece, Some(0.0));
        assert_eq!(metrics.nll.unwrap(), 0.0);
        assert_eq!(metrics.brier_multiclass, Some(0.0));
    }

    #[test]
    fn a_maximally_wrong_confident_forecast_saturates_ece_at_one() {
        let rows = vec![
            sample("A", Some([0.0, 1.0]), Some("A")),
            sample("B", Some([1.0, 0.0]), Some("A")),
        ];
        let metrics = summarize(&rows, 10);
        assert_eq!(metrics.accuracy, Some(0.0));
        assert_eq!(metrics.ece, Some(1.0));
        assert_eq!(metrics.brier_binary, Some(1.0));
    }

    #[test]
    fn the_nll_floor_bounds_an_impossible_gold() {
        // p_gold = 0 must not become +inf.
        let rows = vec![sample("A", Some([0.0, 1.0]), Some("A"))];
        let metrics = summarize(&rows, 10);
        let expected = -NLL_PROBABILITY_FLOOR.ln();
        assert!((metrics.nll.unwrap() - expected).abs() < 1e-9);
        assert!(metrics.nll.unwrap().is_finite());
    }

    #[test]
    fn rows_without_probabilities_are_excluded_from_nll_brier_and_ece() {
        let mut rows = known_four();
        let mut hard = sample("A", None, Some("A"));
        hard.predicted = Some("A".to_string());
        rows.push(hard);
        let metrics = summarize(&rows, 10);
        // Accuracy sees all five rows...
        assert_eq!(metrics.n_predicted, 5);
        assert_eq!(metrics.accuracy, Some(3.0 / 5.0));
        // ...but the probability metrics still see only the four scored ones.
        assert_eq!(metrics.n_scored, 4);
        assert_eq!(metrics.ece, Some(0.5));
        assert_eq!(metrics.brier_binary, Some(0.34375));
    }

    #[test]
    fn missing_rows_are_counted_not_silently_dropped() {
        let rows = vec![
            sample("A", Some([0.5, 0.5]), Some("A")),
            Sample {
                id: "no-answer".into(),
                family: UNLABELLED_FAMILY.into(),
                gold: "A".into(),
                slots: vec!["A".into(), "B".into()],
                predicted: None,
                probs: None,
                positive: None,
            },
        ];
        let metrics = summarize(&rows, 10);
        assert_eq!(metrics.n_items, 2);
        assert_eq!(metrics.n_missing, 1);
        assert_eq!(metrics.n_predicted, 1);
        assert_eq!(metrics.n_scored, 1);
        assert_eq!(metrics.accuracy, Some(1.0));
    }

    #[test]
    fn a_bin_count_of_one_is_plain_calibration_gap() {
        let metrics = summarize(&known_four(), 1);
        // One bin, so everything is clamped into it: mean confidence
        // (0.5 + 0.75 + 0.75 + 1.0) / 4 = 0.75, accuracy 0.5.
        assert_eq!(metrics.reliability.len(), 1);
        assert_eq!(metrics.reliability[0].lower, 0.0);
        assert_eq!(metrics.reliability[0].upper, 1.0);
        assert_eq!(metrics.reliability[0].n, 4);
        assert!((metrics.reliability[0].mean_confidence - 0.75).abs() < 1e-15);
        assert_eq!(metrics.reliability[0].accuracy, 0.5);
        assert!((metrics.ece.unwrap() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn summarize_is_a_pure_function_of_its_input() {
        let rows = known_four();
        assert_eq!(summarize(&rows, 10), summarize(&rows, 10));
        let finer = summarize(&rows, 25);
        assert_ne!(finer, summarize(&rows, 10));
    }

    // ----------------------------------------------------------------------
    // joining
    // ----------------------------------------------------------------------

    fn prediction(
        id: &str,
        probabilities: Option<Vec<f64>>,
        label: Option<&str>,
    ) -> PredictionLine {
        PredictionLine {
            id: id.to_string(),
            option_ids: vec!["yes".into(), "no".into()],
            probabilities,
            label: label.map(str::to_owned),
        }
    }

    fn gold(id: &str, gold: &str) -> GoldLine {
        GoldLine {
            id: id.to_string(),
            gold: gold.to_string(),
            family: None,
            positive: None,
        }
    }

    #[test]
    fn join_pairs_by_id_and_counts_gold_rows_without_predictions() {
        let predictions = vec![
            prediction("a", Some(vec![0.9, 0.1]), None),
            prediction("b", None, Some("no")),
        ];
        let gold = vec![gold("a", "yes"), gold("b", "no"), gold("c", "yes")];
        let samples = join(&predictions, &gold).unwrap();

        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].predicted.as_deref(), Some("yes"));
        assert_eq!(samples[1].predicted.as_deref(), Some("no"));
        assert_eq!(samples[1].probs, None);
        assert_eq!(samples[2].predicted, None);
        assert_eq!(samples[2].probs, None);
        assert_eq!(samples[2].family, UNLABELLED_FAMILY);
    }

    #[test]
    fn join_derives_the_winner_with_the_first_max_rule() {
        let predictions = vec![prediction("a", Some(vec![0.5, 0.5]), None)];
        let samples = join(&predictions, &[gold("a", "yes")]).unwrap();
        assert_eq!(samples[0].predicted.as_deref(), Some("yes"));
    }

    #[test]
    fn join_defaults_noul_positive_to_true() {
        let predictions = vec![PredictionLine {
            id: "a".into(),
            option_ids: vec!["true".into(), "false".into()],
            probabilities: Some(vec![0.8, 0.2]),
            label: None,
        }];
        let samples = join(&predictions, &[gold("a", "true")]).unwrap();
        assert_eq!(samples[0].positive.as_deref(), Some("true"));

        // A non-boolean option set gets no default.
        let predictions = vec![prediction("a", Some(vec![0.8, 0.2]), None)];
        let samples = join(&predictions, &[gold("a", "yes")]).unwrap();
        assert_eq!(samples[0].positive, None);
    }

    #[test]
    fn join_rejects_an_unmatched_prediction() {
        let predictions = vec![prediction("a", Some(vec![0.5, 0.5]), None)];
        let error = join(&predictions, &[gold("b", "yes")]).unwrap_err();
        assert!(error.to_string().contains("no gold counterpart"), "{error}");
    }

    #[test]
    fn join_rejects_duplicate_ids_on_either_side() {
        let predictions = vec![
            prediction("a", Some(vec![0.5, 0.5]), None),
            prediction("a", Some(vec![0.5, 0.5]), None),
        ];
        assert!(join(&predictions, &[gold("a", "yes")]).is_err());

        let predictions = vec![prediction("a", Some(vec![0.5, 0.5]), None)];
        let error = join(&predictions, &[gold("a", "yes"), gold("a", "no")]).unwrap_err();
        assert!(error.to_string().contains("duplicate id"), "{error}");
    }

    #[test]
    fn join_rejects_a_malformed_vector_or_an_unknown_gold() {
        // Length disagrees with the option count.
        let predictions = vec![prediction("a", Some(vec![0.5, 0.3, 0.2]), None)];
        assert!(join(&predictions, &[gold("a", "yes")]).is_err());

        // Negative probability.
        let predictions = vec![prediction("a", Some(vec![1.2, -0.2]), None)];
        assert!(join(&predictions, &[gold("a", "yes")]).is_err());

        // Gold is not one of the row's declared options.
        let predictions = vec![prediction("a", Some(vec![0.5, 0.5]), None)];
        let error = join(&predictions, &[gold("a", "maybe")]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not among the row's declared options"),
            "{error}"
        );
    }

    #[test]
    fn join_rejects_a_positive_that_is_not_a_declared_option() {
        let predictions = vec![prediction("a", Some(vec![0.5, 0.5]), None)];
        let mut line = gold("a", "yes");
        line.positive = Some("maybe".into());
        let error = join(&predictions, &[line]).unwrap_err();
        assert!(error.to_string().contains("positive"), "{error}");
    }

    // ----------------------------------------------------------------------
    // end to end: evaluate + verify
    // ----------------------------------------------------------------------

    fn predictions_document() -> String {
        [
            r#"{"id":"a","option_ids":["yes","no"],"probabilities":[0.9,0.1],"label":null}"#,
            r#"{"id":"b","option_ids":["yes","no"],"probabilities":[0.4,0.6]}"#,
        ]
        .join("\n")
    }

    fn gold_document() -> String {
        [
            r#"{"id":"a","gold":"yes","family":"one"}"#,
            r#"{"id":"b","gold":"no","family":"two"}"#,
        ]
        .join("\n")
    }

    #[test]
    fn evaluate_stratifies_by_family_and_hashes_its_inputs() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();

        assert_eq!(report.schema, REPORT_SCHEMA);
        assert_eq!(report.bins, 10);
        assert_eq!(
            report.predictions_sha256,
            sha256_hex(&predictions_document())
        );
        assert_eq!(report.gold_sha256, sha256_hex(&gold_document()));
        assert_eq!(report.overall.n_items, 2);
        assert_eq!(report.overall.accuracy, Some(1.0));
        assert_eq!(report.by_family.len(), 2);
        assert_eq!(report.by_family["one"].n_items, 1);
        assert_eq!(report.by_family["two"].n_items, 1);
    }

    #[test]
    fn evaluate_rejects_zero_bins() {
        assert!(evaluate(&predictions_document(), &gold_document(), 0).is_err());
    }

    #[test]
    fn evaluate_accepts_blank_lines_and_ignores_extra_fields() {
        let predictions = format!(
            "\n{}\n\n{{\"id\":\"b\",\"option_ids\":[\"yes\",\"no\"],\"probabilities\":[0.4,0.6],\
             \"option_logprobs\":[-0.9,-0.5],\"prompt_sha256\":\"deadbeef\",\"probe\":\"x\"}}\n",
            r#"{"id":"a","option_ids":["yes","no"],"probabilities":[0.9,0.1]}"#
        );
        let report = evaluate(&predictions, &gold_document(), 10).unwrap();
        assert_eq!(report.overall.n_items, 2);
    }

    #[test]
    fn verify_accepts_an_identical_recomputation() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();
        let outcome = verify(&report, &report, 1e-12);
        assert!(outcome.is_ok(), "{}", outcome.report());
        assert!(
            outcome.checks > 0,
            "a passing verify must still count checks"
        );
    }

    #[test]
    fn verify_detects_a_tampered_number() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();
        let mut tampered = report.clone();
        tampered.overall.accuracy = Some(0.25);
        let outcome = verify(&tampered, &report, 1e-12);
        assert!(!outcome.is_ok());
        assert_eq!(outcome.diffs.len(), 1);
        assert_eq!(outcome.diffs[0].path, "report.overall.accuracy");
        assert!(outcome.report().contains("mismatch"));
    }

    #[test]
    fn verify_detects_a_missing_stratum_and_a_tampered_hash() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();

        // A dropped stratum must not pass just because the shared fields match.
        let mut dropped = report.clone();
        dropped.by_family.remove("two");
        assert!(!verify(&dropped, &report, 1e-12).is_ok());

        // A rewritten input hash is a claim about the evidence, and the
        // evidence is what verify recomputes.
        let mut rewritten = report.clone();
        rewritten.gold_sha256 = "0".repeat(64);
        let outcome = verify(&rewritten, &report, 1e-12);
        assert!(!outcome.is_ok());
        assert_eq!(outcome.diffs[0].path, "report.gold_sha256");
    }

    #[test]
    fn verify_tolerance_belongs_to_the_caller() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();
        let mut slightly_off = report.clone();
        // Nudge one number beyond 1e-12 but inside 1e-3.
        slightly_off.overall.nll = slightly_off.overall.nll.map(|value| value + 1e-6);
        assert!(!verify(&slightly_off, &report, 1e-12).is_ok());
        assert!(verify(&slightly_off, &report, 1e-3).is_ok());
    }

    #[test]
    fn reports_round_trip_through_json() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();
        let text = serde_json::to_string_pretty(&report).unwrap();
        let parsed: Report = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, report);
        assert!(verify(&parsed, &report, 1e-12).is_ok());
    }

    #[test]
    fn a_report_with_an_unknown_field_is_rejected() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();
        let mut value = serde_json::to_value(&report).unwrap();
        value["overall"]["total_accuracy"] = serde_json::json!(0.99);
        assert!(
            serde_json::from_value::<Report>(value).is_err(),
            "a fabricated metric must fail to load rather than ride along"
        );
    }

    #[test]
    fn the_table_names_the_overall_stratum_and_every_family() {
        let report = evaluate(&predictions_document(), &gold_document(), 10).unwrap();
        let table = render_table(&report);
        assert!(table.contains("overall"), "{table}");
        assert!(table.contains("one"), "{table}");
        assert!(table.contains("two"), "{table}");
    }
}
