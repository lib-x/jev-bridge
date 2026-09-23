//! End-to-end tests for the offline `--evaluate` path.
//!
//! These drive the built binary the way an operator would — two files in, a
//! report out, `--verify` reproducing it — and pin the failure modes that must
//! stay loud: a tampered report, inputs that are out of sync, and a report
//! that does not belong to the inputs it is checked against.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh scratch directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("jev-bridge-evaluate-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory must be creatable");
    dir
}

fn write(path: &Path, text: &str) {
    std::fs::write(path, text).expect("writing a fixture must succeed");
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jev-bridge"))
        .args(args)
        .output()
        .expect("the binary must be runnable")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Two scored rows, one per family, both correct.
const PREDICTIONS: &str = concat!(
    r#"{"id":"a","option_ids":["yes","no"],"probabilities":[0.9,0.1],"label":null}"#,
    "\n",
    r#"{"id":"b","option_ids":["yes","no"],"probabilities":[0.4,0.6]}"#,
    "\n",
);

const GOLD: &str = concat!(
    r#"{"id":"a","gold":"yes","family":"one"}"#,
    "\n",
    r#"{"id":"b","gold":"no","family":"two"}"#,
    "\n",
);

#[test]
fn evaluate_writes_a_report_and_verify_reproduces_it() {
    let dir = scratch("round-trip");
    let predictions = dir.join("predictions.jsonl");
    let gold = dir.join("gold.jsonl");
    let report = dir.join("report.json");
    write(&predictions, PREDICTIONS);
    write(&gold, GOLD);

    // No --base-url and no --model: evaluation is fully offline, and the CLI
    // must not require them.
    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
        "--out",
        report.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let table = stdout(&output);
    assert!(table.contains("overall"), "{table}");
    assert!(table.contains("one"), "{table}");

    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(written["schema"], "jev-bridge-evaluation-v1");
    assert_eq!(written["overall"]["accuracy"], 1.0);
    assert_eq!(written["overall"]["n_scored"], 2);
    assert_eq!(
        written["predictions_sha256"].as_str().unwrap().len(),
        64,
        "the report must carry a hash of its evidence"
    );
    assert_eq!(written["by_family"]["one"]["n_items"], 1);
    assert_eq!(written["by_family"]["two"]["n_items"], 1);

    // The same inputs reproduce the written report exactly.
    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
        "--verify",
        report.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "stderr: {}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("0 mismatches"), "{text}");
    assert!(text.contains("checks"), "{text}");
}

#[test]
fn a_tampered_report_fails_verification() {
    let dir = scratch("tampered");
    let predictions = dir.join("predictions.jsonl");
    let gold = dir.join("gold.jsonl");
    let report = dir.join("report.json");
    write(&predictions, PREDICTIONS);
    write(&gold, GOLD);

    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
        "--out",
        report.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "stderr: {}", stderr(&output));

    // Rewrite the headline number; verification must refuse it.
    let text = std::fs::read_to_string(&report).unwrap();
    let tampered = text.replacen("\"accuracy\": 1.0", "\"accuracy\": 0.25", 1);
    assert_ne!(
        text, tampered,
        "the fixture must actually contain the number"
    );
    write(&report, &tampered);

    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
        "--verify",
        report.to_str().unwrap(),
    ]);
    assert!(
        !output.status.success(),
        "a tampered report must not verify"
    );
    let text = format!("{}{}", stdout(&output), stderr(&output));
    assert!(text.contains("overall.accuracy"), "{text}");
    assert!(text.contains("mismatch"), "{text}");
}

#[test]
fn a_report_from_other_inputs_fails_verification() {
    let dir = scratch("other-inputs");
    let predictions = dir.join("predictions.jsonl");
    let gold = dir.join("gold.jsonl");
    let report = dir.join("report.json");
    write(&predictions, PREDICTIONS);
    write(&gold, GOLD);

    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
        "--out",
        report.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "stderr: {}", stderr(&output));

    // Change one input file: the report no longer describes the evidence, and
    // the input hashes must catch that even when every metric still matches.
    write(
        &predictions,
        &format!(
            "{PREDICTIONS}{}",
            r#"{"id":"c","option_ids":["yes","no"],"probabilities":[0.5,0.5]}"#
        ),
    );
    write(
        &gold,
        &format!("{GOLD}{}", r#"{"id":"c","gold":"yes","family":"one"}"#),
    );

    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
        "--verify",
        report.to_str().unwrap(),
    ]);
    assert!(!output.status.success(), "changed inputs must not verify");
    let text = format!("{}{}", stdout(&output), stderr(&output));
    assert!(text.contains("predictions_sha256"), "{text}");
}

#[test]
fn out_of_sync_inputs_fail_loudly() {
    let dir = scratch("out-of-sync");
    let predictions = dir.join("predictions.jsonl");
    let gold = dir.join("gold.jsonl");
    write(&predictions, PREDICTIONS);
    // The gold file mentions a row the predictions file never scored.
    write(
        &gold,
        &format!("{GOLD}{}", r#"{"id":"missing","gold":"yes"}"#),
    );

    // A gold row without a prediction is a *missing* row, not an error: it is
    // counted and reported as such.
    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(stdout(&output).contains("1 missing"), "{}", stdout(&output));

    // A prediction row without gold is an error: the files are out of sync.
    write(&gold, GOLD);
    write(
        &predictions,
        &format!(
            "{PREDICTIONS}{}",
            r#"{"id":"extra","option_ids":["yes","no"],"probabilities":[0.5,0.5]}"#
        ),
    );
    let output = run(&[
        "--evaluate",
        "--predictions",
        predictions.to_str().unwrap(),
        "--gold",
        gold.to_str().unwrap(),
    ]);
    assert!(
        !output.status.success(),
        "an unmatched prediction must fail"
    );
    assert!(
        stderr(&output).contains("no gold counterpart"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn evaluate_requires_its_inputs() {
    // --evaluate without --predictions/--gold is a usage error, not a panic.
    let output = run(&["--evaluate"]);
    assert!(!output.status.success());
    let text = stderr(&output);
    assert!(text.contains("--predictions"), "{text}");
}
