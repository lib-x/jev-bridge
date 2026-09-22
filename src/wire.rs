//! System One wire format: request validation and response construction.
//!
//! This mirrors the contract implemented by fastjev's `compat/wire.py`
//! (<https://github.com/chengyongru/fastjev>, MIT). The adapter preserves the
//! public wire shape; it does not claim Jev's model behaviour, and the
//! returned probabilities are conditional option scores, not calibrated
//! decision confidence.

use serde::Serialize;
use serde_json::{Map, Value};
use std::io;

/// The largest number of options a `choice` question may offer.
///
/// Sixteen matches the sixteen uppercase letters the readout can address.
pub const MAX_OPTIONS: usize = 16;

/// The largest number of levels a `score` question may offer.
pub const MAX_SCORE_LEVELS: usize = 10;

/// How the bridge derives its certainty proxy.
///
/// This is `1 - normalized entropy`, documented as such because TypeSafe does
/// not publish the statistic behind its own `confidence` field.
pub const CONFIDENCE_METHOD: &str = "one-minus-normalized-entropy";

/// The disclaimer attached to every returned probability.
pub const PROBABILITY_STATUS: &str = "conditional option scores; uncalibrated as decision confidence";

/// A request that cannot be represented by the scoring backend.
#[derive(Debug, Clone)]
pub struct WireError(pub String);

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for WireError {}

type Result<T> = std::result::Result<T, WireError>;

fn fail<T>(path: &str, message: &str) -> Result<T> {
    Err(WireError(format!("{path}: {message}")))
}

/// Serialize with Python's `json.dumps(..., ensure_ascii=False)` layout.
///
/// fastjev builds its prompt payload with Python's default separators
/// (`", "` and `": "`), so matching that layout byte for byte is what keeps a
/// bridged prompt identical to the reference implementation's.
pub fn to_python_json(value: &Value) -> String {
    let mut buffer = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, PythonFormatter);
    value
        .serialize(&mut serializer)
        .expect("serializing a serde_json::Value cannot fail");
    String::from_utf8(buffer).expect("serde_json emits UTF-8")
}

struct PythonFormatter;

impl serde_json::ser::Formatter for PythonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

/// The three question shapes the System One API defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Pick one of several described options.
    Choice,
    /// Answer a yes/no proposition; maps to the two options `true` and `false`.
    Noul,
    /// Pick one of several ordered levels; the answer is a weighted value.
    Score,
}

impl Kind {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "choice" => Some(Self::Choice),
            "noul" => Some(Self::Noul),
            "score" => Some(Self::Score),
            _ => None,
        }
    }
}

/// What a validated question asks for, kept for building the response.
#[derive(Debug, Clone)]
pub struct QuestionSpec {
    /// The caller's question id, echoed back as the answer key.
    pub id: String,
    /// Which of the three shapes this question is.
    pub kind: Kind,
    /// Option ids in the order they were declared, which fixes the answer order.
    pub option_ids: Vec<String>,
    /// Level texts for a `score` question, empty otherwise.
    pub legend: Vec<String>,
}

/// One answer option handed to the scorer.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RowOption {
    /// Stable option id returned as the winning value.
    pub id: String,
    /// The text the model actually reads, after any id prefix was added.
    pub description: String,
}

/// One decision, in the shape the scorer and the `--score` fixture both use.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Row {
    /// Row id, echoed into the prediction.
    pub id: String,
    /// Arbitrary JSON evidence: a string, an object or an array.
    pub state: Value,
    /// The criterion to apply to the state.
    pub question: String,
    /// Between 2 and [`MAX_OPTIONS`] options.
    pub options: Vec<RowOption>,
}

/// Validate a System One request and convert its questions to scorer rows.
pub fn request_rows(payload: &Value, served_model: &str) -> Result<(Vec<QuestionSpec>, Vec<Row>)> {
    let body = match payload {
        Value::Object(map) => map,
        _ => return fail("body", "must be a JSON object"),
    };
    let state = body.get("state").cloned().unwrap_or(Value::Null);
    validate_state(&state)?;

    match body.get("model") {
        Some(Value::String(model)) if model == served_model => {}
        _ => {
            return fail(
                "model",
                &format!("must be {served_model:?}; this server does not serve Jev aliases"),
            )
        }
    }

    let questions = match body.get("questions") {
        Some(Value::Object(map)) if !map.is_empty() => map,
        _ => return fail("questions", "must be a nonempty object"),
    };

    let mut specs = Vec::with_capacity(questions.len());
    let mut rows = Vec::with_capacity(questions.len());
    for (question_id, question) in questions {
        if question_id.is_empty() {
            return fail("questions", "question IDs must be nonempty strings");
        }
        let base = format!("questions.{question_id}");
        let question = match question {
            Value::Object(map) => map,
            _ => return fail(&base, "must be an object"),
        };
        let kind_name = match question.get("type") {
            Some(Value::String(name)) => name.as_str(),
            _ => return fail(&format!("{base}.type"), "must be one of: choice, noul, score"),
        };
        let kind = match Kind::parse(kind_name) {
            Some(kind) => kind,
            None => return fail(&format!("{base}.type"), "must be one of: choice, noul, score"),
        };
        let instructions = instructions(question.get("instructions"), kind, &format!("{base}.instructions"))?;

        let (option_ids, options, legend) = match kind {
            Kind::Choice => choice_options(question, &base)?,
            Kind::Noul => noul_options(question, &base)?,
            Kind::Score => score_options(question, &base)?,
        };

        specs.push(QuestionSpec {
            id: question_id.clone(),
            kind,
            option_ids,
            legend,
        });
        rows.push(Row {
            id: question_id.clone(),
            state: state.clone(),
            question: instructions,
            options,
        });
    }
    Ok((specs, rows))
}

fn validate_state(state: &Value) -> Result<()> {
    match state {
        Value::String(text) if text.is_empty() => fail("state", "must be a nonempty string, object, or array"),
        Value::String(_) => Ok(()),
        Value::Object(map) if map.is_empty() => {
            fail("state", "must be a nonempty string, object, or array")
        }
        Value::Object(_) => Ok(()),
        Value::Array(items) if items.is_empty() => {
            fail("state", "must be a nonempty string, object, or array")
        }
        Value::Array(_) => Ok(()),
        _ => fail("state", "must be a nonempty string, object, or array"),
    }
}

fn json_text(value: &Value, path: &str) -> Result<String> {
    match value {
        Value::String(text) if text.is_empty() => fail(path, "must not be empty"),
        Value::String(text) => Ok(text.clone()),
        Value::Object(_) | Value::Array(_) => Ok(to_python_json(value)),
        _ => fail(path, "must be a string, object, or array"),
    }
}

fn description(option_id: &str, value: Option<&Value>, path: &str) -> Result<String> {
    match value {
        None | Some(Value::Null) => Ok(option_id.to_string()),
        Some(value) => Ok(format!("{option_id}: {}", json_text(value, path)?)),
    }
}

fn instructions(value: Option<&Value>, kind: Kind, path: &str) -> Result<String> {
    match value {
        Some(Value::Null) | None => Ok(match kind {
            Kind::Choice => "Which option best matches the supplied state?".to_string(),
            Kind::Noul => "Does the true outcome apply to the supplied state?".to_string(),
            Kind::Score => "Which ordered level best matches the supplied state?".to_string(),
        }),
        Some(value) => json_text(value, path),
    }
}

type Options = (Vec<String>, Vec<RowOption>, Vec<String>);

fn choice_options(question: &Map<String, Value>, base: &str) -> Result<Options> {
    let criteria = match question.get("criteria") {
        Some(Value::Object(map)) if (2..=MAX_OPTIONS).contains(&map.len()) => map,
        _ => {
            return fail(
                &format!("{base}.criteria"),
                &format!("must contain 2-{MAX_OPTIONS} options"),
            )
        }
    };
    let mut option_ids = Vec::with_capacity(criteria.len());
    let mut options = Vec::with_capacity(criteria.len());
    for (option_id, value) in criteria {
        if option_id.is_empty() {
            return fail(&format!("{base}.criteria"), "option IDs must be nonempty strings");
        }
        if !matches!(value, Value::Null | Value::String(_) | Value::Object(_) | Value::Array(_)) {
            return fail(
                &format!("{base}.criteria.{option_id}"),
                "must be a string, object, array, or null",
            );
        }
        option_ids.push(option_id.clone());
        options.push(RowOption {
            id: option_id.clone(),
            description: description(option_id, Some(value), &format!("{base}.criteria.{option_id}"))?,
        });
    }
    Ok((option_ids, options, Vec::new()))
}

fn noul_options(question: &Map<String, Value>, base: &str) -> Result<Options> {
    let criteria = match question.get("criteria") {
        None | Some(Value::Null) => &Map::new(),
        Some(Value::Object(map)) => map,
        _ => return fail(&format!("{base}.criteria"), "must be an object when supplied"),
    };
    let unknown: Vec<&String> = criteria
        .keys()
        .filter(|key| key.as_str() != "true" && key.as_str() != "false")
        .collect();
    if !unknown.is_empty() {
        let mut names: Vec<&str> = unknown.iter().map(|key| key.as_str()).collect();
        names.sort_unstable();
        return fail(
            &format!("{base}.criteria"),
            &format!("contains unsupported keys: {:?}", names),
        );
    }
    let option_ids = vec!["true".to_string(), "false".to_string()];
    let defaults = ["The answer is yes.", "The answer is no."];
    let mut options = Vec::with_capacity(2);
    for (index, option_id) in option_ids.iter().enumerate() {
        let path = format!("{base}.criteria.{option_id}");
        // A missing key takes the default text; a key that is present but null
        // renders as the bare option id. Both are then prefixed with the id.
        let fallback;
        let value = match criteria.get(option_id) {
            None => {
                fallback = Value::String(defaults[index].to_string());
                Some(&fallback)
            }
            Some(Value::Null) => None,
            Some(value) => Some(value),
        };
        options.push(RowOption {
            id: option_id.clone(),
            description: description(option_id, value, &path)?,
        });
    }
    Ok((option_ids, options, Vec::new()))
}

fn score_options(question: &Map<String, Value>, base: &str) -> Result<Options> {
    let criteria = match question.get("criteria") {
        Some(Value::Array(items)) if (2..=MAX_SCORE_LEVELS).contains(&items.len()) => items,
        _ => {
            return fail(
                &format!("{base}.criteria"),
                &format!("must contain 2-{MAX_SCORE_LEVELS} ordered levels"),
            )
        }
    };
    let mut option_ids = Vec::with_capacity(criteria.len());
    let mut options = Vec::with_capacity(criteria.len());
    let mut legend = Vec::with_capacity(criteria.len());
    for (index, value) in criteria.iter().enumerate() {
        let path = format!("{base}.criteria.{index}");
        let text = json_text(value, &path)?;
        let option_id = index.to_string();
        legend.push(text.clone());
        option_ids.push(option_id.clone());
        options.push(RowOption {
            id: option_id.clone(),
            description: format!("Level {option_id}: {text}"),
        });
    }
    Ok((option_ids, options, legend))
}

/// A scorer result reduced to what the wire layer needs.
#[derive(Debug, Clone)]
pub struct ScoredAnswer {
    /// Row id, copied from the request.
    pub id: String,
    /// Option ids in the order the probabilities follow.
    pub option_ids: Vec<String>,
    /// One probability per option id, summing to one.
    pub probabilities: Vec<f64>,
    /// Prompt length the runtime reported.
    pub input_tokens: u64,
    /// Prompt contract version, when the backend recorded one.
    pub prompt_version: Option<String>,
}

/// Documented fastjev certainty proxy, not TypeSafe's private statistic.
pub fn distribution_confidence(probabilities: &[f64]) -> f64 {
    let entropy: f64 = probabilities
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

fn normalized_probabilities(spec: &QuestionSpec, result: &ScoredAnswer) -> Result<Vec<f64>> {
    if result.id != spec.id || result.option_ids != spec.option_ids {
        return Err(WireError(format!(
            "Scorer result for {:?} does not match its declared options",
            spec.id
        )));
    }
    if result.probabilities.len() != spec.option_ids.len() {
        return Err(WireError(format!(
            "Scorer result for {:?} has an invalid probability vector",
            spec.id
        )));
    }
    if result
        .probabilities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(WireError(format!(
            "Scorer result for {:?} has non-finite or negative probabilities",
            spec.id
        )));
    }
    let total: f64 = result.probabilities.iter().sum();
    if total <= 0.0 {
        return Err(WireError(format!(
            "Scorer result for {:?} has zero probability mass",
            spec.id
        )));
    }
    Ok(result.probabilities.iter().map(|value| value / total).collect())
}

/// An ordered map that serialises as a JSON object.
///
/// `serde_json::Map` only holds `Value`, and option order matters for
/// readability because it follows the caller's own criteria order, so the pairs
/// stay in a vector and are written out as an object.
#[derive(Debug)]
pub struct OrderedMap<V>(Vec<(String, V)>);

impl<V> OrderedMap<V> {
    /// An empty map.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Append a key, keeping insertion order.
    pub fn insert(&mut self, key: String, value: V) {
        self.0.push((key, value));
    }

    /// Whether no entry has been inserted.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// How many entries were inserted.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<V> Default for OrderedMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Serialize> Serialize for OrderedMap<V> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// One answer in a System One response.
///
/// The `type` tag is what makes the three shapes distinguishable: a `choice`
/// carries a winner and a distribution, a `noul` carries the probability of
/// `true`, and a `score` carries a probability-weighted value plus the legend
/// that maps its levels back to text.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Answer a yes/no question.
    Noul {
        /// Probability assigned to `true`.
        noul: f64,
    },
    /// Pick one of several described options.
    Choice {
        /// The winning option id.
        choice: String,
        /// Full distribution over the declared options.
        probabilities: OrderedMap<f64>,
        /// Certainty proxy; see [`CONFIDENCE_METHOD`].
        confidence: f64,
    },
    /// Pick one of several ordered levels.
    Score {
        /// Probability-weighted level value.
        score: f64,
        /// Level index to level text, in declared order.
        legend: OrderedMap<String>,
        /// Full distribution over the declared levels.
        probabilities: OrderedMap<f64>,
        /// Certainty proxy; see [`CONFIDENCE_METHOD`].
        confidence: f64,
    },
}

#[derive(Debug, Serialize)]
/// Token accounting for one request.
pub struct Usage {
    /// Sum of the per-question prompt lengths.
    pub input_tokens: u64,
    /// Always zero: the bridge reads log probabilities instead of generating.
    pub output_tokens: u64,
}

/// The bridge's own extension block, mirroring fastjev's System One adapter.
#[derive(Debug, Serialize)]
pub struct FastjevMeta {
    /// The uncalibrated-probability disclaimer.
    pub probability_status: &'static str,
    /// How `confidence` was derived.
    pub confidence_method: &'static str,
    /// Distinct prompt versions that produced this response.
    pub prompt_versions: Vec<String>,
}

/// The documented System One response shape.
#[derive(Debug, Serialize)]
pub struct SystemOneResponse {
    /// The served model id, echoed from the request.
    pub model: String,
    /// One answer per question id, in request order.
    pub answers: OrderedMap<Answer>,
    /// Token accounting.
    pub usage: Usage,
    /// Bridge-specific metadata.
    pub fastjev: FastjevMeta,
}

/// Convert scorer results to the documented System One response shape.
pub fn response_from_results(
    served_model: &str,
    specs: &[QuestionSpec],
    results: &[ScoredAnswer],
) -> Result<SystemOneResponse> {
    if results.len() != specs.len() {
        return Err(WireError(
            "Scorer returned a different number of results than requested".to_string(),
        ));
    }
    let mut answers = OrderedMap::new();
    let mut input_tokens: u64 = 0;
    let mut prompt_versions: Vec<String> = Vec::new();
    for (spec, result) in specs.iter().zip(results) {
        let probabilities = normalized_probabilities(spec, result)?;
        let winner_index = probabilities
            .iter()
            .enumerate()
            .fold(0usize, |best, (index, value)| {
                if *value > probabilities[best] {
                    index
                } else {
                    best
                }
            });
        let mut distribution = OrderedMap::new();
        for (option_id, value) in spec.option_ids.iter().zip(&probabilities) {
            distribution.insert(option_id.clone(), *value);
        }
        let answer = match spec.kind {
            Kind::Noul => {
                let true_index = spec
                    .option_ids
                    .iter()
                    .position(|option_id| option_id == "true")
                    .ok_or_else(|| {
                        WireError(format!("Scorer result for {:?} lost its true option", spec.id))
                    })?;
                Answer::Noul {
                    noul: probabilities[true_index],
                }
            }
            Kind::Choice => Answer::Choice {
                choice: spec.option_ids[winner_index].clone(),
                probabilities: distribution,
                confidence: distribution_confidence(&probabilities),
            },
            Kind::Score => {
                let score: f64 = probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, value)| index as f64 * value)
                    .sum();
                let mut legend: OrderedMap<String> = OrderedMap::new();
                for (index, text) in spec.legend.iter().enumerate() {
                    legend.insert(index.to_string(), text.clone());
                }
                Answer::Score {
                    score,
                    legend,
                    probabilities: distribution,
                    confidence: distribution_confidence(&probabilities),
                }
            }
        };
        answers.insert(spec.id.clone(), answer);
        input_tokens += result.input_tokens;
        if let Some(version) = &result.prompt_version
            && !prompt_versions.contains(version)
        {
            prompt_versions.push(version.clone());
        }
    }
    prompt_versions.sort();
    Ok(SystemOneResponse {
        model: served_model.to_string(),
        answers,
        usage: Usage {
            input_tokens,
            output_tokens: 0,
        },
        fastjev: FastjevMeta {
            probability_status: PROBABILITY_STATUS,
            confidence_method: CONFIDENCE_METHOD,
            prompt_versions,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_payload() -> Value {
        json!({
            "model": "bridge-model",
            "state": "Help! My payouts have been failing for three days.",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, and refunds", "technical": null},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the problem?",
                    "criteria": ["Minor", "Degraded", "Blocking"],
                },
            },
        })
    }

    #[test]
    fn python_json_layout_matches_cpython_output() {
        // The expected string is the exact stdout of CPython:
        //   json.dumps(payload, ensure_ascii=False)
        // for the same object, including nesting, booleans, null and non-ASCII.
        let value = json!({
            "evidence": {"message": "已扣款两次 charged twice", "count": 2, "tags": ["a", "b"],
                         "ok": true, "missing": null, "ratio": 0.5},
            "criterion": "Which queue?",
            "options": [{"letter": "A", "description": "Billing, payments, and refunds."}],
        });
        assert_eq!(
            to_python_json(&value),
            r#"{"evidence": {"message": "已扣款两次 charged twice", "count": 2, "tags": ["a", "b"], "ok": true, "missing": null, "ratio": 0.5}, "criterion": "Which queue?", "options": [{"letter": "A", "description": "Billing, payments, and refunds."}]}"#
        );
    }

    #[test]
    fn python_json_layout_matches_python_separators() {
        let value = json!({"a": [1, 2], "b": {"c": "x"}, "d": "中"});
        assert_eq!(to_python_json(&value), r#"{"a": [1, 2], "b": {"c": "x"}, "d": "中"}"#);
    }

    #[test]
    fn rows_render_descriptions_like_fastjev() {
        let (specs, rows) = request_rows(&sample_payload(), "bridge-model").unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(rows[1].options[0].description, "billing: Payments, invoicing, and refunds");
        assert_eq!(rows[1].options[1].description, "technical");
        assert_eq!(rows[2].options[2].description, "Level 2: Blocking");
        assert_eq!(rows[0].options[0].description, "true: The answer is yes.");
        assert_eq!(specs[2].legend, vec!["Minor", "Degraded", "Blocking"]);
        assert_eq!(rows[1].question, "Which team should handle this?");
    }

    #[test]
    fn noul_criteria_defaults_and_nulls_differ() {
        let payload = json!({
            "model": "m",
            "state": "s",
            "questions": {
                "defaults": {"type": "noul"},
                "explicit_null": {"type": "noul", "criteria": {"true": null, "false": "No way."}},
            },
        });
        let (_specs, rows) = request_rows(&payload, "m").unwrap();
        assert_eq!(rows[0].options[0].description, "true: The answer is yes.");
        assert_eq!(rows[0].options[1].description, "false: The answer is no.");
        assert_eq!(rows[1].options[0].description, "true");
        assert_eq!(rows[1].options[1].description, "false: No way.");
    }

    #[test]
    fn model_mismatch_is_rejected() {
        let error = request_rows(&sample_payload(), "other-model").unwrap_err();
        assert!(error.0.starts_with("model: "), "{}", error.0);
    }

    #[test]
    fn empty_state_is_rejected() {
        let payload = json!({"model": "m", "state": "", "questions": {"q": {"type": "noul"}}});
        let error = request_rows(&payload, "m").unwrap_err();
        assert!(error.0.starts_with("state: "), "{}", error.0);
    }

    #[test]
    fn response_uses_probabilities_and_confidence() {
        let (specs, _rows) = request_rows(&sample_payload(), "bridge-model").unwrap();
        let results = vec![
            ScoredAnswer {
                id: "is_urgent".into(),
                option_ids: vec!["true".into(), "false".into()],
                probabilities: vec![0.9, 0.1],
                input_tokens: 10,
                prompt_version: Some("direct-options-v1".into()),
            },
            ScoredAnswer {
                id: "department".into(),
                option_ids: vec!["billing".into(), "technical".into()],
                probabilities: vec![0.25, 0.75],
                input_tokens: 20,
                prompt_version: Some("direct-options-v1".into()),
            },
            ScoredAnswer {
                id: "severity".into(),
                option_ids: vec!["0".into(), "1".into(), "2".into()],
                probabilities: vec![0.0, 1.0, 0.0],
                input_tokens: 30,
                prompt_version: Some("direct-options-v1".into()),
            },
        ];
        let response = serde_json::to_value(
            response_from_results("bridge-model", &specs, &results).unwrap(),
        )
        .unwrap();
        assert_eq!(response["answers"]["is_urgent"]["noul"], 0.9);
        assert_eq!(response["answers"]["department"]["choice"], "technical");
        assert_eq!(response["answers"]["severity"]["score"], 1.0);
        assert_eq!(response["usage"]["input_tokens"], 60);
        assert_eq!(response["usage"]["output_tokens"], 0);
        assert_eq!(response["answers"]["severity"]["legend"]["2"], "Blocking");
        let confidence = response["answers"]["severity"]["confidence"].as_f64().unwrap();
        assert!((confidence - 1.0).abs() < 1e-12);
    }

    #[test]
    fn flat_distribution_has_zero_confidence() {
        assert!(distribution_confidence(&[0.25, 0.25, 0.25, 0.25]).abs() < 1e-12);
    }

    #[test]
    fn mismatched_result_is_rejected() {
        let (specs, _rows) = request_rows(&sample_payload(), "bridge-model").unwrap();
        let results = vec![ScoredAnswer {
            id: "is_urgent".into(),
            option_ids: vec!["false".into(), "true".into()],
            probabilities: vec![0.5, 0.5],
            input_tokens: 1,
            prompt_version: None,
        }];
        assert!(response_from_results("bridge-model", &specs, &results).is_err());
    }
}
