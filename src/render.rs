//! Local chat-template rendering.
//!
//! The serving runtime's own template is the preferred source (`/apply-template`
//! on llama.cpp), but runtimes such as vLLM expose no such endpoint. This module
//! renders the same Jinja template locally with minijinja.
//!
//! llama.cpp renders with the C++ minijinja plus Python-style string methods,
//! which the Rust crate does not implement. Those methods are supplied through
//! [`Environment::set_unknown_method_callback`] so a template written for
//! transformers still renders here. `python_string_methods` mirrors CPython
//! semantics, including treating the argument of `strip`/`lstrip`/`rstrip` as a
//! set of characters rather than a prefix.

use anyhow::{Context, Result};
use minijinja::{Environment, Error, ErrorKind, State, Value as JinjaValue};
use serde_json::{Map, Value};

/// `strip` / `lstrip` / `rstrip` with CPython's character-set semantics.
fn strip_chars(text: &str, chars: Option<&str>, left: bool, right: bool) -> String {
    let mut result = text;
    if left {
        result = result.trim_start_matches(|character: char| match chars {
            Some(set) => set.contains(character),
            None => character.is_whitespace(),
        });
    }
    if right {
        result = result.trim_end_matches(|character: char| match chars {
            Some(set) => set.contains(character),
            None => character.is_whitespace(),
        });
    }
    result.to_string()
}

fn string_argument(method: &str, args: &[JinjaValue], index: usize) -> Result<String, Error> {
    args.get(index)
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("{method}() expects a string argument at position {index}"),
            )
        })
}

/// Methods a transformers chat template may call that the Rust minijinja lacks.
fn python_string_methods(
    _state: &State,
    value: &JinjaValue,
    method: &str,
    args: &[JinjaValue],
) -> Result<JinjaValue, Error> {
    if let Some(text) = value.as_str() {
        return match method {
            "startswith" => Ok(JinjaValue::from(text.starts_with(&string_argument(method, args, 0)?))),
            "endswith" => Ok(JinjaValue::from(text.ends_with(&string_argument(method, args, 0)?))),
            "replace" => Ok(JinjaValue::from(text.replace(
                &string_argument(method, args, 0)?,
                &string_argument(method, args, 1)?,
            ))),
            "split" => {
                let separator = string_argument(method, args, 0)?;
                let parts: Vec<String> = text.split(&separator).map(str::to_string).collect();
                Ok(JinjaValue::from(parts))
            }
            "strip" | "lstrip" | "rstrip" => {
                let chars = if args.is_empty() {
                    None
                } else {
                    Some(string_argument(method, args, 0)?)
                };
                Ok(JinjaValue::from(strip_chars(
                    text,
                    chars.as_deref(),
                    method != "rstrip",
                    method != "lstrip",
                )))
            }
            _ => Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("string has no method named {method}"),
            )),
        };
    }

    if method == "items" && value.as_object().is_some() {
        let mut items = Vec::new();
        for key in value.try_iter()? {
            let entry = value.get_item(&key)?;
            items.push(JinjaValue::from(vec![key, entry]));
        }
        return Ok(JinjaValue::from(items));
    }

    Err(Error::new(
        ErrorKind::UnknownMethod,
        format!("value has no method named {method}"),
    ))
}

/// A chat template rendered in-process.
pub struct LocalRenderer {
    environment: Environment<'static>,
    context: Map<String, Value>,
}

impl LocalRenderer {
    /// Compile `template`, adding `context` to every render.
    ///
    /// The context carries the variables transformers supplies besides the
    /// messages, such as `bos_token` and `eos_token`; a template that starts
    /// with `{{- bos_token }}` needs them.
    pub fn new(template: String, context: Map<String, Value>) -> Result<Self> {
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(python_string_methods);
        environment
            .add_template_owned("chat", template)
            .context("the chat template failed to compile")?;
        Ok(Self {
            environment,
            context,
        })
    }

    /// Render `messages`, with `chat_template_kwargs` taking precedence over the
    /// configured context.
    ///
    /// `messages` is generic over `Serialize` so this module stays independent
    /// of the prompt layer: any chat-shaped value renders.
    pub fn render<M: serde::Serialize + ?Sized>(
        &self,
        messages: &M,
        chat_template_kwargs: &Value,
    ) -> Result<String> {
        let template = self
            .environment
            .get_template("chat")
            .context("the chat template disappeared from the environment")?;
        let mut context = self.context.clone();
        context.insert(
            "messages".to_string(),
            serde_json::to_value(messages).context("messages are not serialisable")?,
        );
        context.insert("add_generation_prompt".to_string(), Value::Bool(true));
        if let Some(extra) = chat_template_kwargs.as_object() {
            for (key, value) in extra {
                context.insert(key.clone(), value.clone());
            }
        }
        template
            .render(Value::Object(context))
            .context("the chat template failed to render")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn render(template: &str, context: Value, kwargs: Value) -> String {
        let renderer = LocalRenderer::new(template.to_string(), context.as_object().unwrap().clone())
            .unwrap();
        renderer
            .render(&json!([{"role": "user", "content": "hi"}]), &kwargs)
            .unwrap()
    }

    #[test]
    fn renders_messages_and_generation_prompt() {
        let template = "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";
        let rendered = render(template, json!({}), json!({}));
        assert_eq!(rendered, "user:hi\nassistant:");
    }

    #[test]
    fn context_variables_are_available() {
        let rendered = render("{{- bos_token }}x", json!({"bos_token": "<s>"}), json!({}));
        assert_eq!(rendered, "<s>x");
    }

    #[test]
    fn kwargs_override_context() {
        let rendered = render(
            "{{ bos_token }}|{{ enable_thinking }}",
            json!({"bos_token": "<s>", "enable_thinking": true}),
            json!({"enable_thinking": false}),
        );
        assert_eq!(rendered, "<s>|False");
    }

    #[test]
    fn python_string_methods_match_cpython() {
        // startswith / endswith, as used to detect tool responses.
        let rendered = render(
            "{{ 'abc'.startswith('ab') }}|{{ 'abc'.endswith('bc') }}",
            json!({}),
            json!({}),
        );
        assert_eq!(rendered, "True|True");

        // split returns an indexable list.
        let rendered = render("{{ 'a</think>b'.split('</think>')[-1] }}", json!({}), json!({}));
        assert_eq!(rendered, "b");

        // strip with a character set, not a prefix: '\n\nx\n\n'.strip('\n') == 'x'
        let rendered = render(
            "{{ '\n\nx\n\n'.strip('\n') }}|{{ '  y  '.lstrip() }}|{{ '  y  '.rstrip() }}",
            json!({}),
            json!({}),
        );
        assert_eq!(rendered, "x|y  |  y");

        // replace substitutes every occurrence.
        let rendered = render("{{ 'a-b-c'.replace('-', '+') }}", json!({}), json!({}));
        assert_eq!(rendered, "a+b+c");
    }

    #[test]
    fn dict_items_are_iterable() {
        let rendered = render(
            "{% for k, v in data.items() %}{{ k }}={{ v }};{% endfor %}",
            json!({"data": {"a": 1, "b": 2}}),
            json!({}),
        );
        assert_eq!(rendered, "a=1;b=2;");
    }

    #[test]
    fn unknown_methods_still_fail_loudly() {
        let renderer =
            LocalRenderer::new("{{ 'x'.frobnicate() }}".to_string(), Map::new()).unwrap();
        let error = renderer.render(&json!([]), &json!({})).unwrap_err();
        assert!(
            format!("{error:#}").contains("no method named frobnicate"),
            "{error:#}"
        );
    }

    #[test]
    fn a_broken_template_fails_at_construction() {
        assert!(LocalRenderer::new("{% for x in %}".to_string(), Map::new()).is_err());
    }
}
