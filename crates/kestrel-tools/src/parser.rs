//! Parsing utilities for extracting structured tool calls from model outputs.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::error::ToolError;
use crate::types::{ModelOutput, ParsedToolCall};

static CALL_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generates a unique tool call ID with format `call_{id}`.
fn next_call_id() -> String {
    let id = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{id}")
}

/// Normalizes arguments field from either a JSON Value (object/array) or a JSON string.
fn normalize_arguments(args_val: Option<&Value>) -> Result<Value, ToolError> {
    match args_val {
        None | Some(Value::Null) => Ok(Value::Object(serde_json::Map::new())),
        Some(Value::Object(map)) => Ok(Value::Object(map.clone())),
        Some(Value::Array(arr)) => Ok(Value::Array(arr.clone())),
        Some(Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed == "{}" {
                return Ok(Value::Object(serde_json::Map::new()));
            }
            serde_json::from_str::<Value>(trimmed).map_err(|e| ToolError::MalformedJson {
                raw: s.clone(),
                reason: format!("invalid JSON in arguments string: {e}"),
            })
        }
        Some(other) => Ok(other.clone()),
    }
}

/// Converts a JSON Object representing a single tool call into `ParsedToolCall`.
fn parse_single_call_object(
    obj: &serde_json::Map<String, Value>,
) -> Result<ParsedToolCall, ToolError> {
    let name = match obj.get("name") {
        Some(Value::String(n)) if !n.trim().is_empty() => n.trim().to_string(),
        _ => {
            return Err(ToolError::MalformedJson {
                raw: serde_json::to_string(obj).unwrap_or_default(),
                reason: "missing or empty 'name' field in tool call".to_string(),
            });
        }
    };

    let id = match obj.get("id") {
        Some(Value::String(id_str)) if !id_str.trim().is_empty() => id_str.trim().to_string(),
        _ => next_call_id(),
    };

    let arguments = normalize_arguments(obj.get("arguments"))?;

    Ok(ParsedToolCall {
        id,
        name,
        arguments,
    })
}

/// Strips enclosing markdown code fences (```json ... ``` or ```tool_call ... ```) if present.
fn strip_markdown_code_fence(s: &str) -> &str {
    let trimmed = s.trim();
    let stripped_opt = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```tool_call"))
        .or_else(|| trimmed.strip_prefix("```"));

    if let Some(stripped) = stripped_opt {
        let after_prefix = stripped.trim_start();
        if let Some(end) = after_prefix.rfind("```") {
            after_prefix[..end].trim()
        } else {
            after_prefix.trim()
        }
    } else {
        trimmed
    }
}

/// Extracts tool calls from `<tool_call> ... </tool_call>` tags.
fn extract_from_tool_call_tags(output: &str) -> Result<Option<Vec<ParsedToolCall>>, ToolError> {
    const OPEN_TAG: &str = "<tool_call>";
    const CLOSE_TAG: &str = "</tool_call>";

    if !output.contains(OPEN_TAG) {
        return Ok(None);
    }

    let mut calls = Vec::new();
    let mut remainder = output;

    while let Some(start_idx) = remainder.find(OPEN_TAG) {
        let after_open = &remainder[start_idx + OPEN_TAG.len()..];
        let (raw_content, advance_len) = if let Some(end_idx) = after_open.find(CLOSE_TAG) {
            (&after_open[..end_idx], end_idx + CLOSE_TAG.len())
        } else if let Some(md_start) = after_open.find("```") {
            let after_md = &after_open[md_start + 3..];
            if let Some(md_close) = after_md.find("```") {
                let end = md_start + 3 + md_close + 3;
                (&after_open[..end], end)
            } else {
                return Err(ToolError::MalformedJson {
                    raw: output.to_string(),
                    reason: "unclosed <tool_call> tag detected in output".to_string(),
                });
            }
        } else {
            return Err(ToolError::MalformedJson {
                raw: output.to_string(),
                reason: "unclosed <tool_call> tag detected in output".to_string(),
            });
        };

        let clean_content = strip_markdown_code_fence(raw_content);
        let parsed_val: Value =
            serde_json::from_str(clean_content).map_err(|e| ToolError::MalformedJson {
                raw: clean_content.to_string(),
                reason: format!("failed to parse JSON inside <tool_call>: {e}"),
            })?;

        match parsed_val {
            Value::Object(map) => {
                calls.push(parse_single_call_object(&map)?);
            }
            Value::Array(arr) => {
                for item in arr {
                    match item {
                        Value::Object(map) => calls.push(parse_single_call_object(&map)?),
                        _ => {
                            return Err(ToolError::MalformedJson {
                                raw: raw_content.to_string(),
                                reason: "tool call array must only contain JSON objects"
                                    .to_string(),
                            });
                        }
                    }
                }
            }
            _ => {
                return Err(ToolError::MalformedJson {
                    raw: raw_content.to_string(),
                    reason: "content inside <tool_call> must be a JSON object or array".to_string(),
                });
            }
        }

        remainder = &after_open[advance_len..];
    }

    Ok(Some(calls))
}

/// Extracts tool calls from markdown code blocks (```` ```tool_call ```` or ```` ```json ````).
fn extract_from_markdown_blocks(output: &str) -> Result<Option<Vec<ParsedToolCall>>, ToolError> {
    for prefix in &["```tool_call", "```json"] {
        if let Some(start_idx) = output.find(prefix) {
            let after_open = &output[start_idx + prefix.len()..];
            let end_idx = match after_open.find("```") {
                Some(idx) => idx,
                None => {
                    return Err(ToolError::MalformedJson {
                        raw: output.to_string(),
                        reason: format!("unclosed markdown block '{prefix}' detected"),
                    });
                }
            };

            let raw_content = after_open[..end_idx].trim();
            match serde_json::from_str::<Value>(raw_content) {
                Ok(Value::Object(map)) => {
                    if map.contains_key("name") {
                        let call = parse_single_call_object(&map)?;
                        return Ok(Some(vec![call]));
                    }
                }
                Ok(Value::Array(arr)) => {
                    let mut calls = Vec::new();
                    let mut all_valid = true;
                    for item in arr {
                        if let Value::Object(map) = item {
                            if map.contains_key("name") {
                                calls.push(parse_single_call_object(&map)?);
                            } else {
                                all_valid = false;
                                break;
                            }
                        } else {
                            all_valid = false;
                            break;
                        }
                    }
                    if all_valid && !calls.is_empty() {
                        return Ok(Some(calls));
                    }
                }
                Err(e) if *prefix == "```tool_call" || raw_content.contains("\"name\"") => {
                    return Err(ToolError::MalformedJson {
                        raw: raw_content.to_string(),
                        reason: format!("invalid JSON in markdown block: {e}"),
                    });
                }
                _ => {}
            }
        }
    }

    Ok(None)
}

/// Extracts tool calls from a raw JSON string at the top level.
fn extract_from_raw_json(output: &str) -> Result<Option<Vec<ParsedToolCall>>, ToolError> {
    let trimmed = output.trim();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return Ok(None);
    }

    match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Object(map)) => {
            if map.contains_key("name") {
                let call = parse_single_call_object(&map)?;
                return Ok(Some(vec![call]));
            }
        }
        Ok(Value::Array(arr)) => {
            let mut calls = Vec::new();
            let mut all_valid = true;
            for item in arr {
                if let Value::Object(map) = item {
                    if map.contains_key("name") {
                        calls.push(parse_single_call_object(&map)?);
                    } else {
                        all_valid = false;
                        break;
                    }
                } else {
                    all_valid = false;
                    break;
                }
            }
            if all_valid && !calls.is_empty() {
                return Ok(Some(calls));
            }
        }
        Err(e) if trimmed.contains("\"name\"") || trimmed.contains("\"arguments\"") => {
            return Err(ToolError::MalformedJson {
                raw: trimmed.to_string(),
                reason: format!("truncated or invalid raw JSON tool call: {e}"),
            });
        }
        _ => {}
    }

    Ok(None)
}

/// Parses raw model completion output into either structured tool calls or plain text.
pub fn parse_model_output(output: &str) -> Result<ModelOutput, ToolError> {
    // 1. Try <tool_call> tags first
    if let Some(calls) = extract_from_tool_call_tags(output)? {
        if !calls.is_empty() {
            return Ok(ModelOutput::ToolCalls(calls));
        }
    }

    // 2. Try markdown blocks (```tool_call or ```json)
    if let Some(calls) = extract_from_markdown_blocks(output)? {
        if !calls.is_empty() {
            return Ok(ModelOutput::ToolCalls(calls));
        }
    }

    // 3. Try raw top-level JSON
    if let Some(calls) = extract_from_raw_json(output)? {
        if !calls.is_empty() {
            return Ok(ModelOutput::ToolCalls(calls));
        }
    }

    // 4. Default to plain text
    Ok(ModelOutput::Text(output.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_tool_call_tag() {
        let output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}\n</tool_call>";
        let parsed = parse_model_output(output).expect("valid parse");
        match parsed {
            ModelOutput::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].arguments, json!({"city": "London"}));
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn test_parse_unclosed_tool_call_tag() {
        let output = "<tool_call>\n{\"name\": \"get_weather\"";
        let res = parse_model_output(output);
        assert!(matches!(res, Err(ToolError::MalformedJson { .. })));
    }

    #[test]
    fn test_parse_stringified_arguments() {
        let output = r#"{"name": "search", "arguments": "{\"query\": \"rust\"}"}"#;
        let parsed = parse_model_output(output).expect("valid parse");
        match parsed {
            ModelOutput::ToolCalls(calls) => {
                assert_eq!(calls[0].arguments, json!({"query": "rust"}));
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn test_parse_plain_text() {
        let output = "Hello there! How can I help you today?";
        let parsed = parse_model_output(output).expect("valid parse");
        assert_eq!(parsed, ModelOutput::Text(output.to_string()));
    }
}
