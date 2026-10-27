//! Native structured output enforcement: JSON Mode and JSON Schema validation with feedback loops.

use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Target mode and constraints for model structured outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructuredOutputMode {
    /// Requires the model to output a valid JSON object (`{"type": "json_object"}`).
    JsonObject,
    /// Requires the model to output JSON conforming to a JSON Schema (`{"type": "json_schema"}`).
    JsonSchema {
        /// Name of the target schema.
        name: String,
        /// Optional human-readable description of the expected output format.
        description: Option<String>,
        /// Expected JSON Schema definition.
        schema: serde_json::Value,
        /// Whether strict schema conformance is enforced without additional properties.
        strict: bool,
    },
}

/// Formats the prompt instructions instructing the model to adhere to the specified structured format.
pub fn format_structured_output_prompt(mode: &StructuredOutputMode) -> String {
    match mode {
        StructuredOutputMode::JsonObject => {
            "Respond ONLY with a valid JSON object. Do not include explanatory text before or after the JSON.\n\n".to_string()
        }
        StructuredOutputMode::JsonSchema {
            name,
            description,
            schema,
            strict,
        } => {
            let schema_json = serde_json::to_string_pretty(schema).unwrap_or_else(|_| "{}".to_string());
            let mut prompt = format!("Respond ONLY with a valid JSON object conforming to the schema for '{name}'.\n");
            if let Some(desc) = description {
                prompt.push_str(&format!("Schema Description: {desc}\n"));
            }
            if *strict {
                prompt.push_str("Schema adherence is strict. Do not include properties not defined in the schema.\n");
            }
            prompt.push_str(&format!("JSON Schema:\n{schema_json}\n\n"));
            prompt
        }
    }
}

/// Formats structured output validation errors into a conversational feedback prompt in ChatML syntax.
pub fn format_structured_feedback_prompt(error: &ToolError) -> String {
    format!(
        "<|im_start|>user\n[JSON Validation Error]: {error}\nPlease correct the errors and output only the valid JSON.<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Extracts JSON from raw model output (handling code fences if present) and validates against the mode.
pub fn extract_and_validate_json(
    raw: &str,
    mode: &StructuredOutputMode,
) -> Result<String, ToolError> {
    let text = raw.trim();
    let json_candidate = if let Some(stripped) = text.strip_prefix("```json") {
        stripped.strip_suffix("```").unwrap_or(stripped).trim()
    } else if let Some(stripped) = text.strip_prefix("```") {
        stripped.strip_suffix("```").unwrap_or(stripped).trim()
    } else if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            after[..end].trim()
        } else {
            after.trim()
        }
    } else if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        if let Some(end) = after.find("```") {
            after[..end].trim()
        } else {
            after.trim()
        }
    } else {
        text
    };

    let value: serde_json::Value =
        serde_json::from_str(json_candidate).map_err(|err| ToolError::MalformedJson {
            raw: raw.to_string(),
            reason: err.to_string(),
        })?;

    match mode {
        StructuredOutputMode::JsonObject => {
            if !value.is_object() {
                return Err(ToolError::MalformedJson {
                    raw: raw.to_string(),
                    reason: "expected a JSON object ({...}), but received a non-object JSON value"
                        .to_string(),
                });
            }
        }
        StructuredOutputMode::JsonSchema { name, schema, .. } => {
            if !schema.is_null() && schema != &serde_json::json!({}) {
                let validator = jsonschema::validator_for(schema).map_err(|e| {
                    ToolError::InvalidSchemaDefinition {
                        tool_name: name.clone(),
                        reason: e.to_string(),
                    }
                })?;

                let mut errors = Vec::new();
                for err in validator.iter_errors(&value) {
                    errors.push(err.to_string());
                }

                if !errors.is_empty() {
                    return Err(ToolError::schema_validation_failed(name.clone(), errors));
                }
            }
        }
    }

    serde_json::to_string(&value).map_err(|err| ToolError::MalformedJson {
        raw: raw.to_string(),
        reason: err.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_and_validate_json_object_success() {
        let raw = "```json\n{\"key\": \"value\", \"count\": 42}\n```";
        let res = extract_and_validate_json(raw, &StructuredOutputMode::JsonObject).unwrap();
        assert_eq!(res, "{\"count\":42,\"key\":\"value\"}");
    }

    #[test]
    fn test_extract_and_validate_json_object_rejects_array() {
        let raw = "[1, 2, 3]";
        let res = extract_and_validate_json(raw, &StructuredOutputMode::JsonObject);
        assert!(matches!(res, Err(ToolError::MalformedJson { .. })));
    }

    #[test]
    fn test_extract_and_validate_json_schema_success() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });

        let mode = StructuredOutputMode::JsonSchema {
            name: "Person".to_string(),
            description: Some("A person record".to_string()),
            schema,
            strict: true,
        };

        let raw = "{\"name\": \"Alice\", \"age\": 30}";
        let res = extract_and_validate_json(raw, &mode).unwrap();
        assert_eq!(res, "{\"age\":30,\"name\":\"Alice\"}");
    }

    #[test]
    fn test_extract_and_validate_json_schema_failure() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });

        let mode = StructuredOutputMode::JsonSchema {
            name: "Person".to_string(),
            description: None,
            schema,
            strict: true,
        };

        let raw = "{\"name\": \"Alice\", \"age\": \"thirty\"}";
        let res = extract_and_validate_json(raw, &mode);
        assert!(matches!(res, Err(ToolError::SchemaValidationFailed { .. })));
    }
}
