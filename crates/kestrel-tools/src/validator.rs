//! JSON Schema validation for tool invocations against tool definitions.

use crate::error::ToolError;
use crate::types::{ParsedToolCall, ToolDefinition};

/// Validates that all tool calls match registered tools and adhere to their parameter schemas.
pub fn validate_tool_calls(
    calls: &[ParsedToolCall],
    tools: &[ToolDefinition],
) -> Result<(), ToolError> {
    for call in calls {
        // 1. Locate tool definition by name
        let tool = match tools.iter().find(|t| t.function.name == call.name) {
            Some(t) => t,
            None => {
                let available = tools.iter().map(|t| t.function.name.clone()).collect();
                return Err(ToolError::UnknownTool {
                    name: call.name.clone(),
                    available,
                });
            }
        };

        // 2. If parameter schema is null or empty object, accept any arguments
        let schema_val = &tool.function.parameters;
        if schema_val.is_null() {
            continue;
        }

        // 3. Compile JSON schema validator
        let validator = jsonschema::validator_for(schema_val).map_err(|e| {
            ToolError::InvalidSchemaDefinition {
                tool_name: call.name.clone(),
                reason: e.to_string(),
            }
        })?;

        // 4. Validate arguments against schema
        let mut errors = Vec::new();
        for err in validator.iter_errors(&call.arguments) {
            errors.push(err.to_string());
        }

        if !errors.is_empty() {
            return Err(ToolError::schema_validation_failed(
                call.name.clone(),
                errors,
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_unknown_tool() {
        let calls = vec![ParsedToolCall::new("c1", "calc", json!({}))];
        let tools = vec![ToolDefinition::new_function(
            "weather",
            None,
            json!({"type": "object"}),
        )];
        let res = validate_tool_calls(&calls, &tools);
        assert!(matches!(res, Err(ToolError::UnknownTool { .. })));
    }

    #[test]
    fn test_validate_schema_success() {
        let calls = vec![ParsedToolCall::new(
            "c1",
            "weather",
            json!({"city": "Tokyo"}),
        )];
        let tools = vec![ToolDefinition::new_function(
            "weather",
            None,
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        )];
        assert!(validate_tool_calls(&calls, &tools).is_ok());
    }

    #[test]
    fn test_validate_schema_failure() {
        let calls = vec![ParsedToolCall::new(
            "c1",
            "weather",
            json!({"city": 123}), // invalid type
        )];
        let tools = vec![ToolDefinition::new_function(
            "weather",
            None,
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        )];
        let res = validate_tool_calls(&calls, &tools);
        assert!(matches!(res, Err(ToolError::SchemaValidationFailed { .. })));
    }
}
