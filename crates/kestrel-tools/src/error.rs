//! Error types for the `kestrel-tools` crate.

use thiserror::Error;

/// Errors arising during tool parsing, schema validation, and retry execution.
#[derive(Debug, Error)]
pub enum ToolError {
    /// The model emitted malformed or unclosed JSON.
    #[error(
        "malformed JSON in tool call '{raw}': {reason}. Ensure valid JSON syntax is produced."
    )]
    MalformedJson {
        /// Raw output snippet attempted to parse.
        raw: String,
        /// Description of the parsing failure.
        reason: String,
    },

    /// The model requested an unregistered tool.
    #[error(
        "tool '{name}' not found; available tools: {available:?}. Verify the tool is registered."
    )]
    UnknownTool {
        /// Requested tool name.
        name: String,
        /// List of registered tool names.
        available: Vec<String>,
    },

    /// The tool's schema definition itself is invalid JSON Schema.
    #[error(
        "invalid JSON schema definition for tool '{tool_name}': {reason}. Check schema syntax."
    )]
    InvalidSchemaDefinition {
        /// Name of the tool.
        tool_name: String,
        /// Schema compilation failure details.
        reason: String,
    },

    /// The arguments passed to a tool violated its JSON Schema.
    #[error("arguments for tool '{tool_name}' failed schema validation: {details}. Ensure all required fields and types match.")]
    SchemaValidationFailed {
        /// Name of the tool.
        tool_name: String,
        /// List of schema violation messages.
        errors: Vec<String>,
        /// Combined error details.
        details: String,
    },

    /// Tool calling was required by the caller but the model produced plain text.
    #[error("tool execution was required, but model emitted plain text: '{output}'. Re-prompt with strict tool instructions.")]
    ToolRequiredButNoneCalled {
        /// Plain text emitted by the model.
        output: String,
    },

    /// Exhausted all retry attempts without obtaining a valid tool call.
    #[error("maximum tool retries exceeded after {attempts} attempts; last error: {last_error}")]
    MaxRetriesExceeded {
        /// Number of attempts executed.
        attempts: usize,
        /// Details of the most recent failure.
        last_error: String,
    },

    /// Underlying error originating from the core engine.
    #[error("core engine error during tool execution: {0}")]
    Core(#[from] kestrel_core::Error),
}

impl ToolError {
    /// Creates a schema validation error with pre-formatted combined details.
    pub fn schema_validation_failed(tool_name: impl Into<String>, errors: Vec<String>) -> Self {
        let details = errors.join("; ");
        Self::SchemaValidationFailed {
            tool_name: tool_name.into(),
            errors,
            details,
        }
    }
}
