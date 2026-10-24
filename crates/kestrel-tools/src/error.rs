//! Error types for the `kestrel-tools` crate.

use thiserror::Error;

/// Errors arising during tool parsing, schema validation, and execution.
#[derive(Debug, Error)]
pub enum ToolError {
    /// The tool definition or arguments contain invalid JSON schema.
    #[error("tool schema validation failed for '{tool_name}': {reason}. Verify parameters match the expected tool signature.")]
    ValidationFailed {
        /// Name of the tool.
        tool_name: String,
        /// Failure details.
        reason: String,
    },

    /// The model emitted a tool call that could not be parsed.
    #[error("failed to parse tool call from output '{output}': {reason}. Consider re-prompting the model with clearer format instructions.")]
    ParseFailed {
        /// Raw output attempted to parse.
        output: String,
        /// Parsing failure details.
        reason: String,
    },

    /// The requested tool was not found in the registry.
    #[error(
        "tool '{tool_name}' not found. Check registered tool definitions in the current session."
    )]
    NotFound {
        /// Requested tool name.
        tool_name: String,
    },
}
