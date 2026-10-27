#![deny(unsafe_code)]

//! Tool calling runtime, schema validation, and retry logic for Kestrel.

pub mod error;
pub mod parser;
pub mod runtime;
pub mod structured;
pub mod types;
pub mod validator;

pub use error::ToolError;
pub use parser::parse_model_output;
pub use runtime::{format_feedback_prompt, format_tools_prompt, ToolRuntime};
pub use structured::{
    extract_and_validate_json, format_structured_feedback_prompt, format_structured_output_prompt,
    StructuredOutputMode,
};
pub use types::{
    FunctionDefinition, ModelOutput, OpenAiFunctionCall, OpenAiToolCall, ParsedToolCall,
    ToolDefinition,
};
pub use validator::validate_tool_calls;
