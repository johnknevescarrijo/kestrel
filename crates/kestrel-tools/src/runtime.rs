//! Tool runtime orchestrating model completion, schema validation, and conversational retry loops.

use kestrel_core::manager::SessionManager;
use kestrel_core::session::SessionId;
use kestrel_core::traits::{Backend, CompletionOptions, CompletionResult, SessionStore};
use tracing::{debug, warn};

use crate::error::ToolError;
use crate::parser::parse_model_output;
use crate::types::{ModelOutput, ToolDefinition};
use crate::validator::validate_tool_calls;

/// Formats the tool definitions into a system prompt instruction for the model.
pub fn format_tools_prompt(tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();
    prompt.push_str("You have access to the following functions:\n\n");

    for tool in tools {
        prompt.push_str(&format!("Function: {}\n", tool.function.name));
        if let Some(ref desc) = tool.function.description {
            prompt.push_str(&format!("Description: {desc}\n"));
        }
        let schema_json = serde_json::to_string_pretty(&tool.function.parameters)
            .unwrap_or_else(|_| "{}".to_string());
        prompt.push_str(&format!("Parameters (JSON Schema):\n{schema_json}\n\n"));
    }

    prompt.push_str(
        "To invoke a tool, output a <tool_call> block with JSON matching the parameter schema:\n",
    );
    prompt.push_str("<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n");
    prompt
}

/// Formats a validation or parsing error into a conversational feedback prompt in ChatML syntax.
pub fn format_feedback_prompt(error: &ToolError) -> String {
    format!(
        "<|im_start|>user\n[Tool Validation Error]: {error}\nPlease fix the error and output a valid <tool_call> JSON.<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Execution runtime for managing tool-aware inference and automatic self-correction retries.
pub struct ToolRuntime {
    /// Maximum number of feedback retries allowed before giving up.
    pub max_retries: usize,
}

impl Default for ToolRuntime {
    fn default() -> Self {
        Self { max_retries: 3 }
    }
}

impl ToolRuntime {
    /// Creates a new `ToolRuntime` with the specified maximum retry count.
    pub fn new(max_retries: usize) -> Self {
        Self { max_retries }
    }

    /// Executes completion with tool-calling capabilities and explicit model override,
    /// retrying with error feedback if output is invalid.
    #[allow(clippy::too_many_arguments)]
    pub async fn complete_with_tools_and_model<B: Backend, S: SessionStore>(
        &self,
        manager: &SessionManager<B, S>,
        session_id: &SessionId,
        model: Option<&str>,
        prompt: &str,
        options: &CompletionOptions,
        tools: &[ToolDefinition],
        require_tool: bool,
    ) -> Result<(ModelOutput, CompletionResult), ToolError> {
        let max_attempts = 1 + self.max_retries;
        let mut attempts = 0;
        let mut total_prefill = 0;
        let mut total_generated = 0;
        let mut current_prompt = prompt.to_string();
        let mut last_error_reason = String::new();

        while attempts < max_attempts {
            attempts += 1;
            debug!(
                session = %session_id,
                attempt = attempts,
                max_attempts = max_attempts,
                "Running completion attempt for tools"
            );

            let res = manager
                .complete_with_model(session_id, model, &current_prompt, options)
                .await
                .map_err(ToolError::Core)?;

            total_prefill += res.prefill_tokens;
            total_generated += res.generated_tokens;

            match parse_model_output(&res.text) {
                Ok(ModelOutput::ToolCalls(calls)) => match validate_tool_calls(&calls, tools) {
                    Ok(()) => {
                        debug!(
                            session = %session_id,
                            attempts,
                            calls_count = calls.len(),
                            "Successfully validated tool calls"
                        );
                        return Ok((
                            ModelOutput::ToolCalls(calls),
                            CompletionResult {
                                text: res.text,
                                prefill_tokens: total_prefill,
                                generated_tokens: total_generated,
                            },
                        ));
                    }
                    Err(validation_err) => {
                        warn!(
                            session = %session_id,
                            attempt = attempts,
                            error = %validation_err,
                            "Tool call schema validation failed"
                        );
                        last_error_reason = validation_err.to_string();
                        if attempts >= max_attempts {
                            return Err(ToolError::MaxRetriesExceeded {
                                attempts,
                                last_error: last_error_reason,
                            });
                        }
                        current_prompt = format_feedback_prompt(&validation_err);
                    }
                },
                Ok(ModelOutput::Text(text)) => {
                    if require_tool {
                        let required_err = ToolError::ToolRequiredButNoneCalled {
                            output: text.clone(),
                        };
                        warn!(
                            session = %session_id,
                            attempt = attempts,
                            "Tool required but model returned plain text"
                        );
                        last_error_reason = required_err.to_string();
                        if attempts >= max_attempts {
                            return Err(ToolError::MaxRetriesExceeded {
                                attempts,
                                last_error: last_error_reason,
                            });
                        }
                        current_prompt = format_feedback_prompt(&required_err);
                    } else {
                        return Ok((
                            ModelOutput::Text(text),
                            CompletionResult {
                                text: res.text,
                                prefill_tokens: total_prefill,
                                generated_tokens: total_generated,
                            },
                        ));
                    }
                }
                Err(parse_err) => {
                    warn!(
                        session = %session_id,
                        attempt = attempts,
                        error = %parse_err,
                        "Tool output parsing failed"
                    );
                    last_error_reason = parse_err.to_string();
                    if attempts >= max_attempts {
                        return Err(ToolError::MaxRetriesExceeded {
                            attempts,
                            last_error: last_error_reason,
                        });
                    }
                    current_prompt = format_feedback_prompt(&parse_err);
                }
            }
        }

        Err(ToolError::MaxRetriesExceeded {
            attempts,
            last_error: last_error_reason,
        })
    }

    /// Executes completion with tool-calling capabilities using default model.
    pub async fn complete_with_tools<B: Backend, S: SessionStore>(
        &self,
        manager: &SessionManager<B, S>,
        session_id: &SessionId,
        prompt: &str,
        options: &CompletionOptions,
        tools: &[ToolDefinition],
        require_tool: bool,
    ) -> Result<(ModelOutput, CompletionResult), ToolError> {
        self.complete_with_tools_and_model(
            manager,
            session_id,
            None,
            prompt,
            options,
            tools,
            require_tool,
        )
        .await
    }

    /// Executes completion enforcing native structured output format (JSON Mode or JSON Schema),
    /// retrying with corrective feedback if the output fails validation.
    pub async fn complete_with_structured_output<B: Backend, S: SessionStore>(
        &self,
        manager: &SessionManager<B, S>,
        session_id: &SessionId,
        model: Option<&str>,
        prompt: &str,
        options: &CompletionOptions,
        mode: &crate::structured::StructuredOutputMode,
    ) -> Result<CompletionResult, ToolError> {
        let max_attempts = 1 + self.max_retries;
        let mut attempts = 0;
        let mut total_prefill = 0;
        let mut total_generated = 0;
        let mut current_prompt = prompt.to_string();
        let mut last_error_reason = String::new();

        while attempts < max_attempts {
            attempts += 1;
            debug!(
                session = %session_id,
                attempt = attempts,
                max_attempts = max_attempts,
                "Running completion attempt for structured output"
            );

            let res = manager
                .complete_with_model(session_id, model, &current_prompt, options)
                .await
                .map_err(ToolError::Core)?;

            total_prefill += res.prefill_tokens;
            total_generated += res.generated_tokens;

            match crate::structured::extract_and_validate_json(&res.text, mode) {
                Ok(valid_json) => {
                    debug!(
                        session = %session_id,
                        attempts,
                        "Successfully validated structured JSON output"
                    );
                    return Ok(CompletionResult {
                        text: valid_json,
                        prefill_tokens: total_prefill,
                        generated_tokens: total_generated,
                    });
                }
                Err(val_err) => {
                    warn!(
                        session = %session_id,
                        attempt = attempts,
                        error = %val_err,
                        "Structured output validation failed"
                    );
                    last_error_reason = val_err.to_string();
                    if attempts >= max_attempts {
                        return Err(ToolError::MaxRetriesExceeded {
                            attempts,
                            last_error: last_error_reason,
                        });
                    }
                    current_prompt = crate::structured::format_structured_feedback_prompt(&val_err);
                }
            }
        }

        Err(ToolError::MaxRetriesExceeded {
            attempts,
            last_error: last_error_reason,
        })
    }
}
