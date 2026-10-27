//! Request handlers for OpenAI compatibility and Kestrel session endpoints.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use kestrel_core::session::SessionId;
use kestrel_core::traits::{Backend, CompletionOptions, SessionStore};
use tracing::debug;

use crate::error::ServerError;
use crate::types::{
    format_chatml_prompt, format_chatml_prompt_with_structured, format_chatml_prompt_with_tools,
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ModelListResponse, ModelObject,
    Usage,
};

/// Shared application state containing session manager, scheduler, and model metadata.
pub struct AppState<B: Backend + 'static, S: SessionStore + 'static> {
    /// Active session manager.
    pub manager: Arc<kestrel_core::manager::SessionManager<B, S>>,
    /// Continuous batching scheduler.
    pub scheduler: Arc<kestrel_core::scheduler::BatchScheduler<B, S>>,
    /// Configured model name.
    pub model_id: String,
    /// Available models served by this instance.
    pub available_models: Vec<String>,
    session_counter: AtomicU64,
}

impl<B: Backend + 'static, S: SessionStore + 'static> AppState<B, S> {
    /// Creates a new `AppState` with default batch configuration.
    pub fn new(
        manager: Arc<kestrel_core::manager::SessionManager<B, S>>,
        model_id: impl Into<String>,
    ) -> Self {
        Self::with_batch_config(
            manager,
            model_id,
            kestrel_core::scheduler::BatchConfig::default(),
        )
    }

    /// Creates a new `AppState` configured with an explicit dynamic batching configuration.
    pub fn with_batch_config(
        manager: Arc<kestrel_core::manager::SessionManager<B, S>>,
        model_id: impl Into<String>,
        config: kestrel_core::scheduler::BatchConfig,
    ) -> Self {
        let scheduler = Arc::new(kestrel_core::scheduler::BatchScheduler::new(
            Arc::clone(&manager),
            config,
        ));
        let model_str = model_id.into();
        Self {
            manager,
            scheduler,
            model_id: model_str.clone(),
            available_models: vec![model_str],
            session_counter: AtomicU64::new(1),
        }
    }

    /// Configures the list of available model IDs.
    pub fn with_models(mut self, models: Vec<String>) -> Self {
        if !models.is_empty() {
            self.available_models = models;
        }
        self
    }

    /// Generates a unique monotonic session identifier for requests without one.
    pub fn next_auto_session_id(&self) -> String {
        let count = self.session_counter.fetch_add(1, Ordering::Relaxed);
        format!("auto-{count}")
    }
}

/// Returns unix timestamp in seconds.
pub(crate) fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `GET /v1/models`
pub async fn list_models<B: Backend, S: SessionStore>(
    State(state): State<Arc<AppState<B, S>>>,
) -> Result<Json<ModelListResponse>, ServerError> {
    let data = state
        .available_models
        .iter()
        .map(|id| ModelObject {
            id: id.clone(),
            object: "model".to_string(),
            created: 1677610602,
            owned_by: "kestrel".to_string(),
        })
        .collect();
    Ok(Json(ModelListResponse {
        object: "list".to_string(),
        data,
    }))
}

pub use crate::types::validate_and_create_session_id;

/// Resolves session identifier from headers, request body, or auto-generation.
fn resolve_session_id<B: Backend, S: SessionStore>(
    headers: &HeaderMap,
    req: &ChatCompletionRequest,
    state: &AppState<B, S>,
) -> Result<SessionId, ServerError> {
    // 1. Header `x-kestrel-session-id`
    if let Some(val) = headers.get("x-kestrel-session-id") {
        if let Ok(id_str) = val.to_str() {
            let trimmed = id_str.trim();
            if !trimmed.is_empty() {
                return validate_and_create_session_id(trimmed);
            }
        }
    }

    // 2. Request body `session_id`
    if let Some(ref body_id) = req.session_id {
        let trimmed = body_id.trim();
        if !trimmed.is_empty() {
            return validate_and_create_session_id(trimmed);
        }
    }

    // 3. Request body `user`
    if let Some(ref user_id) = req.user {
        let trimmed = user_id.trim();
        if !trimmed.is_empty() {
            return validate_and_create_session_id(trimmed);
        }
    }

    // 4. Auto-generated fallback
    let auto_id = state.next_auto_session_id();
    validate_and_create_session_id(&auto_id)
}

/// `POST /v1/chat/completions`
pub async fn chat_completions<B: Backend + 'static, S: SessionStore + 'static>(
    State(state): State<Arc<AppState<B, S>>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ServerError> {
    if req.messages.is_empty() {
        return Err(ServerError::BadRequest {
            reason: "messages array must not be empty".to_string(),
        });
    }

    let session_id = resolve_session_id(&headers, &req, &state)?;
    let prompt = format_chatml_prompt(&req.messages);
    let options = CompletionOptions {
        max_tokens: req.max_tokens.unwrap_or(512),
    };
    let requested_model = if req.model.trim().is_empty() || req.model == "default" {
        state.model_id.clone()
    } else {
        req.model.clone()
    };
    let stream_mode = req.stream.unwrap_or(false);

    debug!(
        session = %session_id,
        messages_count = req.messages.len(),
        stream = stream_mode,
        model = %requested_model,
        "Dispatching chat completion"
    );

    if stream_mode {
        crate::streaming::handle_chat_stream(&state, session_id, requested_model, prompt, options)
    } else if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) {
        let is_none_tool_choice = matches!(
            req.tool_choice.as_ref(),
            Some(serde_json::Value::String(s)) if s == "none"
        );

        if is_none_tool_choice {
            let result = state
                .scheduler
                .submit_with_model(session_id, Some(requested_model.clone()), prompt, options)
                .await?;
            let created = current_timestamp();
            return Ok(Json(make_chat_response(created, requested_model, &result)).into_response());
        }

        let require_tool = match req.tool_choice.as_ref() {
            Some(serde_json::Value::String(s)) => s == "required",
            Some(serde_json::Value::Object(map)) => map.contains_key("function"),
            _ => false,
        };

        let tool_prompt = format_chatml_prompt_with_tools(&req.messages, Some(tools));
        let max_retries = req.max_tool_retries.unwrap_or(3);
        let runtime = kestrel_tools::ToolRuntime::new(max_retries);

        let (model_output, result) = runtime
            .complete_with_tools_and_model(
                &state.manager,
                &session_id,
                Some(&requested_model),
                &tool_prompt,
                &options,
                tools,
                require_tool,
            )
            .await?;

        let created = current_timestamp();
        let choice = match model_output {
            kestrel_tools::ModelOutput::ToolCalls(calls) => {
                let openai_calls: Vec<kestrel_tools::OpenAiToolCall> =
                    calls.iter().map(|c| c.to_openai()).collect();
                ChatChoice {
                    index: 0,
                    message: crate::types::ChatMessage {
                        role: "assistant".to_string(),
                        content: String::new(),
                        tool_calls: Some(openai_calls),
                    },
                    finish_reason: Some("tool_calls".to_string()),
                }
            }
            kestrel_tools::ModelOutput::Text(text) => ChatChoice {
                index: 0,
                message: crate::types::ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            },
        };

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{created}"),
            object: "chat.completion".to_string(),
            created,
            model: requested_model,
            choices: vec![choice],
            usage: Usage {
                prompt_tokens: result.prefill_tokens,
                completion_tokens: result.generated_tokens,
                total_tokens: result.prefill_tokens + result.generated_tokens,
            },
        };

        Ok(Json(response).into_response())
    } else if let Some(ref rf) = req.response_format {
        let mode = match rf.format_type.as_str() {
            "json_object" => kestrel_tools::StructuredOutputMode::JsonObject,
            "json_schema" => {
                let js = rf.json_schema.as_ref().ok_or_else(|| ServerError::BadRequest {
                    reason: "json_schema configuration with 'schema' is required when response_format.type is 'json_schema'".into(),
                })?;
                let schema = js.schema.as_ref().ok_or_else(|| ServerError::BadRequest {
                    reason: "json_schema configuration with 'schema' is required when response_format.type is 'json_schema'".into(),
                })?;
                kestrel_tools::StructuredOutputMode::JsonSchema {
                    name: js.name.clone(),
                    description: js.description.clone(),
                    schema: schema.clone(),
                    strict: js.strict.unwrap_or(false),
                }
            }
            "text" => {
                let result = state
                    .scheduler
                    .submit_with_model(session_id, Some(requested_model.clone()), prompt, options)
                    .await?;
                let created = current_timestamp();
                return Ok(
                    Json(make_chat_response(created, requested_model, &result)).into_response()
                );
            }
            _ => {
                return Err(ServerError::BadRequest {
                    reason: format!("unsupported response_format type '{}'", rf.format_type),
                });
            }
        };

        let structured_prompt = format_chatml_prompt_with_structured(&req.messages, &mode);
        let max_retries = req.max_tool_retries.unwrap_or(3);
        let runtime = kestrel_tools::ToolRuntime::new(max_retries);
        let result = runtime
            .complete_with_structured_output(
                &state.manager,
                &session_id,
                Some(&requested_model),
                &structured_prompt,
                &options,
                &mode,
            )
            .await?;

        let created = current_timestamp();
        Ok(Json(make_chat_response(created, requested_model, &result)).into_response())
    } else {
        let result = state
            .scheduler
            .submit_with_model(session_id, Some(requested_model.clone()), prompt, options)
            .await?;

        let created = current_timestamp();
        Ok(Json(make_chat_response(created, requested_model, &result)).into_response())
    }
}

pub(crate) fn make_chat_response(
    created: u64,
    model: String,
    result: &kestrel_core::traits::CompletionResult,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: crate::types::ChatMessage {
                role: "assistant".to_string(),
                content: result.text.clone(),
                tool_calls: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: result.prefill_tokens,
            completion_tokens: result.generated_tokens,
            total_tokens: result.prefill_tokens + result.generated_tokens,
        },
    }
}

// Re-export session endpoints from session_handlers module.
pub use crate::session_handlers::{delete_session, get_metrics, get_session, list_sessions};
