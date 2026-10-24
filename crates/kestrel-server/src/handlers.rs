//! Request handlers for OpenAI compatibility and Kestrel session endpoints.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use kestrel_core::session::SessionId;
use kestrel_core::traits::{Backend, CompletionOptions, SessionStore};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error};

use crate::error::ServerError;
use crate::types::{
    format_chatml_prompt, format_chatml_prompt_with_tools, ChatChoice, ChatChunkChoice,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatDelta,
    ModelListResponse, ModelObject, Usage,
};

/// Shared application state containing session manager, scheduler, and model metadata.
pub struct AppState<B: Backend + 'static, S: SessionStore + 'static> {
    /// Active session manager.
    pub manager: Arc<kestrel_core::manager::SessionManager<B, S>>,
    /// Continuous batching scheduler.
    pub scheduler: Arc<kestrel_core::scheduler::BatchScheduler<B, S>>,
    /// Configured model name.
    pub model_id: String,
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
        Self {
            manager,
            scheduler,
            model_id: model_id.into(),
            session_counter: AtomicU64::new(1),
        }
    }

    /// Generates a unique monotonic session identifier for requests without one.
    pub fn next_auto_session_id(&self) -> String {
        let count = self.session_counter.fetch_add(1, Ordering::Relaxed);
        format!("auto-{count}")
    }
}

/// Returns unix timestamp in seconds.
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `GET /v1/models`
pub async fn list_models<B: Backend, S: SessionStore>(
    State(state): State<Arc<AppState<B, S>>>,
) -> Result<Json<ModelListResponse>, ServerError> {
    Ok(Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: 1677610602,
            owned_by: "kestrel".to_string(),
        }],
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
    let model = state.model_id.clone();
    let stream_mode = req.stream.unwrap_or(false);

    debug!(
        session = %session_id,
        messages_count = req.messages.len(),
        stream = stream_mode,
        "Dispatching chat completion"
    );

    if stream_mode {
        let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let manager = Arc::clone(&state.manager);
        let sess_id_clone = session_id.clone();
        let prompt_clone = prompt.clone();
        let options_clone = options.clone();

        tokio::spawn(async move {
            if let Err(err) = manager
                .complete_stream(&sess_id_clone, &prompt_clone, &options_clone, token_tx)
                .await
            {
                error!(session = %sess_id_clone, error = %err, "Streaming completion task failed");
            }
        });

        let (sse_tx, sse_rx) =
            tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);
        let chunk_id = format!("chatcmpl-{}", current_timestamp());
        let model_name = model.clone();

        tokio::spawn(async move {
            let created = current_timestamp();

            // 1. Initial role chunk
            let initial_chunk = ChatCompletionChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant".to_string()),
                        content: Some(String::new()),
                    },
                    finish_reason: None,
                }],
            };

            if let Ok(json) = serde_json::to_string(&initial_chunk) {
                let msg = format!("data: {json}\n\n");
                let _ = sse_tx.send(Ok(axum::body::Bytes::from(msg))).await;
            }

            // 2. Stream subsequent tokens
            while let Some(piece) = token_rx.recv().await {
                if piece.is_empty() {
                    continue;
                }
                let chunk = ChatCompletionChunk {
                    id: chunk_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(piece),
                        },
                        finish_reason: None,
                    }],
                };

                if let Ok(json) = serde_json::to_string(&chunk) {
                    let msg = format!("data: {json}\n\n");
                    if sse_tx.send(Ok(axum::body::Bytes::from(msg))).await.is_err() {
                        break;
                    }
                }
            }

            // 3. Final chunk with finish_reason
            let final_chunk = ChatCompletionChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta::default(),
                    finish_reason: Some("stop".to_string()),
                }],
            };

            if let Ok(json) = serde_json::to_string(&final_chunk) {
                let msg = format!("data: {json}\n\n");
                let _ = sse_tx.send(Ok(axum::body::Bytes::from(msg))).await;
            }

            // 4. SSE termination signal
            let _ = sse_tx
                .send(Ok(axum::body::Bytes::from("data: [DONE]\n\n")))
                .await;
        });

        let body = axum::body::Body::from_stream(ReceiverStream::new(sse_rx));
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(body)
            .map_err(|e| ServerError::Internal {
                reason: e.to_string(),
            })?;

        Ok(response)
    } else if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) {
        let is_none_tool_choice = matches!(
            req.tool_choice.as_ref(),
            Some(serde_json::Value::String(s)) if s == "none"
        );

        if is_none_tool_choice {
            let result = state.scheduler.submit(session_id, prompt, options).await?;
            let created = current_timestamp();
            return Ok(Json(make_chat_response(created, model, &result)).into_response());
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
            .complete_with_tools(
                &state.manager,
                &session_id,
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
            model,
            choices: vec![choice],
            usage: Usage {
                prompt_tokens: result.prefill_tokens,
                completion_tokens: result.generated_tokens,
                total_tokens: result.prefill_tokens + result.generated_tokens,
            },
        };

        Ok(Json(response).into_response())
    } else {
        let result = state.scheduler.submit(session_id, prompt, options).await?;

        let created = current_timestamp();
        Ok(Json(make_chat_response(created, model, &result)).into_response())
    }
}

fn make_chat_response(
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
