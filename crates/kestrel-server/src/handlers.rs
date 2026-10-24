//! Request handlers for OpenAI compatibility and Kestrel session endpoints.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use kestrel_core::session::SessionId;
use kestrel_core::traits::{Backend, CompletionOptions, SessionStore};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info};

use crate::error::ServerError;
use crate::types::{
    format_chatml_prompt, ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, DeleteSessionResponse, ModelListResponse, ModelObject,
    SessionStats, SessionsListResponse, Usage,
};

/// Shared application state containing session manager and model metadata.
pub struct AppState<B: Backend, S: SessionStore> {
    /// Active session manager.
    pub manager: Arc<kestrel_core::manager::SessionManager<B, S>>,
    /// Configured model name.
    pub model_id: String,
    session_counter: AtomicU64,
}

impl<B: Backend, S: SessionStore> AppState<B, S> {
    /// Creates a new `AppState`.
    pub fn new(
        manager: Arc<kestrel_core::manager::SessionManager<B, S>>,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            manager,
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
    let response = ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: 1677610602,
            owned_by: "kestrel".to_string(),
        }],
    };
    Ok(Json(response))
}

/// Validates that a session ID string only contains permitted characters.
fn validate_and_create_session_id(id: &str) -> Result<SessionId, ServerError> {
    if id.is_empty() {
        return Err(ServerError::BadRequest {
            reason: "session ID cannot be empty".to_string(),
        });
    }

    let is_valid = id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');

    if !is_valid || id.contains("..") {
        return Err(ServerError::BadRequest {
            reason: format!(
                "invalid session id '{id}': must only contain alphanumeric characters, '-', and '_'"
            ),
        });
    }

    Ok(SessionId::new(id))
}

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
    } else {
        let result = state
            .manager
            .complete(&session_id, &prompt, &options)
            .await?;

        let created = current_timestamp();
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{created}"),
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: crate::types::ChatMessage {
                    role: "assistant".to_string(),
                    content: result.text,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: result.prefill_tokens,
                completion_tokens: result.generated_tokens,
                total_tokens: result.prefill_tokens + result.generated_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

/// `GET /kestrel/sessions`
pub async fn list_sessions<B: Backend, S: SessionStore>(
    State(state): State<Arc<AppState<B, S>>>,
) -> Result<Json<SessionsListResponse>, ServerError> {
    let sessions = state.manager.list_sessions().await?;
    let hot_sessions = state.manager.hot_session_count().await;
    let hot_capacity = state.manager.hot_capacity();

    let stats = SessionStats {
        hot_sessions,
        total_sessions: sessions.len(),
        hot_capacity,
    };

    Ok(Json(SessionsListResponse { sessions, stats }))
}

/// `GET /kestrel/sessions/:id`
pub async fn get_session<B: Backend, S: SessionStore>(
    State(state): State<Arc<AppState<B, S>>>,
    Path(id): Path<String>,
) -> Result<Json<kestrel_core::session::SessionMetadata>, ServerError> {
    let session_id = validate_and_create_session_id(&id)?;

    match state.manager.get_metadata(&session_id).await? {
        Some(meta) => Ok(Json(meta)),
        None => Err(ServerError::NotFound {
            resource: "Session".to_string(),
            id,
        }),
    }
}

/// `DELETE /kestrel/sessions/:id`
pub async fn delete_session<B: Backend, S: SessionStore>(
    State(state): State<Arc<AppState<B, S>>>,
    Path(id): Path<String>,
) -> Result<Json<DeleteSessionResponse>, ServerError> {
    let session_id = validate_and_create_session_id(&id)?;

    let deleted = state.manager.delete_session(&session_id).await?;
    if deleted {
        info!(session = %id, "Deleted session");
        Ok(Json(DeleteSessionResponse { id, deleted: true }))
    } else {
        Err(ServerError::NotFound {
            resource: "Session".to_string(),
            id,
        })
    }
}
