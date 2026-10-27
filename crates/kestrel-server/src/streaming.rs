//! SSE streaming response handler for OpenAI chat completions.

use std::sync::Arc;

use axum::http::{header, StatusCode};
use axum::response::Response;
use kestrel_core::session::SessionId;
use kestrel_core::traits::{Backend, CompletionOptions, SessionStore};
use tokio_stream::wrappers::ReceiverStream;
use tracing::error;

use crate::error::ServerError;
use crate::handlers::{current_timestamp, AppState};
use crate::types::{ChatChunkChoice, ChatCompletionChunk, ChatDelta};

/// Dispatches streaming completion task and returns SSE response stream.
pub fn handle_chat_stream<B: Backend + 'static, S: SessionStore + 'static>(
    state: &Arc<AppState<B, S>>,
    session_id: SessionId,
    model: String,
    prompt: String,
    options: CompletionOptions,
) -> Result<Response, ServerError> {
    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let manager = Arc::clone(&state.manager);
    let sess_id_clone = session_id.clone();
    let prompt_clone = prompt.clone();
    let options_clone = options.clone();
    let requested_model = model.clone();

    tokio::spawn(async move {
        if let Err(err) = manager
            .complete_stream_with_model(
                &sess_id_clone,
                Some(&requested_model),
                &prompt_clone,
                &options_clone,
                token_tx,
            )
            .await
        {
            error!(session = %sess_id_clone, error = %err, "Streaming completion task failed");
        }
    });

    let (sse_tx, sse_rx) =
        tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(32);
    let chunk_id = format!("chatcmpl-{}", current_timestamp());
    let model_name = model;

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
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .map_err(|e| ServerError::Internal {
            reason: e.to_string(),
        })
}
