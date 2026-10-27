//! Request handlers for OpenAI-compatible `/v1/embeddings` endpoint.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use kestrel_core::traits::{Backend, SessionStore};

use crate::error::ServerError;
use crate::handlers::AppState;
use crate::types::{
    EmbeddingInput, EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
};

/// `POST /v1/embeddings`
pub async fn embeddings<B: Backend + 'static, S: SessionStore + 'static>(
    State(state): State<Arc<AppState<B, S>>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ServerError> {
    let inputs: Vec<String> = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Multiple(v) => v,
    };

    if inputs.is_empty() || inputs.iter().any(|s| s.trim().is_empty()) {
        return Err(ServerError::BadRequest {
            reason: "embedding input must not be empty".to_string(),
        });
    }

    if req.dimensions == Some(0) {
        return Err(ServerError::BadRequest {
            reason: "dimensions must be greater than 0".to_string(),
        });
    }

    let mut data = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens = 0;

    for (index, text) in inputs.iter().enumerate() {
        let mut res = state.manager.embed(Some(&req.model), text).await?;

        if let Some(dims) = req.dimensions {
            if res.embedding.len() > dims {
                res.embedding.truncate(dims);
            } else if res.embedding.len() < dims {
                let original = res.embedding.clone();
                while res.embedding.len() < dims {
                    let needed = dims - res.embedding.len();
                    let take = needed.min(original.len());
                    res.embedding.extend_from_slice(&original[..take]);
                }
            }

            let norm: f32 = res.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut res.embedding {
                    *x /= norm;
                }
            }
        }

        total_prompt_tokens += res.prompt_tokens;
        data.push(EmbeddingObject {
            object: "embedding".to_string(),
            embedding: res.embedding,
            index,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: req.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    }))
}
