//! HTTP endpoints for Kestrel session inspection and management.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use kestrel_core::traits::{Backend, SessionStore};
use tracing::info;

use crate::error::ServerError;
use crate::handlers::{validate_and_create_session_id, AppState};
use crate::types::{DeleteSessionResponse, SessionStats, SessionsListResponse};

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
