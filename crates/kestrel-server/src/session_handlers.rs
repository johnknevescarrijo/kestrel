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
    let warm_sessions = state.manager.warm_session_count().await;
    let hot_used_bytes = state.manager.hot_used_bytes().await;
    let warm_used_bytes = state.manager.warm_used_bytes().await;
    let policy = state.manager.policy();

    let stats = SessionStats {
        hot_sessions,
        warm_sessions,
        total_sessions: sessions.len(),
        hot_capacity: state.manager.hot_capacity(),
        hot_used_bytes,
        warm_used_bytes,
        hot_budget_bytes: policy.hot_budget_bytes,
        warm_budget_bytes: policy.warm_budget_bytes,
    };

    Ok(Json(SessionsListResponse { sessions, stats }))
}

/// `GET /kestrel/metrics`
pub async fn get_metrics<B: Backend, S: SessionStore>(
    State(state): State<Arc<AppState<B, S>>>,
) -> Json<kestrel_core::TierMetricsSnapshot> {
    let mut snapshot = state.manager.metrics_snapshot().await;
    snapshot.batch_metrics = Some(state.scheduler.metrics_snapshot());
    Json(snapshot)
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
