//! Error types for the `kestrel-server` crate.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use thiserror::Error;

/// Standard OpenAI-compatible error payload wrapper.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    /// Underlying error details.
    pub error: ApiErrorDetail,
}

/// Details of an API error following OpenAI schema.
#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorDetail {
    /// Human-readable error description.
    pub message: String,
    /// Error category (e.g. invalid_request_error, not_found_error, api_error).
    #[serde(rename = "type")]
    pub error_type: String,
    /// Specific error code if available.
    pub code: String,
}

/// HTTP server and endpoint handling errors.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Failed to bind to requested address/port.
    #[error("failed to bind server to address '{addr}': {reason}. Check if port is in use.")]
    BindFailed {
        /// Socket address.
        addr: String,
        /// Reason for failure.
        reason: String,
    },

    /// Request payload was malformed or could not be decoded.
    #[error("invalid request payload: {reason}. Ensure request follows OpenAI API specification.")]
    BadRequest {
        /// Reason describing why request was invalid.
        reason: String,
    },

    /// Requested resource was not found.
    #[error("resource '{resource}' with identifier '{id}' was not found.")]
    NotFound {
        /// Type of resource.
        resource: String,
        /// Identifier of missing resource.
        id: String,
    },

    /// Internal server error when dispatching to engine.
    #[error("internal server failure: {reason}. Check server logs for full traceback.")]
    Internal {
        /// Reason for failure.
        reason: String,
    },

    /// Error originating from the core engine.
    #[error("core engine error: {0}")]
    Core(#[from] kestrel_core::Error),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, code, message) = match &self {
            ServerError::BadRequest { reason } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
                reason.clone(),
            ),
            ServerError::NotFound { resource, id } => (
                StatusCode::NOT_FOUND,
                "not_found_error",
                "not_found",
                format!("{resource} '{id}' not found"),
            ),
            ServerError::Core(kestrel_core::Error::InvalidSessionId { id, reason }) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_session_id",
                format!("invalid session id '{id}': {reason}"),
            ),
            ServerError::Core(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "internal_error",
                err.to_string(),
            ),
            ServerError::Internal { reason } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "internal_server_error",
                reason.clone(),
            ),
            ServerError::BindFailed { addr, reason } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "bind_failed",
                format!("failed to bind to {addr}: {reason}"),
            ),
        };

        let body = ErrorEnvelope {
            error: ApiErrorDetail {
                message,
                error_type: error_type.to_string(),
                code: code.to_string(),
            },
        };

        (status, Json(body)).into_response()
    }
}
