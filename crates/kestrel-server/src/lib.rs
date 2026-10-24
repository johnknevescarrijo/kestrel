#![deny(unsafe_code)]

//! HTTP server layer for Kestrel, providing OpenAI-compatible endpoints.

pub mod error;
pub mod handlers;
pub mod session_handlers;
pub mod types;

use std::sync::Arc;

use axum::http::{header, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::Router;
use kestrel_core::traits::{Backend, SessionStore};

pub use error::ServerError;
pub use handlers::AppState;

/// Middleware to apply permissive CORS headers and respond to OPTIONS preflight.
async fn cors_middleware(req: Request<axum::body::Body>, next: Next) -> Response {
    if req.method() == Method::OPTIONS {
        let mut res = StatusCode::NO_CONTENT.into_response();
        let headers = res.headers_mut();
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_ORIGIN,
            HeaderValue::from_static("*"),
        );
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, POST, DELETE, OPTIONS"),
        );
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("Content-Type, Authorization, X-Kestrel-Session-Id"),
        );
        return res;
    }

    let mut res = next.run(req).await;
    let headers = res.headers_mut();
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, DELETE, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type, Authorization, X-Kestrel-Session-Id"),
    );
    res
}

/// Constructs the Axum application router with OpenAI and Kestrel routes.
pub fn build_router<B: Backend + 'static, S: SessionStore + 'static>(
    state: Arc<AppState<B, S>>,
) -> Router {
    Router::new()
        // OpenAI compatibility endpoints
        .route("/v1/models", get(handlers::list_models::<B, S>))
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions::<B, S>),
        )
        // Kestrel extensions
        .route("/kestrel/sessions", get(handlers::list_sessions::<B, S>))
        .route("/kestrel/sessions/:id", get(handlers::get_session::<B, S>))
        .route(
            "/kestrel/sessions/:id",
            delete(handlers::delete_session::<B, S>),
        )
        .layer(middleware::from_fn(cors_middleware))
        .with_state(state)
}
