//! Integration tests for Kestrel HTTP server and OpenAI compatibility endpoints.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use kestrel_core::store::FileSessionStore;
use kestrel_core::traits::MockBackend;
use kestrel_core::SessionManager;
use kestrel_server::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    DeleteSessionResponse, ModelListResponse, SessionsListResponse,
};
use kestrel_server::{build_router, AppState};
use serde_json::Value;
use tempfile::tempdir;
use tower::ServiceExt;

/// Creates a test Axum router backed by a MockBackend and temporary file store.
async fn setup_test_router() -> (axum::Router, tempfile::TempDir) {
    let dir = tempdir().expect("Failed to create tempdir");
    let store =
        Arc::new(FileSessionStore::new(dir.path()).expect("Failed to create session store"));
    let backend = Arc::new(MockBackend::new("I am a test assistant."));
    let manager = Arc::new(SessionManager::new(backend, store, 4, "mock-qwen"));
    let state = Arc::new(AppState::new(manager, "mock-qwen"));
    let router = build_router(state);
    (router, dir)
}

#[tokio::test]
async fn test_get_models_endpoint() {
    let (router, _dir) = setup_test_router().await;

    let request = Request::builder()
        .uri("/v1/models")
        .method(Method::GET)
        .body(Body::empty())
        .unwrap();

    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let models_resp: ModelListResponse = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(models_resp.object, "list");
    assert_eq!(models_resp.data.len(), 1);
    assert_eq!(models_resp.data[0].id, "mock-qwen");
    assert_eq!(models_resp.data[0].object, "model");
}

#[tokio::test]
async fn test_chat_completions_non_streaming_and_prefix_reuse() {
    let (router, _dir) = setup_test_router().await;

    let payload1 = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![ChatMessage::user("What is your name?")],
        max_tokens: Some(64),
        stream: Some(false),
        ..Default::default()
    };

    let req1 = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-kestrel-session-id", "test-session-prefix")
        .body(Body::from(serde_json::to_vec(&payload1).unwrap()))
        .unwrap();

    let res1 = router.clone().oneshot(req1).await.unwrap();
    assert_eq!(res1.status(), StatusCode::OK);

    let body1 = res1.into_body().collect().await.unwrap().to_bytes();
    let comp1: ChatCompletionResponse = serde_json::from_slice(&body1).unwrap();
    assert_eq!(comp1.choices[0].message.content, "I am a test assistant.");
    assert!(comp1.usage.prompt_tokens > 0);
    let first_prompt_tokens = comp1.usage.prompt_tokens;

    // Second call with expanded context: reuses session prefix
    let payload2 = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![
            ChatMessage::user("What is your name?"),
            ChatMessage::assistant("I am a test assistant."),
            ChatMessage::user("Tell me more."),
        ],
        max_tokens: Some(64),
        stream: Some(false),
        ..Default::default()
    };

    let req2 = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-kestrel-session-id", "test-session-prefix")
        .body(Body::from(serde_json::to_vec(&payload2).unwrap()))
        .unwrap();

    let res2 = router.oneshot(req2).await.unwrap();
    assert_eq!(res2.status(), StatusCode::OK);

    let body2 = res2.into_body().collect().await.unwrap().to_bytes();
    let comp2: ChatCompletionResponse = serde_json::from_slice(&body2).unwrap();
    assert_eq!(comp2.choices[0].message.content, "I am a test assistant.");
    // Prefill tokens processed on the 2nd request should only be the new turn suffix
    assert!(
        comp2.usage.prompt_tokens <= first_prompt_tokens,
        "Expected prompt tokens on 2nd turn ({}) to be <= first turn ({}) due to prefix reuse",
        comp2.usage.prompt_tokens,
        first_prompt_tokens
    );
}

#[tokio::test]
async fn test_chat_completions_streaming_sse() {
    let (router, _dir) = setup_test_router().await;

    let payload = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![ChatMessage::user("Stream this response")],
        max_tokens: Some(64),
        stream: Some(true),
        session_id: Some("stream-session-1".to_string()),
        ..Default::default()
    };

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

    // Verify SSE lines format
    let lines: Vec<&str> = body_str.lines().filter(|l| !l.is_empty()).collect();
    assert!(lines.len() >= 3);

    // Verify first data chunk
    let first_line = lines[0];
    assert!(first_line.starts_with("data: "));
    let first_chunk: ChatCompletionChunk =
        serde_json::from_str(&first_line["data: ".len()..]).unwrap();
    assert_eq!(first_chunk.object, "chat.completion.chunk");
    assert_eq!(
        first_chunk.choices[0].delta.role.as_deref(),
        Some("assistant")
    );

    // Verify last non-done chunk has finish_reason: "stop"
    let penult_line = lines[lines.len() - 2];
    assert!(penult_line.starts_with("data: "));
    let last_chunk: ChatCompletionChunk =
        serde_json::from_str(&penult_line["data: ".len()..]).unwrap();
    assert_eq!(last_chunk.choices[0].finish_reason.as_deref(), Some("stop"));

    // Verify final message is data: [DONE]
    assert_eq!(lines.last().copied(), Some("data: [DONE]"));
}

#[tokio::test]
async fn test_chat_completions_error_handling() {
    let (router, _dir) = setup_test_router().await;

    // 1. Empty messages array
    let empty_payload = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![],
        ..Default::default()
    };

    let req1 = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&empty_payload).unwrap()))
        .unwrap();

    let res1 = router.clone().oneshot(req1).await.unwrap();
    assert_eq!(res1.status(), StatusCode::BAD_REQUEST);

    let body1 = res1.into_body().collect().await.unwrap().to_bytes();
    let json1: Value = serde_json::from_slice(&body1).unwrap();
    assert_eq!(json1["error"]["type"], "invalid_request_error");
    assert_eq!(json1["error"]["code"], "bad_request");

    // 2. Malicious session ID with path traversal
    let valid_payload = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![ChatMessage::user("Hello")],
        ..Default::default()
    };

    let req2 = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-kestrel-session-id", "../malicious/id")
        .body(Body::from(serde_json::to_vec(&valid_payload).unwrap()))
        .unwrap();

    let res2 = router.oneshot(req2).await.unwrap();
    assert_eq!(res2.status(), StatusCode::BAD_REQUEST);

    let body2 = res2.into_body().collect().await.unwrap().to_bytes();
    let json2: Value = serde_json::from_slice(&body2).unwrap();
    assert_eq!(json2["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_kestrel_sessions_endpoints_crud() {
    let (router, _dir) = setup_test_router().await;

    // 1. Populate a session via completion
    let payload = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![ChatMessage::user("Initiate session")],
        max_tokens: Some(32),
        stream: Some(false),
        session_id: Some("crud-session-1".to_string()),
        ..Default::default()
    };

    let init_req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let init_res = router.clone().oneshot(init_req).await.unwrap();
    assert_eq!(init_res.status(), StatusCode::OK);

    // 2. List sessions
    let list_req = Request::builder()
        .uri("/kestrel/sessions")
        .method(Method::GET)
        .body(Body::empty())
        .unwrap();

    let list_res = router.clone().oneshot(list_req).await.unwrap();
    assert_eq!(list_res.status(), StatusCode::OK);

    let list_bytes = list_res.into_body().collect().await.unwrap().to_bytes();
    let list_resp: SessionsListResponse = serde_json::from_slice(&list_bytes).unwrap();
    assert_eq!(list_resp.stats.total_sessions, 1);
    assert_eq!(list_resp.stats.hot_sessions, 1);
    assert_eq!(list_resp.sessions[0].id.as_str(), "crud-session-1");

    // 3. Get session metadata
    let get_req = Request::builder()
        .uri("/kestrel/sessions/crud-session-1")
        .method(Method::GET)
        .body(Body::empty())
        .unwrap();

    let get_res = router.clone().oneshot(get_req).await.unwrap();
    assert_eq!(get_res.status(), StatusCode::OK);

    // 4. Get non-existent session
    let get_missing_req = Request::builder()
        .uri("/kestrel/sessions/non-existent-id")
        .method(Method::GET)
        .body(Body::empty())
        .unwrap();

    let get_missing_res = router.clone().oneshot(get_missing_req).await.unwrap();
    assert_eq!(get_missing_res.status(), StatusCode::NOT_FOUND);

    // 5. Delete session
    let del_req = Request::builder()
        .uri("/kestrel/sessions/crud-session-1")
        .method(Method::DELETE)
        .body(Body::empty())
        .unwrap();

    let del_res = router.clone().oneshot(del_req).await.unwrap();
    assert_eq!(del_res.status(), StatusCode::OK);

    let del_bytes = del_res.into_body().collect().await.unwrap().to_bytes();
    let del_resp: DeleteSessionResponse = serde_json::from_slice(&del_bytes).unwrap();
    assert!(del_resp.deleted);
    assert_eq!(del_resp.id, "crud-session-1");

    // 6. Delete again returns 404
    let del_again_req = Request::builder()
        .uri("/kestrel/sessions/crud-session-1")
        .method(Method::DELETE)
        .body(Body::empty())
        .unwrap();

    let del_again_res = router.oneshot(del_again_req).await.unwrap();
    assert_eq!(del_again_res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_cors_preflight_options() {
    let (router, _dir) = setup_test_router().await;

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::OPTIONS)
        .body(Body::empty())
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        res.headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .unwrap()
            .to_str()
            .unwrap(),
        "*"
    );
    assert!(res
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_METHODS)
        .unwrap()
        .to_str()
        .unwrap()
        .contains("POST"));
}
