//! Integration tests for post-v1.0 features:
//! - POST /v1/embeddings endpoint (single, multiple, custom dimensions, validation)
//! - Structured Output (JSON Mode and JSON Schema validation with corrective retries)
//! - Multi-Model Routing with ModelRouter and HTTP 404 for missing models
//! - MlxBackend execution, prefix KV reuse, and vector embeddings

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use kestrel_core::session::Session;
use kestrel_core::store::FileSessionStore;
use kestrel_core::traits::{
    compute_deterministic_embedding, Backend, CompletionOptions, CompletionResult, EmbeddingResult,
};
use kestrel_core::{Error, ModelRouter, SessionId, SessionManager};
use kestrel_llama::MlxBackend;
use kestrel_server::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, EmbeddingResponse,
    JsonSchemaConfig, ModelListResponse, ResponseFormat,
};
use kestrel_server::{build_router, AppState};
use serde_json::json;
use tempfile::tempdir;
use tower::ServiceExt;

/// Mock backend with sequential queued responses.
struct QueueMockBackend {
    responses: Arc<Mutex<VecDeque<String>>>,
    call_count: Arc<AtomicUsize>,
    model_name: String,
}

impl QueueMockBackend {
    fn new(model_name: impl Into<String>, responses: Vec<String>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
            call_count: Arc::new(AtomicUsize::new(0)),
            model_name: model_name.into(),
        }
    }

    fn pop_response(&self) -> String {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut queue = self.responses.lock().unwrap();
        queue.pop_front().unwrap_or_else(|| "{}".to_string())
    }
}

#[async_trait]
impl Backend for QueueMockBackend {
    async fn complete_with_options(
        &self,
        _prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<String, Error> {
        Ok(self.pop_response())
    }

    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        let answer = self.pop_response();
        let prefill_tokens =
            if !session.context_text.is_empty() && prompt.starts_with(&session.context_text) {
                let delta = &prompt[session.context_text.len()..];
                delta.split_whitespace().count().max(1)
            } else {
                prompt.split_whitespace().count().max(1)
            };
        session.context_text = format!("{}{}", prompt, answer);
        let gen_tokens = answer
            .split_whitespace()
            .count()
            .max(1)
            .min(options.max_tokens);
        Ok(CompletionResult {
            text: answer,
            prefill_tokens,
            generated_tokens: gen_tokens,
        })
    }

    async fn embed(&self, input: &str) -> Result<EmbeddingResult, Error> {
        self.embed_for_model(&self.model_name, input).await
    }

    async fn embed_for_model(&self, _model: &str, input: &str) -> Result<EmbeddingResult, Error> {
        Ok(compute_deterministic_embedding(input, 128))
    }
}

#[tokio::test]
async fn test_embeddings_endpoint_variations() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let backend = Arc::new(QueueMockBackend::new("mock-embed", vec![]));
    let manager = Arc::new(SessionManager::new(backend, store, 4, "mock-embed"));
    let state = Arc::new(AppState::new(manager, "mock-embed"));
    let router = build_router(state);

    // 1. Single string input
    let payload = json!({
        "model": "mock-embed",
        "input": "Machine learning with local embeddings"
    });
    let req = Request::builder()
        .uri("/v1/embeddings")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let emb_res: EmbeddingResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(emb_res.object, "list");
    assert_eq!(emb_res.data.len(), 1);
    assert_eq!(emb_res.data[0].index, 0);
    assert_eq!(emb_res.data[0].embedding.len(), 128); // default mock dimension
    assert!(emb_res.usage.prompt_tokens > 0);

    // 2. Multiple strings input
    let payload_multi = json!({
        "model": "mock-embed",
        "input": ["First sentence", "Second sentence"]
    });
    let req = Request::builder()
        .uri("/v1/embeddings")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload_multi).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let emb_res: EmbeddingResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(emb_res.data.len(), 2);
    assert_eq!(emb_res.data[0].index, 0);
    assert_eq!(emb_res.data[1].index, 1);

    // 3. Custom dimensions override
    let payload_dim = json!({
        "model": "mock-embed",
        "input": "Vector resizing test",
        "dimensions": 64
    });
    let req = Request::builder()
        .uri("/v1/embeddings")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload_dim).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let emb_res: EmbeddingResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(emb_res.data[0].embedding.len(), 64);

    // 4. Empty input error validation (400)
    let payload_empty = json!({
        "model": "mock-embed",
        "input": ""
    });
    let req = Request::builder()
        .uri("/v1/embeddings")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload_empty).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // 5. Zero dimensions error validation (400)
    let payload_zero_dim = json!({
        "model": "mock-embed",
        "input": "test",
        "dimensions": 0
    });
    let req = Request::builder()
        .uri("/v1/embeddings")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload_zero_dim).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_structured_output_json_object_and_json_schema() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());

    // Prepare responses:
    // 1: valid json object
    // 2: invalid json followed by valid json object (recovery test)
    // 3: valid schema conforming json
    // 4: schema violating json repeated to test retry exhaustion
    let responses = vec![
        "{\"message\": \"success\", \"code\": 200}".to_string(),
        "Not a JSON!".to_string(),
        "{\"recovered\": true}".to_string(),
        "{\"name\": \"Alice\", \"age\": 30}".to_string(),
        "{\"name\": \"Bob\", \"age\": \"not-a-number\"}".to_string(),
        "{\"name\": \"Bob\", \"age\": \"still-not-a-number\"}".to_string(),
    ];
    let backend = Arc::new(QueueMockBackend::new("mock-struct", responses));
    let manager = Arc::new(SessionManager::new(backend, store, 4, "mock-struct"));
    let state = Arc::new(AppState::new(manager, "mock-struct"));
    let router = build_router(state);

    // 1. JSON object mode success
    let req1 = ChatCompletionRequest {
        model: "mock-struct".to_string(),
        messages: vec![ChatMessage::user("Output json object")],
        response_format: Some(ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        }),
        ..Default::default()
    };
    let http_req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&req1).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(http_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let chat_res: ChatCompletionResponse = serde_json::from_slice(&bytes).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&chat_res.choices[0].message.content).unwrap();
    assert_eq!(parsed["code"], 200);

    // 2. JSON object mode with retry recovery
    let req2 = ChatCompletionRequest {
        model: "mock-struct".to_string(),
        messages: vec![ChatMessage::user("Output recovered json")],
        response_format: Some(ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        }),
        max_tool_retries: Some(2),
        ..Default::default()
    };
    let http_req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&req2).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(http_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let chat_res: ChatCompletionResponse = serde_json::from_slice(&bytes).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&chat_res.choices[0].message.content).unwrap();
    assert_eq!(parsed["recovered"], true);

    // 3. JSON Schema conforming output
    let user_schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });

    let req3 = ChatCompletionRequest {
        model: "mock-struct".to_string(),
        messages: vec![ChatMessage::user("Give me user data")],
        response_format: Some(ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(JsonSchemaConfig {
                name: "user_info".to_string(),
                description: Some("User profile information".to_string()),
                schema: Some(user_schema.clone()),
                strict: Some(true),
            }),
        }),
        ..Default::default()
    };
    let http_req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&req3).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(http_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let chat_res: ChatCompletionResponse = serde_json::from_slice(&bytes).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&chat_res.choices[0].message.content).unwrap();
    assert_eq!(parsed["name"], "Alice");
    assert_eq!(parsed["age"], 30);

    // 4. JSON Schema validation retry exhaustion returns 422
    let req4 = ChatCompletionRequest {
        model: "mock-struct".to_string(),
        messages: vec![ChatMessage::user("Invalid schema input")],
        response_format: Some(ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(JsonSchemaConfig {
                name: "user_info".to_string(),
                description: None,
                schema: Some(user_schema),
                strict: Some(true),
            }),
        }),
        max_tool_retries: Some(1),
        ..Default::default()
    };
    let http_req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&req4).unwrap()))
        .unwrap();
    let resp = router.oneshot(http_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_multi_model_router_and_not_found() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());

    let backend_qwen = Arc::new(QueueMockBackend::new(
        "qwen-model",
        vec!["Response from Qwen".to_string()],
    ));
    let backend_llama = Arc::new(QueueMockBackend::new(
        "llama-model",
        vec!["Response from Llama".to_string()],
    ));

    let mut router_backend = ModelRouter::new("qwen-model", backend_qwen);
    router_backend.register("llama-model", backend_llama);
    let model_ids = router_backend.model_ids();
    let router_backend = Arc::new(router_backend);

    let manager = Arc::new(SessionManager::new(router_backend, store, 4, "qwen-model"));
    let state = Arc::new(AppState::new(manager, "qwen-model").with_models(model_ids));
    let app_router = build_router(state);

    // 1. GET /v1/models returns both models
    let req = Request::builder()
        .uri("/v1/models")
        .method(Method::GET)
        .body(Body::empty())
        .unwrap();
    let resp = app_router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let list_res: ModelListResponse = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<String> = list_res.data.into_iter().map(|m| m.id).collect();
    assert!(ids.contains(&"qwen-model".to_string()));
    assert!(ids.contains(&"llama-model".to_string()));

    // 2. Chat with model = "llama-model"
    let req_llama = ChatCompletionRequest {
        model: "llama-model".to_string(),
        messages: vec![ChatMessage::user("Hi Llama")],
        ..Default::default()
    };
    let http_req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&req_llama).unwrap()))
        .unwrap();
    let resp = app_router.clone().oneshot(http_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let chat_res: ChatCompletionResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(chat_res.choices[0].message.content, "Response from Llama");
    assert_eq!(chat_res.model, "llama-model");

    // 3. Chat with unknown model -> 404 Model Not Found
    let req_unknown = ChatCompletionRequest {
        model: "non-existent-model".to_string(),
        messages: vec![ChatMessage::user("Hello")],
        ..Default::default()
    };
    let http_req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&req_unknown).unwrap()))
        .unwrap();
    let resp = app_router.oneshot(http_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_mlx_backend_prefix_reuse_and_embeddings() {
    let dir = tempdir().unwrap();
    let fake_mlx_model = dir.path().join("model.mlx");
    std::fs::write(&fake_mlx_model, b"mlx_weights_v1").unwrap();

    let backend = MlxBackend::load_from_file(&fake_mlx_model).unwrap();

    // 1. Session execution and prefix reuse
    let mut session = Session::new(SessionId::from("mlx-sess-1"), "model.mlx");
    let opts = CompletionOptions { max_tokens: 32 };

    let res1 = backend
        .complete_session(&mut session, "System: Assistant\nUser: Hello", &opts)
        .await
        .unwrap();
    assert!(res1.prefill_tokens > 0);
    assert!(res1.generated_tokens > 0);

    let res2 = backend
        .complete_session(
            &mut session,
            "System: Assistant\nUser: Hello\nAssistant: Hi\nUser: How are you?",
            &opts,
        )
        .await
        .unwrap();
    // Second turn reuses the prefix
    assert!(
        res2.prefill_tokens
            < "System: Assistant\nUser: Hello\nAssistant: Hi\nUser: How are you?".len()
    );

    // 2. Vector embedding generation
    let emb = backend.embed("Test embedding text for MLX").await.unwrap();
    assert_eq!(emb.embedding.len(), 64);
    assert!(emb.prompt_tokens > 0);
    let l2_norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((l2_norm - 1.0).abs() < 1e-3);
}
