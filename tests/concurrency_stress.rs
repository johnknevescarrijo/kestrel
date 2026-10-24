#![deny(unsafe_code)]

//! Concurrency stress tests and M5 continuous batching throughput verification.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use kestrel_core::scheduler::{BatchConfig, BatchScheduler};
use kestrel_core::session::SessionId;
use kestrel_core::store::FileSessionStore;
use kestrel_core::tier::{TierMetricsSnapshot, TierPolicy};
use kestrel_core::traits::{CompletionOptions, MockBackend};
use kestrel_core::SessionManager;
use kestrel_server::types::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage};
use kestrel_server::{build_router, AppState};
use tempfile::tempdir;
use tokio::sync::Barrier;
use tower::ServiceExt;

#[tokio::test]
async fn test_50_concurrent_clients_batch_scheduler() {
    let dir = tempdir().expect("Failed to create tempdir");
    let store = Arc::new(FileSessionStore::new(dir.path()).expect("Failed to create store"));
    let backend = Arc::new(
        MockBackend::new("mock answer from batch")
            .with_forward_pass_delay(Duration::from_millis(5)),
    );

    let policy = TierPolicy::new(64, 128, 50 * 1024 * 1024, 100 * 1024 * 1024);
    let manager = Arc::new(SessionManager::with_policy(
        backend.clone(),
        store,
        policy,
        "test-model",
    ));

    let batch_config = BatchConfig {
        max_batch_size: 16,
        batch_window: Duration::from_millis(15),
    };
    let scheduler = Arc::new(BatchScheduler::new(manager, batch_config));

    let barrier = Arc::new(Barrier::new(50));
    let mut tasks = Vec::with_capacity(50);

    for i in 0..50 {
        let b = Arc::clone(&barrier);
        let sched = Arc::clone(&scheduler);
        let session_id = SessionId::from(format!("session-{i}"));

        tasks.push(tokio::spawn(async move {
            b.wait().await;
            sched
                .submit(
                    session_id,
                    format!("User prompt turn for client {i}"),
                    CompletionOptions::default(),
                )
                .await
        }));
    }

    let mut results = Vec::with_capacity(50);
    for task in tasks {
        let res = task
            .await
            .expect("Task panicked")
            .expect("Inference failed");
        results.push(res);
    }

    assert_eq!(results.len(), 50);
    for r in &results {
        assert_eq!(r.text, "mock answer from batch");
    }

    let metrics = scheduler.metrics_snapshot();
    assert_eq!(metrics.total_batched_requests, 50);
    assert!(
        metrics.max_observed_batch_size > 1,
        "Expected batched requests > 1, got {}",
        metrics.max_observed_batch_size
    );
    assert!(
        backend.batch_call_count() < 20,
        "Expected fewer than 20 batch executions for 50 requests, got {}",
        backend.batch_call_count()
    );
}

#[tokio::test]
async fn test_50_concurrent_http_clients_with_tiering() {
    let dir = tempdir().expect("Failed to create tempdir");
    let store = Arc::new(FileSessionStore::new(dir.path()).expect("Failed to create store"));
    let backend = Arc::new(
        MockBackend::new("mock http answer")
            .with_fixed_kv_size(512 * 1024)
            .with_forward_pass_delay(Duration::from_millis(2)),
    );

    // Restrict hot tier to 4 sessions (2 MB) and warm tier to 8 sessions (4 MB)
    // to trigger cascades and demotions with 50 distinct sessions.
    let policy = TierPolicy::new(4, 8, 2 * 1024 * 1024, 4 * 1024 * 1024);
    let manager = Arc::new(SessionManager::with_policy(
        backend.clone(),
        store,
        policy,
        "stress-model",
    ));

    let batch_config = BatchConfig {
        max_batch_size: 16,
        batch_window: Duration::from_millis(5),
    };
    let state = Arc::new(AppState::with_batch_config(
        manager,
        "stress-model",
        batch_config,
    ));
    let router = build_router(state);

    let barrier = Arc::new(Barrier::new(50));
    let mut tasks = Vec::with_capacity(50);

    for i in 0..50 {
        let b = Arc::clone(&barrier);
        let app = router.clone();
        let session_header = format!("client-session-{i}");

        tasks.push(tokio::spawn(async move {
            b.wait().await;
            let payload = ChatCompletionRequest {
                model: "stress-model".to_string(),
                messages: vec![ChatMessage::user(format!("Hello from client {i}"))],
                stream: Some(false),
                max_tokens: Some(64),
                ..Default::default()
            };

            let req = Request::builder()
                .uri("/v1/chat/completions")
                .method(Method::POST)
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-kestrel-session-id", session_header)
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap();

            app.oneshot(req).await
        }));
    }

    let mut successful_responses = 0;
    for task in tasks {
        let response = task
            .await
            .expect("Task panicked")
            .expect("HTTP dispatch failed");
        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
        let completion: ChatCompletionResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(completion.choices[0].message.content, "mock http answer");
        successful_responses += 1;
    }

    assert_eq!(successful_responses, 50);

    // Verify system metrics
    let metrics_req = Request::builder()
        .uri("/kestrel/metrics")
        .method(Method::GET)
        .body(Body::empty())
        .unwrap();

    let metrics_resp = router.oneshot(metrics_req).await.unwrap();
    assert_eq!(metrics_resp.status(), StatusCode::OK);

    let body_bytes = metrics_resp.into_body().collect().await.unwrap().to_bytes();
    let metrics: TierMetricsSnapshot = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(metrics.total_requests, 50);
    assert!(
        metrics.demotions_hot_to_warm > 0,
        "Expected hot to warm demotions under stress"
    );
    assert!(
        metrics.demotions_warm_to_cold > 0,
        "Expected warm to cold demotions under stress"
    );

    let batch_m = metrics
        .batch_metrics
        .expect("Missing batch metrics in response");
    assert_eq!(batch_m.total_batched_requests, 50);
    assert!(batch_m.max_observed_batch_size > 1);
}

#[tokio::test]
async fn test_m5_throughput_at_least_2x_m4() {
    let forward_delay = Duration::from_millis(4);
    let request_count = 32;

    // 1. Measure M4 sequential / non-batched throughput
    let dir_m4 = tempdir().expect("Failed to create tempdir");
    let store_m4 = Arc::new(FileSessionStore::new(dir_m4.path()).unwrap());
    let backend_m4 = Arc::new(MockBackend::new("answer").with_forward_pass_delay(forward_delay));
    let policy_m4 = TierPolicy::new(64, 128, 50 * 1024 * 1024, 100 * 1024 * 1024);
    let manager_m4 = Arc::new(SessionManager::with_policy(
        backend_m4,
        store_m4,
        policy_m4,
        "bench-model",
    ));

    let start_m4 = Instant::now();
    for i in 0..request_count {
        let sid = SessionId::from(format!("m4-sess-{i}"));
        manager_m4
            .complete(&sid, "test prompt", &CompletionOptions::default())
            .await
            .unwrap();
    }
    let elapsed_m4 = start_m4.elapsed();
    let m4_throughput = request_count as f64 / elapsed_m4.as_secs_f64();

    // 2. Measure M5 continuous batched throughput
    let dir_m5 = tempdir().expect("Failed to create tempdir");
    let store_m5 = Arc::new(FileSessionStore::new(dir_m5.path()).unwrap());
    let backend_m5 = Arc::new(MockBackend::new("answer").with_forward_pass_delay(forward_delay));
    let policy_m5 = TierPolicy::new(64, 128, 50 * 1024 * 1024, 100 * 1024 * 1024);
    let manager_m5 = Arc::new(SessionManager::with_policy(
        backend_m5,
        store_m5,
        policy_m5,
        "bench-model",
    ));

    let batch_config = BatchConfig {
        max_batch_size: 16,
        batch_window: Duration::from_millis(2),
    };
    let scheduler_m5 = Arc::new(BatchScheduler::new(manager_m5, batch_config));

    let barrier = Arc::new(Barrier::new(request_count));
    let mut tasks = Vec::with_capacity(request_count);

    let start_m5 = Instant::now();
    for i in 0..request_count {
        let b = Arc::clone(&barrier);
        let sched = Arc::clone(&scheduler_m5);
        let sid = SessionId::from(format!("m5-sess-{i}"));

        tasks.push(tokio::spawn(async move {
            b.wait().await;
            sched
                .submit(sid, "test prompt".to_string(), CompletionOptions::default())
                .await
        }));
    }

    for task in tasks {
        task.await.unwrap().unwrap();
    }
    let elapsed_m5 = start_m5.elapsed();
    let m5_throughput = request_count as f64 / elapsed_m5.as_secs_f64();

    println!(
        "M4 throughput: {m4_throughput:.1} req/s (elapsed: {elapsed_m4:?}) | \
         M5 throughput: {m5_throughput:.1} req/s (elapsed: {elapsed_m5:?})"
    );

    assert!(
        m5_throughput >= 2.0 * m4_throughput,
        "M5 throughput ({m5_throughput:.1} req/s) must be >= 2x M4 throughput ({m4_throughput:.1} req/s)"
    );
}
