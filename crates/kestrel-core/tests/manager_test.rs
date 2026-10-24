use std::sync::Arc;
use tempfile::tempdir;

use kestrel_core::session::SessionId;
use kestrel_core::store::FileSessionStore;
use kestrel_core::tier::{TierLevel, TierPolicy};
use kestrel_core::traits::{CompletionOptions, MockBackend};
use kestrel_core::SessionManager;

#[tokio::test]
async fn test_manager_complete_lifecycle_and_eviction() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let backend = Arc::new(MockBackend::new("mock answer"));

    let policy = TierPolicy::new(1, 2, 1024 * 1024, 2 * 1024 * 1024);
    let manager = SessionManager::with_policy(backend, store.clone(), policy, "test-model");

    let sess_a = SessionId::from("sess-a");
    let sess_b = SessionId::from("sess-b");

    let res1 = manager
        .complete(&sess_a, "Hello A", &CompletionOptions::default())
        .await
        .unwrap();
    assert_eq!(res1.text, "mock answer");
    assert_eq!(manager.hot_session_count().await, 1);
    assert_eq!(manager.warm_session_count().await, 0);

    let res2 = manager
        .complete(&sess_b, "Hello B", &CompletionOptions::default())
        .await
        .unwrap();
    assert_eq!(res2.text, "mock answer");
    assert_eq!(manager.hot_session_count().await, 1);
    assert_eq!(manager.warm_session_count().await, 1);
    assert_eq!(
        manager.session_tier(&sess_a).await.unwrap(),
        Some(TierLevel::Warm)
    );

    let res3 = manager
        .complete(&sess_a, "Continue A", &CompletionOptions::default())
        .await
        .unwrap();
    assert_eq!(res3.text, "mock answer");
    assert_eq!(
        manager.session_tier(&sess_a).await.unwrap(),
        Some(TierLevel::Hot)
    );

    let metrics = manager.metrics_snapshot().await;
    assert_eq!(metrics.warm_hits, 1);
    assert_eq!(metrics.promotions_warm_to_hot, 1);
    assert_eq!(metrics.demotions_hot_to_warm, 2);
}

#[tokio::test]
async fn test_manager_complete_batch() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let backend = Arc::new(MockBackend::new("mock batch reply"));

    let policy = TierPolicy::new(4, 8, 10 * 1024 * 1024, 20 * 1024 * 1024);
    let manager = SessionManager::with_policy(backend, store.clone(), policy, "test-model");

    let requests = vec![
        (
            SessionId::from("sess-1"),
            "Hello 1".to_string(),
            CompletionOptions::default(),
        ),
        (
            SessionId::from("sess-2"),
            "Hello 2".to_string(),
            CompletionOptions::default(),
        ),
    ];

    let results = manager.complete_batch(&requests).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].text, "mock batch reply");
    assert_eq!(results[1].text, "mock batch reply");
    assert_eq!(manager.hot_session_count().await, 2);
}
