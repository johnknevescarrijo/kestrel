use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tempfile::tempdir;
use tokio::sync::{Barrier, Notify};

use kestrel_core::session::{Session, SessionId, SessionMetadata, SessionSnapshot};
use kestrel_core::store::FileSessionStore;
use kestrel_core::tier::{TierLevel, TierPolicy};
use kestrel_core::traits::{
    Backend, CompletionOptions, CompletionResult, MockBackend, SessionStore,
};
use kestrel_core::{Error, SessionManager};

/// Backend wrapper that verifies per-session mutual exclusion and optional cross-session barrier.
struct ConcurrencyTestBackend {
    inner: MockBackend,
    delay: Duration,
    barrier: Option<Arc<Barrier>>,
    active_total: AtomicUsize,
    max_active_total: AtomicUsize,
    per_session_active: std::sync::Mutex<std::collections::HashMap<SessionId, usize>>,
    observed_prior_tokens: std::sync::Mutex<Vec<usize>>,
}

impl ConcurrencyTestBackend {
    fn new(delay: Duration) -> Self {
        Self {
            inner: MockBackend::new("ok"),
            delay,
            barrier: None,
            active_total: AtomicUsize::new(0),
            max_active_total: AtomicUsize::new(0),
            per_session_active: std::sync::Mutex::new(std::collections::HashMap::new()),
            observed_prior_tokens: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn with_barrier(mut self, barrier: Arc<Barrier>) -> Self {
        self.barrier = Some(barrier);
        self
    }

    fn max_active_total(&self) -> usize {
        self.max_active_total.load(Ordering::SeqCst)
    }

    fn observed_prior_tokens(&self) -> Vec<usize> {
        self.observed_prior_tokens.lock().unwrap().clone()
    }
}

#[async_trait]
impl Backend for ConcurrencyTestBackend {
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<String, Error> {
        self.inner.complete_with_options(prompt, options).await
    }

    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        {
            let mut map = self.per_session_active.lock().unwrap();
            let count = map.entry(session.metadata.id.clone()).or_insert(0);
            *count += 1;
            assert_eq!(
                *count, 1,
                "concurrent inference detected on the same session {}",
                session.metadata.id
            );
        }

        self.observed_prior_tokens
            .lock()
            .unwrap()
            .push(session.tokens.len());

        let prev = self.active_total.fetch_add(1, Ordering::SeqCst);
        self.max_active_total.fetch_max(prev + 1, Ordering::SeqCst);

        if let Some(ref barrier) = self.barrier {
            barrier.wait().await;
        }
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        let res = self.inner.complete_session(session, prompt, options).await;

        self.active_total.fetch_sub(1, Ordering::SeqCst);
        {
            let mut map = self.per_session_active.lock().unwrap();
            if let Some(count) = map.get_mut(&session.metadata.id) {
                *count -= 1;
            }
        }
        res
    }
}

/// SessionStore wrapper introducing controllable save latency/gating around `FileSessionStore`.
struct DelayedSessionStore {
    inner: FileSessionStore,
    save_delay: Duration,
    save_entered: AtomicBool,
    release_save: Option<Arc<Notify>>,
    save_count: AtomicUsize,
}

impl DelayedSessionStore {
    fn new(inner: FileSessionStore, save_delay: Duration) -> Self {
        Self {
            inner,
            save_delay,
            save_entered: AtomicBool::new(false),
            release_save: None,
            save_count: AtomicUsize::new(0),
        }
    }

    fn with_gate(mut self, release_save: Arc<Notify>) -> Self {
        self.release_save = Some(release_save);
        self
    }

    fn save_count(&self) -> usize {
        self.save_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl SessionStore for DelayedSessionStore {
    async fn save(&self, snapshot: &SessionSnapshot) -> Result<(), Error> {
        self.save_entered.store(true, Ordering::SeqCst);
        if let Some(ref gate) = self.release_save {
            gate.notified().await;
        }
        if !self.save_delay.is_zero() {
            tokio::time::sleep(self.save_delay).await;
        }
        self.save_count.fetch_add(1, Ordering::SeqCst);
        self.inner.save(snapshot).await
    }

    async fn load(&self, id: &SessionId) -> Result<Option<SessionSnapshot>, Error> {
        self.inner.load(id).await
    }

    async fn delete(&self, id: &SessionId) -> Result<bool, Error> {
        self.inner.delete(id).await
    }

    async fn list(&self) -> Result<Vec<SessionMetadata>, Error> {
        self.inner.list().await
    }
}

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

#[tokio::test]
async fn test_same_session_concurrent_calls_are_serialized_without_lost_update() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let backend = Arc::new(ConcurrencyTestBackend::new(Duration::from_millis(25)));
    let manager = Arc::new(SessionManager::new(backend.clone(), store, 4, "test-model"));

    let sess_id = SessionId::from("shared-session");
    let opts = CompletionOptions::default();

    let m1 = Arc::clone(&manager);
    let s1 = sess_id.clone();
    let o1 = opts.clone();
    let h1 = tokio::spawn(async move { m1.complete(&s1, "turn one alpha", &o1).await });

    let m2 = Arc::clone(&manager);
    let s2 = sess_id.clone();
    let o2 = opts.clone();
    let h2 = tokio::spawn(async move {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        m2.complete_stream(&s2, "turn two beta", &o2, tx).await
    });

    h1.await.unwrap().unwrap();
    h2.await.unwrap().unwrap();

    // Must never have run concurrently on the same session
    assert_eq!(backend.max_active_total(), 1);

    // First call saw 0 prior tokens; second call MUST have observed tokens produced by the first call.
    let priors = backend.observed_prior_tokens();
    assert_eq!(priors.len(), 2);
    assert_eq!(priors[0], 0);
    assert!(
        priors[1] > 0,
        "second concurrent call must observe tokens from first call, got {:?}",
        priors
    );

    let meta = manager.get_metadata(&sess_id).await.unwrap().unwrap();
    assert_eq!(meta.total_tokens, 8); // (3 + 1) + (3 + 1) tokens across both turns
}

#[tokio::test]
async fn test_distinct_sessions_run_concurrently_without_global_lock() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let barrier = Arc::new(Barrier::new(2));
    let backend =
        Arc::new(ConcurrencyTestBackend::new(Duration::from_millis(10)).with_barrier(barrier));
    let manager = Arc::new(SessionManager::new(backend.clone(), store, 4, "test-model"));

    let m1 = Arc::clone(&manager);
    let m2 = Arc::clone(&manager);
    let opts = CompletionOptions::default();

    let fut = async {
        let h1 = tokio::spawn(async move {
            m1.complete(&SessionId::from("sess-x"), "hello x", &opts)
                .await
        });
        let h2 = tokio::spawn(async move {
            m2.complete(
                &SessionId::from("sess-y"),
                "hello y",
                &CompletionOptions::default(),
            )
            .await
        });
        h1.await.unwrap().unwrap();
        h2.await.unwrap().unwrap();
    };

    // If distinct sessions were globally serialized, the 2-party barrier inside complete_session
    // would deadlock and hit this timeout.
    tokio::time::timeout(Duration::from_secs(2), fut)
        .await
        .expect("distinct sessions must run concurrently without deadlock");

    assert_eq!(backend.max_active_total(), 2);
}

#[tokio::test]
async fn test_concurrent_batches_with_opposite_session_orders_do_not_deadlock() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let backend =
        Arc::new(MockBackend::new("batch ok").with_forward_pass_delay(Duration::from_millis(10)));
    let manager = Arc::new(SessionManager::new(backend, store, 8, "test-model"));

    let a = SessionId::from("batch-a");
    let b = SessionId::from("batch-b");

    let mut handles = Vec::new();
    for i in 0..6 {
        let mgr = Arc::clone(&manager);
        let reqs = if i % 2 == 0 {
            vec![
                (a.clone(), format!("p1-{i}"), CompletionOptions::default()),
                (b.clone(), format!("p2-{i}"), CompletionOptions::default()),
            ]
        } else {
            vec![
                (b.clone(), format!("p2-{i}"), CompletionOptions::default()),
                (a.clone(), format!("p1-{i}"), CompletionOptions::default()),
            ]
        };
        handles.push(tokio::spawn(async move { mgr.complete_batch(&reqs).await }));
    }

    tokio::time::timeout(Duration::from_secs(3), async {
        for h in handles {
            h.await.unwrap().unwrap();
        }
    })
    .await
    .expect("concurrent batches in opposite order must not deadlock");

    assert_eq!(
        manager
            .get_metadata(&a)
            .await
            .unwrap()
            .unwrap()
            .total_tokens,
        18
    );
    assert_eq!(
        manager
            .get_metadata(&b)
            .await
            .unwrap()
            .unwrap()
            .total_tokens,
        18
    );
}

#[tokio::test]
async fn test_async_writeback_non_blocking_and_persisted_after_flush() {
    let dir = tempdir().unwrap();
    let file_store = FileSessionStore::new(dir.path()).unwrap();
    let delayed_store = Arc::new(DelayedSessionStore::new(
        file_store,
        Duration::from_millis(80),
    ));
    let backend = Arc::new(MockBackend::new("async reply"));

    let manager = SessionManager::new(backend, delayed_store.clone(), 4, "test-model");
    let sess = SessionId::from("async-wb-sess");

    let start = Instant::now();
    manager
        .complete(
            &sess,
            "Hello async writeback",
            &CompletionOptions::default(),
        )
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(60),
        "commit_session should not block on slow cold storage save (took {elapsed:?})"
    );
    assert_eq!(
        delayed_store.save_count(),
        0,
        "delayed save should still be in flight right after complete returns"
    );

    manager.flush().await.unwrap();
    assert_eq!(delayed_store.save_count(), 1);

    let persisted = delayed_store
        .load(&sess)
        .await
        .unwrap()
        .expect("snapshot must exist on disk after flush");
    assert!(persisted.context_text.contains("Hello async writeback"));
    assert!(persisted.context_text.contains("async reply"));
}

#[tokio::test]
async fn test_delete_session_does_not_resurrect_in_flight_writeback() {
    let dir = tempdir().unwrap();
    let file_store = FileSessionStore::new(dir.path()).unwrap();
    let release_gate = Arc::new(Notify::new());
    let delayed_store = Arc::new(
        DelayedSessionStore::new(file_store, Duration::from_millis(20))
            .with_gate(Arc::clone(&release_gate)),
    );
    let backend = Arc::new(MockBackend::new("ephemeral reply"));

    let manager = Arc::new(SessionManager::new(
        backend,
        delayed_store.clone(),
        4,
        "test-model",
    ));
    let sess = SessionId::from("delete-race-sess");

    // 1. Complete turn 1 -> spawns background write-back that blocks at `release_gate`.
    manager
        .complete(&sess, "Initial turn", &CompletionOptions::default())
        .await
        .unwrap();

    // Ensure the background task has entered `save()` and is awaiting the gate.
    while !delayed_store.save_entered.load(Ordering::SeqCst) {
        tokio::task::yield_now().await;
    }

    // 2. Start delete_session while write-back is in flight.
    let mgr_del = Arc::clone(&manager);
    let sess_del = sess.clone();
    let del_handle = tokio::spawn(async move { mgr_del.delete_session(&sess_del).await });

    // Give delete_session a moment to reach `cancel_and_wait`, then release the gate.
    tokio::time::sleep(Duration::from_millis(15)).await;
    release_gate.notify_one();

    let deleted = del_handle.await.unwrap().unwrap();
    assert!(deleted, "delete_session must report true");

    manager.flush().await.unwrap();

    assert_eq!(
        manager.session_tier(&sess).await.unwrap(),
        None,
        "deleted session must not reappear in any tier"
    );
    assert_eq!(
        delayed_store.load(&sess).await.unwrap(),
        None,
        "deleted session must not be resurrected on disk by in-flight write-back"
    );
}

#[tokio::test]
async fn test_cold_eviction_guarantees_persistence_without_flush() {
    let dir = tempdir().unwrap();
    let file_store = FileSessionStore::new(dir.path()).unwrap();
    let delayed_store = Arc::new(DelayedSessionStore::new(
        file_store,
        Duration::from_millis(30),
    ));
    let backend = Arc::new(MockBackend::new("evict reply"));

    // 1 hot, 1 warm -> 3rd session forces s1 from warm to cold synchronously.
    let policy = TierPolicy::new(1, 1, 10 * 1024 * 1024, 10 * 1024 * 1024);
    let manager = SessionManager::with_policy(backend, delayed_store.clone(), policy, "test-model");

    let s1 = SessionId::from("evict-s1");
    let s2 = SessionId::from("evict-s2");
    let s3 = SessionId::from("evict-s3");
    let opts = CompletionOptions::default();

    manager.complete(&s1, "one", &opts).await.unwrap();
    manager.complete(&s2, "two", &opts).await.unwrap();
    manager.complete(&s3, "three", &opts).await.unwrap();

    // Without calling flush(), s1 MUST already be durably persisted on disk.
    let loaded = delayed_store.load(&s1).await.unwrap();
    assert!(
        loaded.is_some(),
        "warm->cold eviction must guarantee persistence before returning"
    );
}
