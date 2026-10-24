use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use kestrel_core::session::SessionId;
use kestrel_core::store::FileSessionStore;
use kestrel_core::traits::{CompletionOptions, MockBackend};
use kestrel_core::SessionManager;
use tempfile::tempdir;
use tokio::runtime::Runtime;

fn bench_session_prefill_reuse(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let backend = Arc::new(MockBackend::new("mock response"));
    let manager = SessionManager::new(backend, store, 10, "mock-model");

    let prompt_prefix = "System: You are an intelligent agent.\nContext: Here is a long background text with multiple sentences simulating agent memory and context.";
    let options = CompletionOptions::default();

    let mut group = c.benchmark_group("session_prefill");
    group.measurement_time(Duration::from_secs(3));

    // First call: cold/fresh prompt (full prefill)
    group.bench_function("fresh_session_prefill", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter += 1;
            let sess_id = SessionId::from(format!("fresh-sess-{counter}"));
            let manager = &manager;
            let options = &options;
            rt.block_on(async move {
                let res = manager
                    .complete(&sess_id, prompt_prefix, options)
                    .await
                    .unwrap();
                assert!(res.prefill_tokens > 0);
            });
        });
    });

    // Second call: reused session with continuation (reduced delta prefill)
    let persistent_sess = SessionId::from("persistent-sess");
    rt.block_on(async {
        manager
            .complete(&persistent_sess, prompt_prefix, &options)
            .await
            .unwrap();
    });

    group.bench_function("reused_session_prefill", |b| {
        b.iter(|| {
            let manager = &manager;
            let options = &options;
            let sess_id = &persistent_sess;
            rt.block_on(async move {
                let res = manager
                    .complete(sess_id, "User: Next action?", options)
                    .await
                    .unwrap();
                assert_eq!(res.prefill_tokens, 3);
            });
        });
    });

    group.finish();
}

criterion_group!(benches, bench_session_prefill_reuse);
criterion_main!(benches);
