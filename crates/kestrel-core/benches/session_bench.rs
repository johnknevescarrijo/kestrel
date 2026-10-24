use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use kestrel_core::scheduler::{BatchConfig, BatchScheduler};
use kestrel_core::session::SessionId;
use kestrel_core::store::FileSessionStore;
use kestrel_core::tier::TierPolicy;
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

fn bench_m4_vs_m5_batch_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let delay = Duration::from_millis(2);

    let mut group = c.benchmark_group("m4_vs_m5_batch_throughput");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    // M4 baseline: sequential forward pass per session
    let backend_m4 = Arc::new(MockBackend::new("response").with_forward_pass_delay(delay));
    let policy_m4 = TierPolicy::new(32, 64, 20 * 1024 * 1024, 40 * 1024 * 1024);
    let manager_m4 = Arc::new(SessionManager::with_policy(
        backend_m4,
        store.clone(),
        policy_m4,
        "bench-model",
    ));

    group.bench_function("m4_sequential_inference_16_clients", |b| {
        let options = CompletionOptions::default();
        b.iter(|| {
            let manager = Arc::clone(&manager_m4);
            let opt = options.clone();
            rt.block_on(async move {
                for i in 0..16 {
                    let sid = SessionId::from(format!("m4-bench-{i}"));
                    let _ = manager.complete(&sid, "bench prompt", &opt).await.unwrap();
                }
            });
        });
    });

    // M5 batching: batched forward pass via BatchScheduler
    let backend_m5 = Arc::new(MockBackend::new("response").with_forward_pass_delay(delay));
    let policy_m5 = TierPolicy::new(32, 64, 20 * 1024 * 1024, 40 * 1024 * 1024);
    let manager_m5 = Arc::new(SessionManager::with_policy(
        backend_m5,
        store,
        policy_m5,
        "bench-model",
    ));
    let batch_config = BatchConfig {
        max_batch_size: 16,
        batch_window: Duration::from_millis(2),
    };
    let scheduler_m5 = {
        let _guard = rt.enter();
        Arc::new(BatchScheduler::new(manager_m5, batch_config))
    };

    group.bench_function("m5_batched_inference_16_clients", |b| {
        let options = CompletionOptions::default();
        b.iter(|| {
            let sched = Arc::clone(&scheduler_m5);
            let opt = options.clone();
            rt.block_on(async move {
                let mut tasks = Vec::with_capacity(16);
                for i in 0..16 {
                    let s = Arc::clone(&sched);
                    let sid = SessionId::from(format!("m5-bench-{i}"));
                    let o = opt.clone();
                    tasks.push(tokio::spawn(async move {
                        s.submit(sid, "bench prompt".to_string(), o).await
                    }));
                }
                for t in tasks {
                    t.await.unwrap().unwrap();
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_session_prefill_reuse,
    bench_m4_vs_m5_batch_throughput
);
criterion_main!(benches);
