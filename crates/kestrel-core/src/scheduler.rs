#![deny(unsafe_code)]

//! Continuous batching scheduler for concurrent LLM inference across sessions.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::error::Error;
use crate::manager::SessionManager;
use crate::session::SessionId;
use crate::traits::{Backend, CompletionOptions, CompletionResult, SessionStore};

/// Configuration options for dynamic batching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchConfig {
    /// Maximum number of requests grouped in a single forward pass batch.
    pub max_batch_size: usize,
    /// Maximum time window to wait for incoming requests to form a batch.
    pub batch_window: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            batch_window: Duration::from_millis(2),
        }
    }
}

/// Point-in-time snapshot of batching scheduler metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct BatchMetricsSnapshot {
    /// Total number of batches executed.
    pub total_batches: u64,
    /// Total number of requests processed inside batches.
    pub total_batched_requests: u64,
    /// Maximum observed batch size across all batches.
    pub max_observed_batch_size: usize,
    /// Average batch size (`total_batched_requests / total_batches`).
    pub average_batch_size: f64,
}

#[derive(Debug, Default)]
struct BatchMetricsCounters {
    total_batches: AtomicU64,
    total_batched_requests: AtomicU64,
    max_observed_batch_size: AtomicUsize,
}

impl BatchMetricsCounters {
    fn snapshot(&self) -> BatchMetricsSnapshot {
        let total_batches = self.total_batches.load(Ordering::SeqCst);
        let total_batched_requests = self.total_batched_requests.load(Ordering::SeqCst);
        let max_observed_batch_size = self.max_observed_batch_size.load(Ordering::SeqCst);
        let average_batch_size = if total_batches > 0 {
            total_batched_requests as f64 / total_batches as f64
        } else {
            0.0
        };

        BatchMetricsSnapshot {
            total_batches,
            total_batched_requests,
            max_observed_batch_size,
            average_batch_size,
        }
    }
}

struct BatchRequest {
    session_id: SessionId,
    prompt: String,
    options: CompletionOptions,
    response_tx: oneshot::Sender<Result<CompletionResult, Error>>,
}

/// Continuous batching scheduler orchestrating concurrent requests into batched inference runs.
pub struct BatchScheduler<B: Backend + 'static, S: SessionStore + 'static> {
    tx: mpsc::Sender<BatchRequest>,
    metrics: Arc<BatchMetricsCounters>,
    _phantom: std::marker::PhantomData<(B, S)>,
}

impl<B: Backend + 'static, S: SessionStore + 'static> BatchScheduler<B, S> {
    /// Creates and starts a new `BatchScheduler` worker task.
    pub fn new(manager: Arc<SessionManager<B, S>>, config: BatchConfig) -> Self {
        let queue_bound = config.max_batch_size.max(1) * 32;
        let (tx, rx) = mpsc::channel(queue_bound);
        let metrics = Arc::new(BatchMetricsCounters::default());
        let metrics_clone = Arc::clone(&metrics);

        tokio::spawn(async move {
            run_scheduler_loop(manager, config, rx, metrics_clone).await;
        });

        Self {
            tx,
            metrics,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Submits a request for scheduled batch completion.
    pub async fn submit(
        &self,
        session_id: SessionId,
        prompt: String,
        options: CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        let (response_tx, response_rx) = oneshot::channel();
        let req = BatchRequest {
            session_id,
            prompt,
            options,
            response_tx,
        };

        self.tx.send(req).await.map_err(|_| Error::Internal {
            reason: "batch scheduler worker task has terminated".to_string(),
        })?;

        response_rx.await.map_err(|_| Error::Internal {
            reason: "batch scheduler worker dropped response channel".to_string(),
        })?
    }

    /// Returns a point-in-time metrics snapshot of scheduler batching performance.
    pub fn metrics_snapshot(&self) -> BatchMetricsSnapshot {
        self.metrics.snapshot()
    }
}

async fn run_scheduler_loop<B: Backend + 'static, S: SessionStore + 'static>(
    manager: Arc<SessionManager<B, S>>,
    config: BatchConfig,
    mut rx: mpsc::Receiver<BatchRequest>,
    metrics: Arc<BatchMetricsCounters>,
) {
    let mut carryover: Vec<BatchRequest> = Vec::new();

    loop {
        let mut candidates = Vec::new();
        if carryover.is_empty() {
            match rx.recv().await {
                Some(req) => candidates.push(req),
                None => break,
            }
        } else {
            candidates.append(&mut carryover);
        }

        let deadline = tokio::time::Instant::now() + config.batch_window;
        while candidates.len() < config.max_batch_size {
            let timeout_duration = deadline.saturating_duration_since(tokio::time::Instant::now());
            if timeout_duration.is_zero() {
                break;
            }

            match tokio::time::timeout(timeout_duration, rx.recv()).await {
                Ok(Some(req)) => candidates.push(req),
                Ok(None) => break,
                Err(_) => break,
            }
        }

        let mut current_batch = Vec::new();
        let mut seen_sessions = HashSet::new();

        for req in candidates {
            if current_batch.len() < config.max_batch_size
                && !seen_sessions.contains(&req.session_id)
            {
                seen_sessions.insert(req.session_id.clone());
                current_batch.push(req);
            } else {
                carryover.push(req);
            }
        }

        if current_batch.is_empty() {
            continue;
        }

        let batch_size = current_batch.len();
        metrics.total_batches.fetch_add(1, Ordering::SeqCst);
        metrics
            .total_batched_requests
            .fetch_add(batch_size as u64, Ordering::SeqCst);
        metrics
            .max_observed_batch_size
            .fetch_max(batch_size, Ordering::SeqCst);

        let batch_requests: Vec<(SessionId, String, CompletionOptions)> = current_batch
            .iter()
            .map(|r| (r.session_id.clone(), r.prompt.clone(), r.options.clone()))
            .collect();

        match manager.complete_batch(&batch_requests).await {
            Ok(results) => {
                for (req, res) in current_batch.into_iter().zip(results) {
                    let _ = req.response_tx.send(Ok(res));
                }
            }
            Err(err) => {
                for req in current_batch {
                    let _ = req.response_tx.send(Err(err.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::FileSessionStore;
    use crate::tier::TierPolicy;
    use crate::traits::MockBackend;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_scheduler_groups_concurrent_requests() {
        let dir = tempdir().unwrap();
        let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
        let backend = Arc::new(
            MockBackend::new("batch response").with_forward_pass_delay(Duration::from_millis(5)),
        );

        let policy = TierPolicy::new(8, 16, 10 * 1024 * 1024, 20 * 1024 * 1024);
        let manager = Arc::new(SessionManager::with_policy(
            backend.clone(),
            store,
            policy,
            "test-model",
        ));

        let config = BatchConfig {
            max_batch_size: 4,
            batch_window: Duration::from_millis(20),
        };
        let scheduler = Arc::new(BatchScheduler::new(manager, config));

        let mut handles = Vec::new();
        for i in 0..4 {
            let sched = Arc::clone(&scheduler);
            let sess_id = SessionId::from(format!("session-{i}"));
            handles.push(tokio::spawn(async move {
                sched
                    .submit(sess_id, format!("Prompt {i}"), CompletionOptions::default())
                    .await
            }));
        }

        for handle in handles {
            let res = handle.await.unwrap().unwrap();
            assert_eq!(res.text, "batch response");
        }

        let metrics = scheduler.metrics_snapshot();
        assert_eq!(metrics.total_batched_requests, 4);
        assert_eq!(metrics.max_observed_batch_size, 4);
        assert_eq!(metrics.total_batches, 1);
        assert_eq!(backend.batch_call_count(), 1);
    }

    #[tokio::test]
    async fn test_scheduler_defers_duplicate_session_in_window() {
        let dir = tempdir().unwrap();
        let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
        let backend = Arc::new(
            MockBackend::new("batch response").with_forward_pass_delay(Duration::from_millis(5)),
        );

        let policy = TierPolicy::new(8, 16, 10 * 1024 * 1024, 20 * 1024 * 1024);
        let manager = Arc::new(SessionManager::with_policy(
            backend.clone(),
            store,
            policy,
            "test-model",
        ));

        let config = BatchConfig {
            max_batch_size: 4,
            batch_window: Duration::from_millis(20),
        };
        let scheduler = Arc::new(BatchScheduler::new(manager, config));

        // Submit two requests with the EXACT SAME session ID concurrently
        let sched1 = Arc::clone(&scheduler);
        let sched2 = Arc::clone(&scheduler);
        let same_session = SessionId::from("duplicate-session");

        let h1 = tokio::spawn({
            let sess = same_session.clone();
            async move {
                sched1
                    .submit(sess, "Turn 1".to_string(), CompletionOptions::default())
                    .await
            }
        });

        let h2 = tokio::spawn({
            let sess = same_session.clone();
            async move {
                sched2
                    .submit(sess, "Turn 2".to_string(), CompletionOptions::default())
                    .await
            }
        });

        let (res1, res2) = (h1.await.unwrap().unwrap(), h2.await.unwrap().unwrap());
        assert_eq!(res1.text, "batch response");
        assert_eq!(res2.text, "batch response");

        let metrics = scheduler.metrics_snapshot();
        // Because same session cannot be in the same batch, they were split into 2 batches
        assert_eq!(metrics.total_batches, 2);
        assert_eq!(metrics.total_batched_requests, 2);
        assert_eq!(metrics.max_observed_batch_size, 1);
        assert_eq!(backend.batch_call_count(), 2);
    }
}
