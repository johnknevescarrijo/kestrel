//! Session lifecycle management with 3-tier KV caching (Hot, Warm, Cold) and budget enforcement.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::debug;

use crate::error::Error;
use crate::lru::LruCache;
use crate::session::{Session, SessionId, SessionMetadata, SessionSnapshot};
use crate::tier::{TierLevel, TierMetricsCounters, TierMetricsSnapshot, TierPolicy};
use crate::traits::{Backend, CompletionOptions, CompletionResult, SessionStore};
use crate::writeback::{PersistTicket, WritebackCoordinator};

type SessionLockMap = Arc<StdMutex<HashMap<SessionId, Arc<Mutex<()>>>>>;

/// RAII guard holding an exclusive per-session lock and pruning idle map entries on drop.
struct SessionLockGuard {
    id: SessionId,
    arc: Arc<Mutex<()>>,
    locks: SessionLockMap,
    guard: Option<OwnedMutexGuard<()>>,
}

impl Drop for SessionLockGuard {
    fn drop(&mut self) {
        // Release the session mutex first so any waiter strictly happens-after our critical section.
        drop(self.guard.take());

        let mut map = self.locks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = map.get(&self.id) {
            // Only `map` and `self.arc` hold strong references when no other task is waiting.
            if Arc::ptr_eq(entry, &self.arc) && Arc::strong_count(entry) == 2 {
                map.remove(&self.id);
            }
        }
    }
}

/// Orchestrates session lifecycle across Hot (VRAM), Warm (RAM), and Cold (disk) tiers.
pub struct SessionManager<B: Backend, S: SessionStore> {
    backend: Arc<B>,
    store: Arc<S>,
    hot: Arc<Mutex<LruCache<SessionId, Session>>>,
    warm: Arc<Mutex<LruCache<SessionId, SessionSnapshot>>>,
    session_locks: SessionLockMap,
    writeback: WritebackCoordinator,
    policy: TierPolicy,
    model_id: String,
    metrics: TierMetricsCounters,
}

impl<B: Backend, S: SessionStore + 'static> SessionManager<B, S> {
    /// Creates a new `SessionManager` with default environment-based tier policy and custom hot capacity.
    pub fn new(backend: Arc<B>, store: Arc<S>, hot_cap: usize, model: impl Into<String>) -> Self {
        let mut policy = TierPolicy::from_env();
        policy.max_hot_sessions = hot_cap;
        Self::with_policy(backend, store, policy, model)
    }

    /// Creates a new `SessionManager` configured with an explicit `TierPolicy`.
    pub fn with_policy(
        backend: Arc<B>,
        store: Arc<S>,
        policy: TierPolicy,
        model_id: impl Into<String>,
    ) -> Self {
        let dyn_store: Arc<dyn SessionStore> = Arc::clone(&store) as Arc<dyn SessionStore>;
        Self {
            backend,
            store,
            hot: Arc::new(Mutex::new(LruCache::new(policy.max_hot_sessions.max(1)))),
            warm: Arc::new(Mutex::new(LruCache::new(policy.max_warm_sessions.max(1)))),
            session_locks: Arc::new(StdMutex::new(HashMap::new())),
            writeback: WritebackCoordinator::new(dyn_store),
            policy,
            model_id: model_id.into(),
            metrics: TierMetricsCounters::new(),
        }
    }
}

impl<B: Backend, S: SessionStore> SessionManager<B, S> {
    async fn acquire_session_lock(&self, id: &SessionId) -> SessionLockGuard {
        let arc = {
            let mut map = self.session_locks.lock().unwrap_or_else(|e| e.into_inner());
            map.entry(id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let guard = Arc::clone(&arc).lock_owned().await;
        SessionLockGuard {
            id: id.clone(),
            arc,
            locks: Arc::clone(&self.session_locks),
            guard: Some(guard),
        }
    }

    /// Returns the maximum capacity for hot active sessions.
    pub fn hot_capacity(&self) -> usize {
        self.policy.max_hot_sessions
    }

    /// Returns the active tier policy.
    pub fn policy(&self) -> &TierPolicy {
        &self.policy
    }

    /// Returns the number of sessions currently active in the hot (VRAM) tier.
    pub async fn hot_session_count(&self) -> usize {
        self.hot.lock().await.len()
    }

    /// Returns the number of sessions currently cached in the warm (RAM) tier.
    pub async fn warm_session_count(&self) -> usize {
        self.warm.lock().await.len()
    }

    /// Returns the total KV cache bytes currently resident in the hot tier.
    pub async fn hot_used_bytes(&self) -> usize {
        self.hot
            .lock()
            .await
            .values()
            .map(|s| s.metadata.kv_size_bytes)
            .sum()
    }

    /// Returns the total KV cache bytes currently resident in the warm tier.
    pub async fn warm_used_bytes(&self) -> usize {
        self.warm
            .lock()
            .await
            .values()
            .map(|s| s.metadata.kv_size_bytes)
            .sum()
    }

    /// Flushes all pending asynchronous write-backs to cold storage.
    pub async fn flush(&self) -> Result<(), Error> {
        self.writeback.flush().await
    }

    /// Queries the current storage tier level of a given session.
    pub async fn session_tier(&self, id: &SessionId) -> Result<Option<TierLevel>, Error> {
        if self.hot.lock().await.contains_key(id) {
            return Ok(Some(TierLevel::Hot));
        }
        if self.warm.lock().await.contains_key(id) {
            return Ok(Some(TierLevel::Warm));
        }
        self.writeback.wait_for_session(id).await;
        if self.store.load(id).await?.is_some() {
            Ok(Some(TierLevel::Cold))
        } else {
            Ok(None)
        }
    }

    /// Generates a point-in-time metrics snapshot of tiering performance and utilization.
    pub async fn metrics_snapshot(&self) -> TierMetricsSnapshot {
        let (hot_sessions, hot_used) = {
            let h = self.hot.lock().await;
            (h.len(), h.values().map(|s| s.metadata.kv_size_bytes).sum())
        };
        let (warm_sessions, warm_used) = {
            let w = self.warm.lock().await;
            (w.len(), w.values().map(|s| s.metadata.kv_size_bytes).sum())
        };
        self.metrics.snapshot(
            hot_sessions,
            warm_sessions,
            hot_used,
            warm_used,
            &self.policy,
        )
    }

    /// Resolves a session for inference across Hot, Warm, and Cold tiers.
    async fn resolve_session(&self, id: &SessionId) -> Result<Session, Error> {
        debug!(session = %id, "Resolving session across tiers");
        if let Some(session) = self.hot.lock().await.remove(id) {
            self.metrics.record_hot_hit();
            return Ok(session);
        }
        if let Some(snap) = self.warm.lock().await.remove(id) {
            self.metrics.record_warm_hit();
            self.metrics.record_promotion_warm_to_hot();
            return Ok(Session::from_snapshot(snap));
        }
        self.writeback.wait_for_session(id).await;
        if let Some(snap) = self.store.load(id).await? {
            self.metrics.record_cold_hit();
            self.metrics.record_promotion_cold_to_hot();
            return Ok(Session::from_snapshot(snap));
        }
        self.metrics.record_miss();
        Ok(Session::new(id.clone(), &self.model_id))
    }

    /// Resolves a session and switches model if override specified.
    async fn resolve_session_with_model(
        &self,
        id: &SessionId,
        model: Option<&str>,
    ) -> Result<Session, Error> {
        let mut session = self.resolve_session(id).await?;
        if let Some(target) = model.filter(|m| !m.is_empty() && *m != "default") {
            session.reset_for_model(target);
        }
        Ok(session)
    }

    /// Inserts snapshots into the warm tier while holding the warm mutex, registering
    /// any cold evictions before the warm lock is released.
    async fn insert_warm_locked(
        &self,
        warm: &mut LruCache<SessionId, SessionSnapshot>,
        snapshots: Vec<SessionSnapshot>,
    ) -> Vec<PersistTicket> {
        let mut to_cold = Vec::new();
        for snap in snapshots {
            let id = snap.metadata.id.clone();
            warm.remove(&id);
            while !warm.is_empty()
                && self.policy.exceeds_warm(
                    warm.len(),
                    warm.values().map(|s| s.metadata.kv_size_bytes).sum(),
                    snap.metadata.kv_size_bytes,
                )
            {
                if let Some((_, evicted)) = warm.pop_lru() {
                    self.metrics.record_demotion_warm_to_cold();
                    to_cold.push(evicted);
                }
            }
            warm.put(id, snap);
        }
        self.writeback.enqueue_cold_batch(to_cold).await
    }

    /// Commits an updated session back to the hot tier, scheduling async write-back
    /// and guaranteeing cold persistence for any evicted warm sessions.
    async fn commit_session(&self, id: &SessionId, s: Session) -> Result<(), Error> {
        self.writeback.enqueue_writeback(s.to_snapshot()).await;

        let cold_tickets = {
            let mut hot = self.hot.lock().await;
            hot.remove(id);
            let mut to_warm = Vec::new();
            while !hot.is_empty()
                && self.policy.exceeds_hot(
                    hot.len(),
                    hot.values().map(|item| item.metadata.kv_size_bytes).sum(),
                    s.metadata.kv_size_bytes,
                )
            {
                if let Some((_, evicted)) = hot.pop_lru() {
                    self.metrics.record_demotion_hot_to_warm();
                    to_warm.push(evicted.to_snapshot());
                }
            }
            hot.put(id.clone(), s);

            if to_warm.is_empty() {
                Vec::new()
            } else {
                // Acquire warm before dropping hot so demoted snapshots are never invisible.
                let mut warm = self.warm.lock().await;
                drop(hot);
                self.insert_warm_locked(&mut warm, to_warm).await
            }
        };

        self.writeback.wait_for_tickets(cold_tickets).await
    }

    /// Manually demotes an active session from hot tier to warm tier.
    pub async fn demote_hot_to_warm(&self, id: &SessionId) -> Result<bool, Error> {
        let _session_guard = self.acquire_session_lock(id).await;
        let (demoted, cold_tickets) = {
            let mut hot = self.hot.lock().await;
            match hot.remove(id) {
                Some(s) => {
                    self.metrics.record_demotion_hot_to_warm();
                    let mut warm = self.warm.lock().await;
                    drop(hot);
                    let tickets = self
                        .insert_warm_locked(&mut warm, vec![s.to_snapshot()])
                        .await;
                    (true, tickets)
                }
                None => (false, Vec::new()),
            }
        };
        self.writeback.wait_for_tickets(cold_tickets).await?;
        Ok(demoted)
    }

    /// Manually evicts a session from hot or warm tiers into cold storage.
    pub async fn evict_to_cold(&self, id: &SessionId) -> Result<(), Error> {
        let _session_guard = self.acquire_session_lock(id).await;
        let ticket = {
            let mut hot = self.hot.lock().await;
            if let Some(s) = hot.remove(id) {
                Some(self.writeback.enqueue_cold(s.to_snapshot()).await)
            } else {
                drop(hot);
                let mut warm = self.warm.lock().await;
                if let Some(snap) = warm.remove(id) {
                    Some(self.writeback.enqueue_cold(snap).await)
                } else {
                    None
                }
            }
        };

        if let Some(ticket) = ticket {
            self.writeback.wait_for_ticket(&ticket).await?;
        } else {
            self.writeback.wait_for_session(id).await;
        }
        Ok(())
    }

    /// Executes completion within the specified session with an optional model override.
    pub async fn complete_with_model(
        &self,
        session_id: &SessionId,
        model: Option<&str>,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        let _session_guard = self.acquire_session_lock(session_id).await;
        let mut session = self.resolve_session_with_model(session_id, model).await?;
        let res = self
            .backend
            .complete_session(&mut session, prompt, options)
            .await;
        let c = self.commit_session(session_id, session).await;
        res.and_then(|val| {
            c?;
            Ok(val)
        })
    }

    /// Executes completion within the specified session using default model.
    pub async fn complete(
        &self,
        id: &SessionId,
        prompt: &str,
        opts: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        self.complete_with_model(id, None, prompt, opts).await
    }

    /// Executes streaming completion within the specified session with an optional model override.
    pub async fn complete_stream_with_model(
        &self,
        session_id: &SessionId,
        model: Option<&str>,
        prompt: &str,
        options: &CompletionOptions,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, Error> {
        let _session_guard = self.acquire_session_lock(session_id).await;
        let mut session = self.resolve_session_with_model(session_id, model).await?;
        let res = self
            .backend
            .complete_session_stream(&mut session, prompt, options, token_tx)
            .await;
        let c = self.commit_session(session_id, session).await;
        res.and_then(|val| {
            c?;
            Ok(val)
        })
    }

    /// Executes streaming completion within the specified session using default model.
    pub async fn complete_stream(
        &self,
        id: &SessionId,
        prompt: &str,
        opts: &CompletionOptions,
        tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, Error> {
        self.complete_stream_with_model(id, None, prompt, opts, tx)
            .await
    }

    /// Executes batch completion across distinct sessions with model overrides.
    pub async fn complete_batch_with_models(
        &self,
        requests: &[(SessionId, Option<String>, String, CompletionOptions)],
    ) -> Result<Vec<CompletionResult>, Error> {
        // Acquire per-session locks in deterministic sorted order to prevent deadlocks.
        let mut lock_ids: Vec<SessionId> =
            requests.iter().map(|(id, _, _, _)| id.clone()).collect();
        lock_ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        lock_ids.dedup();

        let mut session_guards = Vec::with_capacity(lock_ids.len());
        for id in &lock_ids {
            session_guards.push(self.acquire_session_lock(id).await);
        }

        let mut sessions = Vec::with_capacity(requests.len());
        for (id, m, _, _) in requests {
            sessions.push(self.resolve_session_with_model(id, m.as_deref()).await?);
        }

        let mut items: Vec<_> = requests
            .iter()
            .zip(sessions.iter_mut())
            .map(
                |((_, _, prompt, options), session)| crate::traits::BatchItem {
                    session,
                    prompt,
                    options,
                },
            )
            .collect();

        let res = self.backend.complete_batch(&mut items).await;
        let mut err = None;
        for ((id, _, _, _), s) in requests.iter().zip(sessions) {
            if let Err(e) = self.commit_session(id, s).await {
                err.get_or_insert(e);
            }
        }
        err.map_or(res, Err)
    }

    /// Executes batch completion across distinct sessions, managing resolution and budget tiering.
    pub async fn complete_batch(
        &self,
        requests: &[(SessionId, String, CompletionOptions)],
    ) -> Result<Vec<CompletionResult>, Error> {
        let reqs: Vec<_> = requests
            .iter()
            .map(|(id, p, o)| (id.clone(), None, p.clone(), o.clone()))
            .collect();
        self.complete_batch_with_models(&reqs).await
    }

    /// Computes a vector embedding for the input text, optionally targeting a specific model.
    pub async fn embed(
        &self,
        model: Option<&str>,
        input: &str,
    ) -> Result<crate::traits::EmbeddingResult, Error> {
        if let Some(m) = model.filter(|m| !m.is_empty() && *m != "default") {
            self.backend.embed_for_model(m, input).await
        } else {
            self.backend.embed(input).await
        }
    }

    /// Retrieves session metadata from hot cache, warm cache, or cold storage.
    pub async fn get_metadata(&self, id: &SessionId) -> Result<Option<SessionMetadata>, Error> {
        if let Some(s) = self.hot.lock().await.peek(id) {
            return Ok(Some(s.metadata.clone()));
        }
        if let Some(s) = self.warm.lock().await.peek(id) {
            return Ok(Some(s.metadata.clone()));
        }
        self.writeback.wait_for_session(id).await;
        Ok(self.store.load(id).await?.map(|s| s.metadata))
    }

    /// Deletes a session from hot cache, warm cache, and cold storage.
    pub async fn delete_session(&self, id: &SessionId) -> Result<bool, Error> {
        let _session_guard = self.acquire_session_lock(id).await;
        let h = self.hot.lock().await.remove(id).is_some();
        let w = self.warm.lock().await.remove(id).is_some();
        let had_pending = self.writeback.cancel_and_wait(id).await;
        let s = self.store.delete(id).await?;
        Ok(h || w || had_pending || s)
    }

    /// Lists metadata for all sessions across cold storage, warm cache, and hot cache.
    pub async fn list_sessions(&self) -> Result<Vec<SessionMetadata>, Error> {
        let mut sessions = self.store.list().await?;
        let in_memory: Vec<_> = {
            let hot = self.hot.lock().await;
            let warm = self.warm.lock().await;
            let wb = self.writeback.pending_metadata().await;
            wb.into_iter()
                .chain(warm.values().map(|s| s.metadata.clone()))
                .chain(hot.values().map(|s| s.metadata.clone()))
                .collect()
        };
        for meta in in_memory {
            if let Some(pos) = sessions.iter().position(|m| m.id == meta.id) {
                sessions[pos] = meta;
            } else {
                sessions.push(meta);
            }
        }
        Ok(sessions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::FileSessionStore;
    use crate::traits::MockBackend;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_session_lock_map_cleans_up_idle_entries() {
        let dir = tempdir().unwrap();
        let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
        let backend = Arc::new(MockBackend::new("ok"));
        let manager = SessionManager::new(backend, store, 4, "test-model");

        let s1 = SessionId::from("sess-cleanup-1");
        let s2 = SessionId::from("sess-cleanup-2");

        manager
            .complete(&s1, "hi", &CompletionOptions::default())
            .await
            .unwrap();
        manager
            .complete(&s2, "hi", &CompletionOptions::default())
            .await
            .unwrap();
        manager.flush().await.unwrap();

        let lock_count = manager
            .session_locks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        assert_eq!(lock_count, 0, "idle session locks should be cleaned up");
    }
}
