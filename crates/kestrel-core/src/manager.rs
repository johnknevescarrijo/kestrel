//! Session lifecycle management with 3-tier KV caching (Hot, Warm, Cold) and budget enforcement.

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::error::Error;
use crate::lru::LruCache;
use crate::session::{Session, SessionId, SessionMetadata, SessionSnapshot};
use crate::tier::{TierLevel, TierMetricsCounters, TierMetricsSnapshot, TierPolicy};
use crate::traits::{Backend, CompletionOptions, CompletionResult, SessionStore};

/// Orchestrates session lifecycle across Hot (VRAM), Warm (RAM), and Cold (disk) tiers.
pub struct SessionManager<B: Backend, S: SessionStore> {
    backend: Arc<B>,
    store: Arc<S>,
    hot: Arc<Mutex<LruCache<SessionId, Session>>>,
    warm: Arc<Mutex<LruCache<SessionId, SessionSnapshot>>>,
    policy: TierPolicy,
    model_id: String,
    metrics: TierMetricsCounters,
}

impl<B: Backend, S: SessionStore> SessionManager<B, S> {
    /// Creates a new `SessionManager` with default environment-based tier policy and custom hot capacity.
    pub fn new(
        backend: Arc<B>,
        store: Arc<S>,
        hot_capacity: usize,
        model_id: impl Into<String>,
    ) -> Self {
        let mut policy = TierPolicy::from_env();
        policy.max_hot_sessions = hot_capacity;
        Self::with_policy(backend, store, policy, model_id)
    }

    /// Creates a new `SessionManager` configured with an explicit `TierPolicy`.
    pub fn with_policy(
        backend: Arc<B>,
        store: Arc<S>,
        policy: TierPolicy,
        model_id: impl Into<String>,
    ) -> Self {
        let hot_cap = policy.max_hot_sessions.max(1);
        let warm_cap = policy.max_warm_sessions.max(1);
        Self {
            backend,
            store,
            hot: Arc::new(Mutex::new(LruCache::new(hot_cap))),
            warm: Arc::new(Mutex::new(LruCache::new(warm_cap))),
            policy,
            model_id: model_id.into(),
            metrics: TierMetricsCounters::new(),
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

    /// Queries the current storage tier level of a given session.
    pub async fn session_tier(&self, session_id: &SessionId) -> Result<Option<TierLevel>, Error> {
        if self.hot.lock().await.contains_key(session_id) {
            return Ok(Some(TierLevel::Hot));
        }
        if self.warm.lock().await.contains_key(session_id) {
            return Ok(Some(TierLevel::Warm));
        }
        if self.store.load(session_id).await?.is_some() {
            return Ok(Some(TierLevel::Cold));
        }
        Ok(None)
    }

    /// Generates a point-in-time metrics snapshot of tiering performance and utilization.
    pub async fn metrics_snapshot(&self) -> TierMetricsSnapshot {
        let hot_sessions = self.hot_session_count().await;
        let warm_sessions = self.warm_session_count().await;
        let hot_used = self.hot_used_bytes().await;
        let warm_used = self.warm_used_bytes().await;
        self.metrics.snapshot(
            hot_sessions,
            warm_sessions,
            hot_used,
            warm_used,
            &self.policy,
        )
    }

    /// Resolves a session for inference across Hot, Warm, and Cold tiers.
    async fn resolve_session(&self, session_id: &SessionId) -> Result<Session, Error> {
        debug!(session = %session_id, "Resolving session across tiers");
        if let Some(session) = self.hot.lock().await.remove(session_id) {
            self.metrics.record_hot_hit();
            return Ok(session);
        }
        if let Some(snapshot) = self.warm.lock().await.remove(session_id) {
            self.metrics.record_warm_hit();
            self.metrics.record_promotion_warm_to_hot();
            info!(session = %session_id, "Promoting session from warm to hot tier");
            return Ok(Session::from_snapshot(snapshot));
        }
        if let Some(snapshot) = self.store.load(session_id).await? {
            self.metrics.record_cold_hit();
            self.metrics.record_promotion_cold_to_hot();
            info!(session = %session_id, "Promoting session from cold storage to hot tier");
            return Ok(Session::from_snapshot(snapshot));
        }
        self.metrics.record_miss();
        info!(session = %session_id, "Creating fresh session");
        Ok(Session::new(session_id.clone(), &self.model_id))
    }

    /// Commits an updated session back to the hot tier, enforcing hot and warm memory budgets.
    async fn commit_session(&self, session_id: &SessionId, session: Session) -> Result<(), Error> {
        let snapshot = session.to_snapshot();
        self.store.save(&snapshot).await?;

        let mut to_warm = Vec::new();
        {
            let mut hot = self.hot.lock().await;
            hot.remove(session_id);
            let needed = session.metadata.kv_size_bytes;
            while !hot.is_empty()
                && (hot.len() >= self.policy.max_hot_sessions
                    || hot
                        .values()
                        .map(|s| s.metadata.kv_size_bytes)
                        .sum::<usize>()
                        + needed
                        > self.policy.hot_budget_bytes)
            {
                if let Some((evicted_id, evicted_session)) = hot.pop_lru() {
                    info!(session = %evicted_id, "Demoting session from hot to warm tier");
                    self.metrics.record_demotion_hot_to_warm();
                    to_warm.push(evicted_session.to_snapshot());
                }
            }
            hot.put(session_id.clone(), session);
        }

        for snap in to_warm {
            self.insert_warm(snap).await?;
        }
        Ok(())
    }

    /// Inserts a snapshot into the warm tier, enforcing warm capacity and byte budgets.
    async fn insert_warm(&self, snap: SessionSnapshot) -> Result<(), Error> {
        let mut to_cold = Vec::new();
        {
            let mut warm = self.warm.lock().await;
            let snap_id = snap.metadata.id.clone();
            warm.remove(&snap_id);
            let needed = snap.metadata.kv_size_bytes;
            while !warm.is_empty()
                && (warm.len() >= self.policy.max_warm_sessions
                    || warm
                        .values()
                        .map(|s| s.metadata.kv_size_bytes)
                        .sum::<usize>()
                        + needed
                        > self.policy.warm_budget_bytes)
            {
                if let Some((evicted_id, evicted_snap)) = warm.pop_lru() {
                    info!(session = %evicted_id, "Demoting session from warm to cold tier");
                    self.metrics.record_demotion_warm_to_cold();
                    to_cold.push(evicted_snap);
                }
            }
            warm.put(snap_id, snap);
        }

        for cold_snap in to_cold {
            self.store.save(&cold_snap).await?;
        }
        Ok(())
    }

    /// Manually demotes an active session from hot tier to warm tier.
    pub async fn demote_hot_to_warm(&self, session_id: &SessionId) -> Result<bool, Error> {
        let session = self.hot.lock().await.remove(session_id);
        if let Some(session) = session {
            self.metrics.record_demotion_hot_to_warm();
            let snap = session.to_snapshot();
            self.insert_warm(snap).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Manually evicts a session from hot or warm tiers into cold storage.
    pub async fn evict_to_cold(&self, session_id: &SessionId) -> Result<(), Error> {
        if let Some(session) = self.hot.lock().await.remove(session_id) {
            self.store.save(&session.to_snapshot()).await?;
        } else if let Some(snap) = self.warm.lock().await.remove(session_id) {
            self.store.save(&snap).await?;
        }
        Ok(())
    }

    /// Executes completion within the specified session, managing promotion and persistence.
    pub async fn complete(
        &self,
        session_id: &SessionId,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        let mut session = self.resolve_session(session_id).await?;
        let result = match self
            .backend
            .complete_session(&mut session, prompt, options)
            .await
        {
            Ok(res) => res,
            Err(err) => {
                let _ = self.commit_session(session_id, session).await;
                return Err(err);
            }
        };
        self.commit_session(session_id, session).await?;
        Ok(result)
    }

    /// Executes streaming completion within the specified session, managing promotion and persistence.
    pub async fn complete_stream(
        &self,
        session_id: &SessionId,
        prompt: &str,
        options: &CompletionOptions,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, Error> {
        let mut session = self.resolve_session(session_id).await?;
        let result = match self
            .backend
            .complete_session_stream(&mut session, prompt, options, token_tx)
            .await
        {
            Ok(res) => res,
            Err(err) => {
                let _ = self.commit_session(session_id, session).await;
                return Err(err);
            }
        };
        self.commit_session(session_id, session).await?;
        Ok(result)
    }

    /// Executes batch completion across distinct sessions, managing resolution and budget tiering.
    pub async fn complete_batch(
        &self,
        requests: &[(SessionId, String, CompletionOptions)],
    ) -> Result<Vec<CompletionResult>, Error> {
        let mut sessions = Vec::with_capacity(requests.len());
        for (session_id, _, _) in requests {
            let session = self.resolve_session(session_id).await?;
            sessions.push(session);
        }

        let mut batch_items: Vec<crate::traits::BatchItem<'_>> = requests
            .iter()
            .zip(sessions.iter_mut())
            .map(|((_, prompt, options), session)| crate::traits::BatchItem {
                session,
                prompt,
                options,
            })
            .collect();

        let batch_result = self.backend.complete_batch(&mut batch_items).await;

        let mut commit_err = None;
        for ((session_id, _, _), session) in requests.iter().zip(sessions) {
            if let Err(err) = self.commit_session(session_id, session).await {
                if commit_err.is_none() {
                    commit_err = Some(err);
                }
            }
        }

        let results = batch_result?;
        if let Some(err) = commit_err {
            return Err(err);
        }
        Ok(results)
    }

    /// Retrieves session metadata from hot cache, warm cache, or cold storage.
    pub async fn get_metadata(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<SessionMetadata>, Error> {
        if let Some(session) = self.hot.lock().await.peek(session_id) {
            return Ok(Some(session.metadata.clone()));
        }
        if let Some(snapshot) = self.warm.lock().await.peek(session_id) {
            return Ok(Some(snapshot.metadata.clone()));
        }
        match self.store.load(session_id).await? {
            Some(snapshot) => Ok(Some(snapshot.metadata)),
            None => Ok(None),
        }
    }

    /// Deletes a session from hot cache, warm cache, and cold storage.
    pub async fn delete_session(&self, session_id: &SessionId) -> Result<bool, Error> {
        let in_hot = self.hot.lock().await.remove(session_id).is_some();
        let in_warm = self.warm.lock().await.remove(session_id).is_some();
        let in_cold = self.store.delete(session_id).await?;
        Ok(in_hot || in_warm || in_cold)
    }

    /// Lists metadata for all sessions across cold storage, warm cache, and hot cache.
    pub async fn list_sessions(&self) -> Result<Vec<SessionMetadata>, Error> {
        let mut sessions = self.store.list().await?;
        {
            let warm = self.warm.lock().await;
            for snap in warm.values() {
                if let Some(pos) = sessions.iter().position(|m| m.id == snap.metadata.id) {
                    sessions[pos] = snap.metadata.clone();
                } else {
                    sessions.push(snap.metadata.clone());
                }
            }
        }
        {
            let hot = self.hot.lock().await;
            for sess in hot.values() {
                if let Some(pos) = sessions.iter().position(|m| m.id == sess.metadata.id) {
                    sessions[pos] = sess.metadata.clone();
                } else {
                    sessions.push(sess.metadata.clone());
                }
            }
        }
        Ok(sessions)
    }
}
