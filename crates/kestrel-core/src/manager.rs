//! Session lifecycle management with 3-tier KV caching (Hot, Warm, Cold) and budget enforcement.

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

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
        Self {
            backend,
            store,
            hot: Arc::new(Mutex::new(LruCache::new(policy.max_hot_sessions.max(1)))),
            warm: Arc::new(Mutex::new(LruCache::new(policy.max_warm_sessions.max(1)))),
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
    pub async fn session_tier(&self, id: &SessionId) -> Result<Option<TierLevel>, Error> {
        if self.hot.lock().await.contains_key(id) {
            Ok(Some(TierLevel::Hot))
        } else if self.warm.lock().await.contains_key(id) {
            Ok(Some(TierLevel::Warm))
        } else if self.store.load(id).await?.is_some() {
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

    async fn evict_hot_lru(&self, needed: usize) -> Vec<SessionSnapshot> {
        let (mut to_warm, mut hot) = (Vec::new(), self.hot.lock().await);
        while !hot.is_empty()
            && self.policy.exceeds_hot(
                hot.len(),
                hot.values().map(|s| s.metadata.kv_size_bytes).sum(),
                needed,
            )
        {
            if let Some((_, s)) = hot.pop_lru() {
                self.metrics.record_demotion_hot_to_warm();
                to_warm.push(s.to_snapshot());
            }
        }
        to_warm
    }

    async fn evict_warm_lru(&self, needed: usize) -> Vec<SessionSnapshot> {
        let (mut to_cold, mut warm) = (Vec::new(), self.warm.lock().await);
        while !warm.is_empty()
            && self.policy.exceeds_warm(
                warm.len(),
                warm.values().map(|s| s.metadata.kv_size_bytes).sum(),
                needed,
            )
        {
            if let Some((_, s)) = warm.pop_lru() {
                self.metrics.record_demotion_warm_to_cold();
                to_cold.push(s);
            }
        }
        to_cold
    }

    /// Commits an updated session back to the hot tier, enforcing hot and warm memory budgets.
    async fn commit_session(&self, id: &SessionId, s: Session) -> Result<(), Error> {
        self.store.save(&s.to_snapshot()).await?;
        self.hot.lock().await.remove(id);
        let to_warm = self.evict_hot_lru(s.metadata.kv_size_bytes).await;
        self.hot.lock().await.put(id.clone(), s);
        for snap in to_warm {
            self.insert_warm(snap).await?;
        }
        Ok(())
    }

    /// Inserts a snapshot into the warm tier, enforcing warm capacity and byte budgets.
    async fn insert_warm(&self, s: SessionSnapshot) -> Result<(), Error> {
        let id = s.metadata.id.clone();
        self.warm.lock().await.remove(&id);
        let to_cold = self.evict_warm_lru(s.metadata.kv_size_bytes).await;
        self.warm.lock().await.put(id, s);
        for snap in to_cold {
            self.store.save(&snap).await?;
        }
        Ok(())
    }

    /// Manually demotes an active session from hot tier to warm tier.
    pub async fn demote_hot_to_warm(&self, id: &SessionId) -> Result<bool, Error> {
        match self.hot.lock().await.remove(id) {
            Some(s) => {
                self.metrics.record_demotion_hot_to_warm();
                self.insert_warm(s.to_snapshot()).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Manually evicts a session from hot or warm tiers into cold storage.
    pub async fn evict_to_cold(&self, id: &SessionId) -> Result<(), Error> {
        if let Some(s) = self.hot.lock().await.remove(id) {
            self.store.save(&s.to_snapshot()).await?;
        } else if let Some(s) = self.warm.lock().await.remove(id) {
            self.store.save(&s).await?;
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
        Ok(self.store.load(id).await?.map(|s| s.metadata))
    }

    /// Deletes a session from hot cache, warm cache, and cold storage.
    pub async fn delete_session(&self, id: &SessionId) -> Result<bool, Error> {
        let h = self.hot.lock().await.remove(id).is_some();
        let w = self.warm.lock().await.remove(id).is_some();
        let s = self.store.delete(id).await?;
        Ok(h || w || s)
    }

    /// Lists metadata for all sessions across cold storage, warm cache, and hot cache.
    pub async fn list_sessions(&self) -> Result<Vec<SessionMetadata>, Error> {
        let mut sessions = self.store.list().await?;
        let in_memory: Vec<_> = {
            let (warm, hot) = (self.warm.lock().await, self.hot.lock().await);
            warm.values()
                .map(|s| s.metadata.clone())
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
