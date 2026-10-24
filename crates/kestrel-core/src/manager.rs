//! Session lifecycle management with hot/cold tiering and LRU cache eviction.

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::error::Error;
use crate::lru::LruCache;
use crate::session::{Session, SessionId, SessionMetadata};
use crate::traits::{Backend, CompletionOptions, CompletionResult, SessionStore};

/// Orchestrates session state, hot (in-memory) caching, cold (disk) persistence,
/// and backend inference delegation.
pub struct SessionManager<B: Backend, S: SessionStore> {
    backend: Arc<B>,
    store: Arc<S>,
    lru: Arc<Mutex<LruCache<SessionId, Session>>>,
    model_id: String,
    hot_capacity: usize,
}

impl<B: Backend, S: SessionStore> SessionManager<B, S> {
    /// Creates a new `SessionManager`.
    pub fn new(
        backend: Arc<B>,
        store: Arc<S>,
        hot_capacity: usize,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            store,
            lru: Arc::new(Mutex::new(LruCache::new(hot_capacity))),
            model_id: model_id.into(),
            hot_capacity,
        }
    }

    /// Returns the configured capacity for hot in-memory sessions.
    pub fn hot_capacity(&self) -> usize {
        self.hot_capacity
    }

    /// Returns the number of sessions currently active in the hot (in-memory) tier.
    pub async fn hot_session_count(&self) -> usize {
        let lru = self.lru.lock().await;
        lru.len()
    }

    /// Executes completion within the specified session, managing promotion and persistence.
    pub async fn complete(
        &self,
        session_id: &SessionId,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        debug!(session = %session_id, "Resolving session for inference");

        // 1. Check hot cache first (remove temporarily while running inference to release LRU lock)
        let hot_session = {
            let mut lru = self.lru.lock().await;
            lru.remove(session_id)
        };

        let mut session = match hot_session {
            Some(hot) => hot,
            None => {
                debug!(session = %session_id, "Hot cache miss, querying cold storage");
                match self.store.load(session_id).await? {
                    Some(snapshot) => {
                        info!(session = %session_id, "Restored session snapshot from cold storage");
                        Session::from_snapshot(snapshot)
                    }
                    None => {
                        info!(session = %session_id, "Creating fresh session");
                        Session::new(session_id.clone(), &self.model_id)
                    }
                }
            }
        };

        // 3. Execute backend inference with session context
        let result = match self
            .backend
            .complete_session(&mut session, prompt, options)
            .await
        {
            Ok(res) => res,
            Err(err) => {
                // Re-insert session into cache before failing so partial state is preserved
                let mut lru = self.lru.lock().await;
                lru.put(session_id.clone(), session);
                return Err(err);
            }
        };

        // 4. Persist updated snapshot to cold store
        let snapshot = session.to_snapshot();
        self.store.save(&snapshot).await?;

        // 5. Place session back into hot LRU cache, handling eviction if capacity reached
        let evicted = {
            let mut lru = self.lru.lock().await;
            lru.put(session_id.clone(), session)
        };

        if let Some((evicted_id, evicted_session)) = evicted {
            info!(session = %evicted_id, "Evicting session from hot cache to cold storage");
            let evicted_snapshot = evicted_session.to_snapshot();
            self.store.save(&evicted_snapshot).await?;
        }

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
        debug!(session = %session_id, "Resolving session for streaming inference");

        // 1. Check hot cache first (remove temporarily while running inference to release LRU lock)
        let hot_session = {
            let mut lru = self.lru.lock().await;
            lru.remove(session_id)
        };

        let mut session = match hot_session {
            Some(hot) => hot,
            None => {
                debug!(session = %session_id, "Hot cache miss, querying cold storage");
                match self.store.load(session_id).await? {
                    Some(snapshot) => {
                        info!(session = %session_id, "Restored session snapshot from cold storage");
                        Session::from_snapshot(snapshot)
                    }
                    None => {
                        info!(session = %session_id, "Creating fresh session");
                        Session::new(session_id.clone(), &self.model_id)
                    }
                }
            }
        };

        // 3. Execute backend streaming inference with session context
        let result = match self
            .backend
            .complete_session_stream(&mut session, prompt, options, token_tx)
            .await
        {
            Ok(res) => res,
            Err(err) => {
                // Re-insert session into cache before failing so partial state is preserved
                let mut lru = self.lru.lock().await;
                lru.put(session_id.clone(), session);
                return Err(err);
            }
        };

        // 4. Persist updated snapshot to cold store
        let snapshot = session.to_snapshot();
        self.store.save(&snapshot).await?;

        // 5. Place session back into hot LRU cache, handling eviction if capacity reached
        let evicted = {
            let mut lru = self.lru.lock().await;
            lru.put(session_id.clone(), session)
        };

        if let Some((evicted_id, evicted_session)) = evicted {
            info!(session = %evicted_id, "Evicting session from hot cache to cold storage");
            let evicted_snapshot = evicted_session.to_snapshot();
            self.store.save(&evicted_snapshot).await?;
        }

        Ok(result)
    }

    /// Manually evicts a session from the hot cache to cold storage.
    pub async fn evict_to_cold(&self, session_id: &SessionId) -> Result<(), Error> {
        let session = {
            let mut lru = self.lru.lock().await;
            lru.remove(session_id)
        };

        if let Some(session) = session {
            let snapshot = session.to_snapshot();
            self.store.save(&snapshot).await?;
            info!(session = %session_id, "Manually evicted session from hot cache to cold storage");
        }

        Ok(())
    }

    /// Retrieves session metadata from hot cache or cold storage.
    pub async fn get_metadata(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<SessionMetadata>, Error> {
        {
            let lru = self.lru.lock().await;
            if let Some(session) = lru.peek(session_id) {
                return Ok(Some(session.metadata.clone()));
            }
        }

        match self.store.load(session_id).await? {
            Some(snapshot) => Ok(Some(snapshot.metadata)),
            None => Ok(None),
        }
    }

    /// Deletes a session from both hot cache and cold storage.
    pub async fn delete_session(&self, session_id: &SessionId) -> Result<bool, Error> {
        let in_hot = {
            let mut lru = self.lru.lock().await;
            lru.remove(session_id).is_some()
        };

        let in_cold = self.store.delete(session_id).await?;
        Ok(in_hot || in_cold)
    }

    /// Lists metadata for all sessions stored in cold storage.
    pub async fn list_sessions(&self) -> Result<Vec<SessionMetadata>, Error> {
        let mut cold_list = self.store.list().await?;

        // Update with freshest metadata from hot cache if present
        let lru = self.lru.lock().await;
        for meta in &mut cold_list {
            if let Some(hot_sess) = lru.peek(&meta.id) {
                *meta = hot_sess.metadata.clone();
            }
        }

        Ok(cold_list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionId;
    use crate::store::FileSessionStore;
    use crate::traits::MockBackend;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_manager_complete_lifecycle_and_eviction() {
        let dir = tempdir().unwrap();
        let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
        let backend = Arc::new(MockBackend::new("mock answer"));

        // Capacity of 1 hot session
        let manager = SessionManager::new(backend, store.clone(), 1, "test-model");

        let sess_a = SessionId::from("sess-a");
        let sess_b = SessionId::from("sess-b");

        let res1 = manager
            .complete(&sess_a, "Hello A", &CompletionOptions::default())
            .await
            .unwrap();
        assert_eq!(res1.text, "mock answer");
        assert_eq!(manager.hot_session_count().await, 1);

        // sess_b will evict sess_a from hot cache to cold storage
        let res2 = manager
            .complete(&sess_b, "Hello B", &CompletionOptions::default())
            .await
            .unwrap();
        assert_eq!(res2.text, "mock answer");
        assert_eq!(manager.hot_session_count().await, 1);

        // Verify sess_a is saved in cold storage
        let loaded_a = store.load(&sess_a).await.unwrap();
        assert!(loaded_a.is_some());

        // Now calling sess_a again should restore it from cold storage
        let res3 = manager
            .complete(&sess_a, "Continue A", &CompletionOptions::default())
            .await
            .unwrap();
        assert_eq!(res3.text, "mock answer");
        assert_eq!(manager.hot_session_count().await, 1);
    }
}
