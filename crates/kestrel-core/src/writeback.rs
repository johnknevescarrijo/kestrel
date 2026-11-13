//! Asynchronous cold-storage write-back coordinator for session persistence.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{oneshot, Mutex};
use tracing::warn;

use crate::error::Error;
use crate::session::{SessionId, SessionMetadata, SessionSnapshot};
use crate::traits::SessionStore;

/// Ticket identifying a queued persistence operation that callers can await.
#[derive(Debug, Clone)]
pub(crate) struct PersistTicket {
    session_id: SessionId,
    seq: u64,
    generation: u64,
}

#[derive(Default)]
struct SessionWritebackState {
    latest: Option<(u64, SessionSnapshot)>,
    saving_meta: Option<SessionMetadata>,
    in_flight: bool,
    next_seq: u64,
    completed_seq: u64,
    generation: u64,
    last_error: Option<(u64, Error)>,
    waiters: Vec<oneshot::Sender<()>>,
}

/// Coordinates per-session background write-back and synchronous eviction persistence.
///
/// Guarantees at most one `SessionStore::save` call runs at any time per `SessionId`,
/// preventing out-of-order overwrites and temporary-file collisions while keeping
/// routine `commit_session` persistence off the critical inference path.
pub(crate) struct WritebackCoordinator {
    store: Arc<dyn SessionStore>,
    state: Arc<Mutex<HashMap<SessionId, SessionWritebackState>>>,
}

impl WritebackCoordinator {
    /// Creates a new coordinator backed by `store`.
    pub(crate) fn new(store: Arc<dyn SessionStore>) -> Self {
        Self {
            store,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Schedules an asynchronous write-back without waiting for disk I/O.
    pub(crate) async fn enqueue_writeback(&self, snapshot: SessionSnapshot) {
        let mut map = self.state.lock().await;
        Self::enqueue_locked(&mut map, &self.store, &self.state, snapshot);
    }

    /// Schedules persistence for a cold-evicted session and returns a wait ticket.
    pub(crate) async fn enqueue_cold(&self, snapshot: SessionSnapshot) -> PersistTicket {
        let mut map = self.state.lock().await;
        Self::enqueue_locked(&mut map, &self.store, &self.state, snapshot)
    }

    /// Schedules persistence for a batch of cold-evicted sessions and returns their tickets.
    pub(crate) async fn enqueue_cold_batch(
        &self,
        snapshots: Vec<SessionSnapshot>,
    ) -> Vec<PersistTicket> {
        if snapshots.is_empty() {
            return Vec::new();
        }
        let mut map = self.state.lock().await;
        snapshots
            .into_iter()
            .map(|snap| Self::enqueue_locked(&mut map, &self.store, &self.state, snap))
            .collect()
    }

    fn enqueue_locked(
        map: &mut HashMap<SessionId, SessionWritebackState>,
        store: &Arc<dyn SessionStore>,
        state: &Arc<Mutex<HashMap<SessionId, SessionWritebackState>>>,
        snapshot: SessionSnapshot,
    ) -> PersistTicket {
        let id = snapshot.metadata.id.clone();
        let entry = map.entry(id.clone()).or_default();
        entry.next_seq += 1;
        let seq = entry.next_seq;
        let generation = entry.generation;
        entry.latest = Some((seq, snapshot));
        entry.last_error = None;

        if !entry.in_flight {
            entry.in_flight = true;
            Self::spawn_worker(id.clone(), Arc::clone(store), Arc::clone(state));
        }

        PersistTicket {
            session_id: id,
            seq,
            generation,
        }
    }

    fn spawn_worker(
        id: SessionId,
        store: Arc<dyn SessionStore>,
        state: Arc<Mutex<HashMap<SessionId, SessionWritebackState>>>,
    ) {
        tokio::spawn(async move {
            loop {
                let (seq, snap, gen) = {
                    let mut map = state.lock().await;
                    let Some(entry) = map.get_mut(&id) else {
                        break;
                    };
                    match entry.latest.take() {
                        Some((seq, snap)) => {
                            entry.saving_meta = Some(snap.metadata.clone());
                            (seq, snap, entry.generation)
                        }
                        None => {
                            entry.in_flight = false;
                            entry.saving_meta = None;
                            for tx in entry.waiters.drain(..) {
                                let _ = tx.send(());
                            }
                            if entry.last_error.is_none() {
                                map.remove(&id);
                            }
                            break;
                        }
                    }
                };

                let res = store.save(&snap).await;

                let mut map = state.lock().await;
                if let Some(entry) = map.get_mut(&id) {
                    entry.saving_meta = None;
                    if entry.generation == gen {
                        match res {
                            Ok(()) => {
                                entry.completed_seq = entry.completed_seq.max(seq);
                            }
                            Err(err) => {
                                warn!(session = %id, error = %err, "Background write-back failed");
                                entry.last_error = Some((seq, err));
                            }
                        }
                    }
                    for tx in entry.waiters.drain(..) {
                        let _ = tx.send(());
                    }
                }
            }
        });
    }

    /// Waits until the specified ticket has been persisted (or superseded/cancelled).
    pub(crate) async fn wait_for_ticket(&self, ticket: &PersistTicket) -> Result<(), Error> {
        loop {
            let rx = {
                let mut map = self.state.lock().await;
                let Some(entry) = map.get_mut(&ticket.session_id) else {
                    return Ok(());
                };
                if entry.generation != ticket.generation {
                    return Ok(());
                }
                if entry.completed_seq >= ticket.seq {
                    return Ok(());
                }
                if let Some((err_seq, ref err)) = entry.last_error {
                    if err_seq >= ticket.seq {
                        let err = err.clone();
                        if !entry.in_flight && entry.latest.is_none() {
                            entry.last_error = None;
                            map.remove(&ticket.session_id);
                        }
                        return Err(err);
                    }
                }
                if !entry.in_flight && entry.latest.is_none() {
                    return Ok(());
                }
                let (tx, rx) = oneshot::channel();
                entry.waiters.push(tx);
                rx
            };
            let _ = rx.await;
        }
    }

    /// Waits for all given tickets to finish persisting.
    pub(crate) async fn wait_for_tickets(&self, tickets: Vec<PersistTicket>) -> Result<(), Error> {
        for ticket in &tickets {
            self.wait_for_ticket(ticket).await?;
        }
        Ok(())
    }

    /// Waits until no write-back or eviction save is pending or in flight for `id`.
    pub(crate) async fn wait_for_session(&self, id: &SessionId) {
        loop {
            let rx = {
                let mut map = self.state.lock().await;
                let Some(entry) = map.get_mut(id) else {
                    return;
                };
                if !entry.in_flight && entry.latest.is_none() {
                    return;
                }
                let (tx, rx) = oneshot::channel();
                entry.waiters.push(tx);
                rx
            };
            let _ = rx.await;
        }
    }

    /// Cancels any queued write-back for `id` and waits for any in-flight save to finish.
    ///
    /// Returns `true` if a pending or in-flight write-back existed for `id`.
    pub(crate) async fn cancel_and_wait(&self, id: &SessionId) -> bool {
        let mut had_pending = false;
        loop {
            let rx = {
                let mut map = self.state.lock().await;
                let Some(entry) = map.get_mut(id) else {
                    return had_pending;
                };
                if entry.latest.take().is_some() {
                    had_pending = true;
                }
                if entry.saving_meta.is_some() || entry.in_flight {
                    had_pending = true;
                }
                entry.generation += 1;
                entry.last_error = None;
                if !entry.in_flight {
                    map.remove(id);
                    return had_pending;
                }
                let (tx, rx) = oneshot::channel();
                entry.waiters.push(tx);
                rx
            };
            let _ = rx.await;
        }
    }

    /// Flushes all pending and in-flight write-backs across all sessions.
    pub(crate) async fn flush(&self) -> Result<(), Error> {
        loop {
            let rx = {
                let mut map = self.state.lock().await;
                let mut wait_rx = None;
                for entry in map.values_mut() {
                    if entry.in_flight || entry.latest.is_some() {
                        let (tx, rx) = oneshot::channel();
                        entry.waiters.push(tx);
                        wait_rx = Some(rx);
                        break;
                    }
                }
                match wait_rx {
                    Some(rx) => rx,
                    None => {
                        let first_err = map
                            .values_mut()
                            .find_map(|e| e.last_error.take().map(|(_, err)| err));
                        map.retain(|_, e| e.in_flight || e.latest.is_some());
                        return match first_err {
                            Some(err) => Err(err),
                            None => Ok(()),
                        };
                    }
                }
            };
            let _ = rx.await;
        }
    }

    /// Returns metadata for all sessions with queued or in-flight writes.
    pub(crate) async fn pending_metadata(&self) -> Vec<SessionMetadata> {
        let map = self.state.lock().await;
        map.values()
            .filter_map(|entry| {
                entry
                    .latest
                    .as_ref()
                    .map(|(_, snap)| snap.metadata.clone())
                    .or_else(|| entry.saving_meta.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct MockFailingOrGatedStore {
        fail: AtomicBool,
        save_entered: AtomicBool,
        gate: Option<Arc<tokio::sync::Notify>>,
        gate_once: AtomicBool,
        saves: Mutex<Vec<SessionSnapshot>>,
        save_calls: AtomicUsize,
    }

    impl MockFailingOrGatedStore {
        fn new(fail: bool, gate: Option<Arc<tokio::sync::Notify>>) -> Self {
            Self {
                fail: AtomicBool::new(fail),
                save_entered: AtomicBool::new(false),
                gate,
                gate_once: AtomicBool::new(true),
                saves: Mutex::new(Vec::new()),
                save_calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl SessionStore for MockFailingOrGatedStore {
        async fn save(&self, snapshot: &SessionSnapshot) -> Result<(), Error> {
            self.save_calls.fetch_add(1, Ordering::SeqCst);
            self.save_entered.store(true, Ordering::SeqCst);
            if let Some(ref g) = self.gate {
                if self.gate_once.swap(false, Ordering::SeqCst) {
                    g.notified().await;
                }
            }
            if self.fail.load(Ordering::SeqCst) {
                return Err(Error::SessionStorageFailed {
                    path: "mock".to_string(),
                    reason: "simulated disk full".to_string(),
                });
            }
            self.saves.lock().await.push(snapshot.clone());
            Ok(())
        }

        async fn load(&self, _id: &SessionId) -> Result<Option<SessionSnapshot>, Error> {
            Ok(None)
        }

        async fn delete(&self, _id: &SessionId) -> Result<bool, Error> {
            Ok(false)
        }

        async fn list(&self) -> Result<Vec<SessionMetadata>, Error> {
            Ok(Vec::new())
        }
    }

    fn make_snap(id: &str, tokens: usize) -> SessionSnapshot {
        let mut meta = SessionMetadata::new(SessionId::from(id), "test");
        meta.total_tokens = tokens;
        SessionSnapshot {
            metadata: meta,
            context_text: format!("ctx-{tokens}"),
            tokens: vec![1; tokens],
            kv_state: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_writeback_coalesces_queued_snapshots() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let store = Arc::new(MockFailingOrGatedStore::new(false, Some(Arc::clone(&gate))));
        let wb = WritebackCoordinator::new(Arc::clone(&store) as Arc<dyn SessionStore>);

        // First write enters save() and blocks on gate.
        wb.enqueue_writeback(make_snap("s1", 1)).await;
        while !store.save_entered.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }

        // Queue several updates while first save is blocked; only latest (tokens=5) should be saved next.
        for t in 2..=5 {
            wb.enqueue_writeback(make_snap("s1", t)).await;
        }

        gate.notify_one();
        wb.flush().await.unwrap();

        let saves = store.saves.lock().await;
        assert_eq!(saves.len(), 2);
        assert_eq!(saves[0].metadata.total_tokens, 1);
        assert_eq!(saves[1].metadata.total_tokens, 5);
    }

    #[tokio::test]
    async fn test_writeback_flush_surfaces_background_error() {
        let store = Arc::new(MockFailingOrGatedStore::new(true, None));
        let wb = WritebackCoordinator::new(Arc::clone(&store) as Arc<dyn SessionStore>);

        wb.enqueue_writeback(make_snap("err-sess", 3)).await;
        let res = wb.flush().await;
        assert!(matches!(res, Err(Error::SessionStorageFailed { .. })));

        // Subsequent flush after error was drained succeeds.
        wb.flush().await.unwrap();
    }
}
