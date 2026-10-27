//! Session models, identifiers, and state snapshot representations.

use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Unique identifier for an agent or user session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Creates a new `SessionId` from a string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns a string slice of the session ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Metadata associated with a persistent or active inference session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Identifier of the session.
    pub id: SessionId,
    /// Associated model identifier.
    pub model_id: String,
    /// Timestamp of session creation in milliseconds since Unix epoch.
    pub created_at_epoch_ms: u64,
    /// Timestamp of the last access in milliseconds since Unix epoch.
    pub last_accessed_epoch_ms: u64,
    /// Total number of tokens currently in context history.
    pub total_tokens: usize,
    /// Size of KV cache or session memory footprint in bytes.
    pub kv_size_bytes: usize,
}

impl SessionMetadata {
    /// Creates a new `SessionMetadata` with current timestamps.
    pub fn new(id: SessionId, model_id: impl Into<String>) -> Self {
        let now = current_epoch_ms();
        Self {
            id,
            model_id: model_id.into(),
            created_at_epoch_ms: now,
            last_accessed_epoch_ms: now,
            total_tokens: 0,
            kv_size_bytes: 0,
        }
    }

    /// Updates the last accessed timestamp to now.
    pub fn touch(&mut self) {
        self.last_accessed_epoch_ms = current_epoch_ms();
    }
}

/// A serialized snapshot of a session saved in cold storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Metadata of the session.
    pub metadata: SessionMetadata,
    /// Cumulative text context of the session prompt and history.
    pub context_text: String,
    /// Token IDs corresponding to the context.
    pub tokens: Vec<u32>,
    /// Serialized KV cache or backend state bytes.
    pub kv_state: Vec<u8>,
}

/// An active in-memory session.
///
/// Holds metadata, context history, and optionally an active backend handle
/// (such as a live `llama.cpp` session) to avoid reloading KV state across requests.
pub struct Session {
    /// Session metadata.
    pub metadata: SessionMetadata,
    /// Cumulative text of prompt and completions.
    pub context_text: String,
    /// List of token IDs in the current context.
    pub tokens: Vec<u32>,
    /// Serialized KV state bytes if persisted or exported.
    pub kv_state: Vec<u8>,
    /// Live backend session handle (e.g. `Arc<Mutex<LlamaSession>>`), if currently resident in hot tier.
    pub backend_handle: Option<Arc<dyn Any + Send + Sync>>,
}

impl Session {
    /// Creates a new empty session.
    pub fn new(id: SessionId, model_id: impl Into<String>) -> Self {
        Self {
            metadata: SessionMetadata::new(id, model_id),
            context_text: String::new(),
            tokens: Vec::new(),
            kv_state: Vec::new(),
            backend_handle: None,
        }
    }

    /// Converts this session into a serializable snapshot.
    pub fn to_snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            metadata: self.metadata.clone(),
            context_text: self.context_text.clone(),
            tokens: self.tokens.clone(),
            kv_state: self.kv_state.clone(),
        }
    }

    /// Restores a session from a serialized snapshot.
    ///
    /// The live `backend_handle` is initialized to `None` and will be reconstructed
    /// by the backend on demand when inference is triggered.
    pub fn from_snapshot(snapshot: SessionSnapshot) -> Self {
        Self {
            metadata: snapshot.metadata,
            context_text: snapshot.context_text,
            tokens: snapshot.tokens,
            kv_state: snapshot.kv_state,
            backend_handle: None,
        }
    }

    /// Updates access timestamp and token counts.
    pub fn record_access(&mut self, total_tokens: usize, kv_size_bytes: usize) {
        self.metadata.touch();
        self.metadata.total_tokens = total_tokens;
        self.metadata.kv_size_bytes = kv_size_bytes;
    }

    /// Resets cached KV state and context text when switching to a different model.
    pub fn reset_for_model(&mut self, new_model: &str) {
        if self.metadata.model_id != new_model {
            self.context_text.clear();
            self.tokens.clear();
            self.kv_state.clear();
            self.backend_handle = None;
            self.metadata.total_tokens = 0;
            self.metadata.kv_size_bytes = 0;
            self.metadata.model_id = new_model.to_string();
        }
    }
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_snapshot_roundtrip() {
        let mut session = Session::new(SessionId::from("sess-1"), "model-test");
        session.context_text = "Hello world".to_string();
        session.tokens = vec![1, 2, 3];
        session.kv_state = vec![0xDE, 0xAD, 0xBE, 0xEF];

        let snapshot = session.to_snapshot();
        assert_eq!(snapshot.metadata.id.as_str(), "sess-1");
        assert_eq!(snapshot.context_text, "Hello world");
        assert_eq!(snapshot.tokens, vec![1, 2, 3]);

        let restored = Session::from_snapshot(snapshot);
        assert_eq!(restored.metadata.id.as_str(), "sess-1");
        assert_eq!(restored.context_text, "Hello world");
        assert_eq!(restored.tokens, vec![1, 2, 3]);
        assert_eq!(restored.kv_state, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(restored.backend_handle.is_none());
    }

    #[test]
    fn test_session_id_traits() {
        let id1 = SessionId::from("test-id");
        let id2: SessionId = "test-id".into();
        assert_eq!(id1, id2);
        assert_eq!(id1.as_str(), "test-id");
        assert_eq!(format!("{id1}"), "test-id");
    }
}
