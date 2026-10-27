#![deny(unsafe_code)]

//! Core logic for Kestrel: session management, scheduler, and backend abstractions.

pub mod error;
pub mod lru;
pub mod manager;
pub mod scheduler;
pub mod session;
pub mod store;
pub mod tier;
pub mod traits;

pub use error::Error;
pub use lru::LruCache;
pub use manager::SessionManager;
pub use scheduler::{BatchConfig, BatchMetricsSnapshot, BatchScheduler};
pub use session::{Session, SessionId, SessionMetadata, SessionSnapshot};
pub use store::{is_windows_reserved_name, FileSessionStore};
pub use tier::{TierLevel, TierMetricsCounters, TierMetricsSnapshot, TierPolicy};
pub use traits::{
    Backend, BatchItem, CompletionOptions, CompletionResult, MockBackend, SessionStore,
};
