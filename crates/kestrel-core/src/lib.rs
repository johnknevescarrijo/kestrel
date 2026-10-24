#![deny(unsafe_code)]

//! Core logic for Kestrel: session management, scheduler, and backend abstractions.

pub mod error;
pub mod lru;
pub mod manager;
pub mod session;
pub mod store;
pub mod traits;

pub use error::Error;
pub use lru::LruCache;
pub use manager::SessionManager;
pub use session::{Session, SessionId, SessionMetadata, SessionSnapshot};
pub use store::FileSessionStore;
pub use traits::{Backend, CompletionOptions, CompletionResult, MockBackend, SessionStore};
