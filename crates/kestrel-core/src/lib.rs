#![deny(unsafe_code)]

//! Core logic for Kestrel: session management, scheduler, and backend abstractions.

pub mod error;
pub mod traits;

pub use error::Error;
pub use traits::{Backend, CompletionOptions, MockBackend};
