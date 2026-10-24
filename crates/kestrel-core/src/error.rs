//! Error types for the `kestrel-core` crate.

use thiserror::Error;

/// Core errors occurring across backend, session, and scheduler operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Failed to execute model inference.
    #[error("backend inference failed: {reason}. Check model configuration or context length.")]
    InferenceFailed {
        /// Underlying failure reason.
        reason: String,
    },

    /// Failed to initialize or load a model from disk.
    #[error("model initialization failed for '{path}': {reason}. Verify that the GGUF file exists and is valid.")]
    ModelInitFailed {
        /// Path to the model file.
        path: String,
        /// Underlying failure reason.
        reason: String,
    },

    /// Requested session could not be found.
    #[error("session '{id}' not found. Ensure the session was created and has not expired.")]
    SessionNotFound {
        /// Identifier of the missing session.
        id: String,
    },

    /// An internal or unexpected task failure.
    #[error("internal runtime error: {reason}. Please check server logs for details.")]
    Internal {
        /// Description of the internal error.
        reason: String,
    },
}
