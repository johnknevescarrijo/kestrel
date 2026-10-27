//! Error types for the `kestrel-core` crate.

use thiserror::Error;

/// Core errors occurring across backend, session, and scheduler operations.
#[derive(Debug, Clone, Error)]
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

    /// Failed to serialize or deserialize session state.
    #[error("session serialization failed: {reason}. Ensure session snapshot is not corrupted.")]
    SessionSerializationFailed {
        /// Underlying failure reason.
        reason: String,
    },

    /// An error occurred during file storage I/O for a session.
    #[error("session storage I/O failed at '{path}': {reason}. Check directory permissions and disk space.")]
    SessionStorageFailed {
        /// Path to the session file or directory.
        path: String,
        /// Underlying failure reason.
        reason: String,
    },

    /// Invalid session identifier supplied (e.g. path traversal attempt).
    #[error("invalid session id '{id}': {reason}. Session IDs must only contain alphanumeric characters, hyphens, and underscores.")]
    InvalidSessionId {
        /// The invalid session ID.
        id: String,
        /// Reason why it is invalid.
        reason: String,
    },

    /// Requested model was not found in the router registry.
    #[error("model '{model}' not found; available models: {available:?}. Specify a registered model identifier.")]
    ModelNotFound {
        /// Requested model name.
        model: String,
        /// List of registered model names.
        available: Vec<String>,
    },

    /// An internal or unexpected task failure.
    #[error("internal runtime error: {reason}. Please check server logs for details.")]
    Internal {
        /// Description of the internal error.
        reason: String,
    },
}
