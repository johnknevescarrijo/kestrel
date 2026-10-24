//! Error types for the `kestrel-llama` crate.

use thiserror::Error;

/// Errors arising from llama.cpp backend operations.
#[derive(Debug, Error)]
pub enum LlamaError {
    /// Failed to load model weights from the specified path.
    #[error("failed to load model from '{path}': {reason}. Check if the file path is correct and accessible.")]
    LoadModel {
        /// Path that was requested.
        path: String,
        /// Underlying error description.
        reason: String,
    },

    /// Failed to create an inference session.
    #[error(
        "failed to create llama.cpp session: {reason}. Ensure sufficient memory is available."
    )]
    SessionCreation {
        /// Underlying error description.
        reason: String,
    },

    /// Failed to evaluate or feed context into the session.
    #[error("failed to advance context: {reason}. Try reducing prompt size.")]
    ContextAdvance {
        /// Underlying error description.
        reason: String,
    },

    /// Failed to execute sampling or prediction.
    #[error("completion failed during generation: {reason}.")]
    Completion {
        /// Underlying error description.
        reason: String,
    },

    /// Blocking task execution panicked or was cancelled.
    #[error("tokio blocking task failed: {reason}.")]
    TaskJoin {
        /// Underlying error description.
        reason: String,
    },
}

impl From<LlamaError> for kestrel_core::Error {
    fn from(err: LlamaError) -> Self {
        match err {
            LlamaError::LoadModel { path, reason } => {
                kestrel_core::Error::ModelInitFailed { path, reason }
            }
            LlamaError::TaskJoin { reason } => kestrel_core::Error::Internal { reason },
            other => kestrel_core::Error::InferenceFailed {
                reason: other.to_string(),
            },
        }
    }
}
