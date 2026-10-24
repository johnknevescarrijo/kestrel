//! Error types for the `kestrel-server` crate.

use thiserror::Error;

/// HTTP server and endpoint handling errors.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Failed to bind to requested address/port.
    #[error(
        "failed to bind server to address '{addr}': {reason}. Check if port is already in use."
    )]
    BindFailed {
        /// Socket address.
        addr: String,
        /// Reason for failure.
        reason: String,
    },

    /// Request payload was malformed or could not be decoded.
    #[error("invalid request payload: {reason}. Ensure request follows OpenAI API specification.")]
    BadRequest {
        /// Reason describing why request was invalid.
        reason: String,
    },

    /// Internal server error when dispatching to engine.
    #[error("internal server failure: {reason}. Check server logs for full traceback.")]
    Internal {
        /// Reason for failure.
        reason: String,
    },
}
