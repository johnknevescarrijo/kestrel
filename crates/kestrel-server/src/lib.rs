#![deny(unsafe_code)]

//! HTTP server layer for Kestrel, providing OpenAI-compatible endpoints.

pub mod error;

pub use error::ServerError;
