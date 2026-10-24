#![deny(unsafe_code)]

//! Tool calling runtime, schema validation, and retry logic for Kestrel.

pub mod error;

pub use error::ToolError;
