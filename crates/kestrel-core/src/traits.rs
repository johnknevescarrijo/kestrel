//! Common abstractions and traits for LLM backends.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Error;

/// Options for text completion requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionOptions {
    /// Maximum number of tokens to predict.
    pub max_tokens: usize,
}

impl Default for CompletionOptions {
    fn default() -> Self {
        Self { max_tokens: 128 }
    }
}

/// LLM backend abstraction interface.
///
/// Implementations may run natively via `llama.cpp` or simulate model output for testing.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Generates a text completion for the provided prompt using default options.
    async fn complete(&self, prompt: &str) -> Result<String, Error> {
        self.complete_with_options(prompt, &CompletionOptions::default())
            .await
    }

    /// Generates a text completion for the provided prompt with custom options.
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<String, Error>;
}

/// Mock backend for testing without requiring real model weights.
#[derive(Debug, Clone)]
pub struct MockBackend {
    fixed_response: String,
    call_count: Arc<AtomicUsize>,
}

impl MockBackend {
    /// Creates a new mock backend returning a fixed response.
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            fixed_response: response.into(),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Returns the number of times `complete_with_options` was called.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Backend for MockBackend {
    async fn complete_with_options(
        &self,
        _prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<String, Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(self.fixed_response.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_backend_default_complete() {
        let mock = MockBackend::new("hello from mock");
        let result = mock.complete("test prompt").await.unwrap();
        assert_eq!(result, "hello from mock");
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_backend_custom_options() {
        let mock = MockBackend::new("custom options response");
        let options = CompletionOptions { max_tokens: 64 };
        let result = mock
            .complete_with_options("test prompt", &options)
            .await
            .unwrap();
        assert_eq!(result, "custom options response");
        assert_eq!(mock.call_count(), 1);
    }
}
