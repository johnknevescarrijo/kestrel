//! Common abstractions, traits, and types for LLM backends and session persistence.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Error;
use crate::session::{Session, SessionId, SessionMetadata, SessionSnapshot};

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

/// Result of an inference completion containing generated text and token metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionResult {
    /// Generated output string.
    pub text: String,
    /// Number of prefill/prompt tokens processed in this inference run.
    pub prefill_tokens: usize,
    /// Number of tokens generated during this completion.
    pub generated_tokens: usize,
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

    /// Executes inference within an existing session context, reusing KV state/prefix.
    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, Error>;

    /// Executes inference within an existing session, streaming generated tokens via channel.
    async fn complete_session_stream(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, Error> {
        let res = self.complete_session(session, prompt, options).await?;
        let _ = token_tx.send(res.text.clone());
        Ok(res)
    }
}

/// Persistence store interface for session snapshots (cold storage tier).
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Saves a session snapshot to cold storage.
    async fn save(&self, snapshot: &SessionSnapshot) -> Result<(), Error>;

    /// Loads a session snapshot from cold storage by ID.
    async fn load(&self, id: &SessionId) -> Result<Option<SessionSnapshot>, Error>;

    /// Deletes a session snapshot from cold storage. Returns true if it existed.
    async fn delete(&self, id: &SessionId) -> Result<bool, Error>;

    /// Lists metadata for all persisted sessions in cold storage.
    async fn list(&self) -> Result<Vec<SessionMetadata>, Error>;
}

/// Mock backend for testing without requiring real model weights.
#[derive(Debug, Clone)]
pub struct MockBackend {
    fixed_response: String,
    call_count: Arc<AtomicUsize>,
    kv_bytes_per_token: usize,
    fixed_kv_size: Option<usize>,
    attach_handle: bool,
}

impl MockBackend {
    /// Creates a new mock backend returning a fixed response.
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            fixed_response: response.into(),
            call_count: Arc::new(AtomicUsize::new(0)),
            kv_bytes_per_token: 4,
            fixed_kv_size: None,
            attach_handle: true,
        }
    }

    /// Configures simulated KV cache bytes generated per context token.
    pub fn with_kv_bytes_per_token(mut self, bytes: usize) -> Self {
        self.kv_bytes_per_token = bytes;
        self
    }

    /// Configures a fixed simulated KV cache footprint in bytes regardless of token count.
    pub fn with_fixed_kv_size(mut self, bytes: usize) -> Self {
        self.fixed_kv_size = Some(bytes);
        self
    }

    /// Configures whether to attach a mock backend handle (`Arc<()>`) to active sessions.
    pub fn with_attach_handle(mut self, attach: bool) -> Self {
        self.attach_handle = attach;
        self
    }

    /// Returns the number of times completion methods were called.
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

    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        if self.attach_handle && session.backend_handle.is_none() {
            session.backend_handle = Some(Arc::new(()));
        }

        let prefill_tokens;
        let generated_tokens = self.fixed_response.split_whitespace().count().max(1);

        if !session.context_text.is_empty() && prompt.starts_with(&session.context_text) {
            let delta = &prompt[session.context_text.len()..];
            prefill_tokens = delta.split_whitespace().count().max(1);
            session.context_text = format!("{}{}", prompt, self.fixed_response);
        } else if session.context_text.is_empty() {
            prefill_tokens = prompt.split_whitespace().count().max(1);
            session.context_text = format!("{}{}", prompt, self.fixed_response);
        } else {
            // Incremental turn appended to session
            prefill_tokens = prompt.split_whitespace().count().max(1);
            session.context_text =
                format!("{}{}{}", session.context_text, prompt, self.fixed_response);
        }

        let start_tok = session.tokens.len() as u32;
        let added_count = prefill_tokens + generated_tokens;
        for i in 0..added_count {
            session.tokens.push(start_tok + i as u32);
        }

        session.kv_state = session
            .tokens
            .iter()
            .flat_map(|&t| t.to_le_bytes())
            .collect();

        let kv_bytes = self
            .fixed_kv_size
            .unwrap_or(session.tokens.len() * self.kv_bytes_per_token);

        session.record_access(session.tokens.len(), kv_bytes);

        Ok(CompletionResult {
            text: self.fixed_response.clone(),
            prefill_tokens,
            generated_tokens,
        })
    }

    async fn complete_session_stream(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, Error> {
        let res = self.complete_session(session, prompt, options).await?;
        let words: Vec<&str> = res.text.split_inclusive(' ').collect();
        if words.is_empty() {
            let _ = token_tx.send(res.text.clone());
        } else {
            for word in words {
                let _ = token_tx.send(word.to_string());
            }
        }
        Ok(res)
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

    #[tokio::test]
    async fn test_mock_backend_complete_session_prefix_reuse() {
        let mock = MockBackend::new("mock response");
        let mut session = Session::new(SessionId::from("sess-1"), "mock-model");

        let res1 = mock
            .complete_session(
                &mut session,
                "User: How are you?",
                &CompletionOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(res1.text, "mock response");
        assert_eq!(res1.prefill_tokens, 4);

        // Second call with expanded prompt containing previous context as prefix
        let continuation = format!("{} User: Tell me more.", session.context_text);
        let res2 = mock
            .complete_session(&mut session, &continuation, &CompletionOptions::default())
            .await
            .unwrap();

        assert_eq!(res2.prefill_tokens, 4);
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn test_mock_backend_complete_session_stream() {
        let mock = MockBackend::new("hello world stream");
        let mut session = Session::new(SessionId::from("sess-stream"), "mock-model");
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let res = mock
            .complete_session_stream(
                &mut session,
                "Prompt stream",
                &CompletionOptions::default(),
                tx,
            )
            .await
            .unwrap();

        assert_eq!(res.text, "hello world stream");
        let mut streamed = Vec::new();
        while let Ok(token) = rx.try_recv() {
            streamed.push(token);
        }
        assert_eq!(streamed, vec!["hello ", "world ", "stream"]);
    }
}
