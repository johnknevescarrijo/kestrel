//! Mock LLM backend implementation for unit and integration testing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Error;
use crate::session::Session;
use crate::traits::{Backend, BatchItem, CompletionOptions, CompletionResult};

/// Mock backend for testing without requiring real model weights.
#[derive(Debug, Clone)]
pub struct MockBackend {
    fixed_response: String,
    call_count: Arc<AtomicUsize>,
    batch_call_count: Arc<AtomicUsize>,
    forward_pass_delay: Option<std::time::Duration>,
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
            batch_call_count: Arc::new(AtomicUsize::new(0)),
            forward_pass_delay: None,
            kv_bytes_per_token: 4,
            fixed_kv_size: None,
            attach_handle: true,
        }
    }

    /// Configures simulated forward pass latency.
    pub fn with_forward_pass_delay(mut self, delay: std::time::Duration) -> Self {
        self.forward_pass_delay = Some(delay);
        self
    }

    /// Returns the number of times batch execution was invoked.
    pub fn batch_call_count(&self) -> usize {
        self.batch_call_count.load(Ordering::SeqCst)
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

    fn process_mock_session(&self, session: &mut Session, prompt: &str) -> CompletionResult {
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

        CompletionResult {
            text: self.fixed_response.clone(),
            prefill_tokens,
            generated_tokens,
        }
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
        if let Some(delay) = self.forward_pass_delay {
            tokio::time::sleep(delay).await;
        }
        Ok(self.process_mock_session(session, prompt))
    }

    async fn complete_batch(
        &self,
        items: &mut [BatchItem<'_>],
    ) -> Result<Vec<CompletionResult>, Error> {
        self.batch_call_count.fetch_add(1, Ordering::SeqCst);
        if let Some(delay) = self.forward_pass_delay {
            tokio::time::sleep(delay).await;
        }
        let mut results = Vec::with_capacity(items.len());
        for item in items.iter_mut() {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            results.push(self.process_mock_session(item.session, item.prompt));
        }
        Ok(results)
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
    use crate::session::SessionId;

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

    #[tokio::test]
    async fn test_mock_backend_complete_batch() {
        let mock = MockBackend::new("mock batch response")
            .with_forward_pass_delay(std::time::Duration::from_millis(5));
        let mut session1 = Session::new(SessionId::from("sess-b1"), "mock-model");
        let mut session2 = Session::new(SessionId::from("sess-b2"), "mock-model");
        let options = CompletionOptions::default();

        let mut items = vec![
            BatchItem {
                session: &mut session1,
                prompt: "Batch prompt 1",
                options: &options,
            },
            BatchItem {
                session: &mut session2,
                prompt: "Batch prompt 2",
                options: &options,
            },
        ];

        let results = mock.complete_batch(&mut items).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].text, "mock batch response");
        assert_eq!(results[1].text, "mock batch response");
        assert_eq!(mock.batch_call_count(), 1);
        assert_eq!(mock.call_count(), 2);
    }
}
