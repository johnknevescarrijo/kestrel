//! Common abstractions, traits, and types for LLM backends and session persistence.

use async_trait::async_trait;

use crate::error::Error;
pub use crate::mock::MockBackend;
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

/// Result of an embedding computation containing the float vector and token count.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResult {
    /// L2-normalized float embedding vector.
    pub embedding: Vec<f32>,
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
}

/// Generates a deterministic normalized embedding vector for testing or fallback backends.
pub fn compute_deterministic_embedding(input: &str, dimensions: usize) -> EmbeddingResult {
    let prompt_tokens = input.split_whitespace().count().max(1);
    let dims = dimensions.max(1);
    let mut vec = vec![0.0f32; dims];

    for (i, byte) in input.bytes().enumerate() {
        let mut hasher: u64 = 0xcbf29ce484222325 ^ (byte as u64);
        hasher = hasher.wrapping_mul(0x100000001b3);
        hasher = hasher.wrapping_add(i as u64);
        let dim = (hasher as usize) % dims;
        let sign = if (hasher & 1) == 0 { 1.0f32 } else { -1.0f32 };
        vec[dim] += sign * ((byte as f32) / 255.0 + 0.1);
    }

    if vec.iter().all(|&v| v == 0.0) {
        for (i, v) in vec.iter_mut().enumerate() {
            *v = ((i + 1) as f32).sin();
        }
    }

    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }

    EmbeddingResult {
        embedding: vec,
        prompt_tokens,
    }
}

/// A single item within a batch completion request.
pub struct BatchItem<'a> {
    /// Mutable reference to the target session context.
    pub session: &'a mut Session,
    /// Prompt text to extend or evaluate.
    pub prompt: &'a str,
    /// Completion parameters.
    pub options: &'a CompletionOptions,
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

    /// Executes completion for a batch of sessions.
    ///
    /// By default, falls back to running `complete_session` sequentially.
    async fn complete_batch(
        &self,
        items: &mut [BatchItem<'_>],
    ) -> Result<Vec<CompletionResult>, Error> {
        let mut results = Vec::with_capacity(items.len());
        for item in items.iter_mut() {
            let res = self
                .complete_session(item.session, item.prompt, item.options)
                .await?;
            results.push(res);
        }
        Ok(results)
    }

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

    /// Generates a vector embedding for the provided input text.
    async fn embed(&self, input: &str) -> Result<EmbeddingResult, Error> {
        Ok(compute_deterministic_embedding(input, 64))
    }

    /// Generates a vector embedding targeting a specific model identifier.
    async fn embed_for_model(&self, _model: &str, input: &str) -> Result<EmbeddingResult, Error> {
        self.embed(input).await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_deterministic_embedding() {
        let res1 = compute_deterministic_embedding("hello world", 64);
        assert_eq!(res1.prompt_tokens, 2);
        assert_eq!(res1.embedding.len(), 64);

        let norm: f32 = res1.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);

        let res2 = compute_deterministic_embedding("hello world", 64);
        assert_eq!(res1.embedding, res2.embedding);
    }
}
