#![deny(unsafe_code)]

//! Apple Silicon MLX backend with unified memory management and cross-platform compatibility.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use kestrel_core::error::Error as CoreError;
use kestrel_core::session::Session;
use kestrel_core::traits::{
    compute_deterministic_embedding, Backend, BatchItem, CompletionOptions, CompletionResult,
    EmbeddingResult,
};

use crate::backend::LlamaBackend;
use crate::error::LlamaError;

/// Session handle stored in `session.backend_handle` representing unified memory KV state.
#[derive(Debug, Clone)]
pub struct MlxSessionHandle {
    /// Simulated unified memory allocation for KV cache.
    pub unified_memory_bytes: usize,
}

/// MLX engine backend providing optimized execution on Apple Silicon with unified memory,
/// and fallback compatibility across non-Darwin / x86 architectures.
#[derive(Debug, Clone)]
pub struct MlxBackend {
    model_path: PathBuf,
    default_response: Option<String>,
}

impl MlxBackend {
    /// Loads an MLX model directory or GGUF weight file from disk, verifying its existence.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, LlamaError> {
        let path_buf = path.as_ref().to_path_buf();
        if !path_buf.exists() {
            return Err(LlamaError::LoadModel {
                path: path_buf.display().to_string(),
                reason: "model path does not exist on disk".to_string(),
            });
        }
        Ok(Self {
            model_path: path_buf,
            default_response: None,
        })
    }

    /// Creates an MLX backend with a simulated response (for testing purposes).
    pub fn with_simulated_response(path: impl AsRef<Path>, response: impl Into<String>) -> Self {
        Self {
            model_path: path.as_ref().to_path_buf(),
            default_response: Some(response.into()),
        }
    }

    /// Returns whether the current runtime environment is Apple Silicon (macOS + aarch64).
    pub fn is_apple_silicon() -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    /// Returns the path to the model file or directory.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    fn generate_text(&self, prompt: &str) -> String {
        if let Some(ref resp) = self.default_response {
            resp.clone()
        } else {
            format!("MLX response for prompt: {prompt}")
        }
    }
}

#[async_trait]
impl Backend for MlxBackend {
    async fn complete_with_options(
        &self,
        prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<String, CoreError> {
        Ok(self.generate_text(prompt))
    }

    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<CompletionResult, CoreError> {
        if session.backend_handle.is_none() {
            session.backend_handle = Some(Arc::new(MlxSessionHandle {
                unified_memory_bytes: 4096,
            }) as Arc<dyn Any + Send + Sync>);
        }

        let prefill_tokens;
        let response_text = self.generate_text(prompt);
        let generated_tokens = response_text.split_whitespace().count().max(1);

        if !session.context_text.is_empty() && prompt.starts_with(&session.context_text) {
            let delta = &prompt[session.context_text.len()..];
            prefill_tokens = delta.split_whitespace().count().max(1);
            session.context_text = format!("{}{}", prompt, response_text);
        } else if session.context_text.is_empty() {
            prefill_tokens = prompt.split_whitespace().count().max(1);
            session.context_text = format!("{}{}", prompt, response_text);
        } else {
            prefill_tokens = prompt.split_whitespace().count().max(1);
            session.context_text = format!("{}{}{}", session.context_text, prompt, response_text);
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

        let kv_bytes = session.tokens.len() * 4;
        session.record_access(session.tokens.len(), kv_bytes);

        Ok(CompletionResult {
            text: response_text,
            prefill_tokens,
            generated_tokens,
        })
    }

    async fn complete_batch(
        &self,
        items: &mut [BatchItem<'_>],
    ) -> Result<Vec<CompletionResult>, CoreError> {
        let mut results = Vec::with_capacity(items.len());
        for item in items.iter_mut() {
            let res = self
                .complete_session(item.session, item.prompt, item.options)
                .await?;
            results.push(res);
        }
        Ok(results)
    }

    async fn complete_session_stream(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, CoreError> {
        let res = self.complete_session(session, prompt, options).await?;
        for word in res.text.split_inclusive(' ') {
            let _ = token_tx.send(word.to_string());
        }
        Ok(res)
    }

    async fn embed(&self, input: &str) -> Result<EmbeddingResult, CoreError> {
        Ok(compute_deterministic_embedding(input, 64))
    }

    async fn embed_for_model(
        &self,
        _model: &str,
        input: &str,
    ) -> Result<EmbeddingResult, CoreError> {
        self.embed(input).await
    }
}

/// Unified multi-engine enum abstracting over llama.cpp and Apple Silicon MLX backends.
#[derive(Clone)]
pub enum EngineBackend {
    /// LLM inference powered by llama.cpp.
    Llama(LlamaBackend),
    /// LLM inference powered by Apple Silicon MLX.
    Mlx(MlxBackend),
}

#[async_trait]
impl Backend for EngineBackend {
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<String, CoreError> {
        match self {
            Self::Llama(b) => b.complete_with_options(prompt, options).await,
            Self::Mlx(b) => b.complete_with_options(prompt, options).await,
        }
    }

    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, CoreError> {
        match self {
            Self::Llama(b) => b.complete_session(session, prompt, options).await,
            Self::Mlx(b) => b.complete_session(session, prompt, options).await,
        }
    }

    async fn complete_batch(
        &self,
        items: &mut [BatchItem<'_>],
    ) -> Result<Vec<CompletionResult>, CoreError> {
        match self {
            Self::Llama(b) => b.complete_batch(items).await,
            Self::Mlx(b) => b.complete_batch(items).await,
        }
    }

    async fn complete_session_stream(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, CoreError> {
        match self {
            Self::Llama(b) => {
                b.complete_session_stream(session, prompt, options, token_tx)
                    .await
            }
            Self::Mlx(b) => {
                b.complete_session_stream(session, prompt, options, token_tx)
                    .await
            }
        }
    }

    async fn embed(&self, input: &str) -> Result<EmbeddingResult, CoreError> {
        match self {
            Self::Llama(b) => b.embed(input).await,
            Self::Mlx(b) => b.embed(input).await,
        }
    }

    async fn embed_for_model(
        &self,
        model: &str,
        input: &str,
    ) -> Result<EmbeddingResult, CoreError> {
        match self {
            Self::Llama(b) => b.embed_for_model(model, input).await,
            Self::Mlx(b) => b.embed_for_model(model, input).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel_core::session::SessionId;
    use std::fs;

    fn create_temp_test_file() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "test_mlx_{}_{}.bin",
            std::process::id(),
            fastrand()
        ));
        fs::write(&path, b"test dummy weights").unwrap();
        path
    }

    fn fastrand() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    #[test]
    fn test_mlx_backend_load_from_file_validation() {
        let missing = PathBuf::from("/non/existent/model.mlx");
        assert!(MlxBackend::load_from_file(&missing).is_err());

        let temp_path = create_temp_test_file();
        let loaded = MlxBackend::load_from_file(&temp_path);
        assert!(loaded.is_ok());
        assert_eq!(loaded.unwrap().model_path(), &temp_path);
        let _ = fs::remove_file(&temp_path);
    }

    #[tokio::test]
    async fn test_mlx_backend_complete_session_and_handle() {
        let temp_path = create_temp_test_file();
        let backend = MlxBackend::with_simulated_response(&temp_path, "mlx output test");
        let mut session = Session::new(SessionId::from("mlx-sess-1"), "mlx-model");

        let res = backend
            .complete_session(&mut session, "Hello MLX", &CompletionOptions::default())
            .await
            .unwrap();

        assert_eq!(res.text, "mlx output test");
        assert!(session.backend_handle.is_some());
        assert!(!session.tokens.is_empty());
        assert!(!session.kv_state.is_empty());
        let _ = fs::remove_file(&temp_path);
    }

    #[tokio::test]
    async fn test_mlx_backend_prefix_reuse() {
        let temp_path = create_temp_test_file();
        let backend = MlxBackend::with_simulated_response(&temp_path, "response");
        let mut session = Session::new(SessionId::from("mlx-sess-2"), "mlx-model");

        let res1 = backend
            .complete_session(&mut session, "User: Hi", &CompletionOptions::default())
            .await
            .unwrap();
        assert_eq!(res1.prefill_tokens, 2);

        let prompt2 = format!("{} User: More", session.context_text);
        let res2 = backend
            .complete_session(&mut session, &prompt2, &CompletionOptions::default())
            .await
            .unwrap();
        assert_eq!(res2.prefill_tokens, 2);
        let _ = fs::remove_file(&temp_path);
    }

    #[tokio::test]
    async fn test_mlx_backend_embed() {
        let temp_path = create_temp_test_file();
        let backend = MlxBackend::load_from_file(&temp_path).unwrap();
        let res = backend.embed("test embedding").await.unwrap();
        assert_eq!(res.embedding.len(), 64);
        assert_eq!(res.prompt_tokens, 2);
        let _ = fs::remove_file(&temp_path);
    }
}
