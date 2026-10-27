//! Multi-model router delegating inference requests to model-specific backends.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Error;
use crate::session::Session;
use crate::traits::{Backend, BatchItem, CompletionOptions, CompletionResult, EmbeddingResult};

/// Multi-model router distributing requests to registered backend engines per model identifier.
pub struct ModelRouter<B: Backend> {
    backends: HashMap<String, Arc<B>>,
    model_order: Vec<String>,
    default_model: String,
}

impl<B: Backend> ModelRouter<B> {
    /// Creates a new router with an initial default model and its backend.
    pub fn new(default_model: impl Into<String>, default_backend: Arc<B>) -> Self {
        let model_str = default_model.into();
        let mut backends = HashMap::new();
        backends.insert(model_str.clone(), default_backend);
        Self {
            backends,
            model_order: vec![model_str.clone()],
            default_model: model_str,
        }
    }

    /// Registers a backend for a specific model identifier.
    pub fn register(&mut self, model_id: impl Into<String>, backend: Arc<B>) {
        let id = model_id.into();
        if !self.backends.contains_key(&id) {
            self.model_order.push(id.clone());
        }
        self.backends.insert(id, backend);
    }

    /// Fluent builder method to attach an additional model backend.
    pub fn with_model(mut self, model_id: impl Into<String>, backend: Arc<B>) -> Self {
        self.register(model_id, backend);
        self
    }

    /// Returns the identifier of the default model.
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// Returns all registered model identifiers in order of registration.
    pub fn model_ids(&self) -> Vec<String> {
        self.model_order.clone()
    }

    /// Resolves the backend for a given model identifier.
    pub fn get_backend(&self, model_id: &str) -> Result<&Arc<B>, Error> {
        let target = if model_id.is_empty() || model_id == "default" {
            &self.default_model
        } else {
            model_id
        };

        self.backends
            .get(target)
            .ok_or_else(|| Error::ModelNotFound {
                model: model_id.to_string(),
                available: self.model_ids(),
            })
    }
}

#[async_trait]
impl<B: Backend + 'static> Backend for ModelRouter<B> {
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<String, Error> {
        let backend = self.get_backend(&self.default_model)?;
        backend.complete_with_options(prompt, options).await
    }

    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        let backend = self.get_backend(&session.metadata.model_id)?;
        backend.complete_session(session, prompt, options).await
    }

    async fn complete_session_stream(
        &self,
        session: &mut Session,
        prompt: &str,
        options: &CompletionOptions,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<CompletionResult, Error> {
        let backend = self.get_backend(&session.metadata.model_id)?;
        backend
            .complete_session_stream(session, prompt, options, token_tx)
            .await
    }

    async fn complete_batch(
        &self,
        items: &mut [BatchItem<'_>],
    ) -> Result<Vec<CompletionResult>, Error> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let first_model = &items[0].session.metadata.model_id;
        let same_model = items
            .iter()
            .all(|it| &it.session.metadata.model_id == first_model);

        if same_model {
            let backend = self.get_backend(first_model)?;
            backend.complete_batch(items).await
        } else {
            let mut results = Vec::with_capacity(items.len());
            for item in items.iter_mut() {
                let backend = self.get_backend(&item.session.metadata.model_id)?;
                let res = backend
                    .complete_session(item.session, item.prompt, item.options)
                    .await?;
                results.push(res);
            }
            Ok(results)
        }
    }

    async fn embed(&self, input: &str) -> Result<EmbeddingResult, Error> {
        let backend = self.get_backend(&self.default_model)?;
        backend.embed(input).await
    }

    async fn embed_for_model(&self, model: &str, input: &str) -> Result<EmbeddingResult, Error> {
        let backend = self.get_backend(model)?;
        backend.embed(input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockBackend;
    use crate::session::SessionId;

    #[tokio::test]
    async fn test_router_dispatch_to_model() {
        let backend1 = Arc::new(MockBackend::new("response-qwen"));
        let backend2 = Arc::new(MockBackend::new("response-llama"));

        let router = ModelRouter::new("qwen-0.5b", backend1.clone())
            .with_model("llama-3b", backend2.clone());

        assert_eq!(router.default_model(), "qwen-0.5b");
        assert_eq!(router.model_ids(), vec!["qwen-0.5b", "llama-3b"]);

        let mut sess1 = Session::new(SessionId::from("s1"), "qwen-0.5b");
        let res1 = router
            .complete_session(&mut sess1, "hi", &CompletionOptions::default())
            .await
            .unwrap();
        assert_eq!(res1.text, "response-qwen");

        let mut sess2 = Session::new(SessionId::from("s2"), "llama-3b");
        let res2 = router
            .complete_session(&mut sess2, "hi", &CompletionOptions::default())
            .await
            .unwrap();
        assert_eq!(res2.text, "response-llama");
    }

    #[tokio::test]
    async fn test_router_model_not_found() {
        let backend = Arc::new(MockBackend::new("ok"));
        let router = ModelRouter::new("default-model", backend);

        let err = router.get_backend("non-existent").unwrap_err();
        match err {
            Error::ModelNotFound { model, available } => {
                assert_eq!(model, "non-existent");
                assert_eq!(available, vec!["default-model"]);
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_router_batch_homogeneous_and_heterogeneous() {
        let b1 = Arc::new(MockBackend::new("res1"));
        let b2 = Arc::new(MockBackend::new("res2"));
        let router = ModelRouter::new("m1", b1.clone()).with_model("m2", b2.clone());

        let mut s1 = Session::new(SessionId::from("s1"), "m1");
        let mut s2 = Session::new(SessionId::from("s2"), "m1");
        let opts = CompletionOptions::default();

        let mut items = vec![
            BatchItem {
                session: &mut s1,
                prompt: "p1",
                options: &opts,
            },
            BatchItem {
                session: &mut s2,
                prompt: "p2",
                options: &opts,
            },
        ];

        let res = router.complete_batch(&mut items).await.unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(b1.batch_call_count(), 1);

        // Heterogeneous batch
        let mut s3 = Session::new(SessionId::from("s3"), "m2");
        let mut items2 = vec![
            BatchItem {
                session: &mut s1,
                prompt: "p1",
                options: &opts,
            },
            BatchItem {
                session: &mut s3,
                prompt: "p3",
                options: &opts,
            },
        ];
        let res2 = router.complete_batch(&mut items2).await.unwrap();
        assert_eq!(res2.len(), 2);
        assert_eq!(res2[0].text, "res1");
        assert_eq!(res2[1].text, "res2");
    }
}
