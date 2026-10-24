//! End-to-end inference test with actual GGUF model.
//!
//! Run with:
//! ```bash
//! ./scripts/fetch-test-model.sh
//! cargo test --test e2e -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use kestrel_core::traits::{Backend, CompletionOptions};
use kestrel_llama::LlamaBackend;

#[tokio::test]
#[ignore = "requires downloaded model weights at ./models/qwen2.5-0.5b-q4.gguf"]
async fn test_qwen_e2e_completion() {
    let candidate_paths = [
        PathBuf::from("./models/qwen2.5-0.5b-q4.gguf"),
        PathBuf::from("../../models/qwen2.5-0.5b-q4.gguf"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen2.5-0.5b-q4.gguf"),
    ];

    let model_path = candidate_paths
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("./models/qwen2.5-0.5b-q4.gguf"));

    if !model_path.exists() {
        panic!(
            "Model file not found at {}. Please run ./scripts/fetch-test-model.sh first.",
            model_path.display()
        );
    }

    let backend = LlamaBackend::load_from_file(&model_path).unwrap_or_else(|err| {
        panic!(
            "Failed to load Qwen2.5-0.5B model from {}: {err}",
            model_path.display()
        )
    });

    let options = CompletionOptions { max_tokens: 32 };
    let prompt = "User: Say hello in one short sentence.\nAssistant:";

    let completion = backend
        .complete_with_options(prompt, &options)
        .await
        .expect("Completion failed");

    println!("Prompt: {prompt}");
    println!("Response: {completion}");

    assert!(
        !completion.trim().is_empty(),
        "Completion should not be empty"
    );
}
