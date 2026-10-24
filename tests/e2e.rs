//! End-to-end inference test with actual GGUF model.
//!
//! Run with:
//! ```bash
//! ./scripts/fetch-test-model.sh
//! cargo test --test e2e -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kestrel_core::session::SessionId;
use kestrel_core::store::FileSessionStore;
use kestrel_core::traits::{Backend, CompletionOptions, SessionStore};
use kestrel_core::SessionManager;
use kestrel_llama::LlamaBackend;
use kestrel_tools::{format_tools_prompt, ToolDefinition, ToolError, ToolRuntime};
use serde_json::json;
use tempfile::tempdir;

fn resolve_model_path() -> PathBuf {
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

    model_path
}

#[tokio::test]
#[ignore = "requires downloaded model weights at ./models/qwen2.5-0.5b-q4.gguf"]
async fn test_qwen_e2e_completion() {
    let model_path = resolve_model_path();
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

#[tokio::test]
#[ignore = "requires downloaded model weights at ./models/qwen2.5-0.5b-q4.gguf"]
async fn test_qwen_e2e_session_prefill_reduction() {
    let model_path = resolve_model_path();
    let backend = LlamaBackend::load_from_file(&model_path).unwrap_or_else(|err| {
        panic!(
            "Failed to load Qwen2.5-0.5B model from {}: {err}",
            model_path.display()
        )
    });

    let dir = tempdir().expect("failed to create temp dir");
    let store = Arc::new(FileSessionStore::new(dir.path()).expect("failed to init store"));
    let manager = SessionManager::new(Arc::new(backend), store.clone(), 4, "qwen2.5-0.5b");

    let session_id = SessionId::from("e2e-persistent-sess-1");
    let options = CompletionOptions { max_tokens: 32 };

    // Turn 1: Long prompt establishing agent context
    let prompt_1 = "User: You are an autonomous coding agent. Our target project is named Kestrel, written in Rust, aiming to provide persistent KV cache sessions for local LLM inference on consumer workstations. Please acknowledge.\nAssistant:";
    let res1 = manager
        .complete(&session_id, prompt_1, &options)
        .await
        .expect("turn 1 failed");

    println!("\n=== Turn 1 ===");
    println!("Prompt: {prompt_1}");
    println!("Response: {}", res1.text);
    println!("Prefill tokens: {}", res1.prefill_tokens);
    println!("Generated tokens: {}", res1.generated_tokens);

    assert!(
        res1.prefill_tokens > 20,
        "Turn 1 prefill should exceed 20 tokens"
    );

    // Turn 2: Short follow-up question in the same session
    let prompt_2 = "User: What is the target language of Kestrel?\nAssistant:";
    let res2 = manager
        .complete(&session_id, prompt_2, &options)
        .await
        .expect("turn 2 failed");

    println!("\n=== Turn 2 ===");
    println!("Prompt: {prompt_2}");
    println!("Response: {}", res2.text);
    println!("Prefill tokens: {}", res2.prefill_tokens);
    println!("Generated tokens: {}", res2.generated_tokens);

    assert!(
        res2.prefill_tokens < res1.prefill_tokens,
        "Turn 2 prefill tokens ({}) MUST be fewer than Turn 1 prefill tokens ({})",
        res2.prefill_tokens,
        res1.prefill_tokens
    );

    // Verify kv_state is populated and persisted on disk after inference
    let snapshot = store
        .load(&session_id)
        .await
        .expect("failed to load snapshot")
        .expect("snapshot must exist");
    assert!(
        !snapshot.kv_state.is_empty(),
        "E2E session kv_state must not be empty after inference"
    );
    assert_eq!(snapshot.kv_state.len(), snapshot.tokens.len() * 4);
}

#[tokio::test]
#[ignore = "requires downloaded model weights at ./models/qwen2.5-0.5b-q4.gguf"]
async fn test_qwen_e2e_tool_calling() {
    let model_path = resolve_model_path();
    let backend = LlamaBackend::load_from_file(&model_path).unwrap_or_else(|err| {
        panic!(
            "Failed to load Qwen2.5-0.5B model from {}: {err}",
            model_path.display()
        )
    });

    let dir = tempdir().expect("failed to create temp dir");
    let store = Arc::new(FileSessionStore::new(dir.path()).expect("failed to init store"));
    let manager = SessionManager::new(Arc::new(backend), store.clone(), 4, "qwen2.5-0.5b");

    let tool = ToolDefinition::new_function(
        "get_weather",
        Some("Get current weather for a specified city".to_string()),
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "The city name, e.g. Tokyo" }
            },
            "required": ["city"]
        }),
    );

    let tools = vec![tool];
    let tools_prompt = format_tools_prompt(&tools);
    let prompt = format!(
        "<|im_start|>system\n{tools_prompt}\nUse the tool to answer user requests.<|im_end|>\n<|im_start|>user\nWhat is the weather in Tokyo? Call get_weather tool.<|im_end|>\n<|im_start|>assistant\n"
    );

    let session_id = SessionId::from("e2e-tool-calling-sess");
    let options = CompletionOptions { max_tokens: 32 };

    // 1. Direct session completion to verify raw inference
    let raw_res = manager
        .complete(&session_id, &prompt, &options)
        .await
        .expect("Raw completion should succeed");

    println!("\n=== Raw E2E Model Output ===");
    println!("{}", raw_res.text);

    let parsed = kestrel_tools::parse_model_output(&raw_res.text);
    println!("Parsed result: {:?}", parsed);

    // 2. Validate ToolRuntime execution with real model
    let runtime = ToolRuntime::new(1);
    let session_id_2 = SessionId::from("e2e-tool-calling-runtime-sess");
    let result = runtime
        .complete_with_tools(&manager, &session_id_2, &prompt, &options, &tools, false)
        .await;

    match result {
        Ok((output, res)) => {
            println!("ToolRuntime succeeded: {:?}", output);
            assert!(res.generated_tokens > 0);
        }
        Err(ToolError::MaxRetriesExceeded {
            attempts,
            last_error,
        }) => {
            println!(
                "ToolRuntime executed retries (attempts: {attempts}) and signaled structured error: {last_error}"
            );
            assert_eq!(attempts, 2);
        }
        Err(other) => panic!("Unexpected error from ToolRuntime: {other:?}"),
    }
}
