//! Integration tests for session persistence, LRU caching, and prefix reuse.

use std::sync::Arc;

use kestrel_core::session::SessionId;
use kestrel_core::store::FileSessionStore;
use kestrel_core::traits::{CompletionOptions, MockBackend, SessionStore};
use kestrel_core::SessionManager;
use tempfile::tempdir;

#[tokio::test]
async fn test_session_prefix_reuse_reduces_prefill_tokens() {
    let dir = tempdir().expect("failed to create temp dir");
    let store = Arc::new(FileSessionStore::new(dir.path()).expect("store init failed"));
    let backend = Arc::new(MockBackend::new("Assistant: I am ready."));
    let manager = SessionManager::new(backend, store, 4, "mock-model");

    let session_id = SessionId::from("agent-dialog-1");
    let options = CompletionOptions::default();

    // Turn 1: Initial full prompt
    let prompt_turn_1 = "User: Hello, let us start our task.";
    let res1 = manager
        .complete(&session_id, prompt_turn_1, &options)
        .await
        .expect("turn 1 completion failed");

    assert_eq!(res1.text, "Assistant: I am ready.");
    assert_eq!(res1.prefill_tokens, 7); // "User: Hello, let us start our task." is 7 words

    // Turn 2: Incremental turn on the same session
    let prompt_turn_2 = "User: Step 1 is done.";
    let res2 = manager
        .complete(&session_id, prompt_turn_2, &options)
        .await
        .expect("turn 2 completion failed");

    // Only the new prompt tokens should be counted as prefill tokens
    assert_eq!(res2.prefill_tokens, 5); // "User: Step 1 is done." is 5 words
    assert!(
        res2.prefill_tokens < res1.prefill_tokens,
        "prefill tokens for incremental turn ({}) should be less than initial prompt ({})",
        res2.prefill_tokens,
        res1.prefill_tokens
    );

    // Verify session metadata is persisted in cold storage
    let metadata = manager
        .get_metadata(&session_id)
        .await
        .expect("get_metadata failed")
        .expect("session metadata should exist");

    assert_eq!(metadata.id, session_id);
    assert!(metadata.total_tokens > 0);
}

#[tokio::test]
async fn test_session_lru_eviction_and_restoration_from_disk() {
    let dir = tempdir().expect("failed to create temp dir");
    let store = Arc::new(FileSessionStore::new(dir.path()).expect("store init failed"));
    let backend = Arc::new(MockBackend::new("Ack"));

    // Hot cache capacity = 1
    let manager = SessionManager::new(backend, store.clone(), 1, "mock-model");

    let sess_a = SessionId::from("sess-a");
    let sess_b = SessionId::from("sess-b");
    let options = CompletionOptions::default();

    // 1. Run inference on session A
    let res_a1 = manager
        .complete(
            &sess_a,
            "Task A prompt with many words for prefill",
            &options,
        )
        .await
        .expect("sess_a turn 1 failed");
    assert_eq!(manager.hot_session_count().await, 1);
    assert_eq!(res_a1.prefill_tokens, 8);

    // 2. Run inference on session B -> evicts session A to disk
    let _res_b = manager
        .complete(&sess_b, "Task B short", &options)
        .await
        .expect("sess_b turn 1 failed");
    assert_eq!(manager.hot_session_count().await, 1);

    // Confirm session A snapshot was saved to disk
    let loaded_a = store.load(&sess_a).await.expect("store read failed");
    assert!(loaded_a.is_some(), "session A must be in cold storage");
    let snapshot_a = loaded_a.unwrap();
    assert!(snapshot_a
        .context_text
        .contains("Task A prompt with many words"));

    // 3. Run inference on session A again -> restored from disk, preserving prefix history
    let res_a2 = manager
        .complete(&sess_a, "Next A", &options)
        .await
        .expect("sess_a turn 2 failed");

    // Reused prefix from disk: only the delta "Next A" counted
    assert_eq!(res_a2.prefill_tokens, 2);
    assert!(
        res_a2.prefill_tokens < res_a1.prefill_tokens,
        "restored session A should only prefill delta tokens"
    );

    // Session A is now back in hot cache, evicting session B to disk
    assert_eq!(manager.hot_session_count().await, 1);
    let loaded_b = store.load(&sess_b).await.expect("store read failed");
    assert!(loaded_b.is_some(), "session B must now be in cold storage");
}
