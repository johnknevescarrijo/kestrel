//! Integration tests using MockBackend.

use kestrel_core::traits::{Backend, CompletionOptions, MockBackend};

#[tokio::test]
async fn test_mock_backend_integration_flow() {
    let mock = MockBackend::new("The capital of France is Paris.");
    let options = CompletionOptions { max_tokens: 32 };

    let response = mock
        .complete_with_options("What is the capital of France?", &options)
        .await
        .expect("MockBackend should not fail");

    assert_eq!(response, "The capital of France is Paris.");
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test]
async fn test_mock_backend_multiple_invocations() {
    let mock = MockBackend::new("Repeated response");

    for i in 1..=5 {
        let response = mock
            .complete("Hello")
            .await
            .expect("MockBackend should succeed");
        assert_eq!(response, "Repeated response");
        assert_eq!(mock.call_count(), i);
    }
}
