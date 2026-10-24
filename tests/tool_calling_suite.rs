//! Comprehensive integration test suite for Milestone 3 (Reliable Tool Calling).
//!
//! Covers:
//! - Syntax parsing (<tool_call>, markdown blocks, raw json, stringified arguments, plain text)
//! - JSON Schema validation and error recovery
//! - Multi-turn retries on malformed syntax, unknown tools, and schema violations
//! - HTTP endpoint integration with OpenAI schema compatibility and HTTP 422 error signaling

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use kestrel_core::manager::SessionManager;
use kestrel_core::session::{Session, SessionId};
use kestrel_core::store::FileSessionStore;
use kestrel_core::traits::{Backend, CompletionOptions, CompletionResult};
use kestrel_core::Error;
use kestrel_server::handlers::AppState;
use kestrel_server::types::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage};
use kestrel_tools::error::ToolError;
use kestrel_tools::parser::parse_model_output;
use kestrel_tools::runtime::ToolRuntime;
use kestrel_tools::types::{ModelOutput, ToolDefinition};
use serde_json::{json, Value};
use tempfile::tempdir;
use tower::ServiceExt;

/// Mock backend that serves a queue of predefined responses sequentially.
struct SequentialMockBackend {
    responses: Arc<Mutex<VecDeque<String>>>,
    call_count: Arc<AtomicUsize>,
}

impl SequentialMockBackend {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Backend for SequentialMockBackend {
    async fn complete_with_options(
        &self,
        _prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<String, Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut queue = self.responses.lock().unwrap();
        if queue.len() > 1 {
            Ok(queue.pop_front().unwrap())
        } else {
            Ok(queue.front().cloned().unwrap_or_default())
        }
    }

    async fn complete_session(
        &self,
        session: &mut Session,
        prompt: &str,
        _options: &CompletionOptions,
    ) -> Result<CompletionResult, Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let resp = {
            let mut queue = self.responses.lock().unwrap();
            if queue.len() > 1 {
                queue.pop_front().unwrap()
            } else {
                queue.front().cloned().unwrap_or_default()
            }
        };

        let prefill_tokens = prompt.split_whitespace().count().max(1);
        let generated_tokens = resp.split_whitespace().count().max(1);
        session.context_text = format!("{}{}{}", session.context_text, prompt, resp);

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

        Ok(CompletionResult {
            text: resp,
            prefill_tokens,
            generated_tokens,
        })
    }
}

fn sample_calc_tool() -> ToolDefinition {
    ToolDefinition::new_function(
        "calculate",
        Some("Perform calculation".to_string()),
        json!({
            "type": "object",
            "properties": {
                "val": { "type": "integer" }
            },
            "required": ["val"]
        }),
    )
}

fn sample_weather_tool() -> ToolDefinition {
    ToolDefinition::new_function(
        "get_weather",
        Some("Get weather details".to_string()),
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["city"]
        }),
    )
}

// ==========================================
// 1. Parsing Tests
// ==========================================

#[test]
fn test_01_parse_tool_call_tag_simple() {
    let raw = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "get_weather");
            assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_02_parse_tool_call_tag_multiple() {
    let raw = "<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"val\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"val\": 2}}\n</tool_call>";
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].arguments, json!({"val": 1}));
            assert_eq!(calls[1].arguments, json!({"val": 2}));
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_03_parse_tool_call_tag_array() {
    let raw = "<tool_call>\n[{\"name\": \"t1\", \"arguments\": {}}, {\"name\": \"t2\", \"arguments\": {}}]\n</tool_call>";
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].name, "t1");
            assert_eq!(calls[1].name, "t2");
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_04_parse_stringified_arguments() {
    let raw = r#"<tool_call>{"name": "search", "arguments": "{\"query\": \"rust\"}"}</tool_call>"#;
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls[0].arguments, json!({"query": "rust"}));
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_05_parse_markdown_tool_call_block() {
    let raw = "```tool_call\n{\"name\": \"fetch\", \"arguments\": {\"url\": \"https://example.com\"}}\n```";
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "fetch");
            assert_eq!(calls[0].arguments, json!({"url": "https://example.com"}));
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_06_parse_markdown_json_block() {
    let raw = "```json\n{\"name\": \"calculate\", \"arguments\": {\"val\": 99}}\n```";
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "calculate");
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_07_parse_raw_json_object() {
    let raw = "{\"name\": \"calculate\", \"arguments\": {\"val\": 5}}";
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls[0].name, "calculate");
            assert_eq!(calls[0].arguments, json!({"val": 5}));
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_08_parse_raw_json_array() {
    let raw = "[{\"name\": \"t1\", \"arguments\": {}}, {\"name\": \"t2\", \"arguments\": {}}]";
    let parsed = parse_model_output(raw).expect("valid parse");
    match parsed {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
        }
        _ => panic!("expected ToolCalls"),
    }
}

#[test]
fn test_09_parse_plain_text_no_tools() {
    let raw = "Hello, world! I am a helpful AI assistant.";
    let parsed = parse_model_output(raw).expect("valid parse");
    assert_eq!(parsed, ModelOutput::Text(raw.to_string()));
}

// ==========================================
// 2. Retry and Schema Validation Tests
// ==========================================

#[tokio::test]
async fn test_10_retry_recovers_from_unclosed_tool_call_tag() {
    let responses = vec![
        "<tool_call>\n{\"name\": \"calculate\", \"arguments\": {\"val\": 10}".to_string(), // Unclosed tag!
        "<tool_call>\n{\"name\": \"calculate\", \"arguments\": {\"val\": 10}}\n</tool_call>"
            .to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-10"),
            "Calculate 10",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].name, "calculate"),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_11_retry_recovers_from_truncated_raw_json() {
    let responses = vec![
        "{\"name\": \"calculate\", \"arguments\": {\"val\":".to_string(), // Truncated raw JSON!
        "{\"name\": \"calculate\", \"arguments\": {\"val\": 42}}".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-11"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments, json!({"val": 42})),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_12_retry_recovers_from_invalid_argument_type() {
    let responses = vec![
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"val\": \"not-a-number\"}}</tool_call>".to_string(),
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"val\": 42}}</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-12"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments, json!({"val": 42})),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_13_retry_recovers_from_missing_required_property() {
    let responses = vec![
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {}}</tool_call>".to_string(), // Missing 'val'
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"val\": 7}}</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-13"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments, json!({"val": 7})),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_14_retry_recovers_from_unknown_tool() {
    let responses = vec![
        "<tool_call>{\"name\": \"nonexistent_tool\", \"arguments\": {}}</tool_call>".to_string(),
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"val\": 100}}</tool_call>"
            .to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-14"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].name, "calculate"),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_15_retry_recovers_from_enum_violation() {
    let responses = vec![
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\", \"unit\": \"kelvin\"}}</tool_call>".to_string(),
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\", \"unit\": \"celsius\"}}</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_weather_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-15"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments["unit"], "celsius"),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_16_retry_recovers_from_number_bounds_violation() {
    let bounds_tool = ToolDefinition::new_function(
        "rate",
        None,
        json!({
            "type": "object",
            "properties": {
                "score": { "type": "integer", "minimum": 1, "maximum": 5 }
            },
            "required": ["score"]
        }),
    );

    let responses = vec![
        "<tool_call>{\"name\": \"rate\", \"arguments\": {\"score\": 10}}</tool_call>".to_string(),
        "<tool_call>{\"name\": \"rate\", \"arguments\": {\"score\": 4}}</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-16"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &[bounds_tool],
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments["score"], 4),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_17_retry_recovers_from_string_length_violation() {
    let str_tool = ToolDefinition::new_function(
        "tag",
        None,
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "minLength": 5 }
            },
            "required": ["name"]
        }),
    );

    let responses = vec![
        "<tool_call>{\"name\": \"tag\", \"arguments\": {\"name\": \"abc\"}}</tool_call>"
            .to_string(),
        "<tool_call>{\"name\": \"tag\", \"arguments\": {\"name\": \"kestrel\"}}</tool_call>"
            .to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-17"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &[str_tool],
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments["name"], "kestrel"),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_18_retry_recovers_from_nested_object_schema_violation() {
    let nested_tool = ToolDefinition::new_function(
        "register_user",
        None,
        json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": { "type": "integer" }
                    },
                    "required": ["age"]
                }
            },
            "required": ["user"]
        }),
    );

    let responses = vec![
        "<tool_call>{\"name\": \"register_user\", \"arguments\": {\"user\": {\"age\": \"thirty\"}}}</tool_call>".to_string(),
        "<tool_call>{\"name\": \"register_user\", \"arguments\": {\"user\": {\"age\": 30}}}</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-18"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &[nested_tool],
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments["user"]["age"], 30),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_19_retry_recovers_from_array_items_schema_violation() {
    let array_tool = ToolDefinition::new_function(
        "set_tags",
        None,
        json!({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["tags"]
        }),
    );

    let responses = vec![
        "<tool_call>{\"name\": \"set_tags\", \"arguments\": {\"tags\": [1, 2, 3]}}</tool_call>".to_string(),
        "<tool_call>{\"name\": \"set_tags\", \"arguments\": {\"tags\": [\"rust\", \"ai\"]}}</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-19"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &[array_tool],
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => {
            assert_eq!(calls[0].arguments["tags"], json!(["rust", "ai"]));
        }
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_20_multiple_consecutive_retries_recovers() {
    let responses = vec![
        "<tool_call>{\"name\": \"calculate\"".to_string(), // 1. Unclosed tag
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {}}</tool_call>".to_string(), // 2. Missing required
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"val\": 500}}</tool_call>"
            .to_string(), // 3. Valid!
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-20"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 3);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments["val"], 500),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_21_exhausted_retries_returns_max_retries_exceeded() {
    let responses = vec![
        "<tool_call>{\"name\": \"calculate\"".to_string(), // Unclosed forever
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(2); // 1 initial + 2 retries = 3 attempts
    let tools = vec![sample_calc_tool()];
    let err = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-21"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .unwrap_err();

    assert_eq!(backend.call_count(), 3);
    match err {
        ToolError::MaxRetriesExceeded { attempts, .. } => assert_eq!(attempts, 3),
        other => panic!("expected MaxRetriesExceeded, got {other:?}"),
    }
}

#[tokio::test]
async fn test_22_required_tool_choice_rejects_plain_text() {
    let responses = vec![
        "I will just answer in plain English.".to_string(), // Rejected when require_tool = true
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"val\": 12}}</tool_call>"
            .to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let res = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-22"),
            "prompt",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            true, // require_tool = true!
        )
        .await
        .expect("recovery succeeded");

    assert_eq!(backend.call_count(), 2);
    match res.0 {
        ModelOutput::ToolCalls(calls) => assert_eq!(calls[0].arguments["val"], 12),
        _ => panic!("expected tool calls"),
    }
}

#[tokio::test]
async fn test_23_token_usage_accumulates_across_retries() {
    let responses = vec![
        "<tool_call>{\"name\": \"calculate\"".to_string(),
        "<tool_call>{\"name\": \"calculate\", \"arguments\": {\"val\": 1}}</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = SessionManager::new(backend.clone(), store, 4, "test-model");

    let runtime = ToolRuntime::new(3);
    let tools = vec![sample_calc_tool()];
    let (_, result) = runtime
        .complete_with_tools(
            &manager,
            &SessionId::new("retry-sess-23"),
            "Calculate 1",
            &CompletionOptions { max_tokens: 64 },
            &tools,
            false,
        )
        .await
        .expect("succeeded");

    assert_eq!(backend.call_count(), 2);
    // Prefill and generated tokens must sum both attempts
    assert!(result.prefill_tokens > 2);
    assert!(result.generated_tokens > 2);
}

// ==========================================
// 3. HTTP Server Tool Integration Tests
// ==========================================

#[tokio::test]
async fn test_24_http_chat_completions_tool_calling_success() {
    let responses = vec![
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Madrid\"}}\n</tool_call>".to_string(),
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = Arc::new(SessionManager::new(backend, store, 4, "mock-qwen"));
    let state = Arc::new(AppState::new(manager, "mock-qwen"));
    let router = kestrel_server::build_router(state);

    let payload = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![ChatMessage::user("What's the weather in Madrid?")],
        tools: Some(vec![sample_weather_tool()]),
        tool_choice: Some(json!("auto")),
        ..Default::default()
    };

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body = res.into_body().collect().await.unwrap().to_bytes();
    let resp: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));

    let tool_calls = resp.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool calls present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Madrid");
}

#[tokio::test]
async fn test_25_http_chat_completions_tool_validation_failure_422() {
    let responses = vec![
        "<tool_call>{\"name\": \"get_weather\"".to_string(), // Truncated forever
    ];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = Arc::new(SessionManager::new(backend, store, 4, "mock-qwen"));
    let state = Arc::new(AppState::new(manager, "mock-qwen"));
    let router = kestrel_server::build_router(state);

    let payload = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![ChatMessage::user("Weather please")],
        tools: Some(vec![sample_weather_tool()]),
        max_tool_retries: Some(1),
        ..Default::default()
    };

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json_resp: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json_resp["error"]["type"], "tool_call_validation_error");
    assert_eq!(json_resp["error"]["code"], "invalid_tool_call");
}

#[tokio::test]
async fn test_26_http_chat_completions_tool_choice_none_ignores_tools() {
    let responses = vec!["I will not use any tools.".to_string()];
    let backend = Arc::new(SequentialMockBackend::new(responses));
    let dir = tempdir().unwrap();
    let store = Arc::new(FileSessionStore::new(dir.path()).unwrap());
    let manager = Arc::new(SessionManager::new(backend, store, 4, "mock-qwen"));
    let state = Arc::new(AppState::new(manager, "mock-qwen"));
    let router = kestrel_server::build_router(state);

    let payload = ChatCompletionRequest {
        model: "mock-qwen".to_string(),
        messages: vec![ChatMessage::user("Hello")],
        tools: Some(vec![sample_weather_tool()]),
        tool_choice: Some(json!("none")), // Disables tools!
        ..Default::default()
    };

    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method(Method::POST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body = res.into_body().collect().await.unwrap().to_bytes();
    let resp: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    assert!(resp.choices[0].message.tool_calls.is_none());
    assert_eq!(resp.choices[0].message.content, "I will not use any tools.");
}
