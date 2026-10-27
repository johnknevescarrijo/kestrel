//! End-to-end contract test validating Kestrel against the official Python `openai` library.

use std::sync::Arc;
use std::time::Duration;

use kestrel_core::store::FileSessionStore;
use kestrel_core::traits::MockBackend;
use kestrel_core::SessionManager;
use kestrel_server::{build_router, AppState};
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_official_python_openai_contract() {
    let dir = tempdir().expect("Failed to create temporary session directory");
    let store =
        Arc::new(FileSessionStore::new(dir.path()).expect("Failed to initialize session store"));
    let backend = Arc::new(MockBackend::new("Hello from Kestrel OpenAI server!"));
    let model_name = "mock-qwen";
    let manager = Arc::new(SessionManager::new(backend, store, 8, model_name));
    let state = Arc::new(AppState::new(manager, model_name));
    let router = build_router(state);

    // Bind to ephemeral port 0
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind ephemeral TCP listener");
    let addr = listener.local_addr().expect("Failed to get local address");
    let port = addr.port();

    // Spawn server in background
    let server_handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, router).await {
            eprintln!("Test server stopped with error: {e}");
        }
    });

    // Give server a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let script_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/test-openai-contract.py")
        .canonicalize()
        .expect("Failed to find test-openai-contract.py");

    let python_bin = if cfg!(windows) { "python" } else { "python3" };
    let uv_available = Command::new("uv")
        .arg("--version")
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);

    let output = if uv_available {
        let mut cmd = Command::new("uv");
        cmd.args([
            "run",
            "--with",
            "openai",
            python_bin,
            script_path.to_str().expect("valid utf8 path"),
            "--port",
            &port.to_string(),
            "--model",
            model_name,
        ]);
        cmd.output()
            .await
            .expect("Failed to execute uv contract script")
    } else {
        let has_openai = Command::new(python_bin)
            .args(["-c", "import openai"])
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false);

        if !has_openai {
            println!("Skipping Python OpenAI contract external script because neither 'uv' nor 'python openai' package is installed in this environment");
            server_handle.abort();
            return;
        }

        let mut direct = Command::new(python_bin);
        direct.args([
            script_path.to_str().expect("valid utf8 path"),
            "--port",
            &port.to_string(),
            "--model",
            model_name,
        ]);
        direct
            .output()
            .await
            .expect("Failed to execute python contract script")
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("Python Contract Stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("Python Contract Stderr:\n{stderr}");
    }

    server_handle.abort();

    assert!(
        output.status.success(),
        "Python OpenAI contract test failed with status: {:?}",
        output.status
    );
}
