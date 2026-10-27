#![deny(unsafe_code)]

//! CLI binary for Kestrel inference engine.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use kestrel_core::traits::{Backend, CompletionOptions};
use kestrel_core::{FileSessionStore, ModelRouter, SessionId, SessionManager, TierPolicy};
use kestrel_llama::{EngineBackend, LlamaBackend, MlxBackend};
use kestrel_server::{build_router, AppState};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    Llama,
    Mlx,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Llama => write!(f, "llama"),
            BackendKind::Mlx => write!(f, "mlx"),
        }
    }
}

#[derive(Parser)]
#[command(
    name = "kestrel",
    version,
    about = "Agent-native local LLM inference engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a text completion for a prompt using a local model
    Complete(CompleteArgs),

    /// Start an OpenAI-compatible HTTP inference server
    Serve(ServeArgs),
}

#[derive(Args)]
struct ServeArgs {
    /// Path to the primary model file
    #[arg(short, long, default_value = "./models/qwen2.5-0.5b-q4.gguf")]
    model: PathBuf,

    /// Execution backend engine (llama.cpp or mlx)
    #[arg(long, value_enum, default_value_t = BackendKind::Llama)]
    backend: BackendKind,

    /// Additional model paths to register for multi-model routing
    #[arg(long)]
    extra_model: Vec<PathBuf>,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Host address to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Directory for storing cold session snapshots
    #[arg(long, default_value = "./.kestrel/sessions")]
    session_dir: PathBuf,

    /// Hot in-memory session capacity
    #[arg(long, default_value_t = 16)]
    hot_capacity: usize,

    /// Hot KV cache memory budget in megabytes (falls back to KESTREL_HOT_BUDGET_MB env var)
    #[arg(long)]
    hot_budget_mb: Option<usize>,

    /// Warm KV cache memory budget in megabytes (falls back to KESTREL_WARM_BUDGET_MB env var)
    #[arg(long)]
    warm_budget_mb: Option<usize>,
}

#[derive(Args)]
struct CompleteArgs {
    /// The prompt to complete
    prompt: String,

    /// Path to the model file
    #[arg(short, long, default_value = "./models/qwen2.5-0.5b-q4.gguf")]
    model: PathBuf,

    /// Execution backend engine (llama.cpp or mlx)
    #[arg(long, value_enum, default_value_t = BackendKind::Llama)]
    backend: BackendKind,

    /// Optional session ID for persistent KV cache and context reuse
    #[arg(short, long)]
    session: Option<String>,

    /// Directory for storing cold session snapshots
    #[arg(long, default_value = "./.kestrel/sessions")]
    session_dir: PathBuf,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
}

fn load_engine_backend(kind: BackendKind, path: &std::path::Path) -> Result<EngineBackend> {
    match kind {
        BackendKind::Llama => {
            let b = LlamaBackend::load_from_file(path)
                .with_context(|| format!("failed to load GGUF model from '{}'", path.display()))?;
            Ok(EngineBackend::Llama(b))
        }
        BackendKind::Mlx => {
            let b = MlxBackend::load_from_file(path)
                .with_context(|| format!("failed to load MLX model from '{}'", path.display()))?;
            Ok(EngineBackend::Mlx(b))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Complete(args) => {
            if !args.model.exists() {
                anyhow::bail!(
                    "Model file not found at '{}'. Run ./scripts/fetch-test-model.sh or specify --model <path>.",
                    args.model.display()
                );
            }

            let backend = load_engine_backend(args.backend, &args.model)?;

            let options = CompletionOptions {
                max_tokens: args.max_tokens,
            };

            if let Some(session_id) = args.session {
                let store =
                    Arc::new(FileSessionStore::new(&args.session_dir).with_context(|| {
                        format!(
                            "failed to initialize session store at '{}'",
                            args.session_dir.display()
                        )
                    })?);

                let model_name = extract_model_name(&args.model);

                let manager = SessionManager::new(Arc::new(backend), store, 8, &model_name);
                let sess_id = SessionId::from(session_id);

                let result = manager
                    .complete(&sess_id, &args.prompt, &options)
                    .await
                    .with_context(|| {
                        format!("inference completion failed for session '{sess_id}'")
                    })?;

                info!(
                    session_id = %sess_id,
                    prefill_tokens = result.prefill_tokens,
                    generated_tokens = result.generated_tokens,
                    "Session completion succeeded"
                );

                println!("{}", result.text);
            } else {
                let completion = backend
                    .complete_with_options(&args.prompt, &options)
                    .await
                    .with_context(|| "inference completion failed")?;

                println!("{completion}");
            }
        }
        Commands::Serve(args) => {
            if !args.model.exists() {
                anyhow::bail!(
                    "Model file not found at '{}'. Run ./scripts/fetch-test-model.sh or specify --model <path>.",
                    args.model.display()
                );
            }

            let default_model_name = extract_model_name(&args.model);
            let default_backend = Arc::new(load_engine_backend(args.backend, &args.model)?);

            let mut router = ModelRouter::new(&default_model_name, default_backend);

            for extra in &args.extra_model {
                if !extra.exists() {
                    anyhow::bail!("Extra model file not found at '{}'.", extra.display());
                }
                let extra_name = extract_model_name(extra);
                let extra_backend = Arc::new(load_engine_backend(args.backend, extra)?);
                router.register(&extra_name, extra_backend);
            }

            let model_ids = router.model_ids();
            let router_backend = Arc::new(router);

            let store = Arc::new(FileSessionStore::new(&args.session_dir).with_context(|| {
                format!(
                    "failed to initialize session store at '{}'",
                    args.session_dir.display()
                )
            })?);

            let mut policy = TierPolicy::from_env();
            policy.max_hot_sessions = args.hot_capacity;
            if let Some(hot_mb) = args.hot_budget_mb {
                policy.hot_budget_bytes = hot_mb * 1024 * 1024;
            }
            if let Some(warm_mb) = args.warm_budget_mb {
                policy.warm_budget_bytes = warm_mb * 1024 * 1024;
            }

            let manager = Arc::new(SessionManager::with_policy(
                router_backend,
                store,
                policy,
                &default_model_name,
            ));

            let state =
                Arc::new(AppState::new(manager, &default_model_name).with_models(model_ids));
            let router = build_router(state);

            let addr = format!("{}:{}", args.host, args.port);
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .with_context(|| format!("failed to bind TCP listener to '{addr}'"))?;

            info!("Kestrel OpenAI HTTP server running at http://{addr}");
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .with_context(|| "HTTP server runtime encountered an error")?;
        }
    }

    Ok(())
}

/// Extracts the model name from a path in a cross-platform manner,
/// normalizing both Unix ('/') and Windows ('\') directory separators.
fn extract_model_name(path: &std::path::Path) -> String {
    let path_str = path.to_string_lossy();
    path_str
        .split(['/', '\\'])
        .rfind(|part| !part.is_empty())
        .unwrap_or("default-model")
        .to_string()
}

/// Listens for a CTRL+C signal across platforms (Linux, macOS, Windows) for graceful server shutdown.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
    info!("Shutdown signal received, shutting down gracefully...");
}
