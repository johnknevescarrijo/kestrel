#![deny(unsafe_code)]

//! CLI binary for Kestrel inference engine.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use kestrel_core::traits::{Backend, CompletionOptions};
use kestrel_core::{FileSessionStore, SessionId, SessionManager};
use kestrel_llama::LlamaBackend;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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
    /// Generate a text completion for a prompt using a local GGUF model
    Complete(CompleteArgs),
}

#[derive(Args)]
struct CompleteArgs {
    /// The prompt to complete
    prompt: String,

    /// Path to the GGUF model file
    #[arg(short, long, default_value = "./models/qwen2.5-0.5b-q4.gguf")]
    model: PathBuf,

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

            let backend = LlamaBackend::load_from_file(&args.model).with_context(|| {
                format!("failed to load GGUF model from '{}'", args.model.display())
            })?;

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

                let model_name = args
                    .model
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("default-model");

                let manager = SessionManager::new(Arc::new(backend), store, 8, model_name);
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
    }

    Ok(())
}
