# 🦅 kestrel

> **Agent-native local inference engine for consumer hardware (8–24 GB VRAM).**

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Backend](https://img.shields.io/badge/llama.cpp-GGUF-blue.svg)](https://github.com/ggerganov/llama.cpp)
[![API](https://img.shields.io/badge/API-OpenAI%20Compatible-green.svg)](https://platform.openai.com/docs/api-reference)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-lightgrey.svg)](LICENSE)

---

## 📌 Overview & Value Proposition

While traditional inference servers (such as vLLM or plain llama.cpp) treat each request as ephemeral, **kestrel** is designed from day one for **autonomous agent** workflows. In agentic interactions, an agent performs dozens of incremental turns (executing commands, reading files, reasoning). Recomputing the entire context on every turn introduces latency and massive computational waste.

**kestrel** flips this paradigm: **Session > Request**.

- 🧠 **Persistent Sessions & Prefix Reuse:** KV cache state is indexed by a `SessionId`. Subsequent calls within the same session process only the new tokens (delta prefill), dropping time-to-first-token (TTFT) to near zero.
- ⚡ **3-Tier KV Cache Management:**
  - **Hot (VRAM):** Active sessions keep their backend handles on GPU for zero-latency execution.
  - **Warm (RAM):** Idle sessions keep snapshots in host system memory, freeing VRAM for active sessions.
  - **Cold (Disk via `bincode`):** Sessions inactive for longer periods are demoted atomically to local disk, restorable at any time.
  - Automatic governance driven by configurable memory limits (`KESTREL_HOT_BUDGET_MB`, `KESTREL_WARM_BUDGET_MB`) and an LRU eviction policy.
- 🛠️ **Structured & Resilient Tool Calling:** Flexible parser capable of extracting calls from `<tool_call>` tags, markdown code blocks, or raw JSON. Strict validation using JSON Schema (`jsonschema`) with an **automatic feedback & retry loop**: if the model provides invalid arguments or violates the schema, Kestrel feeds the error back into the model for self-correction.
- 🚀 **Continuous Batching:** The `BatchScheduler` groups concurrent requests across hot sessions into micro-batch windows, maximizing GPU throughput without degrading single-request latency.
- 🔌 **Drop-in OpenAI Compatibility:** `/v1/chat/completions` (JSON and SSE streaming) and `/v1/models` endpoints, integrating seamlessly with Open WebUI, Cursor, Continue, LangChain, or the official OpenAI SDK.
- 📊 **Native Observability:** Structured spans powered by `tracing`, contextual logs, and full operational metrics exposed at `/kestrel/metrics` (tier hit rates, promotions, demotions, and batching stats).

---

## 🧱 Workspace Architecture

The project is structured as a modular Cargo workspace with clear boundaries of responsibility and `#![deny(unsafe_code)]` across all pure modules:

```
kestrel/
├── Cargo.toml                # Workspace manifest (Rust 2021, rust-version 1.80)
├── crates/
│   ├── kestrel-core/         # Pure logic: sessions, LRU, tiering, BatchScheduler, and traits
│   ├── kestrel-llama/        # Isolated llama.cpp bindings (GGUF, KV cache restore)
│   ├── kestrel-tools/        # Tool call parsing, JSON Schema validation, and retry loop
│   ├── kestrel-server/       # Axum HTTP server, OpenAI compatibility, and /kestrel/* routes
│   └── kestrel-cli/          # Main binary: 'serve' and 'complete' CLI commands
├── scripts/                  # Helper scripts (e.g., test model download)
├── tests/                    # Integration, stress, and E2E test suite
├── benches/                  # Criterion benchmarks (KV cache reuse)
└── models/                   # Local directory for GGUF models (.gitignore)
```

| Crate | Primary Responsibility |
|---|---|
| **`kestrel-core`** | Lifecycle management (`SessionManager`), tiering policies (`TierPolicy`), memory budget calculations, disk persistence (`FileSessionStore`), queueing and continuous scheduling (`BatchScheduler`). |
| **`kestrel-llama`** | Direct integration with `llama_cpp`. Isolates low-level calls, manages context handles, and executes forward passes with token prefix restoration. |
| **`kestrel-tools`** | Tool invocation extraction, JSON Schema compilation (`jsonschema`), argument validation, and multi-turn corrective feedback injection. |
| **`kestrel-server`** | Axum-based HTTP layer. Provides OpenAI-compatible endpoints, Server-Sent Events (SSE) streaming support, automatic `SessionId` resolution, and session/metrics administration endpoints. |
| **`kestrel-cli`** | CLI entry point (via `clap`). Supports both interactive/direct CLI completions and launching the production HTTP server. |

---

## 🚀 Quickstart

### 1. Prerequisites
- **Rust:** Version 1.80 or higher (`rustup default stable`).
- **Build tools:** `cmake` (>= 3.14) and a compatible C/C++ compiler (GCC, Clang, or MSVC) to compile `llama.cpp`.

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential cmake

# Fedora
sudo dnf install -y gcc-c++ cmake
```

### 2. Download the Test Model
We provide a helper script to download **Qwen2.5-0.5B-Instruct (Q4_K_M GGUF)** (~398 MB), ideal for local development and testing:

```bash
./scripts/fetch-test-model.sh
```
The model will be saved to `./models/qwen2.5-0.5b-q4.gguf`.

### 3. Build the Project (Cross-Platform)

Kestrel compiles into a single, self-contained binary across **Linux**, **macOS**, and **Windows**:

```bash
# Build release binary (native target)
cargo build --release -p kestrel-cli

# The executable will be available at:
# - Linux / macOS: target/release/kestrel
# - Windows:       target/release/kestrel.exe
```

#### Cross-Platform Support
- **Linux:** Tested on `x86_64-unknown-linux-gnu` with GCC/Clang and CMake.
- **macOS:** Tested on Apple Silicon (`aarch64-apple-darwin`) with Xcode Command Line Tools.
- **Windows:** Tested on `x86_64-pc-windows-msvc` with MSVC and CMake. Graceful shutdown (`CTRL+C`), directory separator normalization (`/` and `\`), atomic session persistence, and Windows reserved device name validation (`CON`, `PRN`, `AUX`, `NUL`, etc.) are built-in.

---

## 💻 Command Line Usage (CLI)

The `kestrel` binary provides two main subcommands: `complete` and `serve`.

### Direct Completion (`complete`)

```bash
# One-shot completion without session
./target/release/kestrel complete "Explain in one sentence what Rust is."

# Completion with persistent session (KV cache preserved across runs!)
./target/release/kestrel complete "My name is Alice and I like coffee." --session session-user-1
./target/release/kestrel complete "What is my name and what do I like?" --session session-user-1
```
> **Note:** On the second invocation with the same `--session`, Kestrel reuses the already processed KV cache, running prefill only on the newly added question.

### Starting the HTTP Server (`serve`)

```bash
./target/release/kestrel serve \
  --model ./models/qwen2.5-0.5b-q4.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  --hot-budget-mb 2048 \
  --warm-budget-mb 4096 \
  --hot-capacity 16
```

---

## 🌐 HTTP API Usage Examples

Kestrel starts on default port `8080` with full support for OpenAI request/response schemas and headers.

### 1. Standard Completion (JSON) via `curl`

Specify the session using the `X-Kestrel-Session-Id` header (or via the `user` field in the request body):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Kestrel-Session-Id: ag-sess-001" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [
      {"role": "user", "content": "Hello! I am setting up a local inference server."}
    ],
    "max_tokens": 128
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-1729792000",
  "object": "chat.completion",
  "created": 1729792000,
  "model": "qwen2.5-0.5b-q4.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! Great to see you exploring local inference. How can I help with your setup?",
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 22,
    "total_tokens": 41
  }
}
```

### 2. Real-Time Streaming (SSE)

Set `"stream": true`:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Kestrel-Session-Id: ag-sess-001" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [
      {"role": "user", "content": "Count to 5 slowly."}
    ],
    "stream": true
  }'
```

The server emits standard SSE events:
```
data: {"id":"chatcmpl-1729792010","object":"chat.completion.chunk","created":1729792010,"model":"qwen2.5-0.5b-q4.gguf","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-1729792010","object":"chat.completion.chunk","created":1729792010,"model":"qwen2.5-0.5b-q4.gguf","choices":[{"index":0,"delta":{"role":null,"content":"1"},"finish_reason":null}]}

...
data: [DONE]
```

### 3. Integration with Official OpenAI Python SDK

You can use the official `openai` Python SDK without any changes to client code other than setting the base URL:

```python
from openai import OpenAI

# Initialize pointing to local Kestrel server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",  # Kestrel is single-user local by default
)

# Send request specifying persistent session
response = client.chat.completions.create(
    model="qwen2.5-0.5b-q4.gguf",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    extra_headers={"X-Kestrel-Session-Id": "research-sess-01"},
)

print(response.choices[0].message.content)

# Second turn reuses the same session and benefits from KV cache reuse:
response2 = client.chat.completions.create(
    model="qwen2.5-0.5b-q4.gguf",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": response.choices[0].message.content},
        {"role": "user", "content": "What is its estimated population?"},
    ],
    extra_headers={"X-Kestrel-Session-Id": "research-sess-01"},
)

print(response2.choices[0].message.content)
```

### 4. Reliable Tool Calling with JSON Schema

Kestrel parses JSON schemas declared in `tools` and validates syntax and constraints before responding to the client:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Kestrel-Session-Id: agent-tool-sess" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [
      {"role": "user", "content": "What is the weather like in Curitiba?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather conditions for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": { "type": "string" },
              "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

When the model generates a valid tool invocation, Kestrel returns `finish_reason: "tool_calls"`:

```json
{
  "id": "chatcmpl-1729792030",
  "object": "chat.completion",
  "created": 1729792030,
  "model": "qwen2.5-0.5b-q4.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_1729792030",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Curitiba\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": { "prompt_tokens": 142, "completion_tokens": 28, "total_tokens": 170 }
}
```

> 🔁 **Automatic Self-Correction (Retry Loop):** If the model outputs malformed JSON or parameters violating the schema, Kestrel captures the validation error against `jsonschema`, injects a corrective feedback prompt explaining the error, and prompts the model to regenerate the response (up to the configured retry limit, default: 3).

### 5. Native Structured Output (JSON Mode & JSON Schema)

Kestrel natively supports guaranteed structured output conforming to arbitrary JSON Schemas or raw JSON objects via the standard OpenAI `response_format` parameter:

```bash
# Enforce strict JSON Schema compliance
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [
      {"role": "user", "content": "Extract the person: Alice is 30 years old and lives in Berlin."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person_info",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "city": { "type": "string" }
          },
          "required": ["name", "age", "city"],
          "additionalProperties": false
        }
      }
    }
  }'
```

```bash
# Raw JSON mode
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [{"role": "user", "content": "Return a JSON map with 3 colors and hex codes"}],
    "response_format": { "type": "json_object" }
  }'
```

### 6. Vector Embeddings (`POST /v1/embeddings`)

Generate dense vector embeddings with optional dimension overrides and L2 normalization:

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "input": ["Local semantic search vector", "Document retrieval"],
    "dimensions": 64
  }'
```

### 7. Multi-Model Routing & Apple Silicon MLX Backend

Serve multiple models simultaneously and choose between `llama.cpp` or Apple Silicon `mlx` backends:

```bash
# Serve primary model with extra models registered in ModelRouter
kestrel serve \
  --model ./models/qwen2.5-0.5b-q4.gguf \
  --extra-model ./models/llama-3.2-1b.gguf \
  --backend llama \
  --port 8080

# On Apple Silicon (macOS) using native MLX:
kestrel serve \
  --model ./models/qwen2.5-mlx \
  --backend mlx \
  --port 8080
```

---

## 🛠️ Admin & Observability Endpoints

In addition to the standard OpenAI endpoints, Kestrel provides custom routes for agent session governance and observability:

### `GET /v1/models`
Lists models currently loaded on the server.

### `GET /kestrel/sessions`
Lists all active and stored sessions with aggregated memory metrics.

```bash
curl -s http://localhost:8080/kestrel/sessions
```
```json
{
  "sessions": [
    {
      "id": "ag-sess-001",
      "tier": "hot",
      "tokens": 41,
      "kv_size_bytes": 1048576,
      "last_accessed": 1729792000,
      "model": "qwen2.5-0.5b-q4.gguf"
    }
  ],
  "stats": {
    "hot_sessions": 1,
    "warm_sessions": 0,
    "total_sessions": 1,
    "hot_capacity": 16,
    "hot_used_bytes": 1048576,
    "warm_used_bytes": 0,
    "hot_budget_bytes": 2147483648,
    "warm_budget_bytes": 4294967296
  }
}
```

### `GET /kestrel/sessions/:id`
Displays metadata for a specific session (current tier, KV cache byte size, token count).

### `DELETE /kestrel/sessions/:id`
Explicitly evicts a session from memory (VRAM/RAM) and deletes its disk snapshot.

```bash
curl -s -X DELETE http://localhost:8080/kestrel/sessions/ag-sess-001
```

### `GET /kestrel/metrics`
Real-time operational metrics for KV cache tiering and continuous batching:

```bash
curl -s http://localhost:8080/kestrel/metrics
```
```json
{
  "hot_hits": 142,
  "warm_hits": 28,
  "cold_hits": 9,
  "misses": 5,
  "total_requests": 184,
  "hit_rate": 0.9728,
  "hot_hit_rate": 0.7717,
  "promotions_warm_to_hot": 28,
  "promotions_cold_to_hot": 9,
  "demotions_hot_to_warm": 35,
  "demotions_warm_to_cold": 12,
  "hot_sessions": 12,
  "warm_sessions": 30,
  "hot_used_bytes": 1610612736,
  "warm_used_bytes": 2147483648,
  "hot_budget_bytes": 2147483648,
  "warm_budget_bytes": 4294967296,
  "batch_metrics": {
    "total_batches": 64,
    "total_batched_requests": 150,
    "max_observed_batch_size": 8,
    "average_batch_size": 2.34
  }
}
```

---

## ⚙️ Configuration & Environment Variables

### `SessionId` Resolution
Kestrel resolves the session identifier in the following precedence order:
1. HTTP Header `X-Kestrel-Session-Id`
2. `session_id` field in the JSON request body
3. `user` field in the request body (OpenAI standard)
4. Automatic generation (`auto-1`, `auto-2`, etc.) for anonymous requests

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KESTREL_HOT_BUDGET_MB` | `2048` (2 GiB) | Maximum VRAM memory budget for sessions in the **Hot** tier. |
| `KESTREL_WARM_BUDGET_MB` | `4096` (4 GiB) | Maximum RAM memory budget for sessions in the **Warm** tier. |
| `RUST_LOG` | `info` | Filter level for trace logging (`error`, `warn`, `info`, `debug`, `trace`). |

### Command Line Options (`kestrel serve`)

```
Options:
  -m, --model <PATH>             Path to primary model file [default: ./models/qwen2.5-0.5b-q4.gguf]
      --backend <llama|mlx>      Backend inference engine [default: llama]
      --extra-model <PATH>...    Additional model paths to register for multi-model routing
  -p, --port <PORT>              TCP port for the server [default: 8080]
      --host <HOST>              Bind IP address [default: 0.0.0.0]
      --session-dir <DIR>        Directory for Cold snapshots [default: ./.kestrel/sessions]
      --hot-capacity <N>         Maximum number of concurrent sessions resident in VRAM [default: 16]
      --hot-budget-mb <MB>       Override Hot memory budget in MB
      --warm-budget-mb <MB>      Override Warm memory budget in MB
```

---

## 🧪 Testing & Benchmarks

The project includes a comprehensive test suite covering unit tests, heavy-load concurrency stress tests, and real end-to-end inference:

```bash
# 1. Run all unit and integration tests (using MockBackend)
cargo test --workspace

# 2. Run real E2E tests against downloaded Qwen2.5-0.5B model
# (Requires ./scripts/fetch-test-model.sh executed beforehand)
cargo test --test e2e -- --ignored

# 3. Run concurrency stress test (50 concurrent clients via BatchScheduler)
cargo test --test concurrency_stress -- --ignored

# 4. Benchmark session latency and KV cache prefix reuse (Criterion)
cargo bench -p kestrel-core
```

---

## 🗺️ Roadmap & Next Steps

### Completed Milestones (v1.0)
- [x] **M0 — Hello, llama**: Baseline GGUF inference via `llama.cpp` bindings and basic CLI completion.
- [x] **M1 — Persistent Sessions**: Stateful `SessionId`, persistent KV cache save/load on disk, prefix reuse verification.
- [x] **M2 — OpenAI-Compatible API**: Drop-in `/v1/chat/completions` (JSON and SSE) and `/v1/models` endpoints, plus `/kestrel/sessions` management.
- [x] **M3 — Resilient Tool Calling**: Robust tool call parsing, strict JSON Schema validation (`jsonschema`), and automatic model self-correction retry loop.
- [x] **M4 — Tiered KV Cache**: Three-tier storage hierarchy (**Hot** in VRAM, **Warm** in RAM, **Cold** on disk) governed by configurable budgets and LRU promotion/demotion.
- [x] **M5 — Concurrency & Batching**: Continuous micro-batching via `BatchScheduler` supporting concurrent hot sessions and 50+ simultaneous clients.
- [x] **Single Cross-Platform Binary**: Cross-platform single binary (`kestrel` / `kestrel.exe`) tested on Linux, macOS, and Windows with graceful shutdown.

### Post-v1.0 (Backlog Features)
- [x] **MLX Backend Support** (Apple Silicon unified memory optimization)
- [x] **Embeddings Endpoint** (`/v1/embeddings`)
- [x] **Native Structured Output** (JSON mode / strict JSON Schema response format)
- [x] **Multi-Model Routing** (per-session model routing)

---

## 📄 License

Distributed under dual **MIT** or **Apache-2.0** license. See [LICENSE](LICENSE) for details.
