# 🦅 kestrel

> **Motor de inferência local *agent-native* para hardware de consumidor (8–24 GB de VRAM).**

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Backend](https://img.shields.io/badge/llama.cpp-GGUF-blue.svg)](https://github.com/ggerganov/llama.cpp)
[![API](https://img.shields.io/badge/API-OpenAI%20Compatible-green.svg)](https://platform.openai.com/docs/api-reference)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-lightgrey.svg)](LICENSE)

---

## 📌 Visão Geral & Proposta de Valor

Enquanto servidores de inferência tradicionais (como vLLM ou llama.cpp puro) tratam cada requisição como efêmera, o **kestrel** foi projetado desde o primeiro dia para fluxos de trabalho com **agentes autônomos**. Em interações agênticas, um agente realiza dezenas de turnos incrementais (executando comandos, lendo arquivos e raciocinando). Recomputar o contexto inteiro a cada turno gera latência e desperdício massivo de computação.

O **kestrel** inverte essa lógica: **Sessão > Requisição**.

- 🧠 **Sessão Persistente & Reuso de Prefixo:** O estado do KV cache é indexado por um `SessionId`. Chamadas subsequentes na mesma sessão processam apenas os novos tokens (*prefill* delta), reduzindo a latência de primeiro token a zero ou próximo disso.
- ⚡ **Tiering de KV Cache em 3 Níveis:**
  - **Hot (VRAM):** Sessões ativas mantêm o handle do backend na GPU para latência imediata.
  - **Warm (RAM):** Sessões ociosas têm seu snapshot mantido na memória do sistema, liberando a VRAM para sessões ativas.
  - **Cold (Disco via `bincode`):** Sessões inativas há mais tempo são descarregadas atomicamente para o disco local, restauráveis a qualquer momento.
  - Governança automática por limites de memória configuráveis (`KESTREL_HOT_BUDGET_MB`, `KESTREL_WARM_BUDGET_MB`) e algoritmo LRU.
- 🛠️ **Tool Calling Estruturado & Resiliente:** Parser flexível capaz de extrair chamadas a partir de tags `<tool_call>`, blocos markdown de código ou JSON cru. Validação estrita via JSON Schema (`jsonschema`) com **loop automático de feedback e retry**: se o modelo errar argumentos ou alucinar o schema, o Kestrel realimenta o erro reflexivamente de volta ao modelo.
- 🚀 **Continuous Batching:** O `BatchScheduler` agrupa requisições concorrentes entre diferentes sessões hot em janelas de micro-lotes, maximizando a taxa de transferência da GPU sem aumentar a latência individual.
- 🔌 **Compatibilidade OpenAI Pronta para Uso:** Endpoints `/v1/chat/completions` (JSON completo e SSE streaming) e `/v1/models`, integrando perfeitamente com Open WebUI, Cursor, Continue, LangChain ou o SDK oficial da OpenAI.
- 📊 **Observabilidade Nativa:** Spans estruturados com `tracing`, logs contextualizados e métricas operacionais completas via `/kestrel/metrics` (taxas de hit dos tiers, promoções, demissões e batching).

---

## 🧱 Arquitetura do Workspace

O projeto é organizado como um workspace Cargo modular, com limites claros de responsabilidade e `#![deny(unsafe_code)]` em todos os módulos puros:

```
kestrel/
├── Cargo.toml                # Workspace manifest (Rust 2021, rust-version 1.80)
├── crates/
│   ├── kestrel-core/         # Lógica pura: sessões, LRU, tiering, BatchScheduler e traits
│   ├── kestrel-llama/        # Bindings isolados para llama.cpp (GGUF, restauração de KV)
│   ├── kestrel-tools/        # Parsing de tool calls, validação JSON Schema e loop de retry
│   ├── kestrel-server/       # Servidor HTTP Axum, compatibilidade OpenAI e rotas /kestrel/*
│   └── kestrel-cli/          # Binário principal: comandos 'serve' e 'complete'
├── scripts/                  # Scripts auxiliares (ex: download de modelos)
├── tests/                    # Suíte de testes de integração, stress e E2E
├── benches/                  # Benchmarks Criterion (reuso de KV cache)
└── models/                   # Diretório local para modelos GGUF (.gitignore)
```

| Crate | Responsabilidade Principal |
|---|---|
| **`kestrel-core`** | Gerenciador de ciclo de vida (`SessionManager`), políticas de tiering (`TierPolicy`), cálculo de orçamentos de memória, persistência em disco (`FileSessionStore`), fila e escalonador contínuo (`BatchScheduler`). |
| **`kestrel-llama`** | Integração direta com a biblioteca `llama_cpp`. Isola chamadas de baixo nível, gerencia handles de contexto e executa forward passes com restauração parcial de tokens. |
| **`kestrel-tools`** | Extração de invocação de ferramentas, compilação de esquemas JSON (`jsonschema`), validação de argumentos e injeção de feedback corretivo multi-tentativas. |
| **`kestrel-server`** | Camada HTTP construída com Axum. Oferece endpoints compatíveis com a OpenAI, suporte a Server-Sent Events (SSE) para streaming, resolução automática de `SessionId` e endpoints de métricas e sessões. |
| **`kestrel-cli`** | Ponto de entrada CLI (via `clap`). Permite tanto inferência interativa/direta quanto o lançamento do servidor de produção. |

---

## 🚀 Quickstart

### 1. Pré-requisitos
- **Rust:** Versão 1.80 ou superior (`rustup default stable`).
- **Ferramentas de compilação:** `cmake` (>= 3.14) e um compilador C/C++ compatível (GCC, Clang ou MSVC) para compilar o `llama.cpp`.

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential cmake

# Fedora
sudo dnf install -y gcc-c++ cmake
```

### 2. Baixar o Modelo de Teste
Disponibilizamos um script que faz o download do **Qwen2.5-0.5B-Instruct (Q4_K_M GGUF)** (~398 MB), ideal para desenvolvimento local e testes:

```bash
./scripts/fetch-test-model.sh
```
O modelo será salvo em `./models/qwen2.5-0.5b-q4.gguf`.

### 3. Compilar o Projeto

```bash
# Compilar binário release
cargo build --release -p kestrel-cli

# O executável estará disponível em target/release/kestrel
```

---

## 💻 Uso via Linha de Comando (CLI)

O binário `kestrel` oferece dois subcomandos principais: `complete` e `serve`.

### Inferência Direta (`complete`)

```bash
# Inferência pontual sem sessão
./target/release/kestrel complete "Explique em uma frase o que é Rust."

# Inferência com sessão persistente (KV cache preservado entre execuções!)
./target/release/kestrel complete "Meu nome é João e eu gosto de café." --session session-user-1
./target/release/kestrel complete "Qual é o meu nome e o que eu gosto?" --session session-user-1
```
> **Nota:** Na segunda chamada com o mesmo `--session`, o Kestrel reaproveita o KV cache já processado, executando prefill apenas na nova pergunta.

### Subindo o Servidor HTTP (`serve`)

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

## 🌐 Exemplos de Uso da API HTTP

O Kestrel inicia na porta padrão `8080` com suporte integral aos headers e esquemas da OpenAI.

### 1. Chamada Padrão (JSON) via `curl`

Identifique a sessão passando o header `X-Kestrel-Session-Id` (ou pelo campo `user` no corpo da requisição):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Kestrel-Session-Id: ag-sess-001" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [
      {"role": "user", "content": "Olá! Estou configurando um servidor de inferência local."}
    ],
    "max_tokens": 128
  }'
```

**Resposta:**
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
        "content": "Olá! Muito bom ver você explorando inferência local. Como posso ajudar com a sua configuração?",
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

### 2. Streaming em Tempo Real (SSE)

Basta definir `"stream": true`:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Kestrel-Session-Id: ag-sess-001" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [
      {"role": "user", "content": "Conte até 5 devagar."}
    ],
    "stream": true
  }'
```

O servidor emitirá eventos padrão SSE:
```
data: {"id":"chatcmpl-1729792010","object":"chat.completion.chunk","created":1729792010,"model":"qwen2.5-0.5b-q4.gguf","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-1729792010","object":"chat.completion.chunk","created":1729792010,"model":"qwen2.5-0.5b-q4.gguf","choices":[{"index":0,"delta":{"role":null,"content":"1"},"finish_reason":null}]}

...
data: [DONE]
```

### 3. Integração com SDK Oficial da OpenAI (Python)

Você pode usar o SDK padrão da OpenAI sem nenhuma alteração no código do cliente, além de apontar a URL base:

```python
from openai import OpenAI

# Inicializa apontando para o servidor Kestrel local
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",  # Kestrel é single-user local por padrão
)

# Envia requisição informando a sessão persistente
response = client.chat.completions.create(
    model="qwen2.5-0.5b-q4.gguf",
    messages=[
        {"role": "system", "content": "Você é um assistente conciso."},
        {"role": "user", "content": "Qual a capital da França?"},
    ],
    extra_headers={"X-Kestrel-Session-Id": "sessao-pesquisa-01"},
)

print(response.choices[0].message.content)

# Segundo turno reutiliza a mesma sessão e aproveita o KV cache:
response2 = client.chat.completions.create(
    model="qwen2.5-0.5b-q4.gguf",
    messages=[
        {"role": "system", "content": "Você é um assistente conciso."},
        {"role": "user", "content": "Qual a capital da França?"},
        {"role": "assistant", "content": response.choices[0].message.content},
        {"role": "user", "content": "Qual a população estimada dela?"},
    ],
    extra_headers={"X-Kestrel-Session-Id": "sessao-pesquisa-01"},
)

print(response2.choices[0].message.content)
```

### 4. Tool Calling Confiável com JSON Schema

O Kestrel analisa esquemas JSON declarados em `tools` e força a validação sintática e semântica antes de responder ao cliente:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Kestrel-Session-Id: agent-tool-sess" \
  -d '{
    "model": "qwen2.5-0.5b-q4.gguf",
    "messages": [
      {"role": "user", "content": "Como está o clima em Curitiba?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Obtém as condições climáticas atuais de uma cidade",
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

Se o modelo gerar a chamada corretamente, o Kestrel retorna `finish_reason: "tool_calls"`:

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

> 🔁 **Recuperação Automática (Self-Correction):** Caso o modelo emita parâmetros inválidos ou JSON malformado, o Kestrel detecta o erro contra o `jsonschema`, injeta um prompt de feedback explicando a violação e solicita ao modelo que reescreva a resposta (até o limite de retries configurado, padrão: 3).

---

## 🛠️ Endpoints de Administração & Observabilidade

Além da API padrão da OpenAI, o Kestrel expõe rotas exclusivas para controle agêntico:

### `GET /v1/models`
Lista os modelos atualmente carregados no servidor.

### `GET /kestrel/sessions`
Lista todas as sessões ativas e em armazenamento, com métricas agregadas de uso de memória.

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
Exibe os metadados de uma sessão específica (tier atual, tamanho do KV cache, tokens armazenados).

### `DELETE /kestrel/sessions/:id`
Remove a sessão explicitamente da memória (VRAM/RAM) e apaga seu snapshot do disco.

```bash
curl -s -X DELETE http://localhost:8080/kestrel/sessions/ag-sess-001
```

### `GET /kestrel/metrics`
Métricas operacionais em tempo real sobre o sistema de tiering e continuous batching:

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

## ⚙️ Configurações & Variáveis de Ambiente

### Resolução de `SessionId`
O Kestrel resolve a sessão da requisição na seguinte ordem de precedência:
1. Header HTTP `X-Kestrel-Session-Id`
2. Campo `session_id` no corpo da requisição JSON
3. Campo `user` no corpo da requisição (padrão OpenAI)
4. Geração automática (`auto-1`, `auto-2`, etc.) para requisições anônimas

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `KESTREL_HOT_BUDGET_MB` | `2048` (2 GiB) | Orçamento máximo de memória VRAM para sessões no tier **Hot**. |
| `KESTREL_WARM_BUDGET_MB` | `4096` (4 GiB) | Orçamento máximo de memória RAM para sessões no tier **Warm**. |
| `RUST_LOG` | `info` | Nível de filtragem do coletor de traces (`error`, `warn`, `info`, `debug`, `trace`). |

### Opções de Linha de Comando (`kestrel serve`)

```
Opções:
  -m, --model <PATH>             Caminho para o arquivo GGUF [default: ./models/qwen2.5-0.5b-q4.gguf]
  -p, --port <PORT>              Porta TCP do servidor [default: 8080]
      --host <HOST>              Endereço IP para bind [default: 0.0.0.0]
      --session-dir <DIR>        Diretório para snapshots Cold [default: ./.kestrel/sessions]
      --hot-capacity <N>         Número máximo de sessões residentes na VRAM [default: 16]
      --hot-budget-mb <MB>       Sobrescreve o orçamento Hot em MB
      --warm-budget-mb <MB>      Sobrescreve o orçamento Warm em MB
```

---

## 🧪 Testes & Benchmarks

O projeto possui uma suíte rigorosa cobrindo desde testes unitários isolados até concorrência sob carga pesada e testes fim-a-fim reais:

```bash
# 1. Executar todos os testes unitários e de integração (com MockBackend)
cargo test --workspace

# 2. Executar testes E2E reais contra o modelo Qwen2.5-0.5B baixado
# (Requer ./scripts/fetch-test-model.sh executado previamente)
cargo test --test e2e -- --ignored

# 3. Executar teste de estresse de concorrência (50 clientes simultâneos via BatchScheduler)
cargo test --test concurrency_stress -- --ignored

# 4. Benchmarking de latência de sessão e reuso de KV cache (Criterion)
cargo bench -p kestrel-core
```

---

## 📄 Licença

Distribuído sob licença dual **MIT** ou **Apache-2.0**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.