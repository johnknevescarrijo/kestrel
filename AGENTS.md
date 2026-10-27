# AGENTS.md — kestrel

> Motor de inferência **agent-native** para hardware de consumidor.
> Este documento é a fonte de verdade para qualquer agente de código que
> trabalhe neste repositório. Leia-o antes de qualquer tarefa e atualize-o
> quando uma decisão arquitetural mudar.

---

## 1. Visão Geral

**kestrel** é um servidor de inferência local, escrito em Rust, que trata
sessões agênticas como cidadãs de primeira classe. Enquanto motores
tradicionais (vLLM, llama.cpp puro) tratam cada requisição como efêmera,
kestrel mantém **KV cache persistente por sessão**, com tiering entre VRAM
e disco, e oferece **tool calling estruturado e confiável** via uma API
compatível com OpenAI.

**Público-alvo:** desenvolvedores rodando agentes em workstation/edge
(1 GPU de consumidor, 8–24 GB de VRAM).

**Não-é:** um framework de agentes, uma UI, um orquestrador de múltiplos
modelos. É a *engine* que agentes (CLIs, SDKs) consomem.

---

## 2. Objetivos e Não-Objetivos

### Objetivos (v1.0)
- [x] API HTTP compatível com `/v1/chat/completions` e `/v1/models`.
- [x] Sessões persistentes: `SessionId` → estado de KV cache salvo/restaurado.
- [x] Tiering automático de sessões: **hot** (VRAM), **warm** (RAM), **cold** (disco).
- [x] Tool calling com parsing de JSON estruturado, validação de schema e retry.

- [x] Suporte a modelos GGUF via `llama.cpp`.
- [x] Observabilidade via `tracing` (spans por requisição e por sessão).
- [x] Binário único, multiplataforma (Linux, macOS, Windows).

### Não-Objetivos (v1.0)
- Multi-GPU / tensor parallelism.
- Treino ou fine-tuning.
- Suporte a modelos não-GGUF (safetensors etc.).
- Interface web própria (use `curl`, SDKs existentes, ou Open WebUI).
- Autenticação multiusuário (single-user local por padrão).

---

## 3. Princípios de Design

1. **Sessão > Requisição.** Toda decisão arquitetural deve favorecer a
   continuidade do estado entre chamadas. Se algo força o cliente a
   reenviar contexto, é um bug de design.
2. **Compatibilidade OpenAI como contrato.** Qualquer extensão é feita via
   campos opcionais ou endpoints extra (ex: `/kestrel/sessions`). Nunca
   quebrar o subset OpenAI.
3. **Rust idiomático, sem heroísmos.** Prefira `tokio`, `axum`, `serde`,
   `thiserror`, `tracing`. Se precisar de `unsafe`, isole num crate
   separado e documente.
4. **Falhe de forma explícita.** Erros de tool call, schema inválido,
   timeout de sessão — tudo retorna erro estruturado, nunca "resposta
   vazia". Agentes downstream dependem de sinal claro.
5. **Edge-first.** Orçamento de VRAM é restrito. Toda feature deve ser
   justificável em 8 GB. Otimizações que assumem H100 estão fora.
6. **Observável por padrão.** Se um agente não consegue debugar uma sessão
   que falhou, a ferramenta falhou.

---

## 4. Arquitetura

### Layout do workspace (Cargo workspace)

```
kestrel/
├── Cargo.toml                # workspace manifest
├── crates/
│   ├── kestrel-core/         # lógica pura: sessões, scheduler, tipos
│   ├── kestrel-llama/        # bindings para llama.cpp (isolado)
│   ├── kestrel-server/       # API HTTP (axum)
│   ├── kestrel-tools/        # runtime de tool calling
│   └── kestrel-cli/          # binário principal + ferramentas admin
├── tests/                    # testes de integração end-to-end
├── benches/                  # benchmarks (criterion)
├── models/                   # (gitignored) GGUF para testes locais
└── AGENTS.md
```

### Fluxo de uma requisição

```
HTTP /v1/chat/completions
    │
    ▼
[kestrel-server] valida payload + resolve SessionId
    │
    ▼
[kestrel-core] SessionManager:
    - hit? → retorna handle
    - miss? → carrega KV cache do tier cold/warm
    - evict se necessário (LRU por sessão)
    │
    ▼
[kestrel-llama] executa forward pass com KV cache restaurado
    │
    ▼
[kestrel-tools] detecta tool call na saída → valida → retorna
    │
    ▼
[kestrel-core] persiste KV cache atualizado (write-back async)
    │
    ▼
HTTP response (SSE ou JSON)
```

### Componentes-chave

- **`SessionManager`** — dono do ciclo de vida das sessões. Mantém um
  `LruCache<SessionId, Session>` para tiers hot/warm, e um `SessionStore`
  (trait) para o tier cold.
- **`Session`** — encapsula `LlamaContext` + estado de conversa + metadados
  (último acesso, tamanho do KV, modelo associado).
- **`TierPolicy`** — decide promoção/demoção de sessão. Configurável por
  variável de ambiente (`KESTREL_HOT_BUDGET_MB`, `KESTREL_WARM_BUDGET_MB`).
- **`ToolRuntime`** — parsing de tool call da saída do modelo, validação
  de JSON Schema, execução (in-process ou subprocess), retry com feedback.
- **`Backend`** (trait) — abstração sobre llama.cpp. Permite mock em testes
  e futura troca por outro backend (ex: MLX) sem tocar em `kestrel-core`.

---

## 5. Stack Técnico

| Camada | Escolha | Justificativa |
|---|---|---|
| Runtime async | `tokio` | padrão de fato, bem integrado |
| HTTP | `axum` | ergonômico, tokio-native, SSE fácil |
| Serialização | `serde` + `serde_json` | obrigatório |
| Erros (lib) | `thiserror` | erros tipados por crate |
| Erros (bin) | `anyhow` | aplicação permite contexto |
| Logs/traces | `tracing` + `tracing-subscriber` | structured, spans |
| CLI | `clap` (derive) | padrão |
| Backend LLM | `llama-cpp-rs` (crate `llama_cpp`) | bindings maduros, GGUF |
| JSON Schema | `jsonschema` crate | validação de tool calls |
| Storage cold | `sled` ou arquivos + `bincode` | simples, local |
| Testes | `cargo test`, `insta` (snapshots) | padrão |
| Bench | `criterion` | padrão |

**Adicionar dependência nova exige justificativa no PR/commit e aprovação
do humano.** Não adicione crates "só por conveniência".

---

## 6. Convenções de Código

### Rust
- Edition 2021, `rust-version` fixado no `Cargo.toml` do workspace.
- `cargo fmt` e `cargo clippy -- -D warnings` **sempre** passam antes de
  commit. O agente deve rodar ambos.
- `#![deny(unsafe_code)]` em todos os crates exceto `kestrel-llama`.
- Nomes: `snake_case` (fn/var), `PascalCase` (tipo), `SCREAMING_SNAKE`
  (const). Siglas em nomes compostos: `SessionId`, `LlmBackend`.
- Documentação: `///` em toda API pública. Doc-tests quando o exemplo
  ajudar. Não documente o óbvio (`/// Retorna o id.`).

### Estrutura de módulo
- Um conceito por arquivo. Se passar de ~400 LOC, quebre.
- `mod.rs` só re-exporta; lógica fica nos submódulos.
- Traits públicas em `traits.rs` do crate.

### Erros
- Cada crate tem seu `Error` enum com `thiserror`.
- Nada de `unwrap()`/`expect()` fora de testes e init de binário. Use
  `?` e propague.
- Mensagens de erro: **o que falhou + contexto + ação sugerida**.
  Ex: `"session {id} evicted during inference; increase KESTREL_HOT_BUDGET_MB"`.

### Async
- Toda I/O é async. Blocking (llama.cpp, disco) roda em
  `tokio::task::spawn_blocking`.
- Nunca `.await` segurando um `Mutex` std. Use `tokio::sync::Mutex` ou
  `parking_lot` fora de async.
- Cancellation-safe: `select!` e timeouts devem ser testados.

### API HTTP
- Versionamento por header ou path (`/v1/...`).
- Extensões próprias sob `/kestrel/...` (sessões, admin, métricas).
- Streaming via SSE (`text/event-stream`), formato idêntico ao OpenAI.
- Erros: `{"error": {"message": ..., "type": ..., "code": ...}}`.

---

## 7. Testing

### Camadas
1. **Unit** — em cada módulo, `#[cfg(test)]`.
2. **Integration** — em `tests/`, sem rede. Use `Backend` mock para
   `kestrel-core` e `kestrel-server`.
3. **E2E** — em `tests/e2e.rs`, com um modelo GGUF minúsculo
   (TinyLlama 1.1B Q4 ou Qwen2.5-0.5B Q4) baixado por script.
   Marcados com `#[ignore]`, rodados via `cargo test -- --ignored`.
4. **Bench** — em `benches/`, `criterion`. Metas: latência P50/P99,
   hit-rate de KV cache, uso de VRAM.

### Regras
- Todo bugfix acompanha um teste que falha antes e passa depois.
- Snapshots (`insta`) para saídas JSON estruturadas (ex: payload de tool
  call). Revise snapshots no PR.
- Cobertura mínima: **não** perseguir %. Persiga casos de borda:
  sessão evictada no meio, tool call malformada, cliente desconecta no
  meio do stream, OOM simulado.

---

## 8. Comandos

```bash
# setup
rustup toolchain install stable
cargo install cargo-watch cargo-nextest

# build
cargo build --workspace
cargo build --release -p kestrel-cli

# lint + format (obrigatório antes de commit)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# testes
cargo nextest run --workspace
cargo test --workspace -- --ignored          # E2E (precisa de modelo)

# baixar modelo de teste
./scripts/fetch-test-model.sh                 # Qwen2.5-0.5B Q4

# rodar local
cargo run --release -p kestrel-cli -- serve \
    --model ./models/qwen2.5-0.5b-q4.gguf \
    --port 8080

# bench
cargo bench -p kestrel-core
```

---

## 9. Workflow do Agente

### Antes de começar qualquer tarefa
1. Leia a seção relevante deste `AGENTS.md`.
2. Verifique `git status` e `git log -5` para contexto.
3. Se a tarefa envolve decisão arquitetural (novo componente, novo
   trait, mudança de dependência), **pare e pergunte ao humano** antes
   de codar.

### Durante
- Trabalhe em commits pequenos e coesos. Mensagem: `area: verbo curto`
  (ex: `session: persist kv cache on drop`).
- Rode `cargo fmt` + `cargo clippy` + `cargo nextest run` a cada
  milestone lógico, não só no final.
- Se um teste falhar de forma inesperada e não-trivial, **não "conserte"
  o teste** sem discutir. Pode ser bug de design.
- Atualize o `AGENTS.md` se mudar arquitetura, adicionar crate ou
  alterar comando.

### Depois
- Resumo do que mudou, o que testou, o que ficou pendente.
- Se o milestone do roadmap mudou de status, atualize a seção 10.

### Nunca faça
- Adicionar dependência sem aprovação.
- Adicionar `unsafe` fora de `kestrel-llama`.
- Quebrar compatibilidade OpenAI sem discussão.
- Fazer `git push`, `git reset --hard`, ou reescrever histórico.
- Commitar segredos, modelos GGUF, ou arquivos > 10 MB.
- Remover testes que "incomodam".

---

## 10. Roadmap

Cada milestone tem critério de aceite claro. **Não avance sem fechar o
anterior.**

### M0 — Hello, llama
- [x] `kestrel-llama` expõe `Backend` trait com `complete(prompt) -> String`.
- [x] `kestrel-cli` tem `kestrel complete "hello"` que roda localmente.
- **Aceite:** rodar Qwen2.5-0.5B no CI local, imprimir resposta. (Concluído)

### M1 — Sessões persistentes
- [x] `Session` com `SessionId`, save/load de KV cache em disco.
- [x] `SessionManager` com LRU e `SessionStore` (trait + impl em arquivo).
- [x] Teste: duas chamadas com mesmo `SessionId` reutilizam prefixo
      (validar via métrica de tokens processados).
- **Aceite:** benchmark mostra redução de tokens de prefill na 2ª chamada. (Concluído)

### M2 — API OpenAI-compatible
- [x] `/v1/chat/completions` (JSON e SSE) e `/v1/models`.
- [x] Extensão `/kestrel/sessions` (list, delete, stats).
- [x] Teste de contrato comparando com cliente oficial `openai` (Python).
- **Aceite:** Open WebUI consegue conversar com kestrel sem patches. (Concluído)

### M3 — Tool calling confiável
- [x] Parsing de tool call estruturado, validação JSON Schema.
- [x] Retry com feedback ao modelo em caso de falha (max N).
- [x] Teste com cenários: JSON truncado, tipo errado, campo faltando,
      tool inexistente.
- **Aceite:** suite de 20 casos passa com >= 95% de acerto. (Concluído)

### M4 — Tiering de KV cache
- [x] Hot (VRAM) / warm (RAM) / cold (disco) com budgets configuráveis.
- [x] Promoção/demoção por LRU + política configurável.
- [x] Métrica de hit-rate exposta em `/kestrel/metrics`.
- **Aceite:** 20 sessões simultâneas em 8 GB de VRAM sem OOM. (Concluído)


### M5 — Concorrência e batching
- [x] Continuous batching de sessões hot (uma forward pass, N sessões).
- [x] Stress test com 50 clientes concorrentes.
- **Aceite:** throughput >= 2x M4 em benchmark padrão. (Concluído)

### Pós-v1.0 (backlog)
- [x] Suporte a MLX (Apple Silicon).
- [x] Embeddings endpoint.
- [x] Structured output (JSON mode) nativo.
- [x] Multi-modelo (roteamento por sessão).

---

## 11. Anti-patterns (o que este projeto NÃO é)

- ❌ "Vamos adicionar um framework de agentes aqui dentro." Não. kestrel é
  engine. Agentes vêm de fora.
- ❌ "Vamos otimizar para H100 agora." Não. Edge-first.
- ❌ "Vamos reescrever em Python para prototipar mais rápido." Não. Rust é
  parte da proposta (latência, footprint, deployment).
- ❌ "Vamos usar `panic!` porque 'nunca vai acontecer'." Não. Servidor
  local precisa ser robusto a inputs malformados de agentes.
- ❌ "Vamos guardar KV cache em variável global." Não. `SessionManager` é
  dono do estado. Global state é proibido.

---

## 12. Contato / Escalação

- Decisões arquiteturais: **pergunte ao humano** antes de implementar.
- Dúvida em requisito: **pare e pergunte**, não invente.
- Bug bloqueante de dependência (`llama-cpp-rs`, `axum`): documente no
  issue tracker e proponha workaround.
