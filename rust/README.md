# Forge CLI — Rust Workspace

The Rust implementation of Forge CLI. This is the high-performance core runtime.

## Crates

| Crate          | Description                                      |
|---------------|--------------------------------------------------|
| `forge-cli`    | Main binary — REPL, one-shot prompts, sessions  |
| `runtime`      | Session management, tool orchestration, hooks    |
| `api`          | LLM provider clients (Anthropic, OpenAI-compat) |
| `tools`        | Tool definitions and execution engine            |
| `plugins`      | Plugin system with pre/post hooks                |
| `commands`     | Slash command registry and dispatch              |
| `telemetry`    | Usage tracking and metrics                       |
| `compat-harness` | Compatibility layer                           |

## Build

```bash
cargo build --release
```

## Test

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Run

```bash
./target/release/forge
```
