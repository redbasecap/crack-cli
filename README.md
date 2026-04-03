# 🔥 Forge CLI

**Self-improving agent harness — forge better agents through iterative refinement.**

[![Rust](https://img.shields.io/badge/Rust-1.78+-orange?logo=rust&style=flat-square)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What is Forge?

Forge is an open-source CLI that combines a **high-performance Rust agent runtime** with a **self-improving experiment loop**. Instead of manually tuning your agent harness, Forge runs benchmarks, measures scores, and hill-climbs toward better performance — automatically.

**Core ideas:**
- 🦀 **Rust runtime** — fast, memory-safe agent execution with streaming, tool orchestration, and session management
- 🔄 **Self-improving loop** — automated benchmark → score → keep/discard cycle that iteratively refines the harness
- 📝 **Human-in-the-loop via Markdown** — steer the meta-agent through `program.md`, not code
- 🧰 **Rich tool ecosystem** — file ops, bash, web search, MCP integration, plugins
- 🐳 **Sandboxed execution** — tasks run in isolation, nothing can damage your host

## Quick Start

### Build the Rust CLI

```bash
cd rust
cargo build --release
./target/release/forge
```

### Interactive mode

```bash
forge                           # Start REPL
forge prompt "explain this"     # One-shot prompt
forge --resume                  # Resume last session
```

### Run the self-improving loop

```bash
# 1. Initialize with example tasks
python3 -m src.main forge-init-tasks

# 2. Run benchmarks once
python3 -m src.main forge-bench

# 3. Start the improvement loop
python3 -m src.main forge-improve

# 4. Check results
python3 -m src.main forge-score
python3 -m src.main forge-experiment-log
```

## How Self-Improvement Works

```
┌─────────────────────────────────────────────────┐
│                 program.md                       │
│           (human writes directives)              │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Meta-Agent                          │
│  1. Read directives + past results               │
│  2. Diagnose failures from last run              │
│  3. Propose harness changes                      │
│  4. Apply changes + git commit                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│            Benchmark Runner                      │
│  - Discover tasks in tasks/                      │
│  - Run each task in subprocess isolation         │
│  - Collect scores (0.0 – 1.0)                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│             Hill Climber                         │
│  - Score improved? → KEEP (commit stays)         │
│  - Score same + simpler? → KEEP                  │
│  - Otherwise → DISCARD (git revert)              │
│  - Log to results.tsv                            │
└─────────────────────────────────────────────────┘
                     │
                     └──── loop ────┐
                                    ▼
                              next iteration
```

## Task Format

Tasks live in `tasks/` and follow this structure:

```
tasks/my-task/
├── task.toml            # Config (timeout, metadata)
├── instruction.md       # Prompt sent to the agent
└── tests/
    └── test.sh          # Verifier — writes score to stdout
```

**Example `task.toml`:**
```toml
name = "hello-world"
timeout_seconds = 60
```

**Example `test.sh`:**
```bash
#!/bin/bash
if [ -f /task/output.txt ] && grep -q "Hello" /task/output.txt; then
    echo "1.0"  # pass
else
    echo "0.0"  # fail
fi
```

## Architecture

```
forge/
├── rust/                    # Rust workspace
│   └── crates/
│       ├── forge-cli/       # CLI binary (forge)
│       ├── runtime/         # Session, tools, permissions, hooks
│       ├── api/             # LLM provider clients (Anthropic, OpenAI-compat)
│       ├── tools/           # Tool definitions + execution
│       ├── plugins/         # Plugin system + hooks
│       ├── commands/        # Slash commands
│       └── telemetry/       # Usage tracking
├── src/                     # Python harness layer
│   ├── self_improve/        # Self-improvement engine
│   │   ├── engine.py        # ExperimentLoop (hill-climbing)
│   │   ├── scorer.py        # Score aggregation + keep/discard
│   │   ├── task_runner.py   # Task discovery + execution
│   │   └── meta_agent.py    # Failure diagnosis + proposals
│   ├── main.py              # CLI entry point
│   ├── runtime.py           # Routing + session bootstrap
│   ├── tools.py             # Tool registry
│   └── commands.py          # Command registry
├── tasks/                   # Benchmark tasks
├── program.md               # Meta-agent directives
└── results.tsv              # Experiment log (auto-generated)
```

## Configuration

| Env Variable       | Description                    | Default          |
|-------------------|--------------------------------|------------------|
| `FORGE_MODEL`      | LLM model to use              | `claude-sonnet-4-20250514`  |
| `FORGE_API_KEY`    | Provider API key              | —                |
| `FORGE_MAX_TURNS`  | Max turns per task             | `30`             |
| `FORGE_TASKS_DIR`  | Path to benchmark tasks        | `./tasks`        |

Config files: `~/.forge/settings.json` or `.forge.json` in project root.

## CLI Reference

### Rust CLI (`forge`)

| Command                     | Description                          |
|----------------------------|--------------------------------------|
| `forge`                     | Interactive REPL                    |
| `forge prompt "..."`        | One-shot prompt                     |
| `forge --resume`            | Resume last session                 |
| `forge --model <model>`     | Use specific model                  |

### Python CLI (`python3 -m src.main`)

| Command                     | Description                          |
|----------------------------|--------------------------------------|
| `forge-improve`             | Start self-improvement loop          |
| `forge-bench`               | Run benchmark suite once             |
| `forge-score`               | Show current scores                  |
| `forge-experiment-log`      | Show experiment history              |
| `forge-init-tasks`          | Scaffold tasks/ with example         |
| `summary`                   | Render workspace summary             |
| `manifest`                  | Print workspace manifest             |
| `route <prompt>`            | Route prompt to tools/commands       |
| `bootstrap <prompt>`        | Build a runtime session              |

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run tests: `cd rust && cargo test --workspace`
5. Submit a PR

## License

MIT — see [LICENSE](LICENSE) for details.
