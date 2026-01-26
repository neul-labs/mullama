# Mullama

**Drop-in Ollama replacement. All-in-one LLM toolkit.**

[![Crates.io](https://img.shields.io/crates/v/mullama)](https://crates.io/crates/mullama)
[![PyPI](https://img.shields.io/pypi/v/mullama)](https://pypi.org/project/mullama/)
[![npm](https://img.shields.io/npm/v/mullama)](https://www.npmjs.com/package/mullama)
[![Documentation](https://img.shields.io/badge/docs-neullabs.com-blue)](https://docs.neullabs.com/mullama/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Mullama is a local LLM server and library that works just like Ollama -- same CLI commands, same model format, same Modelfile syntax -- but with native language bindings for Rust, Python, Node.js, Go, PHP, and C/C++.

**Coming from Ollama?** Your commands work unchanged:

```bash
mullama run llama3.2:1b "Hello!"
mullama pull qwen2.5:7b
mullama serve
mullama list
```

## Install

**One-liner (Linux/macOS):**

```bash
curl -fsSL https://mullama.dev/install.sh | sh
```

**Windows (PowerShell):**

```powershell
iwr -useb https://mullama.dev/install.ps1 | iex
```

**Package managers:**

```bash
# Rust library
cargo add mullama

# Python
pip install mullama

# Node.js
npm install mullama

# Go
go get github.com/neul-labs/mullama-go

# PHP
composer require neul-labs/mullama
```

## Quick Start

```bash
# Run a model (daemon auto-starts)
mullama run llama3.2:1b "What is the capital of France?"

# Interactive chat
mullama chat

# Start OpenAI-compatible server
mullama serve --model llama3.2:1b

# Use with OpenAI SDK (any language)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Use as a Library (No Server Required)

Unlike Ollama, Mullama can be embedded directly in your application -- no HTTP overhead, no separate daemon.

**Python:**

```python
from mullama import Model, Context

model = Model.load('llama3.2-1b.gguf', n_gpu_layers=32)
ctx = Context(model, n_ctx=4096)
response = ctx.generate('Hello, AI!')
print(response)
```

**Node.js:**

```javascript
const { Model, Context } = require('mullama');

const model = await Model.load('llama3.2-1b.gguf', { gpuLayers: 32 });
const ctx = new Context(model, { contextSize: 4096 });
const response = await ctx.generate('Hello, AI!');
console.log(response);
```

**Rust:**

```rust
use mullama::{Model, Context, ContextParams};

let model = Model::load("llama3.2-1b.gguf")?;
let mut ctx = Context::new(&model, ContextParams::default())?;
let response = ctx.generate("Hello, AI!", 256)?;
println!("{}", response);
```

See [bindings documentation](./documentation/docs/bindings/) for Go, PHP, and C/C++.

## Ollama Compatibility + More

| Feature | Mullama | Ollama |
|---------|:-------:|:------:|
| CLI commands (`run`, `pull`, `serve`, etc.) | Same syntax | -- |
| Modelfile format | Compatible | -- |
| GGUF models | Yes | Yes |
| OpenAI API compatibility | Yes | Yes |
| Anthropic API compatibility | Yes | No |
| Native language bindings | 6 languages | HTTP only |
| Embed in your app (no daemon) | Yes | No |
| Web UI | Built-in | No |
| TUI chat | Built-in | No |
| ColBERT/late interaction | Yes | No |
| Streaming audio input | Yes | No |

[Full comparison](./documentation/docs/comparison/vs-ollama.md) | [Migration guide](./documentation/docs/comparison/migration-from-ollama.md)

## All-in-One LLM Toolkit

Mullama includes everything you need:

| Component | Description |
|-----------|-------------|
| **CLI** | Familiar commands: `run`, `pull`, `serve`, `list`, `chat`, `create` |
| **Daemon** | Multi-model server with auto-spawn, hot-reload, session persistence |
| **REST API** | OpenAI + Anthropic compatible endpoints |
| **Web UI** | Model management, chat interface, API playground |
| **TUI** | Terminal-based chat client |
| **Library** | Native bindings for Rust, Python, Node.js, Go, PHP, C/C++ |

## CLI Commands

```bash
# Model management
mullama list              # List local models
mullama pull llama3.2:1b  # Download model
mullama rm old-model      # Remove model
mullama ps                # Show running models

# Generation
mullama run llama3.2:1b "prompt"  # One-shot generation
mullama chat                       # Interactive TUI

# Daemon
mullama serve --model llama3.2:1b  # Start server
mullama daemon start               # Start in background
mullama daemon stop                # Stop daemon
mullama daemon status              # Show status

# Model creation (Ollama-compatible Modelfile)
mullama create my-assistant -f ./Modelfile
mullama show my-assistant --modelfile
```

## API Endpoints

**OpenAI-compatible:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Anthropic-compatible:**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello!"}]}'
```

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI chat completions |
| `POST /v1/completions` | OpenAI text completions |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/embeddings` | Generate embeddings |
| `GET /v1/models` | List available models |
| `GET /ui/` | Web UI |

## GPU Acceleration

```bash
# NVIDIA CUDA
export LLAMA_CUDA=1

# Apple Metal (macOS)
export LLAMA_METAL=1

# AMD ROCm (Linux)
export LLAMA_HIPBLAS=1

# Intel OpenCL
export LLAMA_CLBLAST=1
```

## Feature Flags (Rust Library)

```toml
[dependencies.mullama]
version = "0.1.1"
features = [
    "async",              # Async/await support
    "streaming",          # Token streaming
    "web",                # Axum web framework
    "websockets",         # WebSocket support
    "multimodal",         # Image and audio processing
    "streaming-audio",    # Real-time audio capture
    "parallel",           # Rayon parallel processing
    "late-interaction",   # ColBERT-style embeddings
    "daemon",             # Daemon mode with TUI client
    "full"                # All features
]
```

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](./docs/GETTING_STARTED.md) | Installation and first steps |
| [Platform Setup](./docs/PLATFORM_SETUP.md) | OS-specific setup |
| [Daemon Guide](./docs/DAEMON.md) | Daemon, CLI, API reference |
| [Migration from Ollama](./docs/MIGRATION_FROM_OLLAMA.md) | Quick migration checklist |
| [Full Documentation](https://docs.neullabs.com/mullama/) | Complete docs site |

## Examples

```bash
# Basic text generation
cargo run --example simple --features async

# Streaming responses
cargo run --example streaming_generation --features "async,streaming"

# Web service
cargo run --example web_service --features "web,websockets"

# Late interaction / ColBERT retrieval
cargo run --example late_interaction --features late-interaction
```

## Contributing

```bash
git clone --recurse-submodules https://github.com/neul-labs/mullama.git
cd mullama
cargo test --all-features
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
