# Mullama

**Comprehensive Rust bindings for llama.cpp with advanced integration features**

[![Crates.io](https://img.shields.io/crates/v/mullama)](https://crates.io/crates/mullama)
[![Documentation](https://docs.rs/mullama/badge.svg)](https://docs.rs/mullama)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Mullama provides memory-safe Rust bindings for llama.cpp with production-ready features including async/await support, real-time streaming, multimodal processing, and web framework integration.

## Why Mullama?

Most llama.cpp Rust bindings expose low-level C APIs directly. Mullama provides an **idiomatic Rust experience**:

```rust
// Other wrappers: manual memory management, raw pointers, verbose setup
let params = llama_context_default_params();
let ctx = unsafe { llama_new_context_with_model(model, params) };
let tokens = unsafe { llama_tokenize(model, text.as_ptr(), ...) };
// Don't forget to free everything...

// Mullama: builder patterns, async/await, automatic resource management
let model = ModelBuilder::new()
    .path("model.gguf")
    .gpu_layers(35)
    .build().await?;

let response = model.generate("Hello", 100).await?;
```

**Developer experience improvements:**

| Feature | Other Wrappers | Mullama |
|---------|---------------|---------|
| API Style | Raw FFI / C-like | Builder patterns, fluent API |
| Async Support | Manual threading | Native async/await with Tokio |
| Error Handling | Error codes / panics | `Result<T, MullamaError>` with context |
| Memory Management | Manual free/cleanup | Automatic RAII |
| Streaming | Callbacks | `Stream` trait, async iterators |
| Configuration | Struct fields | Type-safe builders with validation |
| Web Integration | DIY | Built-in Axum routes |

## Key Features

- **Async/Await Native** - Full Tokio integration for non-blocking operations
- **Real-time Streaming** - Token-by-token generation with backpressure handling
- **Multimodal Processing** - Text, image, and audio in a single pipeline
- **Web Framework Ready** - Direct Axum integration with REST APIs
- **WebSocket Support** - Real-time bidirectional communication
- **Parallel Processing** - Work-stealing parallelism for batch operations
- **GPU Acceleration** - CUDA, Metal, ROCm, and OpenCL support
- **Memory Safe** - Zero unsafe operations in public API

## Quick Start

### Installation

```toml
[dependencies]
mullama = "0.1.0"

# With all features
mullama = { version = "0.1.0", features = ["full"] }
```

### Prerequisites

**Linux (Ubuntu/Debian):**
```bash
sudo apt install -y build-essential cmake pkg-config libasound2-dev libpulse-dev
```

**macOS:**
```bash
brew install cmake pkg-config portaudio
```

**Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) and [CMake](https://cmake.org/download/).

See [Platform Setup Guide](./docs/PLATFORM_SETUP.md) for detailed instructions.

### Basic Example

```rust
use mullama::prelude::*;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    let model = ModelBuilder::new()
        .path("model.gguf")
        .context_size(4096)
        .build().await?;

    let response = model.generate("The future of AI is", 100).await?;
    println!("{}", response);

    Ok(())
}
```

## Feature Flags

```toml
[dependencies.mullama]
version = "0.1.0"
features = [
    "async",              # Async/await support
    "streaming",          # Token streaming
    "web",                # Axum web framework
    "websockets",         # WebSocket support
    "multimodal",         # Image and audio processing
    "streaming-audio",    # Real-time audio capture
    "format-conversion",  # Audio/image format conversion
    "parallel",           # Rayon parallel processing
    "daemon",             # Daemon mode with TUI client
    "full"                # All features
]
```

### Common Combinations

```toml
# Web applications
features = ["web", "websockets", "async", "streaming"]

# Multimodal AI
features = ["multimodal", "streaming-audio", "format-conversion"]

# High-performance batch processing
features = ["parallel", "async"]

# Daemon with TUI chat interface
features = ["daemon"]
```

## Daemon Mode

Mullama includes a multi-model daemon with OpenAI-compatible HTTP API and TUI client:

```bash
# Build the CLI
cargo build --release --features daemon

# Start daemon with local model
mullama serve --model llama:./llama.gguf

# Start with HuggingFace model (auto-downloads and caches)
mullama serve --model hf:TheBloke/Llama-2-7B-GGUF

# Multiple models with custom aliases
mullama serve \
  --model llama:hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf \
  --model mistral:hf:TheBloke/Mistral-7B-v0.1-GGUF

# Interactive TUI chat
mullama chat

# One-shot generation
mullama run "What is the meaning of life?"

# Model management
mullama models            # List loaded models
mullama load phi:./phi.gguf  # Load a model
mullama unload phi        # Unload a model
mullama default llama     # Set default model

# Cache management
mullama pull hf:TheBloke/Llama-2-7B-GGUF  # Pre-download model
mullama cache list        # List cached models
mullama cache size        # Show cache size
mullama cache clear       # Clear cache

# Use OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### HuggingFace Model Format

```
hf:<owner>/<repo>:<filename>   # Specific file
hf:<owner>/<repo>              # Auto-detect best GGUF
<alias>:hf:<owner>/<repo>      # With custom alias
```

Set `HF_TOKEN` environment variable for gated/private models.

Architecture:
```
                                   ┌──────────────────────────────────┐
                                   │           Daemon                 │
┌─────────────┐                    │  ┌────────────────────────────┐  │
│  TUI Client │◄── nng (IPC) ─────►│  │     Model Manager          │  │
└─────────────┘                    │  │  ┌───────┐  ┌───────┐      │  │
                                   │  │  │Model 1│  │Model 2│ ...  │  │
┌─────────────┐                    │  │  └───────┘  └───────┘      │  │
│   curl/app  │◄── HTTP/REST ─────►│  └────────────────────────────┘  │
└─────────────┘   (OpenAI API)     │                                  │
                                   │  Endpoints:                      │
┌─────────────┐                    │  • /v1/chat/completions          │
│ Other Client│◄── nng (IPC) ─────►│  • /v1/completions               │
└─────────────┘                    │  • /v1/models                    │
                                   │  • /v1/embeddings                │
                                   └──────────────────────────────────┘
```

Programmatic usage:
```rust
use mullama::daemon::{DaemonClient, DaemonBuilder};

// Connect as client
let client = DaemonClient::connect_default()?;
let result = client.chat("Hello, AI!", None, 100, 0.7)?;
println!("{} ({:.1} tok/s)", result.text, result.tokens_per_second());

// List models
for model in client.list_models()? {
    println!("{}: {}M params", model.alias, model.info.parameters / 1_000_000);
}
```

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

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](./docs/GETTING_STARTED.md) | Installation and first application |
| [Platform Setup](./docs/PLATFORM_SETUP.md) | OS-specific setup instructions |
| [Features Guide](./docs/FEATURES.md) | Integration features overview |
| [Use Cases](./docs/USE_CASES.md) | Real-world application examples |
| [API Reference](./docs/API_REFERENCE.md) | Complete API documentation |
| [Sampling Guide](./docs/sampling.md) | Sampling strategies and configuration |
| [GPU Guide](./docs/gpu.md) | GPU acceleration setup |
| [Feature Status](./docs/FEATURE_STATUS.md) | Implementation status and roadmap |

## Examples

```bash
# Basic text generation
cargo run --example simple --features async

# Streaming responses
cargo run --example streaming_generation --features "async,streaming"

# Web service
cargo run --example web_service --features "web,websockets"

# Audio processing
cargo run --example streaming_audio_demo --features "streaming-audio,multimodal"
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

```bash
git clone --recurse-submodules https://github.com/username/mullama.git
cd mullama
cargo test --all-features
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The underlying inference engine
- [ggml](https://github.com/ggerganov/ggml) - Tensor operations library
