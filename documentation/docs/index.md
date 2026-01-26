---
title: Mullama - Drop-in Ollama Replacement with Native Language Bindings
description: Drop-in Ollama replacement with native Node.js, Python, Go, PHP, and C/C++ bindings. OpenAI and Anthropic-compatible API server. All-in-one LLM toolkit.
---

<div class="hero" markdown>

# Drop-in Ollama replacement. Native language bindings. Production-ready.

Mullama is a local LLM server and library that works just like Ollama -- same CLI commands, same Modelfile syntax -- but with native bindings for **Python, Node.js, Go, PHP, Rust, and C/C++**.

```bash
curl -fsSL https://mullama.dev/install.sh | sh
mullama run llama3.2:1b "Hello!"
```

<div class="hero-actions" markdown>

[Get Started](getting-started/index.md){ .md-button .md-button--primary }
[Compare to Ollama](comparison/vs-ollama.md){ .md-button }
[View on GitHub](https://github.com/neul-labs/mullama){ .md-button }

</div>

</div>

---

## Two Ways to Use Mullama

<div class="grid cards" markdown>

-   **:material-code-braces: Use as a Library**

    ---

    Embed LLM inference directly in your application with native bindings. No HTTP overhead, no separate process -- just import and generate.

    **Supported:** Node.js, Python, Rust, Go, PHP, C/C++

    [:octicons-arrow-right-24: Library Guide](guide/index.md)

-   **:material-server: Use as a Server**

    ---

    Run mullama as a daemon with OpenAI-compatible APIs, a web UI, and multi-model management. Drop-in replacement for Ollama with more power.

    **Compatible:** OpenAI SDK, Anthropic SDK, curl, any HTTP client

    [:octicons-arrow-right-24: Daemon & CLI](daemon/index.md)

</div>

---

## Quick Start

=== "Node.js"

    ```bash
    npm install mullama
    ```

    ```javascript
    const { Model, Context } = require('mullama');

    async function main() {
      const model = await Model.load('llama3.2-1b.gguf', { gpuLayers: 32 });
      const ctx = new Context(model, { contextSize: 4096 });

      const response = await ctx.generate('Explain quantum computing in one sentence.');
      console.log(response);
    }

    main();
    ```

=== "Python"

    ```bash
    pip install mullama
    ```

    ```python
    from mullama import Model, Context

    model = Model.load('llama3.2-1b.gguf', n_gpu_layers=32)
    ctx = Context(model, n_ctx=4096)

    response = ctx.generate('Explain quantum computing in one sentence.')
    print(response)
    ```

=== "Rust"

    ```toml
    # Cargo.toml
    [dependencies]
    mullama = { version = "0.1", features = ["async", "streaming"] }
    ```

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::sync::Arc;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let model = Arc::new(Model::load("llama3.2-1b.gguf")?);
        let params = ContextParams { n_ctx: 4096, ..Default::default() };
        let mut ctx = Context::new(model, params)?;

        let response = ctx.generate("Explain quantum computing in one sentence.", 256)?;
        println!("{}", response);
        Ok(())
    }
    ```

=== "CLI"

    ```bash
    # Install and run in one command
    mullama run llama3.2:1b "Explain quantum computing in one sentence."
    ```

    ```bash
    # Or start a server with OpenAI-compatible API
    mullama serve --model llama3.2:1b

    # Use with any OpenAI SDK
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hello!"}]}'
    ```

---

## Why Mullama?

<div class="grid cards" markdown>

-   **:material-lightning-bolt: Native Performance**

    ---

    Direct function calls instead of HTTP roundtrips. Microseconds of overhead instead of milliseconds. Your LLM runs in-process, not in a separate server.

-   **:material-language-javascript: Multi-Language Bindings**

    ---

    First-class support for Node.js, Python, Go, PHP, and C/C++. All bindings share the same high-performance Rust core via a unified FFI layer.

-   **:material-gpu: GPU Accelerated**

    ---

    NVIDIA CUDA, Apple Metal, AMD ROCm, and OpenCL. Automatic detection and configuration. Full GPU offload or partial layer offloading.

-   **:material-shield-check: Production Ready**

    ---

    Memory-safe Rust core. Comprehensive error handling. Prometheus metrics. Graceful shutdown. Session persistence. Zero unsafe in the public API.

-   **:material-image-multiple: Multimodal**

    ---

    Process text, images, and audio in a unified pipeline. Real-time audio capture with voice activity detection. Vision-language model support with CLIP and DINOv2.

-   **:material-api: API Compatible**

    ---

    OpenAI and Anthropic-compatible API endpoints. Use your existing SDKs and tools without changes. Drop-in replacement for cloud APIs in development.

</div>

---

## How It Compares

| Capability | Mullama | Ollama | Raw llama.cpp |
|:-----------|:-------:|:------:|:-------------:|
| Native language bindings | Node.js, Python, Go, PHP, C | -- | C/C++ only |
| Embed in your application | Yes | No (HTTP only) | Yes (C API) |
| OpenAI-compatible API | Yes | Yes | -- |
| Anthropic-compatible API | Yes | -- | -- |
| Streaming generation | Native + SSE | SSE only | Callback |
| Async/await support | Native | -- | -- |
| Real-time audio input | Yes (VAD) | -- | -- |
| Web framework integration | Axum built-in | -- | -- |
| WebSocket server | Built-in | -- | -- |
| Grammar constraints | GBNF | -- | GBNF |
| JSON structured output | Schema-based | JSON mode | -- |
| Embeddings | Multi-strategy | Basic | Basic |
| LoRA adapters | Hot-swap | Modelfile | CLI flag |
| ColBERT late interaction | Yes | -- | -- |
| SIMD-accelerated sampling | AVX2/512, NEON | -- | Inference only |
| Batch parallel processing | Rayon | -- | -- |
| Memory-mapped models | Yes | Yes | Yes |
| Web UI | Built-in | -- | -- |
| TUI chat interface | Built-in | -- | -- |
| Model aliases | 40+ pre-configured | Yes | -- |

[:octicons-arrow-right-24: Full comparison with Ollama](comparison/vs-ollama.md)

---

## Built for Real Applications

<div class="grid cards" markdown>

-   **:material-chat: Chatbot**

    Build conversational AI with streaming responses, multi-turn context, and chat templates for any model format.

    [:octicons-arrow-right-24: Tutorial](examples/chatbot.md)

-   **:material-database-search: RAG Pipeline**

    Semantic search with embeddings, ColBERT late interaction scoring, and grammar-constrained generation for structured answers.

    [:octicons-arrow-right-24: Tutorial](examples/rag.md)

-   **:material-microphone: Voice Assistant**

    Real-time audio capture with voice activity detection, speech-to-text processing, and streaming text generation.

    [:octicons-arrow-right-24: Tutorial](examples/voice-assistant.md)

-   **:material-cloud: API Server**

    Production API server with OpenAI compatibility, streaming SSE, rate limiting, and Prometheus metrics.

    [:octicons-arrow-right-24: Tutorial](examples/api-server.md)

</div>

---

## Architecture

Mullama is built in three layers, each providing progressively higher-level abstractions:

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                       │
├──────────┬──────────┬─────────┬──────────┬──────────────┤
│ Node.js  │  Python  │   Go    │   PHP    │    C/C++     │
│  (NAPI)  │  (PyO3)  │  (cgo)  │  (FFI)   │   (Header)  │
├──────────┴──────────┴─────────┴──────────┴──────────────┤
│                                                          │
│   Integration Layer                                      │
│   Async | Streaming | Web | WebSocket | Multimodal       │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   Core API Layer                                         │
│   Model | Context | Sampler | Batch | Embedding          │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   Foundation Layer                                       │
│   FFI Bindings | Memory Management | Platform Detection  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                     llama.cpp (C++)                       │
├──────────────────────────────────────────────────────────┤
│              CUDA | Metal | ROCm | OpenCL                │
└──────────────────────────────────────────────────────────┘
```

---

## By the Numbers

| | |
|:--|:--|
| **14,000+** | Lines of Rust integration code |
| **6** | Native language bindings |
| **4** | GPU acceleration backends |
| **40+** | Pre-configured model aliases |
| **2** | API compatibility layers (OpenAI + Anthropic) |
| **10+** | Sampling strategies with SIMD acceleration |

---

## Get Started

<div class="grid cards" markdown>

-   **:material-download: Install**

    Get mullama running on your platform in under 2 minutes.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   **:material-rocket-launch: First Project**

    Build a working chatbot from scratch in 15 minutes.

    [:octicons-arrow-right-24: Tutorial](getting-started/first-project.md)

-   **:material-compare: Why Mullama?**

    See how mullama compares to Ollama and when to use each.

    [:octicons-arrow-right-24: Comparison](comparison/vs-ollama.md)

-   **:material-book-open: API Reference**

    Complete reference for all types, methods, and configuration options.

    [:octicons-arrow-right-24: API Docs](api/index.md)

</div>
