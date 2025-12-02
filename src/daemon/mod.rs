//! # Mullama Daemon
//!
//! A multi-model daemon for serving LLMs with IPC and OpenAI-compatible HTTP API.
//!
//! ## Architecture
//!
//! ```text
//!                                    ┌──────────────────────────────────┐
//!                                    │           Daemon                 │
//! ┌─────────────┐                    │  ┌────────────────────────────┐  │
//! │  TUI Client │◄── nng (IPC) ─────►│  │     Model Manager          │  │
//! └─────────────┘                    │  │  ┌───────┐  ┌───────┐      │  │
//!                                    │  │  │Model 1│  │Model 2│ ...  │  │
//! ┌─────────────┐                    │  │  └───────┘  └───────┘      │  │
//! │   curl/app  │◄── HTTP/REST ─────►│  └────────────────────────────┘  │
//! └─────────────┘   (OpenAI API)     │                                  │
//!                                    │  Endpoints:                      │
//! ┌─────────────┐                    │  • /v1/chat/completions          │
//! │ Other Client│◄── nng (IPC) ─────►│  • /v1/completions               │
//! └─────────────┘                    │  • /v1/models                    │
//!                                    │  • /v1/embeddings                │
//!                                    └──────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Start the daemon
//! mullama serve --model llama:./models/llama.gguf --model mistral:./models/mistral.gguf
//!
//! # Interactive TUI chat
//! mullama chat
//!
//! # One-shot generation
//! mullama run "What is the meaning of life?"
//!
//! # Use with curl (OpenAI compatible)
//! curl http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model": "llama", "messages": [{"role": "user", "content": "Hello!"}]}'
//! ```

mod protocol;
mod models;
mod server;
mod openai;
mod client;
mod tui;
mod hf;

// Protocol types (IPC)
pub use protocol::{
    Request, Response, ErrorCode, DaemonStatus, DaemonStats,
    ModelInfo, ModelStatus, ChatMessage, Usage, EmbeddingInput,
    ChatCompletionResponse as IpcChatCompletionResponse,
    CompletionResponse as IpcCompletionResponse,
    ChatChoice as IpcChatChoice,
    CompletionChoice as IpcCompletionChoice,
    generate_completion_id,
};

// Model management
pub use models::{LoadedModel, ModelLoadConfig, ModelManager, RequestGuard};

// Server
pub use server::{Daemon, DaemonBuilder, DaemonConfig};

// OpenAI-compatible HTTP API types
pub use openai::{
    create_openai_router, AppState, ApiError,
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    ChatChoice, CompletionChoice,
    ModelsResponse, ModelObject,
    EmbeddingsRequest, EmbeddingsResponse, EmbeddingObject,
    ErrorResponse, ErrorDetail,
};

// Client
pub use client::{DaemonClient, ChatResult, CompletionResult};

// TUI
pub use tui::TuiApp;

// HuggingFace downloader
pub use hf::{HfDownloader, HfModelSpec, CachedModel, resolve_model_path};

/// Default IPC socket path
#[cfg(unix)]
pub const DEFAULT_SOCKET: &str = "ipc:///tmp/mullama.sock";
#[cfg(windows)]
pub const DEFAULT_SOCKET: &str = "ipc://mullama";

/// Default HTTP port for OpenAI API
pub const DEFAULT_HTTP_PORT: u16 = 8080;

/// Default model alias when none specified
pub const DEFAULT_MODEL: &str = "default";
