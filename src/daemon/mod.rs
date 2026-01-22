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

mod anthropic;
mod client;
pub mod defaults;
mod hf;
mod models;
mod openai;
mod protocol;
pub mod registry;
mod server;
pub mod spawn;
mod tui;
pub mod ui;

// Protocol types (IPC)
pub use protocol::{
    generate_completion_id, ChatChoice as IpcChatChoice,
    ChatCompletionResponse as IpcChatCompletionResponse, ChatMessage,
    CompletionChoice as IpcCompletionChoice, CompletionResponse as IpcCompletionResponse,
    DaemonStats, DaemonStatus, EmbeddingInput, ErrorCode, ModelInfo, ModelStatus, Request,
    Response, Usage,
};

// Model management
pub use models::{LoadedModel, ModelLoadConfig, ModelManager, RequestGuard};

// Server
pub use server::{Daemon, DaemonBuilder, DaemonConfig};

// OpenAI-compatible HTTP API types
pub use openai::{
    create_openai_router, ApiError, AppState, ChatChoice, ChatCompletionRequest,
    ChatCompletionResponse, CompletionChoice, CompletionRequest, CompletionResponse,
    EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, ErrorDetail, ErrorResponse,
    ModelObject, ModelsResponse,
};

// Client
pub use client::{ChatResult, CompletionResult, DaemonClient};

// TUI
pub use tui::TuiApp;

// HuggingFace downloader
pub use hf::{
    resolve_model_path, CachedModel, GgufFileInfo, HfDownloader, HfModelSpec, HfSearchResult,
};

// Model registry (aliases)
pub use registry::{
    registry, resolve_model_name, ModelAlias, ModelRegistry, ParsedModelSpec, RegistryError,
    ResolvedModel,
};

// Anthropic API types
pub use anthropic::{
    AnthropicMessage, AnthropicUsage, ContentBlock, MessageContent, MessagesRequest,
    MessagesResponse, ResponseContentBlock,
};

// Daemon spawn utilities
pub use spawn::{
    daemon_status, ensure_daemon_running, is_daemon_running, spawn_daemon, stop_daemon,
    DaemonInfo, SpawnConfig, SpawnResult,
};

// Embedded Web UI
pub use ui::{serve_ui, ui_available};

// Default models
pub use defaults::{get_default, list_default_infos, list_defaults, DefaultModel, DefaultModelInfo};

/// Default IPC socket path
#[cfg(unix)]
pub const DEFAULT_SOCKET: &str = "ipc:///tmp/mullama.sock";
#[cfg(windows)]
pub const DEFAULT_SOCKET: &str = "ipc://mullama";

/// Default HTTP port for OpenAI API
pub const DEFAULT_HTTP_PORT: u16 = 8080;

/// Default model alias when none specified
pub const DEFAULT_MODEL: &str = "default";
