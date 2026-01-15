//! IPC Protocol for daemon communication
//!
//! Uses a JSON-based protocol over nng sockets for high-performance IPC.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request messages from client to daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Request {
    /// Ping to check daemon status
    Ping,

    /// Get daemon status and loaded models
    Status,

    /// List all loaded models
    ListModels,

    /// Load a model with alias
    LoadModel {
        alias: String,
        path: String,
        #[serde(default)]
        gpu_layers: i32,
        #[serde(default)]
        context_size: u32,
    },

    /// Unload a model by alias
    UnloadModel { alias: String },

    /// Set the default model
    SetDefaultModel { alias: String },

    /// Chat completion (OpenAI-style)
    ChatCompletion {
        model: Option<String>,
        messages: Vec<ChatMessage>,
        #[serde(default = "default_max_tokens")]
        max_tokens: u32,
        #[serde(default = "default_temperature")]
        temperature: f32,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        stop: Vec<String>,
    },

    /// Text completion
    Completion {
        model: Option<String>,
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: u32,
        #[serde(default = "default_temperature")]
        temperature: f32,
        #[serde(default)]
        stream: bool,
    },

    /// Generate embeddings
    Embeddings {
        model: Option<String>,
        input: EmbeddingInput,
    },

    /// Tokenize text
    Tokenize { model: Option<String>, text: String },

    /// Cancel ongoing generation for a request
    Cancel { request_id: String },

    /// Shutdown the daemon
    Shutdown,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

/// Chat message for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content - can be simple text or array of content parts (for vision)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),
    /// Array of content parts (text and/or images)
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the text content from the message
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// Check if this content contains images
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }

    /// Extract image URLs/data from the content
    pub fn images(&self) -> Vec<&ImageUrl> {
        match self {
            MessageContent::Text(_) => vec![],
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url),
                    _ => None,
                })
                .collect(),
        }
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// Content part in a message (OpenAI vision API format)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content (can be URL or base64 data URI)
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL or base64 data URI (data:image/jpeg;base64,...)
    pub url: String,
    /// Optional detail level for image processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Input for embeddings (can be string or array)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Response messages from daemon to client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Response {
    /// Pong response
    Pong { uptime_secs: u64, version: String },

    /// Daemon status
    Status(DaemonStatus),

    /// List of models
    Models(Vec<ModelStatus>),

    /// Model loaded
    ModelLoaded { alias: String, info: ModelInfo },

    /// Model unloaded
    ModelUnloaded { alias: String },

    /// Default model set
    DefaultModelSet { alias: String },

    /// Chat completion response
    ChatCompletion(ChatCompletionResponse),

    /// Text completion response
    Completion(CompletionResponse),

    /// Streaming chunk
    StreamChunk(StreamChunk),

    /// Stream finished
    StreamEnd { request_id: String, usage: Usage },

    /// Embeddings response
    Embeddings(EmbeddingsResponse),

    /// Tokenization result
    Tokens { tokens: Vec<i32>, count: usize },

    /// Request cancelled
    Cancelled { request_id: String },

    /// Shutting down
    ShuttingDown,

    /// Error
    Error {
        code: ErrorCode,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
    },
}

/// Daemon status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub version: String,
    pub uptime_secs: u64,
    pub models_loaded: usize,
    pub default_model: Option<String>,
    pub http_endpoint: Option<String>,
    pub ipc_endpoint: String,
    pub stats: DaemonStats,
}

/// Daemon statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonStats {
    pub requests_total: u64,
    pub tokens_generated: u64,
    pub active_requests: u32,
    pub memory_used_mb: u64,
    pub gpu_available: bool,
}

/// Model status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub alias: String,
    pub info: ModelInfo,
    pub is_default: bool,
    pub active_requests: u32,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: String,
    pub parameters: u64,
    pub context_size: u32,
    pub vocab_size: u32,
    pub gpu_layers: i32,
    pub quantization: Option<String>,
}

/// Chat completion response (OpenAI compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Text completion response (OpenAI compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub request_id: String,
    pub index: u32,
    pub delta: String,
    pub token_id: i32,
}

/// Token usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Embeddings response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Error codes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    ModelNotFound,
    ModelLoadFailed,
    NoDefaultModel,
    InvalidRequest,
    GenerationFailed,
    Cancelled,
    RateLimited,
    Internal,
    Timeout,
}

impl Request {
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

impl Response {
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        Response::Error {
            code,
            message: message.into(),
            request_id: None,
        }
    }

    pub fn error_with_id(code: ErrorCode, message: impl Into<String>, request_id: String) -> Self {
        Response::Error {
            code,
            message: message.into(),
            request_id: Some(request_id),
        }
    }
}

/// Generate a unique request ID
pub fn generate_request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("req_{:x}", ts)
}

/// Generate a unique completion ID
pub fn generate_completion_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("cmpl_{:x}", ts)
}
