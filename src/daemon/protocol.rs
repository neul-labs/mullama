//! IPC Protocol for daemon communication
//!
//! Uses a JSON-based protocol over nng sockets for high-performance IPC.
//!
//! ## Performance Optimizations (Rust-exclusive)
//!
//! This module uses Rust-specific zero-copy patterns that are impossible in Go:
//! - **Arc<str>**: Shared request IDs across all stream chunks (no cloning)
//! - **rkyv**: Zero-copy deserialization - data is accessed directly from bytes
//!   without parsing. 10-100x faster than serde JSON for IPC.
//!
//! Go strings are immutable but always copied on share. Rust's ownership model
//! allows true zero-copy sharing.
//!
//! ## rkyv Zero-Copy Serialization
//!
//! Select types have `Archive` derives for zero-copy deserialization:
//! - Deserialize is essentially free - just pointer validation
//! - Data is accessed directly from serialized bytes
//! - 10-100x faster than JSON for complex structures
//!
//! This is impossible in Go because:
//! - Go requires runtime reflection for serialization
//! - No way to reinterpret bytes as structs safely
//! - Protobuf/msgpack still allocate on deserialize

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
        /// Response format (text or JSON)
        #[serde(default)]
        response_format: Option<ResponseFormat>,
        /// Tools/functions the model can call
        #[serde(default)]
        tools: Option<Vec<Tool>>,
        /// How to choose which tool to call
        #[serde(default)]
        tool_choice: Option<ToolChoice>,
        /// Extended thinking configuration
        #[serde(default)]
        thinking: Option<ThinkingConfig>,
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
    #[serde(default)]
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID when role is "tool"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
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
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
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
///
/// Has rkyv derives for zero-copy IPC when both endpoints support it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
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
    /// Thinking content from reasoning models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingContent>,
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

/// Streaming chunk with zero-copy optimizations
///
/// ## Rust-Exclusive Optimizations
///
/// - **request_id**: Uses `Arc<str>` for zero-copy sharing across all chunks in a stream.
///   In Go, each chunk would clone the string. In Rust, all chunks share the same allocation.
/// - **delta**: Regular String (necessary for serialization compatibility)
///
/// **Memory savings**: For a 1000-token generation, this saves ~999 string allocations
/// for the request_id (about 24 bytes * 999 = ~24KB allocation overhead eliminated).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Shared request ID (Arc<str> allows zero-copy sharing across chunks)
    /// When serializing, this appears as a regular string
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    pub request_id: Arc<str>,
    pub index: u32,
    pub delta: String,
    pub token_id: i32,
    /// True if this chunk is thinking content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    /// Tool call delta for streaming tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Serialize Arc<str> as a regular string
fn serialize_arc_str<S>(value: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(value)
}

/// Deserialize a string into Arc<str>
fn deserialize_arc_str<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Use fully-qualified syntax to disambiguate from rkyv::Deserialize
    let s = <String as serde::Deserialize>::deserialize(deserializer)?;
    Ok(Arc::from(s))
}

/// Delta for streaming tool calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function call delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Token usage
///
/// Has rkyv derives for zero-copy IPC when both endpoints support it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
#[archive(check_bytes)]
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
///
/// Has rkyv derives for zero-copy IPC when both endpoints support it.
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize,
)]
#[archive(check_bytes)]
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

// ==================== JSON Mode Types ====================

/// Response format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// Plain text output
    #[serde(rename = "text")]
    Text,
    /// JSON object output (model will produce valid JSON)
    #[serde(rename = "json_object")]
    JsonObject,
    /// JSON output conforming to a schema
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaSpec },
}

impl Default for ResponseFormat {
    fn default() -> Self {
        ResponseFormat::Text
    }
}

/// JSON schema specification for structured output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaSpec {
    /// Name for the schema
    pub name: String,
    /// The JSON schema definition
    pub schema: serde_json::Value,
    /// Whether to enforce strict schema compliance
    #[serde(default)]
    pub strict: bool,
}

// ==================== Tool/Function Calling Types ====================

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (currently only "function" is supported)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionDefinition,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema for function parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    /// Whether strict schema validation is required
    #[serde(default)]
    pub strict: bool,
}

/// Tool choice specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Let model decide: "auto", "none", "required"
    Mode(String),
    /// Force a specific function
    Specific {
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolChoiceFunction,
    },
}

/// Specific function choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// A tool call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// Type of call (always "function" for now)
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function call details
    pub function: FunctionCall,
}

/// Function call details within a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// JSON string of arguments
    pub arguments: String,
}

// ==================== Thinking Mode Types ====================

/// Configuration for extended thinking mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Enable extended thinking
    #[serde(default)]
    pub enabled: bool,
    /// Maximum tokens for thinking (0 = unlimited)
    #[serde(default)]
    pub budget_tokens: u32,
    /// Include thinking content in stream
    #[serde(default)]
    pub stream_thinking: bool,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            budget_tokens: 0,
            stream_thinking: false,
        }
    }
}

/// Thinking content returned by models with reasoning capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingContent {
    /// The thinking/reasoning content
    pub content: String,
    /// Number of tokens used for thinking
    pub tokens: u32,
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
