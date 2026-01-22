//! Anthropic-compatible API
//!
//! Provides REST endpoints compatible with the Anthropic Claude API specification.
//! This allows applications built for Claude to work with local models via Mullama.
//!
//! ## Endpoint
//!
//! ```
//! POST /v1/messages
//! ```
//!
//! ## Example Request
//!
//! ```json
//! {
//!   "model": "llama3.2:1b",
//!   "max_tokens": 1024,
//!   "messages": [
//!     {"role": "user", "content": "Hello, Claude!"}
//!   ],
//!   "stream": false
//! }
//! ```
//!
//! ## Example Response
//!
//! ```json
//! {
//!   "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
//!   "type": "message",
//!   "role": "assistant",
//!   "content": [
//!     {"type": "text", "text": "Hello! How can I help you today?"}
//!   ],
//!   "model": "llama3.2:1b",
//!   "stop_reason": "end_turn",
//!   "usage": {
//!     "input_tokens": 10,
//!     "output_tokens": 25
//!   }
//! }
//! ```

use std::sync::Arc;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use super::protocol::ChatMessage;
use super::server::Daemon;

/// Shared state for the HTTP server
pub type AppState = Arc<Daemon>;

// ==================== Request Types ====================

/// Anthropic Messages API request
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    /// Model to use
    pub model: Option<String>,

    /// Maximum tokens to generate
    pub max_tokens: u32,

    /// Messages in the conversation
    pub messages: Vec<AnthropicMessage>,

    /// System prompt (optional, can also be first message with role "system")
    #[serde(default)]
    pub system: Option<String>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Temperature for sampling (0.0 to 1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p sampling
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k sampling
    #[serde(default)]
    pub top_k: Option<i32>,

    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,

    /// Metadata (ignored but accepted for compatibility)
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

fn default_temperature() -> f32 {
    1.0
}

/// A message in the Anthropic format
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    /// Role: "user" or "assistant"
    pub role: String,

    /// Content - can be string or array of content blocks
    pub content: MessageContent,
}

/// Message content - either a simple string or array of content blocks
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text string
    Text(String),

    /// Array of content blocks (for multimodal)
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    /// Get the text content as a string
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => {
                blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
    }
}

/// A content block in a message
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },

    /// Image content (base64 encoded)
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
    },

    /// Tool use block (for function calling)
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    /// Tool result block
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// Image source for multimodal messages
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64"
    pub media_type: String,  // "image/jpeg", "image/png", etc.
    pub data: String,        // base64-encoded image data
}

// ==================== Response Types ====================

/// Anthropic Messages API response
#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    /// Unique message ID
    pub id: String,

    /// Object type (always "message")
    #[serde(rename = "type")]
    pub object_type: String,

    /// Role (always "assistant")
    pub role: String,

    /// Content blocks
    pub content: Vec<ResponseContentBlock>,

    /// Model used
    pub model: String,

    /// Stop reason
    pub stop_reason: Option<String>,

    /// Stop sequence that triggered stop (if any)
    pub stop_sequence: Option<String>,

    /// Token usage
    pub usage: AnthropicUsage,
}

/// Response content block
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },
}

/// Token usage in Anthropic format
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Error response in Anthropic format
#[derive(Debug, Serialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ==================== Streaming Types ====================

/// Streaming event types for Anthropic SSE
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStartData,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: AnthropicUsage,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "error")]
    Error { error: AnthropicErrorDetail },
}

#[derive(Debug, Serialize)]
pub struct MessageStartData {
    pub id: String,
    #[serde(rename = "type")]
    pub object_type: String,
    pub role: String,
    pub content: Vec<serde_json::Value>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStartData {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaData {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

// ==================== Handlers ====================

/// Handle POST /v1/messages
pub async fn messages_handler(
    State(daemon): State<AppState>,
    Json(request): Json<MessagesRequest>,
) -> Response {
    if request.stream {
        match handle_messages_streaming(daemon, request).await {
            Ok(stream) => stream.into_response(),
            Err(e) => e.into_response(),
        }
    } else {
        match handle_messages(daemon, request).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(e) => e.into_response(),
        }
    }
}

/// Handle non-streaming messages request
async fn handle_messages(
    daemon: Arc<Daemon>,
    request: MessagesRequest,
) -> Result<MessagesResponse, ApiError> {
    // Convert Anthropic messages to internal format
    let messages = convert_messages(&request)?;

    // Get stop sequences
    let stop = request.stop_sequences.unwrap_or_default();

    // Call the daemon
    let loaded = daemon
        .models
        .get(request.model.as_deref())
        .await
        .map_err(|e| ApiError::model_not_found(e.to_string()))?;

    let model_alias = loaded.alias.clone();

    // Build prompt
    let prompt = daemon.build_chat_prompt(&loaded.model, &messages);

    // Get stop sequences from chat template
    let mut all_stops = loaded.model.get_chat_stop_sequences();
    all_stops.extend(stop);

    // Generate
    let result = daemon
        .generate_text(
            &loaded,
            &prompt,
            request.max_tokens,
            request.temperature,
            &all_stops,
            None, // Anthropic API doesn't support response_format yet
        )
        .await
        .map_err(|e| ApiError::generation_failed(e.to_string()))?;

    let (text, prompt_tokens, completion_tokens) = result;

    Ok(MessagesResponse {
        id: generate_message_id(),
        object_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![ResponseContentBlock::Text { text }],
        model: model_alias,
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: prompt_tokens,
            output_tokens: completion_tokens,
        },
    })
}

/// Handle streaming messages request
async fn handle_messages_streaming(
    daemon: Arc<Daemon>,
    request: MessagesRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    // Convert Anthropic messages to internal format
    let messages = convert_messages(&request)?;

    // Get stop sequences
    let stop = request.stop_sequences.unwrap_or_default();

    // Get model
    let loaded = daemon
        .models
        .get(request.model.as_deref())
        .await
        .map_err(|e| ApiError::model_not_found(e.to_string()))?;

    let model_alias = loaded.alias.clone();
    let message_id = generate_message_id();

    // Build prompt
    let prompt = daemon.build_chat_prompt(&loaded.model, &messages);

    // Get stop sequences from chat template
    let mut all_stops = loaded.model.get_chat_stop_sequences();
    all_stops.extend(stop);

    // Start streaming generation
    let (rx, prompt_tokens, _request_id) = daemon
        .generate_text_streaming(
            loaded,
            prompt,
            request.max_tokens,
            request.temperature,
            all_stops,
        )
        .await
        .map_err(|e| ApiError::generation_failed(e.to_string()))?;

    // Create the initial events
    let message_start = StreamEvent::MessageStart {
        message: MessageStartData {
            id: message_id.clone(),
            object_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: model_alias.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: prompt_tokens,
                output_tokens: 0,
            },
        },
    };

    let content_block_start = StreamEvent::ContentBlockStart {
        index: 0,
        content_block: ContentBlockStartData {
            block_type: "text".to_string(),
            text: String::new(),
        },
    };

    // Create the stream
    let initial_events = vec![
        Ok(Event::default()
            .event("message_start")
            .data(serde_json::to_string(&message_start).unwrap())),
        Ok(Event::default()
            .event("content_block_start")
            .data(serde_json::to_string(&content_block_start).unwrap())),
    ];

    let initial_stream = stream::iter(initial_events);

    // Transform the token stream
    let _model_alias_clone = model_alias.clone();
    let token_stream = ReceiverStream::new(rx).map(move |chunk| {
        // StreamChunk is a struct with delta field containing the text
        let delta = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta { text: chunk.delta },
        };
        Ok(Event::default()
            .event("content_block_delta")
            .data(serde_json::to_string(&delta).unwrap()))
    });

    // Final events
    let final_events = vec![
        Ok(Event::default()
            .event("content_block_stop")
            .data(serde_json::to_string(&StreamEvent::ContentBlockStop { index: 0 }).unwrap())),
        Ok(Event::default()
            .event("message_stop")
            .data(serde_json::to_string(&StreamEvent::MessageStop).unwrap())),
    ];

    let final_stream = stream::iter(final_events);

    // Combine all streams
    let combined = initial_stream.chain(token_stream).chain(final_stream);

    Ok(Sse::new(combined).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    ))
}

// ==================== Helpers ====================

/// Convert Anthropic messages to internal ChatMessage format
fn convert_messages(request: &MessagesRequest) -> Result<Vec<ChatMessage>, ApiError> {
    let mut messages = Vec::new();

    // Add system message if provided
    if let Some(ref system) = request.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system.clone().into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert each message
    for msg in &request.messages {
        messages.push(ChatMessage {
            role: msg.role.clone(),
            content: msg.content.as_text().into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    Ok(messages)
}

/// Generate a unique message ID in Anthropic format
fn generate_message_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let chars: String = (0..24)
        .map(|_| {
            let idx = rng.gen_range(0..36);
            if idx < 10 {
                (b'0' + idx) as char
            } else {
                (b'a' + idx - 10) as char
            }
        })
        .collect();
    format!("msg_{}", chars)
}

/// Map internal finish reason to Anthropic format
fn map_finish_reason(reason: &str) -> String {
    match reason {
        "stop" | "eos" => "end_turn".to_string(),
        "length" | "max_tokens" => "max_tokens".to_string(),
        "tool_use" => "tool_use".to_string(),
        _ => "end_turn".to_string(),
    }
}

// ==================== Error Handling ====================

pub struct ApiError {
    status: StatusCode,
    error_type: String,
    message: String,
}

impl ApiError {
    fn model_not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "not_found_error".to_string(),
            message,
        }
    }

    fn generation_failed(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "api_error".to_string(),
            message,
        }
    }

    fn invalid_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error".to_string(),
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = AnthropicError {
            error_type: "error".to_string(),
            error: AnthropicErrorDetail {
                error_type: self.error_type,
                message: self.message,
            },
        };

        (self.status, Json(body)).into_response()
    }
}
