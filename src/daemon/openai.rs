//! OpenAI-compatible HTTP API
//!
//! Provides REST endpoints compatible with the OpenAI API specification.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;
use tower_http::cors::{Any, CorsLayer};

use super::protocol::{ChatMessage, EmbeddingInput, Response as ProtoResponse, StreamChunk, Usage};
use super::server::Daemon;

/// Shared state for the HTTP server
pub type AppState = Arc<Daemon>;

/// Create the OpenAI-compatible router
pub fn create_openai_router(daemon: Arc<Daemon>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // OpenAI API endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model", get(get_model))
        .route("/v1/embeddings", post(embeddings))
        // Health and status
        .route("/health", get(health))
        .route("/status", get(status))
        .with_state(daemon)
        .layer(cors)
}

// ==================== Request/Response Types ====================

/// Chat completion request (OpenAI compatible)
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
}

/// Chat completion response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Streaming chat completion chunk (OpenAI compatible)
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoiceDelta {
    pub index: u32,
    pub delta: DeltaContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Text completion request
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Text completion response
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Models list response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Embeddings request
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

/// Embeddings response
#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

// ==================== Handlers ====================

/// POST /v1/chat/completions
async fn chat_completions(
    State(daemon): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    // Check if any message contains images (vision request)
    let has_images = req.messages.iter().any(|m| m.content.has_images());

    // Handle streaming requests
    if req.stream {
        if has_images {
            #[cfg(feature = "multimodal")]
            return chat_completions_vision_stream(daemon, req).await;
            #[cfg(not(feature = "multimodal"))]
            return Err(ApiError::new("Vision support requires multimodal feature"));
        }
        return chat_completions_stream(daemon, req).await;
    }

    // Handle vision requests (non-streaming)
    if has_images {
        #[cfg(feature = "multimodal")]
        {
            match daemon
                .handle_vision_chat_completion(
                    req.model,
                    req.messages,
                    req.max_tokens,
                    req.temperature,
                    req.stop.unwrap_or_default(),
                )
                .await
            {
                super::protocol::Response::ChatCompletion(resp) => {
                    return Ok(Json(ChatCompletionResponse {
                        id: resp.id,
                        object: resp.object,
                        created: resp.created,
                        model: resp.model,
                        choices: resp
                            .choices
                            .into_iter()
                            .map(|c| ChatChoice {
                                index: c.index,
                                message: c.message,
                                finish_reason: c.finish_reason,
                            })
                            .collect(),
                        usage: resp.usage,
                    })
                    .into_response());
                }
                super::protocol::Response::Error { code, message, .. } => {
                    return Err(ApiError::new(format!("{:?}: {}", code, message)));
                }
                _ => return Err(ApiError::new("Unexpected response")),
            }
        }
        #[cfg(not(feature = "multimodal"))]
        return Err(ApiError::new("Vision support requires multimodal feature"));
    }

    // Non-streaming text-only request
    let request = super::protocol::Request::ChatCompletion {
        model: req.model,
        messages: req.messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: false,
        stop: req.stop.unwrap_or_default(),
        response_format: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    match daemon.handle_request(request).await {
        super::protocol::Response::ChatCompletion(resp) => Ok(Json(ChatCompletionResponse {
            id: resp.id,
            object: resp.object,
            created: resp.created,
            model: resp.model,
            choices: resp
                .choices
                .into_iter()
                .map(|c| ChatChoice {
                    index: c.index,
                    message: c.message,
                    finish_reason: c.finish_reason,
                })
                .collect(),
            usage: resp.usage,
        })
        .into_response()),
        super::protocol::Response::Error { code, message, .. } => {
            Err(ApiError::new(format!("{:?}: {}", code, message)))
        }
        _ => Err(ApiError::new("Unexpected response")),
    }
}

/// Handle streaming chat completions with SSE
async fn chat_completions_stream(
    daemon: AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Start streaming generation
    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_chat_completion_streaming(
            req.model,
            req.messages,
            req.max_tokens,
            req.temperature,
            req.stop.unwrap_or_default(),
        )
        .await
        .map_err(|resp| {
            if let super::protocol::Response::Error { message, .. } = resp {
                ApiError::new(message)
            } else {
                ApiError::new("Failed to start streaming")
            }
        })?;

    // Convert mpsc receiver to SSE stream
    let stream = ReceiverStream::new(rx);
    let request_id_clone = request_id.clone();
    let model_clone = model_alias.clone();

    let sse_stream = stream
        .map(move |chunk| {
            let sse_chunk = ChatCompletionChunk {
                id: request_id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_clone.clone(),
                choices: vec![ChatChoiceDelta {
                    index: chunk.index,
                    delta: DeltaContent {
                        role: if chunk.index == 0 {
                            Some("assistant".to_string())
                        } else {
                            None
                        },
                        content: Some(chunk.delta),
                    },
                    finish_reason: None,
                }],
            };

            Event::default().data(serde_json::to_string(&sse_chunk).unwrap_or_default())
        })
        .chain(stream::once(async move {
            // Send final chunk with finish_reason
            let final_chunk = ChatCompletionChunk {
                id: request_id,
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_alias,
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: DeltaContent {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            Event::default().data(serde_json::to_string(&final_chunk).unwrap_or_default())
        }))
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);

    Ok(Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response())
}

/// Handle streaming vision chat completions with SSE
#[cfg(feature = "multimodal")]
async fn chat_completions_vision_stream(
    daemon: AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Start streaming vision generation
    let (rx, _prompt_tokens, request_id, model_alias) = daemon
        .handle_vision_chat_completion_streaming(
            req.model,
            req.messages,
            req.max_tokens,
            req.temperature,
            req.stop.unwrap_or_default(),
        )
        .await
        .map_err(|resp| {
            if let super::protocol::Response::Error { message, .. } = resp {
                ApiError::new(message)
            } else {
                ApiError::new("Failed to start vision streaming")
            }
        })?;

    // Convert mpsc receiver to SSE stream
    let stream = ReceiverStream::new(rx);
    let request_id_clone = request_id.clone();
    let model_clone = model_alias.clone();

    let sse_stream = stream
        .map(move |chunk| {
            let sse_chunk = ChatCompletionChunk {
                id: request_id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_clone.clone(),
                choices: vec![ChatChoiceDelta {
                    index: chunk.index,
                    delta: DeltaContent {
                        role: if chunk.index == 0 {
                            Some("assistant".to_string())
                        } else {
                            None
                        },
                        content: Some(chunk.delta),
                    },
                    finish_reason: None,
                }],
            };

            Event::default().data(serde_json::to_string(&sse_chunk).unwrap_or_default())
        })
        .chain(stream::once(async move {
            // Send final chunk with finish_reason
            let final_chunk = ChatCompletionChunk {
                id: request_id,
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_alias,
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: DeltaContent {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            Event::default().data(serde_json::to_string(&final_chunk).unwrap_or_default())
        }))
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);

    Ok(Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response())
}

/// POST /v1/completions
async fn completions(
    State(daemon): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    let request = super::protocol::Request::Completion {
        model: req.model,
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: req.stream,
    };

    match daemon.handle_request(request).await {
        super::protocol::Response::Completion(resp) => Ok(Json(CompletionResponse {
            id: resp.id,
            object: resp.object,
            created: resp.created,
            model: resp.model,
            choices: resp
                .choices
                .into_iter()
                .map(|c| CompletionChoice {
                    index: c.index,
                    text: c.text,
                    finish_reason: c.finish_reason,
                })
                .collect(),
            usage: resp.usage,
        })),
        super::protocol::Response::Error { code, message, .. } => {
            Err(ApiError::new(format!("{:?}: {}", code, message)))
        }
        _ => Err(ApiError::new("Unexpected response")),
    }
}

/// GET /v1/models
async fn list_models(State(daemon): State<AppState>) -> Json<ModelsResponse> {
    let models = daemon.models.list().await;

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models
            .into_iter()
            .map(|(alias, info, _, _)| ModelObject {
                id: alias,
                object: "model".to_string(),
                created: 0, // TODO: track creation time
                owned_by: "local".to_string(),
            })
            .collect(),
    })
}

/// GET /v1/models/:model
async fn get_model(
    State(daemon): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<ModelObject>, ApiError> {
    match daemon.models.get(Some(&model_id)).await {
        Ok(model) => Ok(Json(ModelObject {
            id: model.alias.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "local".to_string(),
        })),
        Err(_) => Err(ApiError::not_found(&model_id)),
    }
}

/// POST /v1/embeddings
async fn embeddings(
    State(daemon): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    match daemon.handle_embeddings(req.model, req.input).await {
        ProtoResponse::Embeddings(resp) => {
            // Convert from protocol types to openai types
            let data = resp
                .data
                .into_iter()
                .map(|d| EmbeddingObject {
                    object: d.object,
                    embedding: d.embedding,
                    index: d.index,
                })
                .collect();
            Ok(Json(EmbeddingsResponse {
                object: resp.object,
                data,
                model: resp.model,
                usage: resp.usage,
            }))
        }
        ProtoResponse::Error { message, .. } => Err(ApiError::new(&message)),
        _ => Err(ApiError::new("Unexpected response from embeddings handler")),
    }
}

/// GET /health
async fn health() -> &'static str {
    "ok"
}

/// GET /status
async fn status(State(daemon): State<AppState>) -> Json<serde_json::Value> {
    let models = daemon.models.list().await;
    let default = daemon.models.default_alias().await;

    Json(serde_json::json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": daemon.start_time.elapsed().as_secs(),
        "models_loaded": models.len(),
        "default_model": default,
        "models": models.iter().map(|(alias, info, is_default, active)| {
            serde_json::json!({
                "alias": alias,
                "parameters": info.parameters,
                "context_size": info.context_size,
                "is_default": is_default,
                "active_requests": active,
            })
        }).collect::<Vec<_>>(),
        "stats": {
            "total_requests": daemon.total_requests.load(std::sync::atomic::Ordering::Relaxed),
            "tokens_generated": daemon.models.total_tokens(),
            "gpu_available": crate::supports_gpu_offload(),
        }
    }))
}

// ==================== Error Handling ====================

pub struct ApiError {
    message: String,
    status: StatusCode,
}

impl ApiError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn not_found(model: &str) -> Self {
        Self {
            message: format!("Model '{}' not found", model),
            status: StatusCode::NOT_FOUND,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message: self.message,
                error_type: "api_error".to_string(),
                code: None,
            },
        });
        (self.status, body).into_response()
    }
}
