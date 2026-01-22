//! OpenAI-compatible HTTP API
//!
//! Provides REST endpoints compatible with the OpenAI API specification.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{delete, get, post},
    Router,
};
use futures::stream::{self};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;
use tower_http::cors::{Any, CorsLayer};

use super::anthropic::messages_handler;
use super::protocol::{ChatMessage, EmbeddingInput, Response as ProtoResponse, ResponseFormat, Usage};
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
        // Anthropic API endpoint
        .route("/v1/messages", post(messages_handler))
        // Model management API
        .route("/api/models", get(api_list_models))
        .route("/api/models/pull", post(api_pull_model))
        .route("/api/models/load", post(api_load_model))
        .route("/api/models/:name/unload", post(api_unload_model))
        .route("/api/models/:name", delete(api_delete_model))
        .route("/api/models/:name", get(api_get_model))
        // System API
        .route("/api/system/status", get(api_system_status))
        // Default models API
        .route("/api/defaults", get(api_list_defaults))
        .route("/api/defaults/:name/use", post(api_use_default))
        // Health and status
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/metrics", get(metrics))
        // Embedded Web UI
        .route("/ui", get(ui_redirect))
        .route("/ui/", get(serve_ui_handler))
        .route("/ui/*path", get(serve_ui_handler))
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
    /// Response format for structured outputs (JSON Schema validation)
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
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
        response_format: req.response_format,
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
    let models = daemon.models.list();

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
    let models = daemon.models.list();
    let default = daemon.models.default_alias();

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

/// Prometheus-compatible metrics endpoint
async fn metrics(State(daemon): State<AppState>) -> impl IntoResponse {
    let models = daemon.models.list();
    let uptime = daemon.start_time.elapsed().as_secs();
    let total_requests = daemon.total_requests.load(std::sync::atomic::Ordering::Relaxed);
    let active_requests = daemon.active_requests.load(std::sync::atomic::Ordering::Relaxed);
    let tokens_generated = daemon.models.total_tokens();

    let mut output = String::new();

    // Help text and type definitions
    output.push_str("# HELP mullama_info Mullama daemon information\n");
    output.push_str("# TYPE mullama_info gauge\n");
    output.push_str(&format!(
        "mullama_info{{version=\"{}\"}} 1\n",
        env!("CARGO_PKG_VERSION")
    ));

    output.push_str("\n# HELP mullama_uptime_seconds Daemon uptime in seconds\n");
    output.push_str("# TYPE mullama_uptime_seconds counter\n");
    output.push_str(&format!("mullama_uptime_seconds {}\n", uptime));

    output.push_str("\n# HELP mullama_models_loaded Number of loaded models\n");
    output.push_str("# TYPE mullama_models_loaded gauge\n");
    output.push_str(&format!("mullama_models_loaded {}\n", models.len()));

    output.push_str("\n# HELP mullama_requests_total Total number of requests processed\n");
    output.push_str("# TYPE mullama_requests_total counter\n");
    output.push_str(&format!("mullama_requests_total {}\n", total_requests));

    output.push_str("\n# HELP mullama_requests_active Number of active requests\n");
    output.push_str("# TYPE mullama_requests_active gauge\n");
    output.push_str(&format!("mullama_requests_active {}\n", active_requests));

    output.push_str("\n# HELP mullama_tokens_generated_total Total tokens generated\n");
    output.push_str("# TYPE mullama_tokens_generated_total counter\n");
    output.push_str(&format!("mullama_tokens_generated_total {}\n", tokens_generated));

    output.push_str("\n# HELP mullama_gpu_available Whether GPU offload is available\n");
    output.push_str("# TYPE mullama_gpu_available gauge\n");
    output.push_str(&format!(
        "mullama_gpu_available {}\n",
        if crate::supports_gpu_offload() { 1 } else { 0 }
    ));

    // Per-model metrics
    if !models.is_empty() {
        output.push_str("\n# HELP mullama_model_parameters Model parameter count\n");
        output.push_str("# TYPE mullama_model_parameters gauge\n");
        for (alias, info, _, _) in &models {
            output.push_str(&format!(
                "mullama_model_parameters{{model=\"{}\"}} {}\n",
                alias, info.parameters
            ));
        }

        output.push_str("\n# HELP mullama_model_context_size Model context size\n");
        output.push_str("# TYPE mullama_model_context_size gauge\n");
        for (alias, info, _, _) in &models {
            output.push_str(&format!(
                "mullama_model_context_size{{model=\"{}\"}} {}\n",
                alias, info.context_size
            ));
        }

        output.push_str("\n# HELP mullama_model_gpu_layers Model GPU layers\n");
        output.push_str("# TYPE mullama_model_gpu_layers gauge\n");
        for (alias, info, _, _) in &models {
            output.push_str(&format!(
                "mullama_model_gpu_layers{{model=\"{}\"}} {}\n",
                alias, info.gpu_layers
            ));
        }

        output.push_str("\n# HELP mullama_model_active_requests Active requests per model\n");
        output.push_str("# TYPE mullama_model_active_requests gauge\n");
        for (alias, _, _, active) in &models {
            output.push_str(&format!(
                "mullama_model_active_requests{{model=\"{}\"}} {}\n",
                alias, active
            ));
        }
    }

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
}

// ==================== Model Management API ====================

/// Request to pull a model
#[derive(Debug, Deserialize)]
pub struct PullModelRequest {
    /// Model name or HuggingFace spec
    pub name: String,
}

/// Request to load a model into the daemon
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    /// Model name, path, or alias
    pub name: String,
    /// GPU layers to offload (optional, uses daemon default if not specified)
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    /// Context size (optional, uses daemon default if not specified)
    #[serde(default)]
    pub context_size: Option<u32>,
}

/// Response for model operations
#[derive(Debug, Serialize)]
pub struct ModelOperationResponse {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<serde_json::Value>,
}

/// Detailed model information
#[derive(Debug, Serialize)]
pub struct ModelDetails {
    pub name: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub size: u64,
    pub size_formatted: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downloaded: Option<String>,
}

/// List all models (cached + running)
async fn api_list_models(State(daemon): State<AppState>) -> Json<serde_json::Value> {
    use super::hf::HfDownloader;
    use super::registry::registry;

    let mut models = Vec::new();

    // Get cached HuggingFace models
    if let Ok(downloader) = HfDownloader::new() {
        for cached in downloader.list_cached() {
            let short_name = format!(
                "{}:{}",
                cached.repo_id.split('/').last().unwrap_or(&cached.repo_id),
                cached.filename.trim_end_matches(".gguf")
            );

            models.push(serde_json::json!({
                "name": short_name,
                "source": "huggingface",
                "repo_id": cached.repo_id,
                "filename": cached.filename,
                "size": cached.size_bytes,
                "size_formatted": format_size(cached.size_bytes),
                "path": cached.local_path.display().to_string(),
                "downloaded": cached.downloaded_at,
                "loaded": false,
            }));
        }
    }

    // Get loaded models
    let loaded = daemon.models.list();
    for (alias, info, is_default, active_requests) in loaded {
        // Check if already in list
        let already_listed = models.iter().any(|m| {
            m.get("path")
                .and_then(|p| p.as_str())
                .map(|p| p == info.path)
                .unwrap_or(false)
        });

        if already_listed {
            // Update the existing entry to mark it as loaded
            for model in &mut models {
                if model.get("path").and_then(|p| p.as_str()) == Some(info.path.as_str()) {
                    model["loaded"] = serde_json::json!(true);
                    model["is_default"] = serde_json::json!(is_default);
                    model["active_requests"] = serde_json::json!(active_requests);
                    model["context_size"] = serde_json::json!(info.context_size);
                    model["gpu_layers"] = serde_json::json!(info.gpu_layers);
                }
            }
        } else {
            models.push(serde_json::json!({
                "name": alias,
                "source": "local",
                "size": 0,
                "size_formatted": "unknown",
                "path": info.path,
                "loaded": true,
                "is_default": is_default,
                "active_requests": active_requests,
                "context_size": info.context_size,
                "gpu_layers": info.gpu_layers,
            }));
        }
    }

    // Get available aliases
    let reg = registry();
    let aliases: Vec<_> = reg.list_aliases().iter().map(|a| a.to_string()).collect();

    Json(serde_json::json!({
        "models": models,
        "available_aliases": aliases,
        "total_cached": models.len(),
    }))
}

/// Format bytes as human-readable size
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Pull a model from HuggingFace
async fn api_pull_model(
    State(_daemon): State<AppState>,
    Json(request): Json<PullModelRequest>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    use super::hf::{HfDownloader, HfModelSpec};
    use super::registry::{resolve_model_name, ResolvedModel};

    // Resolve the model name
    let resolved = resolve_model_name(&request.name);

    let hf_spec = match resolved {
        ResolvedModel::HuggingFace { spec, .. } => spec,
        ResolvedModel::LocalPath(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ModelOperationResponse {
                    success: false,
                    message: "Cannot pull a local path".to_string(),
                    model: None,
                }),
            ));
        }
        ResolvedModel::Unknown(name) => {
            // Try as direct HF spec
            if name.starts_with("hf:") || name.contains('/') {
                if name.starts_with("hf:") {
                    name
                } else {
                    format!("hf:{}", name)
                }
            } else {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!("Unknown model '{}'. Use hf:owner/repo format or a known alias.", name),
                        model: None,
                    }),
                ));
            }
        }
    };

    // Parse and download
    let spec = HfModelSpec::parse(&hf_spec).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Invalid HuggingFace spec: {}", hf_spec),
                model: None,
            }),
        )
    })?;

    let downloader = HfDownloader::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to initialize downloader: {}", e),
                model: None,
            }),
        )
    })?;

    // Download without progress (for API)
    let path = downloader.download_spec(&spec, false).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Download failed: {}", e),
                model: None,
            }),
        )
    })?;

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    Ok(Json(ModelOperationResponse {
        success: true,
        message: format!("Model '{}' downloaded successfully", request.name),
        model: Some(serde_json::json!({
            "name": spec.get_alias(),
            "source": "huggingface",
            "repo_id": spec.repo_id,
            "filename": spec.filename,
            "size": size,
            "size_formatted": format_size(size),
            "path": path.display().to_string(),
            "downloaded": chrono::Utc::now().to_rfc3339(),
        })),
    }))
}

/// Delete a model
async fn api_delete_model(
    State(_daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    use super::hf::HfDownloader;

    let downloader = HfDownloader::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to initialize: {}", e),
                model: None,
            }),
        )
    })?;

    let cached = downloader.list_cached();

    // Find the model
    let mut found = None;
    for model in &cached {
        let short_name = format!(
            "{}:{}",
            model.repo_id.split('/').last().unwrap_or(&model.repo_id),
            model.filename.trim_end_matches(".gguf")
        );

        if model.filename == name
            || model.repo_id == name
            || short_name == name
            || model.filename.trim_end_matches(".gguf") == name
        {
            found = Some(model);
            break;
        }
    }

    let model = found.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Model '{}' not found", name),
                model: None,
            }),
        )
    })?;

    let size = model.size_bytes;
    let path = model.local_path.display().to_string();

    // Delete the file
    std::fs::remove_file(&model.local_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to delete: {}", e),
                model: None,
            }),
        )
    })?;

    Ok(Json(ModelOperationResponse {
        success: true,
        message: format!("Model '{}' deleted, freed {}", name, format_size(size)),
        model: Some(serde_json::json!({
            "name": name,
            "source": "huggingface",
            "repo_id": model.repo_id,
            "filename": model.filename,
            "size": size,
            "size_formatted": format_size(size),
            "path": path,
        })),
    }))
}

/// Get model details
async fn api_get_model(
    State(daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ModelOperationResponse>)> {
    use super::hf::HfDownloader;

    // Check if model is loaded
    let loaded = daemon.models.list();
    for (alias, info, is_default, active_requests) in &loaded {
        if alias == &name {
            return Ok(Json(serde_json::json!({
                "name": alias,
                "source": "loaded",
                "path": info.path,
                "parameters": info.parameters,
                "context_size": info.context_size,
                "gpu_layers": info.gpu_layers,
                "is_default": is_default,
                "active_requests": active_requests,
                "loaded": true,
            })));
        }
    }

    // Check cached models
    if let Ok(downloader) = HfDownloader::new() {
        for model in downloader.list_cached() {
            let short_name = format!(
                "{}:{}",
                model.repo_id.split('/').last().unwrap_or(&model.repo_id),
                model.filename.trim_end_matches(".gguf")
            );

            if model.filename == name
                || model.repo_id == name
                || short_name == name
                || model.filename.trim_end_matches(".gguf") == name
            {
                return Ok(Json(serde_json::json!({
                    "name": short_name,
                    "source": "huggingface",
                    "repo_id": model.repo_id,
                    "filename": model.filename,
                    "size": model.size_bytes,
                    "size_formatted": format_size(model.size_bytes),
                    "path": model.local_path.display().to_string(),
                    "downloaded": model.downloaded_at,
                    "loaded": false,
                })));
            }
        }
    }

    Err((
        StatusCode::NOT_FOUND,
        Json(ModelOperationResponse {
            success: false,
            message: format!("Model '{}' not found", name),
            model: None,
        }),
    ))
}

/// Load a model into the daemon
async fn api_load_model(
    State(daemon): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    use super::hf::HfDownloader;
    use super::registry::{resolve_model_name, ResolvedModel};

    // Resolve the model name to find the actual path
    let resolved = resolve_model_name(&request.name);

    let (path, alias): (String, String) = match resolved {
        ResolvedModel::HuggingFace { spec, .. } => {
            // Check if it's already downloaded
            let downloader = HfDownloader::new().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!("Failed to initialize downloader: {}", e),
                        model: None,
                    }),
                )
            })?;

            // Parse the spec to get repo info
            if let Some(hf_spec) = super::hf::HfModelSpec::parse(&spec) {
                // Look for cached model
                let cached = downloader.list_cached();
                let found = cached.iter().find(|m| {
                    m.repo_id == hf_spec.repo_id
                        && (hf_spec.filename.is_none() || Some(&m.filename) == hf_spec.filename.as_ref())
                });

                if let Some(model) = found {
                    // Use the alias from the spec or the request name
                    let model_alias = hf_spec.alias.unwrap_or_else(|| request.name.clone());
                    (model.local_path.display().to_string(), model_alias)
                } else {
                    return Err((
                        StatusCode::NOT_FOUND,
                        Json(ModelOperationResponse {
                            success: false,
                            message: format!("Model '{}' not downloaded. Pull it first.", request.name),
                            model: None,
                        }),
                    ));
                }
            } else {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!("Invalid model spec: {}", spec),
                        model: None,
                    }),
                ));
            }
        }
        ResolvedModel::LocalPath(path) => {
            (path.display().to_string(), request.name.clone())
        }
        ResolvedModel::Unknown(name) => {
            // Try to find in cached models
            let downloader = HfDownloader::new().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ModelOperationResponse {
                        success: false,
                        message: format!("Failed to initialize: {}", e),
                        model: None,
                    }),
                )
            })?;

            let cached = downloader.list_cached();
            let found = cached.iter().find(|m| {
                let short_name = format!(
                    "{}:{}",
                    m.repo_id.split('/').last().unwrap_or(&m.repo_id),
                    m.filename.trim_end_matches(".gguf")
                );
                m.filename == name
                    || m.repo_id == name
                    || short_name == name
                    || m.filename.trim_end_matches(".gguf") == name
            });

            if let Some(model) = found {
                let short_name = format!(
                    "{}:{}",
                    model.repo_id.split('/').last().unwrap_or(&model.repo_id),
                    model.filename.trim_end_matches(".gguf")
                );
                (model.local_path.display().to_string(), short_name)
            } else {
                // Check if it's a direct path
                if std::path::Path::new(&name).exists() {
                    (name.clone(), name)
                } else {
                    return Err((
                        StatusCode::NOT_FOUND,
                        Json(ModelOperationResponse {
                            success: false,
                            message: format!("Model '{}' not found. Pull it first or provide a valid path.", name),
                            model: None,
                        }),
                    ));
                }
            }
        }
    };

    // Load the model
    let gpu_layers = request.gpu_layers.unwrap_or(daemon.config.default_gpu_layers);
    let context_size = request.context_size.unwrap_or(daemon.config.default_context_size);

    let config = super::models::ModelLoadConfig {
        alias: alias.clone(),
        path: path.clone(),
        gpu_layers,
        context_size,
        threads: num_cpus::get() as i32,
        mmproj_path: None,
    };

    match daemon.models.load(config).await {
        Ok(info) => Ok(Json(ModelOperationResponse {
            success: true,
            message: format!("Model '{}' loaded successfully", alias),
            model: Some(serde_json::json!({
                "alias": alias,
                "path": path,
                "parameters": info.parameters,
                "context_size": info.context_size,
                "gpu_layers": info.gpu_layers,
            })),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ModelOperationResponse {
                success: false,
                message: format!("Failed to load model: {}", e),
                model: None,
            }),
        )),
    }
}

/// Unload a model from the daemon
async fn api_unload_model(
    State(daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ModelOperationResponse>, (StatusCode, Json<ModelOperationResponse>)> {
    match daemon.models.unload(&name).await {
        Ok(_) => Ok(Json(ModelOperationResponse {
            success: true,
            message: format!("Model '{}' unloaded successfully", name),
            model: None,
        })),
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            Err((
                status,
                Json(ModelOperationResponse {
                    success: false,
                    message: format!("Failed to unload model: {}", e),
                    model: None,
                }),
            ))
        }
    }
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

// ==================== System API ====================

/// System status response
#[derive(Debug, Serialize)]
pub struct SystemStatus {
    pub version: String,
    pub uptime_secs: u64,
    pub models_loaded: usize,
    pub http_endpoint: Option<String>,
}

/// Get system status
async fn api_system_status(State(daemon): State<AppState>) -> Json<SystemStatus> {
    let models = daemon.models.list();
    let uptime = daemon.start_time.elapsed().as_secs();

    let http_endpoint = daemon.config.http_port.map(|port| {
        format!("http://{}:{}", daemon.config.http_addr, port)
    });

    Json(SystemStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
        models_loaded: models.len(),
        http_endpoint,
    })
}

// ==================== Default Models API ====================

/// Response for listing default models
#[derive(Debug, Serialize)]
struct DefaultsResponse {
    models: Vec<super::defaults::DefaultModelInfo>,
}

/// Response for using a default model
#[derive(Debug, Serialize)]
struct UseDefaultResponse {
    success: bool,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<serde_json::Value>,
}

/// List all available default model templates
async fn api_list_defaults() -> Json<DefaultsResponse> {
    let infos = super::defaults::list_default_infos();
    Json(DefaultsResponse { models: infos })
}

/// Use a default model (download if needed and load)
async fn api_use_default(
    State(daemon): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<UseDefaultResponse>, (StatusCode, Json<UseDefaultResponse>)> {
    use super::defaults::get_default;
    use super::hf::{HfDownloader, HfModelSpec};

    // Get the default model
    let default = get_default(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Default model '{}' not found", name),
                model: None,
            }),
        )
    })?;

    // Parse the FROM directive to get the HuggingFace spec
    let from = &default.modelfile.from;

    // Check if it's a HuggingFace model
    if !from.starts_with("hf:") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Default model '{}' is not a HuggingFace model", name),
                model: None,
            }),
        ));
    }

    let spec = HfModelSpec::parse(from).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Invalid HuggingFace spec in modelfile: {}", from),
                model: None,
            }),
        )
    })?;

    // Initialize downloader
    let downloader = HfDownloader::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Failed to initialize downloader: {}", e),
                model: None,
            }),
        )
    })?;

    // Download the model (will use cache if already downloaded)
    let model_path = downloader.download_spec(&spec, false).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Failed to download model: {}", e),
                model: None,
            }),
        )
    })?;

    // Extract parameters from the modelfile
    let context_size = default
        .modelfile
        .parameters
        .get("num_ctx")
        .and_then(|v| v.as_i64())
        .unwrap_or(4096) as u32;

    let gpu_layers = default.modelfile.gpu_layers.unwrap_or(0);

    // Get mmproj path for vision models
    let mmproj_path = default.modelfile.vision_projector.as_ref().map(|p| {
        // If it's a relative path, resolve it relative to the model directory
        if p.is_relative() {
            model_path.parent()
                .map(|parent| parent.join(p).display().to_string())
                .unwrap_or_else(|| p.display().to_string())
        } else {
            p.display().to_string()
        }
    });

    // Load the model
    let load_config = super::models::ModelLoadConfig {
        path: model_path.display().to_string(),
        alias: name.clone(),
        context_size,
        gpu_layers,
        threads: num_cpus::get() as i32,
        mmproj_path,
    };

    daemon.models.load(load_config).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(UseDefaultResponse {
                success: false,
                message: format!("Failed to load model: {}", e),
                model: None,
            }),
        )
    })?;

    // Set as default model
    daemon.models.set_default(&name);

    // Get the model info
    let model_info = daemon.models.get(Some(name.as_str())).await.ok().map(|_m| {
        serde_json::json!({
            "alias": name,
            "path": model_path.display().to_string(),
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "description": default.info.description,
            "has_thinking": default.info.has_thinking,
            "has_vision": default.info.has_vision,
        })
    });

    Ok(Json(UseDefaultResponse {
        success: true,
        message: format!("Model '{}' is now ready to use", name),
        model: model_info,
    }))
}

// ==================== Embedded Web UI ====================

use axum::http::{header, Uri};

/// Redirect /ui to /ui/
async fn ui_redirect() -> impl IntoResponse {
    (
        StatusCode::TEMPORARY_REDIRECT,
        [(header::LOCATION, "/ui/")],
        "",
    )
}

/// Serve embedded UI assets
async fn serve_ui_handler(uri: Uri) -> impl IntoResponse {
    super::ui::serve_ui(uri).await
}
