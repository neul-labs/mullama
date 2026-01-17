//! Daemon server implementation
//!
//! Core daemon that manages models and handles requests from IPC and HTTP.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, RwLock};

use super::models::{ModelLoadConfig, ModelManager, RequestGuard};
use super::protocol::*;
use crate::embedding::{EmbeddingConfig, EmbeddingGenerator};
use crate::{MullamaError, SamplerParams};

/// Daemon server configuration
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// IPC socket address
    pub ipc_addr: String,
    /// HTTP port (None to disable)
    pub http_port: Option<u16>,
    /// HTTP bind address
    pub http_addr: String,
    /// Default context size for new models
    pub default_context_size: u32,
    /// Default GPU layers
    pub default_gpu_layers: i32,
    /// Number of threads per model
    pub threads_per_model: i32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            ipc_addr: super::DEFAULT_SOCKET.to_string(),
            http_port: Some(super::DEFAULT_HTTP_PORT),
            http_addr: "0.0.0.0".to_string(),
            default_context_size: 4096,
            default_gpu_layers: 0,
            threads_per_model: (num_cpus::get() / 2).max(1) as i32,
        }
    }
}

/// The daemon server
pub struct Daemon {
    pub config: DaemonConfig,
    pub models: Arc<ModelManager>,
    pub start_time: Instant,
    pub shutdown: Arc<AtomicBool>,
    pub active_requests: AtomicU32,
    pub total_requests: AtomicU64,
}

impl Daemon {
    /// Create a new daemon
    pub fn new(config: DaemonConfig) -> Self {
        Self {
            config,
            models: Arc::new(ModelManager::new()),
            start_time: Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            active_requests: AtomicU32::new(0),
            total_requests: AtomicU64::new(0),
        }
    }

    /// Handle a request
    pub async fn handle_request(&self, request: Request) -> Response {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match request {
            Request::Ping => Response::Pong {
                uptime_secs: self.start_time.elapsed().as_secs(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },

            Request::Status => self.handle_status().await,
            Request::ListModels => self.handle_list_models().await,

            Request::LoadModel {
                alias,
                path,
                gpu_layers,
                context_size,
            } => {
                self.handle_load_model(alias, path, gpu_layers, context_size)
                    .await
            }

            Request::UnloadModel { alias } => self.handle_unload_model(&alias).await,
            Request::SetDefaultModel { alias } => self.handle_set_default(&alias).await,

            Request::ChatCompletion {
                model,
                messages,
                max_tokens,
                temperature,
                stream,
                stop,
                response_format: _,
                tools: _,
                tool_choice: _,
                thinking: _,
            } => {
                self.handle_chat_completion(model, messages, max_tokens, temperature, stream, stop)
                    .await
            }

            Request::Completion {
                model,
                prompt,
                max_tokens,
                temperature,
                stream,
            } => {
                self.handle_completion(model, prompt, max_tokens, temperature, stream)
                    .await
            }

            Request::Embeddings { model, input } => self.handle_embeddings(model, input).await,

            Request::Tokenize { model, text } => self.handle_tokenize(model, &text).await,

            Request::Cancel { request_id } => {
                // TODO: Implement cancellation
                Response::Cancelled { request_id }
            }

            Request::Shutdown => {
                self.shutdown.store(true, Ordering::SeqCst);
                Response::ShuttingDown
            }
        }
    }

    async fn handle_status(&self) -> Response {
        let default_model = self.models.default_alias().await;

        Response::Status(DaemonStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
            models_loaded: self.models.count().await,
            default_model,
            http_endpoint: self
                .config
                .http_port
                .map(|p| format!("http://{}:{}", self.config.http_addr, p)),
            ipc_endpoint: self.config.ipc_addr.clone(),
            stats: DaemonStats {
                requests_total: self.total_requests.load(Ordering::Relaxed),
                tokens_generated: self.models.total_tokens(),
                active_requests: self.active_requests.load(Ordering::Relaxed),
                memory_used_mb: 0, // TODO
                gpu_available: crate::supports_gpu_offload(),
            },
        })
    }

    async fn handle_list_models(&self) -> Response {
        let models = self.models.list().await;
        Response::Models(
            models
                .into_iter()
                .map(|(alias, info, is_default, active)| ModelStatus {
                    alias,
                    info,
                    is_default,
                    active_requests: active,
                })
                .collect(),
        )
    }

    async fn handle_load_model(
        &self,
        alias: String,
        path: String,
        gpu_layers: i32,
        context_size: u32,
    ) -> Response {
        let config = ModelLoadConfig::new(&alias, &path)
            .gpu_layers(if gpu_layers == 0 {
                self.config.default_gpu_layers
            } else {
                gpu_layers
            })
            .context_size(if context_size == 0 {
                self.config.default_context_size
            } else {
                context_size
            })
            .threads(self.config.threads_per_model);

        match self.models.load(config).await {
            Ok(info) => Response::ModelLoaded { alias, info },
            Err(e) => Response::error(ErrorCode::ModelLoadFailed, e.to_string()),
        }
    }

    async fn handle_unload_model(&self, alias: &str) -> Response {
        match self.models.unload(alias).await {
            Ok(()) => Response::ModelUnloaded {
                alias: alias.to_string(),
            },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }

    async fn handle_set_default(&self, alias: &str) -> Response {
        match self.models.set_default(alias).await {
            Ok(()) => Response::DefaultModelSet {
                alias: alias.to_string(),
            },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }

    /// Handle streaming chat completion - returns receiver for SSE
    pub async fn handle_chat_completion_streaming(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        stop: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Build prompt from messages using model's chat template
        let prompt = self.build_chat_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();

        // Get stop sequences from chat template and merge with user-provided ones
        let mut all_stops = loaded.model.get_chat_stop_sequences();
        all_stops.extend(stop);

        // Start streaming generation
        match self
            .generate_text_streaming(loaded, prompt, max_tokens, temperature, all_stops)
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }

    async fn handle_chat_completion(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        _stream: bool,
        stop: Vec<String>,
    ) -> Response {
        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Build prompt from messages using model's chat template
        let prompt = self.build_chat_prompt(&loaded.model, &messages);

        // Get stop sequences from chat template and merge with user-provided ones
        let mut all_stops = loaded.model.get_chat_stop_sequences();
        all_stops.extend(stop);

        // Generate
        let result = self
            .generate_text(&loaded, &prompt, max_tokens, temperature, &all_stops)
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Response::ChatCompletion(ChatCompletionResponse {
                    id: generate_completion_id(),
                    object: "chat.completion".to_string(),
                    created,
                    model: loaded.alias.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text.into(),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                    thinking: None,
                })
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    async fn handle_completion(
        &self,
        model: Option<String>,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        _stream: bool,
    ) -> Response {
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let result = self
            .generate_text(&loaded, &prompt, max_tokens, temperature, &[])
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Response::Completion(CompletionResponse {
                    id: generate_completion_id(),
                    object: "text_completion".to_string(),
                    created,
                    model: loaded.alias.clone(),
                    choices: vec![CompletionChoice {
                        index: 0,
                        text,
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                })
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    /// Handle vision chat completion (images + text)
    #[cfg(feature = "multimodal")]
    pub async fn handle_vision_chat_completion(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        stop: Vec<String>,
    ) -> Response {
        use crate::{Bitmap, MtmdContext};
        use base64::Engine;

        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        // Check if model has multimodal support
        if !loaded.has_multimodal() {
            return Response::error(
                ErrorCode::InvalidRequest,
                "Model does not have multimodal support. Load with --mmproj to enable vision.",
            );
        }

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Extract images from messages and decode base64
        let mut bitmaps: Vec<Bitmap> = Vec::new();
        let mtmd_guard = loaded.mtmd_context.as_ref().unwrap().read().await;

        for msg in &messages {
            for img_url in msg.content.images() {
                // Parse data URI: data:image/jpeg;base64,/9j/4AAQ...
                let url = &img_url.url;
                if let Some(base64_data) = url.strip_prefix("data:").and_then(|s| {
                    // Find the base64 part after the comma
                    s.split_once(',').map(|(_, data)| data)
                }) {
                    // Decode base64
                    match base64::engine::general_purpose::STANDARD.decode(base64_data) {
                        Ok(image_bytes) => {
                            // Create bitmap from decoded image data
                            match mtmd_guard.bitmap_from_buffer(&image_bytes) {
                                Ok(bitmap) => bitmaps.push(bitmap),
                                Err(e) => {
                                    self.active_requests.fetch_sub(1, Ordering::Relaxed);
                                    return Response::error(
                                        ErrorCode::InvalidRequest,
                                        format!("Failed to load image: {}", e),
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            self.active_requests.fetch_sub(1, Ordering::Relaxed);
                            return Response::error(
                                ErrorCode::InvalidRequest,
                                format!("Invalid base64 image data: {}", e),
                            );
                        }
                    }
                } else {
                    self.active_requests.fetch_sub(1, Ordering::Relaxed);
                    return Response::error(
                        ErrorCode::InvalidRequest,
                        "Image URL must be a base64 data URI (data:image/...;base64,...)",
                    );
                }
            }
        }

        drop(mtmd_guard);

        // Build prompt with <__media__> markers for images
        // For VLMs, we need to place image markers where images should be processed
        let prompt = self.build_vision_prompt(&loaded.model, &messages);

        // Get stop sequences
        let mut all_stops = loaded.model.get_chat_stop_sequences();
        all_stops.extend(stop);

        // Process with multimodal context
        let result = self
            .generate_vision_text(
                &loaded,
                &prompt,
                &bitmaps,
                max_tokens,
                temperature,
                &all_stops,
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Response::ChatCompletion(ChatCompletionResponse {
                    id: generate_completion_id(),
                    object: "chat.completion".to_string(),
                    created,
                    model: loaded.alias.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text.into(),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                    thinking: None,
                })
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    /// Build prompt for vision models with image markers
    #[cfg(feature = "multimodal")]
    fn build_vision_prompt(&self, model: &crate::Model, messages: &[ChatMessage]) -> String {
        // Extract text content, replacing images with <__media__> markers
        let mut processed_messages: Vec<(String, String)> = Vec::new();

        for msg in messages {
            let mut content = String::new();
            match &msg.content {
                MessageContent::Text(s) => content = s.clone(),
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => content.push_str(text),
                            ContentPart::ImageUrl { .. } => {
                                // Insert media marker where image should go
                                content.push_str("<__media__>");
                            }
                        }
                    }
                }
            }
            processed_messages.push((msg.role.clone(), content));
        }

        // Try to use the model's built-in chat template
        let msg_tuples: Vec<(&str, &str)> = processed_messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect();

        match model.apply_chat_template(None, &msg_tuples, true) {
            Ok(formatted) => formatted,
            Err(_) => {
                // Fallback to simple format
                let mut prompt = String::new();
                for (role, content) in &processed_messages {
                    match role.as_str() {
                        "system" => prompt.push_str(&format!("System: {}\n\n", content)),
                        "user" => prompt.push_str(&format!("User: {}\n\n", content)),
                        "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", content)),
                        _ => prompt.push_str(&format!("{}: {}\n\n", role, content)),
                    }
                }
                prompt.push_str("Assistant:");
                prompt
            }
        }
    }

    /// Generate text with vision input
    #[cfg(feature = "multimodal")]
    async fn generate_vision_text(
        &self,
        loaded: &super::models::LoadedModel,
        prompt: &str,
        bitmaps: &[crate::Bitmap],
        max_tokens: u32,
        temperature: f32,
        stop_sequences: &[String],
    ) -> Result<(String, u32, u32), MullamaError> {
        // Get locks on context and mtmd_context
        let mut ctx_guard = loaded.context.write().await;
        let mut mtmd_guard = loaded.mtmd_context.as_ref().unwrap().write().await;

        let model = loaded.model.clone();
        let stop_sequences = stop_sequences.to_vec();

        // Run CPU-bound generation in blocking context
        let (generated, prompt_tokens, completion_tokens) = tokio::task::block_in_place(|| {
            // Clear KV cache
            ctx_guard.kv_cache_clear();

            // Create bitmap references for tokenize
            let bitmap_refs: Vec<&crate::Bitmap> = bitmaps.iter().collect();

            // Tokenize the prompt with images
            let chunks = mtmd_guard.tokenize(prompt, &bitmap_refs)?;

            // Evaluate chunks (processes both text and images)
            // Use a reasonable default batch size
            let n_batch = 512;
            let n_past = mtmd_guard.eval_chunks(&mut ctx_guard, &chunks, 0, 0, n_batch, true)?;

            let prompt_tokens = n_past as u32;

            // Set up sampler
            let mut sampler_params = SamplerParams::default();
            sampler_params.temperature = temperature;
            sampler_params.top_p = 0.9;
            sampler_params.top_k = 40;
            let mut sampler = sampler_params.build_chain(model.clone())?;

            // Generate tokens
            let mut generated = String::with_capacity((max_tokens as usize) * 6);
            let mut completion_tokens = 0u32;

            for _ in 0..max_tokens {
                // Sample next token
                let next_token = sampler.sample(&mut *ctx_guard, -1);

                // Check for end of generation
                if model.vocab_is_eog(next_token) {
                    break;
                }

                // Get token text
                if let Ok(text) = model.token_to_str(next_token, 0, false) {
                    generated.push_str(&text);

                    // Check for stop sequences
                    if !stop_sequences.is_empty() {
                        for stop in &stop_sequences {
                            if let Some(pos) = generated.find(stop) {
                                generated.truncate(pos);
                                return Ok((generated, prompt_tokens, completion_tokens));
                            }
                        }
                    }
                }

                // Accept the token and evaluate
                sampler.accept(next_token);
                ctx_guard.decode_single(next_token)?;
                completion_tokens += 1;
            }

            Ok::<_, MullamaError>((generated, prompt_tokens, completion_tokens))
        })?;

        self.models.add_tokens(completion_tokens as u64);

        Ok((generated, prompt_tokens, completion_tokens))
    }

    /// Handle streaming vision chat completion
    #[cfg(feature = "multimodal")]
    pub async fn handle_vision_chat_completion_streaming(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        stop: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        use crate::Bitmap;
        use base64::Engine;

        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        // Check if model has multimodal support
        if !loaded.has_multimodal() {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                "Model does not have multimodal support. Load with --mmproj to enable vision.",
            ));
        }

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Extract and decode images
        let mut bitmaps: Vec<Bitmap> = Vec::new();
        {
            let mtmd_guard = loaded.mtmd_context.as_ref().unwrap().read().await;

            for msg in &messages {
                for img_url in msg.content.images() {
                    let url = &img_url.url;
                    if let Some(base64_data) = url
                        .strip_prefix("data:")
                        .and_then(|s| s.split_once(',').map(|(_, data)| data))
                    {
                        match base64::engine::general_purpose::STANDARD.decode(base64_data) {
                            Ok(image_bytes) => match mtmd_guard.bitmap_from_buffer(&image_bytes) {
                                Ok(bitmap) => bitmaps.push(bitmap),
                                Err(e) => {
                                    self.active_requests.fetch_sub(1, Ordering::Relaxed);
                                    return Err(Response::error(
                                        ErrorCode::InvalidRequest,
                                        format!("Failed to load image: {}", e),
                                    ));
                                }
                            },
                            Err(e) => {
                                self.active_requests.fetch_sub(1, Ordering::Relaxed);
                                return Err(Response::error(
                                    ErrorCode::InvalidRequest,
                                    format!("Invalid base64 image data: {}", e),
                                ));
                            }
                        }
                    } else {
                        self.active_requests.fetch_sub(1, Ordering::Relaxed);
                        return Err(Response::error(
                            ErrorCode::InvalidRequest,
                            "Image URL must be a base64 data URI",
                        ));
                    }
                }
            }
        }

        // Build prompt with image markers
        let prompt = self.build_vision_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();

        // Get stop sequences
        let mut all_stops = loaded.model.get_chat_stop_sequences();
        all_stops.extend(stop);

        // Start streaming generation with vision
        match self
            .generate_vision_text_streaming(
                loaded,
                prompt,
                bitmaps,
                max_tokens,
                temperature,
                all_stops,
            )
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }

    /// Generate streaming text with vision input
    #[cfg(feature = "multimodal")]
    async fn generate_vision_text_streaming(
        &self,
        loaded: std::sync::Arc<super::models::LoadedModel>,
        prompt: String,
        bitmaps: Vec<crate::Bitmap>,
        max_tokens: u32,
        temperature: f32,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        let request_id = generate_completion_id();
        let request_id_clone = request_id.clone();
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        let model = loaded.model.clone();
        let models_ref = self.models.clone();

        // Process image chunks first, then spawn streaming task
        // We need to get both locks, process images, then stream text generation

        tokio::spawn(async move {
            let mut context = loaded.context.write().await;
            let mut mtmd_context = loaded.mtmd_context.as_ref().unwrap().write().await;

            let result = tokio::task::block_in_place(|| {
                // Clear KV cache
                context.kv_cache_clear();

                // Create bitmap references
                let bitmap_refs: Vec<&crate::Bitmap> = bitmaps.iter().collect();

                // Tokenize and evaluate chunks (processes images)
                let chunks = mtmd_context.tokenize(&prompt, &bitmap_refs)?;
                let n_batch = 512;
                let _n_past =
                    mtmd_context.eval_chunks(&mut context, &chunks, 0, 0, n_batch, true)?;

                // Set up sampler
                let mut sampler_params = SamplerParams::default();
                sampler_params.temperature = temperature;
                sampler_params.top_p = 0.9;
                sampler_params.top_k = 40;
                let mut sampler = sampler_params.build_chain(model.clone())?;

                let mut generated = String::new();
                let mut index = 0u32;

                for _ in 0..max_tokens {
                    let next_token = sampler.sample(&mut *context, -1);

                    if model.vocab_is_eog(next_token) {
                        break;
                    }

                    if let Ok(text) = model.token_to_str(next_token, 0, false) {
                        generated.push_str(&text);

                        // Check stop sequences before sending chunk
                        if !stop_sequences.is_empty() {
                            for stop in &stop_sequences {
                                if let Some(pos) = generated.find(stop) {
                                    let final_text = &generated[generated.len() - text.len()..];
                                    let stop_pos_in_text = if pos >= generated.len() - text.len() {
                                        pos - (generated.len() - text.len())
                                    } else {
                                        0
                                    };

                                    if stop_pos_in_text > 0 {
                                        let partial = &final_text[..stop_pos_in_text];
                                        let chunk = StreamChunk {
                                            request_id: request_id_clone.clone(),
                                            index,
                                            delta: partial.to_string(),
                                            token_id: next_token,
                                        };
                                        let _ = tx.blocking_send(chunk);
                                    }
                                    return Ok::<_, MullamaError>(index + 1);
                                }
                            }
                        }

                        // Send chunk
                        let chunk = StreamChunk {
                            request_id: request_id_clone.clone(),
                            index,
                            delta: text,
                            token_id: next_token,
                            thinking: None,
                            tool_calls: None,
                        };

                        if tx.blocking_send(chunk).is_err() {
                            break;
                        }

                        index += 1;
                    }

                    sampler.accept(next_token);
                    context.decode_single(next_token)?;
                }

                Ok::<_, MullamaError>(index)
            });

            if let Ok(tokens) = result {
                models_ref.add_tokens(tokens as u64);
            }
        });

        // Return prompt_tokens as 0 for now since we process asynchronously
        // The actual prompt token count will be computed inside the task
        Ok((rx, 0, request_id))
    }

    pub async fn handle_embeddings(
        &self,
        model: Option<String>,
        input: EmbeddingInput,
    ) -> Response {
        // Get the model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        // Create an EmbeddingGenerator with default config (Mean pooling, normalized)
        let config = EmbeddingConfig::default();
        let mut generator = match EmbeddingGenerator::new(loaded.model.clone(), config) {
            Ok(g) => g,
            Err(e) => {
                return Response::error(
                    ErrorCode::Internal,
                    format!("Failed to create embedding generator: {}", e),
                )
            }
        };

        // Collect texts and generate embeddings
        let texts: Vec<String> = match &input {
            EmbeddingInput::Single(text) => vec![text.clone()],
            EmbeddingInput::Multiple(texts) => texts.clone(),
        };

        // Count tokens for usage stats
        let mut total_tokens = 0usize;
        for text in &texts {
            if let Ok(tokens) = loaded.model.tokenize(text, true, false) {
                total_tokens += tokens.len();
            }
        }

        // Generate embeddings
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = match generator.embed_batch(&text_refs) {
            Ok(emb) => emb,
            Err(e) => {
                return Response::error(
                    ErrorCode::GenerationFailed,
                    format!("Failed to generate embeddings: {}", e),
                )
            }
        };

        // Build response
        let data: Vec<EmbeddingData> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, embedding)| EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index: i as u32,
            })
            .collect();

        Response::Embeddings(EmbeddingsResponse {
            object: "list".to_string(),
            data,
            model: loaded.alias.clone(),
            usage: Usage {
                prompt_tokens: total_tokens as u32,
                completion_tokens: 0,
                total_tokens: total_tokens as u32,
            },
        })
    }

    async fn handle_tokenize(&self, model: Option<String>, text: &str) -> Response {
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        match loaded.model.tokenize(text, false, false) {
            Ok(tokens) => {
                let count = tokens.len();
                Response::Tokens { tokens, count }
            }
            Err(e) => Response::error(ErrorCode::Internal, e.to_string()),
        }
    }

    fn build_chat_prompt(&self, model: &crate::Model, messages: &[ChatMessage]) -> String {
        // Convert ChatMessage to the format expected by apply_chat_template
        // Extract text content from each message (ignoring images for the prompt template)
        let text_contents: Vec<String> = messages.iter().map(|m| m.content.text()).collect();
        let msg_tuples: Vec<(&str, &str)> = messages
            .iter()
            .zip(text_contents.iter())
            .map(|(m, content)| (m.role.as_str(), content.as_str()))
            .collect();

        // Try to use the model's built-in chat template
        match model.apply_chat_template(None, &msg_tuples, true) {
            Ok(formatted) => formatted,
            Err(e) => {
                // Log warning about template fallback
                eprintln!(
                    "[WARN] Chat template failed: {}. Using generic format. \
                    Model may produce suboptimal output.",
                    e
                );

                // Fallback to simple format if template fails
                let mut prompt = String::new();

                for (msg, content) in messages.iter().zip(text_contents.iter()) {
                    match msg.role.as_str() {
                        "system" => {
                            prompt.push_str(&format!("System: {}\n\n", content));
                        }
                        "user" => {
                            prompt.push_str(&format!("User: {}\n\n", content));
                        }
                        "assistant" => {
                            prompt.push_str(&format!("Assistant: {}\n\n", content));
                        }
                        _ => {
                            prompt.push_str(&format!("{}: {}\n\n", msg.role, content));
                        }
                    }
                }

                prompt.push_str("Assistant:");
                prompt
            }
        }
    }

    async fn generate_text(
        &self,
        loaded: &super::models::LoadedModel,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        stop_sequences: &[String],
    ) -> Result<(String, u32, u32), MullamaError> {
        // Tokenize - respect model's BOS token setting to avoid double BOS
        let add_bos = loaded.model.add_bos_token();
        let tokens = loaded.model.tokenize(prompt, add_bos, false)?;
        let prompt_tokens = tokens.len() as u32;

        // Get context lock
        let mut context = loaded.context.write().await;

        // Run CPU-bound generation in a blocking context to not block the async runtime
        // block_in_place allows blocking while keeping the current task context
        let model = loaded.model.clone();
        let stop_sequences = stop_sequences.to_vec();

        let (generated, completion_tokens) = tokio::task::block_in_place(|| {
            // Clear KV cache to start fresh for each request
            context.kv_cache_clear();

            // Setup sampler
            let mut sampler_params = SamplerParams::default();
            sampler_params.temperature = temperature;
            sampler_params.top_p = 0.9;
            sampler_params.top_k = 40;
            let mut sampler = sampler_params.build_chain(model.clone())?;

            // Decode prompt
            context.decode(&tokens)?;

            // Generate tokens - pre-allocate with estimated capacity
            let mut generated = String::with_capacity((max_tokens as usize) * 6);
            let mut completion_tokens = 0u32;

            for _ in 0..max_tokens {
                // Use -1 to sample from the last token's logits
                let next_token = sampler.sample(&mut *context, -1);

                if model.vocab_is_eog(next_token) {
                    break;
                }

                if let Ok(text) = model.token_to_str(next_token, 0, false) {
                    generated.push_str(&text);

                    // Check for stop sequences (check if contained, not just ends_with)
                    if !stop_sequences.is_empty() {
                        for stop in &stop_sequences {
                            if let Some(pos) = generated.find(stop) {
                                // Truncate at the stop sequence position
                                generated.truncate(pos);
                                return Ok((generated, completion_tokens));
                            }
                        }
                    }
                }

                // Accept the token to update sampler state (grammar, repetition, etc.)
                sampler.accept(next_token);
                context.decode_single(next_token)?;
                completion_tokens += 1;
            }

            Ok::<_, MullamaError>((generated, completion_tokens))
        })?;

        self.models.add_tokens(completion_tokens as u64);

        Ok((generated, prompt_tokens, completion_tokens))
    }

    /// Generate text with streaming - yields tokens as they're generated
    pub async fn generate_text_streaming(
        &self,
        loaded: Arc<super::models::LoadedModel>,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        // Tokenize - respect model's BOS token setting
        let add_bos = loaded.model.add_bos_token();
        let tokens = loaded.model.tokenize(&prompt, add_bos, false)?;
        let prompt_tokens = tokens.len() as u32;

        // Generate request ID
        let request_id = generate_completion_id();
        let request_id_clone = request_id.clone();

        // Create channel for streaming chunks
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        // Clone what we need for the spawned task
        let model = loaded.model.clone();
        let models_ref = self.models.clone();

        // Spawn the generation task
        tokio::spawn(async move {
            // Get context lock
            let mut context = loaded.context.write().await;

            // Run CPU-bound generation in blocking context
            let result = tokio::task::block_in_place(|| {
                context.kv_cache_clear();

                let mut sampler_params = SamplerParams::default();
                sampler_params.temperature = temperature;
                sampler_params.top_p = 0.9;
                sampler_params.top_k = 40;
                let mut sampler = sampler_params.build_chain(model.clone())?;

                context.decode(&tokens)?;

                let mut generated = String::new();
                let mut index = 0u32;

                for _ in 0..max_tokens {
                    let next_token = sampler.sample(&mut *context, -1);

                    if model.vocab_is_eog(next_token) {
                        break;
                    }

                    if let Ok(text) = model.token_to_str(next_token, 0, false) {
                        generated.push_str(&text);

                        // Check stop sequences BEFORE sending chunk
                        if !stop_sequences.is_empty() {
                            for stop in &stop_sequences {
                                if let Some(pos) = generated.find(stop) {
                                    // Found stop sequence - send final partial chunk if needed
                                    let final_text = &generated[generated.len() - text.len()..];
                                    let stop_pos_in_text = if pos >= generated.len() - text.len() {
                                        pos - (generated.len() - text.len())
                                    } else {
                                        0 // Stop sequence was in previous text
                                    };

                                    if stop_pos_in_text > 0 {
                                        // Send the part before the stop sequence
                                        let partial = &final_text[..stop_pos_in_text];
                                        let chunk = StreamChunk {
                                            request_id: request_id_clone.clone(),
                                            index,
                                            delta: partial.to_string(),
                                            token_id: next_token,
                                            thinking: None,
                                            tool_calls: None,
                                        };
                                        let _ = tx.blocking_send(chunk);
                                    }
                                    return Ok::<_, MullamaError>(index + 1);
                                }
                            }
                        }

                        // Send chunk
                        let chunk = StreamChunk {
                            request_id: request_id_clone.clone(),
                            index,
                            delta: text,
                            token_id: next_token,
                            thinking: None,
                            tool_calls: None,
                        };

                        // If receiver dropped, stop generation
                        if tx.blocking_send(chunk).is_err() {
                            break;
                        }

                        index += 1;
                    }

                    sampler.accept(next_token);
                    context.decode_single(next_token)?;
                }

                Ok::<_, MullamaError>(index)
            });

            if let Ok(tokens_generated) = result {
                models_ref.add_tokens(tokens_generated as u64);
            }
        });

        Ok((rx, prompt_tokens, request_id))
    }

    /// Check if shutdown was requested
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}

/// Builder for daemon configuration
pub struct DaemonBuilder {
    config: DaemonConfig,
    initial_models: Vec<ModelLoadConfig>,
}

impl DaemonBuilder {
    pub fn new() -> Self {
        Self {
            config: DaemonConfig::default(),
            initial_models: Vec::new(),
        }
    }

    pub fn ipc_socket(mut self, addr: impl Into<String>) -> Self {
        self.config.ipc_addr = addr.into();
        self
    }

    pub fn http_port(mut self, port: u16) -> Self {
        self.config.http_port = Some(port);
        self
    }

    pub fn disable_http(mut self) -> Self {
        self.config.http_port = None;
        self
    }

    pub fn http_addr(mut self, addr: impl Into<String>) -> Self {
        self.config.http_addr = addr.into();
        self
    }

    pub fn default_context_size(mut self, size: u32) -> Self {
        self.config.default_context_size = size;
        self
    }

    pub fn default_gpu_layers(mut self, layers: i32) -> Self {
        self.config.default_gpu_layers = layers;
        self
    }

    pub fn threads_per_model(mut self, threads: i32) -> Self {
        self.config.threads_per_model = threads;
        self
    }

    /// Add a model to load on startup (format: "alias:path" or just "path")
    pub fn model(mut self, spec: impl Into<String>) -> Self {
        let spec = spec.into();
        let (alias, path) = if let Some(pos) = spec.find(':') {
            (spec[..pos].to_string(), spec[pos + 1..].to_string())
        } else {
            // Use filename without extension as alias
            let path = std::path::Path::new(&spec);
            let alias = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "default".to_string());
            (alias, spec)
        };

        self.initial_models.push(
            ModelLoadConfig::new(alias, path)
                .gpu_layers(self.config.default_gpu_layers)
                .context_size(self.config.default_context_size)
                .threads(self.config.threads_per_model),
        );
        self
    }

    pub fn build(self) -> (Daemon, Vec<ModelLoadConfig>) {
        (Daemon::new(self.config), self.initial_models)
    }
}

impl Default for DaemonBuilder {
    fn default() -> Self {
        Self::new()
    }
}
