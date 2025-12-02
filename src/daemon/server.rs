//! Daemon server implementation
//!
//! Core daemon that manages models and handles requests from IPC and HTTP.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use super::models::{ModelLoadConfig, ModelManager, RequestGuard};
use super::protocol::*;
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
            } => self.handle_load_model(alias, path, gpu_layers, context_size).await,

            Request::UnloadModel { alias } => self.handle_unload_model(&alias).await,
            Request::SetDefaultModel { alias } => self.handle_set_default(&alias).await,

            Request::ChatCompletion {
                model,
                messages,
                max_tokens,
                temperature,
                stream,
                stop,
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

            Request::Embeddings { model, input } => {
                self.handle_embeddings(model, input).await
            }

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
            http_endpoint: self.config.http_port.map(|p| format!("http://{}:{}", self.config.http_addr, p)),
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
            .gpu_layers(if gpu_layers == 0 { self.config.default_gpu_layers } else { gpu_layers })
            .context_size(if context_size == 0 { self.config.default_context_size } else { context_size })
            .threads(self.config.threads_per_model);

        match self.models.load(config).await {
            Ok(info) => Response::ModelLoaded { alias, info },
            Err(e) => Response::error(ErrorCode::ModelLoadFailed, e.to_string()),
        }
    }

    async fn handle_unload_model(&self, alias: &str) -> Response {
        match self.models.unload(alias).await {
            Ok(()) => Response::ModelUnloaded { alias: alias.to_string() },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }

    async fn handle_set_default(&self, alias: &str) -> Response {
        match self.models.set_default(alias).await {
            Ok(()) => Response::DefaultModelSet { alias: alias.to_string() },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }

    async fn handle_chat_completion(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        _stream: bool,
        _stop: Vec<String>,
    ) -> Response {
        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::SeqCst);

        // Build prompt from messages
        let prompt = self.build_chat_prompt(&messages);

        // Generate
        let result = self.generate_text(&loaded, &prompt, max_tokens, temperature).await;

        self.active_requests.fetch_sub(1, Ordering::SeqCst);

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
                            content: text,
                            name: None,
                        },
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
        self.active_requests.fetch_add(1, Ordering::SeqCst);

        let result = self.generate_text(&loaded, &prompt, max_tokens, temperature).await;

        self.active_requests.fetch_sub(1, Ordering::SeqCst);

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

    async fn handle_embeddings(
        &self,
        _model: Option<String>,
        _input: EmbeddingInput,
    ) -> Response {
        Response::error(ErrorCode::Internal, "Embeddings not yet implemented")
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

    fn build_chat_prompt(&self, messages: &[ChatMessage]) -> String {
        // Simple chat template - could be enhanced with model-specific templates
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str(&format!("System: {}\n\n", msg.content));
                }
                "user" => {
                    prompt.push_str(&format!("User: {}\n\n", msg.content));
                }
                "assistant" => {
                    prompt.push_str(&format!("Assistant: {}\n\n", msg.content));
                }
                _ => {
                    prompt.push_str(&format!("{}: {}\n\n", msg.role, msg.content));
                }
            }
        }

        prompt.push_str("Assistant:");
        prompt
    }

    async fn generate_text(
        &self,
        loaded: &super::models::LoadedModel,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<(String, u32, u32), MullamaError> {
        // Tokenize
        let tokens = loaded.model.tokenize(prompt, true, false)?;
        let prompt_tokens = tokens.len() as u32;

        // Get context lock
        let mut context = loaded.context.write().await;

        // Setup sampler
        let mut sampler_params = SamplerParams::default();
        sampler_params.temperature = temperature;
        sampler_params.top_p = 0.9;
        sampler_params.top_k = 40;

        let mut sampler = sampler_params.build_chain(loaded.model.clone())?;

        // Decode prompt
        context.decode(&tokens)?;

        // Generate
        let mut generated = String::new();
        let mut completion_tokens = 0u32;

        for _ in 0..max_tokens {
            let next_token = sampler.sample(&mut *context, 0);

            if loaded.model.vocab_is_eog(next_token) {
                break;
            }

            if let Ok(text) = loaded.model.token_to_str(next_token, 0, false) {
                generated.push_str(&text);
            }

            context.decode(&[next_token])?;
            completion_tokens += 1;
        }

        self.models.add_tokens(completion_tokens as u64);

        Ok((generated, prompt_tokens, completion_tokens))
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
