//! Daemon client for IPC communication
//!
//! Provides a high-level API for communicating with the Mullama daemon.

use std::time::Duration;

use nng::{Protocol, Socket};

use super::protocol::*;
use crate::error::MullamaError;

/// Client for communicating with the Mullama daemon
pub struct DaemonClient {
    socket: Socket,
    timeout: Duration,
}

impl DaemonClient {
    /// Connect to the daemon at the default socket
    pub fn connect_default() -> Result<Self, MullamaError> {
        Self::connect(super::DEFAULT_SOCKET)
    }

    /// Connect to a daemon at the given address
    pub fn connect(addr: &str) -> Result<Self, MullamaError> {
        Self::connect_with_timeout(addr, Duration::from_secs(30))
    }

    /// Connect with a custom timeout
    pub fn connect_with_timeout(addr: &str, timeout: Duration) -> Result<Self, MullamaError> {
        let socket = Socket::new(Protocol::Req0)
            .map_err(|e| MullamaError::DaemonError(format!("Failed to create socket: {}", e)))?;

        socket.dial(addr).map_err(|e| {
            MullamaError::DaemonError(format!("Failed to connect to {}: {}", addr, e))
        })?;

        Ok(Self { socket, timeout })
    }

    /// Send a request and wait for response
    pub fn request(&self, request: &Request) -> Result<Response, MullamaError> {
        let req_bytes = request
            .to_bytes()
            .map_err(|e| MullamaError::DaemonError(format!("Serialization failed: {}", e)))?;

        self.socket
            .send(nng::Message::from(req_bytes.as_slice()))
            .map_err(|(_, e)| MullamaError::DaemonError(format!("Send failed: {}", e)))?;

        let msg = self
            .socket
            .recv()
            .map_err(|e| MullamaError::DaemonError(format!("Receive failed: {}", e)))?;

        Response::from_bytes(&msg)
            .map_err(|e| MullamaError::DaemonError(format!("Deserialization failed: {}", e)))
    }

    /// Ping the daemon
    pub fn ping(&self) -> Result<(u64, String), MullamaError> {
        match self.request(&Request::Ping)? {
            Response::Pong {
                uptime_secs,
                version,
            } => Ok((uptime_secs, version)),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Get daemon status
    pub fn status(&self) -> Result<DaemonStatus, MullamaError> {
        match self.request(&Request::Status)? {
            Response::Status(status) => Ok(status),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// List loaded models
    pub fn list_models(&self) -> Result<Vec<ModelStatus>, MullamaError> {
        match self.request(&Request::ListModels)? {
            Response::Models(models) => Ok(models),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Load a model (format: "alias:path" or just "path")
    pub fn load_model(&self, spec: &str) -> Result<(String, ModelInfo), MullamaError> {
        let (alias, path) = if let Some(pos) = spec.find(':') {
            (spec[..pos].to_string(), spec[pos + 1..].to_string())
        } else {
            let p = std::path::Path::new(spec);
            let alias = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string());
            (alias, spec.to_string())
        };

        match self.request(&Request::LoadModel {
            alias: alias.clone(),
            path,
            gpu_layers: 0,
            context_size: 0,
        })? {
            Response::ModelLoaded { alias, info } => Ok((alias, info)),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Load a model with full options
    pub fn load_model_with_options(
        &self,
        alias: &str,
        path: &str,
        gpu_layers: i32,
        context_size: u32,
    ) -> Result<(String, ModelInfo), MullamaError> {
        match self.request(&Request::LoadModel {
            alias: alias.to_string(),
            path: path.to_string(),
            gpu_layers,
            context_size,
        })? {
            Response::ModelLoaded { alias, info } => Ok((alias, info)),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Unload a model
    pub fn unload_model(&self, alias: &str) -> Result<(), MullamaError> {
        match self.request(&Request::UnloadModel {
            alias: alias.to_string(),
        })? {
            Response::ModelUnloaded { .. } => Ok(()),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Set the default model
    pub fn set_default_model(&self, alias: &str) -> Result<(), MullamaError> {
        match self.request(&Request::SetDefaultModel {
            alias: alias.to_string(),
        })? {
            Response::DefaultModelSet { .. } => Ok(()),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Chat with a model
    pub fn chat(
        &self,
        message: &str,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ChatResult, MullamaError> {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: message.to_string(),
            name: None,
        }];

        self.chat_completion(messages, model, max_tokens, temperature)
    }

    /// Chat completion with message history
    pub fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ChatResult, MullamaError> {
        let start = std::time::Instant::now();

        match self.request(&Request::ChatCompletion {
            model: model.map(String::from),
            messages,
            max_tokens,
            temperature,
            stream: false,
            stop: vec![],
        })? {
            Response::ChatCompletion(resp) => Ok(ChatResult {
                text: resp
                    .choices
                    .first()
                    .map(|c| c.message.content.clone())
                    .unwrap_or_default(),
                model: resp.model,
                prompt_tokens: resp.usage.prompt_tokens,
                completion_tokens: resp.usage.completion_tokens,
                duration_ms: start.elapsed().as_millis() as u64,
            }),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Text completion
    pub fn complete(
        &self,
        prompt: &str,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<CompletionResult, MullamaError> {
        let start = std::time::Instant::now();

        match self.request(&Request::Completion {
            model: model.map(String::from),
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            stream: false,
        })? {
            Response::Completion(resp) => Ok(CompletionResult {
                text: resp
                    .choices
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default(),
                model: resp.model,
                prompt_tokens: resp.usage.prompt_tokens,
                completion_tokens: resp.usage.completion_tokens,
                duration_ms: start.elapsed().as_millis() as u64,
            }),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Tokenize text
    pub fn tokenize(&self, text: &str, model: Option<&str>) -> Result<Vec<i32>, MullamaError> {
        match self.request(&Request::Tokenize {
            model: model.map(String::from),
            text: text.to_string(),
        })? {
            Response::Tokens { tokens, .. } => Ok(tokens),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }

    /// Shutdown the daemon
    pub fn shutdown(&self) -> Result<(), MullamaError> {
        match self.request(&Request::Shutdown)? {
            Response::ShuttingDown => Ok(()),
            Response::Error { message, .. } => Err(MullamaError::DaemonError(message)),
            _ => Err(MullamaError::DaemonError("Unexpected response".into())),
        }
    }
}

/// Result of chat completion
#[derive(Debug, Clone)]
pub struct ChatResult {
    pub text: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub duration_ms: u64,
}

impl ChatResult {
    pub fn tokens_per_second(&self) -> f64 {
        if self.duration_ms == 0 {
            0.0
        } else {
            (self.completion_tokens as f64) / (self.duration_ms as f64 / 1000.0)
        }
    }
}

/// Result of text completion
#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub text: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub duration_ms: u64,
}

impl CompletionResult {
    pub fn tokens_per_second(&self) -> f64 {
        if self.duration_ms == 0 {
            0.0
        } else {
            (self.completion_tokens as f64) / (self.duration_ms as f64 / 1000.0)
        }
    }
}
