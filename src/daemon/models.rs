//! Multi-model manager for the daemon
//!
//! Handles loading, unloading, and managing multiple models simultaneously.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::RwLock;

use super::protocol::ModelInfo;
use crate::{Context, ContextParams, Model, ModelParams, MullamaError};

#[cfg(feature = "multimodal")]
use crate::{MtmdContext, MtmdParams};

/// A loaded model instance with its context
pub struct LoadedModel {
    pub alias: String,
    pub model: Arc<Model>,
    pub context: RwLock<Context>,
    pub info: ModelInfo,
    pub active_requests: AtomicU32,
    /// Multimodal context for vision/audio models (requires mmproj file)
    #[cfg(feature = "multimodal")]
    pub mtmd_context: Option<RwLock<MtmdContext>>,
}

impl LoadedModel {
    /// Create a new loaded model
    #[cfg(feature = "multimodal")]
    pub fn new(
        alias: String,
        model: Arc<Model>,
        context: Context,
        info: ModelInfo,
        mtmd_context: Option<MtmdContext>,
    ) -> Self {
        Self {
            alias,
            model,
            context: RwLock::new(context),
            info,
            active_requests: AtomicU32::new(0),
            mtmd_context: mtmd_context.map(RwLock::new),
        }
    }

    /// Create a new loaded model (non-multimodal build)
    #[cfg(not(feature = "multimodal"))]
    pub fn new(alias: String, model: Arc<Model>, context: Context, info: ModelInfo) -> Self {
        Self {
            alias,
            model,
            context: RwLock::new(context),
            info,
            active_requests: AtomicU32::new(0),
        }
    }

    /// Check if this model has multimodal (vision/audio) support
    #[cfg(feature = "multimodal")]
    pub fn has_multimodal(&self) -> bool {
        self.mtmd_context.is_some()
    }

    #[cfg(not(feature = "multimodal"))]
    pub fn has_multimodal(&self) -> bool {
        false
    }

    /// Increment active request count
    pub fn acquire(&self) {
        self.active_requests.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement active request count
    pub fn release(&self) {
        self.active_requests.fetch_sub(1, Ordering::SeqCst);
    }

    /// Get active request count
    pub fn active_count(&self) -> u32 {
        self.active_requests.load(Ordering::SeqCst)
    }
}

/// Configuration for loading a model
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    pub alias: String,
    pub path: String,
    pub gpu_layers: i32,
    pub context_size: u32,
    pub threads: i32,
    /// Path to multimodal projector file (mmproj) for vision/audio models
    pub mmproj_path: Option<String>,
}

impl ModelLoadConfig {
    pub fn new(alias: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
            path: path.into(),
            gpu_layers: 0,
            context_size: 4096,
            threads: num_cpus::get() as i32,
            mmproj_path: None,
        }
    }

    pub fn gpu_layers(mut self, layers: i32) -> Self {
        self.gpu_layers = layers;
        self
    }

    pub fn context_size(mut self, size: u32) -> Self {
        self.context_size = size;
        self
    }

    pub fn threads(mut self, threads: i32) -> Self {
        self.threads = threads;
        self
    }

    /// Set the multimodal projector path for vision/audio models
    pub fn mmproj(mut self, path: impl Into<String>) -> Self {
        self.mmproj_path = Some(path.into());
        self
    }
}

/// Multi-model manager
pub struct ModelManager {
    models: RwLock<HashMap<String, Arc<LoadedModel>>>,
    default_model: RwLock<Option<String>>,
    total_tokens: AtomicU64,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            default_model: RwLock::new(None),
            total_tokens: AtomicU64::new(0),
        }
    }

    /// Load a model with the given configuration
    pub async fn load(&self, config: ModelLoadConfig) -> Result<ModelInfo, MullamaError> {
        // Check if alias already exists
        {
            let models = self.models.read().await;
            if models.contains_key(&config.alias) {
                return Err(MullamaError::OperationFailed(format!(
                    "Model with alias '{}' already loaded",
                    config.alias
                )));
            }
        }

        // Load the model
        let mut model_params = ModelParams::default();
        model_params.n_gpu_layers = config.gpu_layers;

        let model = Arc::new(Model::load_with_params(&config.path, model_params)?);

        // Create context
        let mut ctx_params = ContextParams::default();
        ctx_params.n_ctx = config.context_size;
        ctx_params.n_threads = config.threads;
        ctx_params.n_threads_batch = config.threads;

        let context = Context::new(model.clone(), ctx_params)?;

        let info = ModelInfo {
            path: config.path.clone(),
            parameters: model.n_params() as u64,
            context_size: config.context_size,
            vocab_size: model.n_vocab() as u32,
            gpu_layers: config.gpu_layers,
            quantization: None, // TODO: detect from model
        };

        // Create multimodal context if mmproj path provided
        #[cfg(feature = "multimodal")]
        let mtmd_context = if let Some(ref mmproj_path) = config.mmproj_path {
            let mut mtmd_params = MtmdParams::default();
            mtmd_params.n_threads = config.threads;
            match MtmdContext::new(mmproj_path, &model, mtmd_params) {
                Ok(ctx) => {
                    eprintln!(
                        "  Multimodal: vision={}, audio={}",
                        ctx.supports_vision(),
                        ctx.supports_audio()
                    );
                    Some(ctx)
                }
                Err(e) => {
                    eprintln!("  Warning: Failed to load mmproj: {}", e);
                    None
                }
            }
        } else {
            None
        };

        #[cfg(feature = "multimodal")]
        let loaded = Arc::new(LoadedModel::new(
            config.alias.clone(),
            model,
            context,
            info.clone(),
            mtmd_context,
        ));

        #[cfg(not(feature = "multimodal"))]
        let loaded = Arc::new(LoadedModel::new(
            config.alias.clone(),
            model,
            context,
            info.clone(),
        ));

        // Add to models
        {
            let mut models = self.models.write().await;
            models.insert(config.alias.clone(), loaded);
        }

        // Set as default if first model
        {
            let mut default = self.default_model.write().await;
            if default.is_none() {
                *default = Some(config.alias);
            }
        }

        Ok(info)
    }

    /// Unload a model by alias
    pub async fn unload(&self, alias: &str) -> Result<(), MullamaError> {
        let mut models = self.models.write().await;

        if let Some(model) = models.get(alias) {
            if model.active_count() > 0 {
                return Err(MullamaError::OperationFailed(format!(
                    "Model '{}' has {} active requests",
                    alias,
                    model.active_count()
                )));
            }
        }

        if models.remove(alias).is_none() {
            return Err(MullamaError::OperationFailed(format!(
                "Model '{}' not found",
                alias
            )));
        }

        // Update default if needed
        {
            let mut default = self.default_model.write().await;
            if default.as_deref() == Some(alias) {
                *default = models.keys().next().cloned();
            }
        }

        Ok(())
    }

    /// Get a model by alias, or the default model
    pub async fn get(&self, alias: Option<&str>) -> Result<Arc<LoadedModel>, MullamaError> {
        let models = self.models.read().await;

        let key = match alias {
            Some(a) => a.to_string(),
            None => {
                let default = self.default_model.read().await;
                default.clone().ok_or_else(|| {
                    MullamaError::OperationFailed("No default model set".to_string())
                })?
            }
        };

        models
            .get(&key)
            .cloned()
            .ok_or_else(|| MullamaError::OperationFailed(format!("Model '{}' not found", key)))
    }

    /// Set the default model
    pub async fn set_default(&self, alias: &str) -> Result<(), MullamaError> {
        let models = self.models.read().await;
        if !models.contains_key(alias) {
            return Err(MullamaError::OperationFailed(format!(
                "Model '{}' not found",
                alias
            )));
        }

        let mut default = self.default_model.write().await;
        *default = Some(alias.to_string());
        Ok(())
    }

    /// Get the default model alias
    pub async fn default_alias(&self) -> Option<String> {
        self.default_model.read().await.clone()
    }

    /// List all loaded models
    pub async fn list(&self) -> Vec<(String, ModelInfo, bool, u32)> {
        let models = self.models.read().await;
        let default = self.default_model.read().await;

        models
            .iter()
            .map(|(alias, model)| {
                (
                    alias.clone(),
                    model.info.clone(),
                    default.as_deref() == Some(alias),
                    model.active_count(),
                )
            })
            .collect()
    }

    /// Get the number of loaded models
    pub async fn count(&self) -> usize {
        self.models.read().await.len()
    }

    /// Add to total tokens generated
    pub fn add_tokens(&self, count: u64) {
        self.total_tokens.fetch_add(count, Ordering::Relaxed);
    }

    /// Get total tokens generated
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Check if any models are loaded
    pub async fn has_models(&self) -> bool {
        !self.models.read().await.is_empty()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard for tracking active requests
pub struct RequestGuard {
    model: Arc<LoadedModel>,
}

impl RequestGuard {
    pub fn new(model: Arc<LoadedModel>) -> Self {
        model.acquire();
        Self { model }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.model.release();
    }
}
