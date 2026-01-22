//! Builder patterns for complex configurations
//!
//! This module provides fluent, ergonomic builder patterns for configuring
//! Mullama components. Builders enforce correct construction order and
//! provide sensible defaults while allowing fine-grained control.
//!
//! ## Features
//!
//! - **Fluent API**: Chainable method calls for readable configuration
//! - **Type safety**: Compile-time guarantees about required parameters
//! - **Validation**: Built-in validation with helpful error messages
//! - **Defaults**: Sensible defaults for all optional parameters
//! - **Presets**: Quick configuration for common use cases
//! - **Progressive disclosure**: Start simple, add complexity as needed
//!
//! ## Example
//!
//! ```rust,ignore
//! use mullama::builder::{ModelBuilder, ContextBuilder, SamplerBuilder};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mullama::MullamaError> {
//!     // Build a complete model with fluent API
//!     let model = ModelBuilder::new()
//!         .path("path/to/model.gguf")
//!         .gpu_layers(32)
//!         .context_size(4096)
//!         .memory_mapping(true)
//!         .build()
//!         .await?;
//!
//!     // Build context with advanced configuration
//!     let context = ContextBuilder::new(model.clone())
//!         .context_size(4096)
//!         .batch_size(512)
//!         .threads(8)
//!         .embeddings(true)
//!         .flash_attention(true)
//!         .build()
//!         .await?;
//!
//!     // Build sophisticated sampling strategy
//!     let sampler = SamplerBuilder::new()
//!         .temperature(0.8)
//!         .top_k(50)
//!         .nucleus(0.95)
//!         .penalties(|p| p
//!             .repetition(1.1)
//!             .frequency(0.1)
//!             .presence(0.1)
//!         )
//!         .build(model.clone())?;
//!
//!     Ok(())
//! }
//! ```

use crate::{
    context::KvCacheType, Context, ContextParams, Model, ModelParams, MullamaError, SamplerChain,
    SamplerParams,
};
use std::sync::Arc;

#[cfg(feature = "async")]
use crate::async_support::AsyncModel;

/// Builder for creating models with fluent API
#[derive(Debug, Clone)]
pub struct ModelBuilder {
    path: Option<String>,
    gpu_layers: i32,
    context_size: Option<u32>,
    use_mmap: bool,
    use_mlock: bool,
    check_tensors: bool,
    vocab_only: bool,
}

impl ModelBuilder {
    /// Create a new model builder
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            path: None,
            gpu_layers: 0,
            context_size: None,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
            vocab_only: false,
        }
    }

    /// Set the path to the model file (required)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .path("path/to/model.gguf");
    /// ```
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the number of GPU layers to offload
    ///
    /// # Arguments
    ///
    /// * `layers` - Number of layers to offload to GPU (0 = CPU only)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .gpu_layers(32); // Offload 32 layers to GPU
    /// ```
    pub fn gpu_layers(mut self, layers: i32) -> Self {
        self.gpu_layers = layers;
        self
    }

    /// Set the context size for the model
    ///
    /// # Arguments
    ///
    /// * `size` - Context size in tokens
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .context_size(4096); // 4K context
    /// ```
    pub fn context_size(mut self, size: u32) -> Self {
        self.context_size = Some(size);
        self
    }

    /// Enable or disable memory mapping
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to use memory mapping
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .memory_mapping(true); // Enable mmap
    /// ```
    pub fn memory_mapping(mut self, enable: bool) -> Self {
        self.use_mmap = enable;
        self
    }

    /// Enable or disable memory locking
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to use memory locking
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .memory_locking(true); // Enable mlock
    /// ```
    pub fn memory_locking(mut self, enable: bool) -> Self {
        self.use_mlock = enable;
        self
    }

    /// Enable or disable tensor validation
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to validate tensors
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .tensor_validation(false); // Disable validation for faster loading
    /// ```
    pub fn tensor_validation(mut self, enable: bool) -> Self {
        self.check_tensors = enable;
        self
    }

    /// Set vocabulary-only mode
    ///
    /// In vocabulary-only mode, only the tokenizer is loaded,
    /// which is useful for tokenization-only tasks.
    ///
    /// # Arguments
    ///
    /// * `vocab_only` - Whether to load only vocabulary
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let builder = ModelBuilder::new()
    ///     .vocabulary_only(true); // Load only tokenizer
    /// ```
    pub fn vocabulary_only(mut self, vocab_only: bool) -> Self {
        self.vocab_only = vocab_only;
        self
    }

    /// Apply a preset configuration
    ///
    /// # Arguments
    ///
    /// * `preset` - Preset configuration function
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{ModelBuilder, presets};
    ///
    /// let builder = ModelBuilder::new()
    ///     .preset(presets::performance_optimized);
    /// ```
    pub fn preset<F>(self, preset: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        preset(self)
    }

    /// Build the model synchronously
    ///
    /// # Returns
    ///
    /// An `Arc<Model>` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if model loading fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// let model = ModelBuilder::new()
    ///     .path("model.gguf")
    ///     .gpu_layers(16)
    ///     .build()?;
    /// # Ok::<(), mullama::MullamaError>(())
    /// ```
    pub fn build(self) -> Result<Arc<Model>, MullamaError> {
        let path = self
            .path
            .ok_or_else(|| MullamaError::ConfigError("Model path is required".to_string()))?;

        let params = ModelParams {
            n_gpu_layers: self.gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            check_tensors: self.check_tensors,
            vocab_only: self.vocab_only,
            ..Default::default()
        };

        let model = Model::load_with_params(&path, params)?;
        Ok(Arc::new(model))
    }

    /// Build the model asynchronously
    ///
    /// # Returns
    ///
    /// An `AsyncModel` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if model loading fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::ModelBuilder;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), mullama::MullamaError> {
    ///     let model = ModelBuilder::new()
    ///         .path("model.gguf")
    ///         .gpu_layers(16)
    ///         .build_async()
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    #[cfg(feature = "async")]
    pub async fn build_async(self) -> Result<AsyncModel, MullamaError> {
        let path = self
            .path
            .ok_or_else(|| MullamaError::ConfigError("Model path is required".to_string()))?;

        let params = ModelParams {
            n_gpu_layers: self.gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            check_tensors: self.check_tensors,
            vocab_only: self.vocab_only,
            ..Default::default()
        };

        AsyncModel::load_with_params(path, params).await
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating contexts with fluent API
#[derive(Debug, Clone)]
pub struct ContextBuilder {
    model: Arc<Model>,
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    embeddings: bool,
    flash_attn_type: crate::sys::llama_flash_attn_type,
    offload_kqv: bool,
    type_k: KvCacheType,
    type_v: KvCacheType,
}

impl ContextBuilder {
    /// Create a new context builder
    ///
    /// # Arguments
    ///
    /// * `model` - The model to create a context for
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{ModelBuilder, ContextBuilder};
    ///
    /// # async fn example() -> Result<(), mullama::MullamaError> {
    /// let model = ModelBuilder::new().path("model.gguf").build()?;
    /// let builder = ContextBuilder::new(model);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model: Arc<Model>) -> Self {
        Self {
            model,
            n_ctx: 2048,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: num_cpus::get() as i32,
            n_threads_batch: num_cpus::get() as i32,
            embeddings: false,
            flash_attn_type: crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO,
            offload_kqv: false,
            type_k: KvCacheType::default(),
            type_v: KvCacheType::default(),
        }
    }

    /// Set the context size
    ///
    /// # Arguments
    ///
    /// * `size` - Context size in tokens
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .context_size(4096);
    /// ```
    pub fn context_size(mut self, size: u32) -> Self {
        self.n_ctx = size;
        self
    }

    /// Set the batch size
    ///
    /// # Arguments
    ///
    /// * `size` - Batch size for prompt processing
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .batch_size(1024);
    /// ```
    pub fn batch_size(mut self, size: u32) -> Self {
        self.n_batch = size;
        self
    }

    /// Set the physical batch size
    ///
    /// # Arguments
    ///
    /// * `size` - Physical batch size
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .physical_batch_size(256);
    /// ```
    pub fn physical_batch_size(mut self, size: u32) -> Self {
        self.n_ubatch = size;
        self
    }

    /// Set maximum number of sequences
    ///
    /// # Arguments
    ///
    /// * `max_seq` - Maximum number of parallel sequences
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .max_sequences(4);
    /// ```
    pub fn max_sequences(mut self, max_seq: u32) -> Self {
        self.n_seq_max = max_seq;
        self
    }

    /// Set number of threads for generation
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .threads(8);
    /// ```
    pub fn threads(mut self, threads: i32) -> Self {
        self.n_threads = threads;
        self
    }

    /// Set number of threads for batch processing
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads for batch processing
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .batch_threads(4);
    /// ```
    pub fn batch_threads(mut self, threads: i32) -> Self {
        self.n_threads_batch = threads;
        self
    }

    /// Enable or disable embeddings
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable embeddings
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .embeddings(true);
    /// ```
    pub fn embeddings(mut self, enable: bool) -> Self {
        self.embeddings = enable;
        self
    }

    /// Enable or disable flash attention
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable flash attention
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .flash_attention(true);
    /// ```
    pub fn flash_attention(mut self, enable: bool) -> Self {
        self.flash_attn_type = if enable {
            crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED
        } else {
            crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED
        };
        self
    }

    /// Enable or disable KQV offloading
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to offload KQV operations to GPU
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .kqv_offload(true);
    /// ```
    pub fn kqv_offload(mut self, enable: bool) -> Self {
        self.offload_kqv = enable;
        self
    }

    /// Set KV-cache quantization type for both K and V caches
    ///
    /// Lower precision types reduce memory usage but may slightly affect quality:
    /// - F16 (default): Best quality, baseline memory
    /// - Q8_0: ~50% memory savings
    /// - Q4_0: ~75% memory savings
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The quantization type for both K and V caches
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use mullama::context::KvCacheType;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .kv_cache_type(KvCacheType::Q8_0); // 50% memory savings
    /// ```
    pub fn kv_cache_type(mut self, cache_type: KvCacheType) -> Self {
        self.type_k = cache_type;
        self.type_v = cache_type;
        self
    }

    /// Set Key cache quantization type separately
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The quantization type for the K cache
    pub fn key_cache_type(mut self, cache_type: KvCacheType) -> Self {
        self.type_k = cache_type;
        self
    }

    /// Set Value cache quantization type separately
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The quantization type for the V cache
    pub fn value_cache_type(mut self, cache_type: KvCacheType) -> Self {
        self.type_v = cache_type;
        self
    }

    /// Apply performance optimizations
    ///
    /// This enables flash attention, optimizes thread counts,
    /// and sets efficient batch sizes.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_for_performance();
    /// ```
    pub fn optimize_for_performance(mut self) -> Self {
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self.n_batch = 1024;
        self.n_ubatch = 512;
        self.offload_kqv = true;
        self
    }

    /// Optimize for memory usage
    ///
    /// This reduces batch sizes, uses Q4 quantized KV cache (~75% memory savings),
    /// and disables memory-intensive features.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_for_memory();
    /// ```
    pub fn optimize_for_memory(mut self) -> Self {
        self.n_ctx = 2048;
        self.n_batch = 256;
        self.n_ubatch = 256;
        self.type_k = KvCacheType::Q4_0;
        self.type_v = KvCacheType::Q4_0;
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self
    }

    /// Balanced optimization (Q8 KV cache)
    ///
    /// Uses Q8 quantized KV cache for ~50% memory savings with minimal quality loss.
    /// Good balance between memory usage and output quality.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_balanced();
    /// ```
    pub fn optimize_balanced(mut self) -> Self {
        self.type_k = KvCacheType::Q8_0;
        self.type_v = KvCacheType::Q8_0;
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        self.offload_kqv = true;
        self
    }

    /// Optimize for quality (F16 KV cache)
    ///
    /// Uses full F16 precision KV cache for best output quality.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::ContextBuilder;
    /// # use std::sync::Arc;
    /// # let model = Arc::new(mullama::Model::load("").unwrap());
    /// let builder = ContextBuilder::new(model)
    ///     .optimize_for_quality();
    /// ```
    pub fn optimize_for_quality(mut self) -> Self {
        self.type_k = KvCacheType::F16;
        self.type_v = KvCacheType::F16;
        self.flash_attn_type = crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO;
        self
    }

    /// Build the context
    ///
    /// # Returns
    ///
    /// A `Context` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if context creation fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use mullama::builder::{ModelBuilder, ContextBuilder};
    /// # fn example() -> Result<(), mullama::MullamaError> {
    /// let model = ModelBuilder::new().path("model.gguf").build()?;
    /// let context = ContextBuilder::new(model)
    ///     .context_size(2048)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Context, MullamaError> {
        let params = ContextParams {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_seq_max: self.n_seq_max,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            embeddings: self.embeddings,
            flash_attn_type: self.flash_attn_type,
            offload_kqv: self.offload_kqv,
            type_k: self.type_k,
            type_v: self.type_v,
            ..Default::default()
        };

        Context::new(self.model, params)
    }

    /// Build the context asynchronously
    ///
    /// # Returns
    ///
    /// An `AsyncContext` ready for use
    ///
    /// # Errors
    ///
    /// Returns `MullamaError` if context creation fails
    #[cfg(feature = "async")]
    pub async fn build_async(self) -> Result<crate::async_support::AsyncContext, MullamaError> {
        use tokio::task;

        let params = ContextParams {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_seq_max: self.n_seq_max,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            embeddings: self.embeddings,
            flash_attn_type: self.flash_attn_type,
            offload_kqv: self.offload_kqv,
            type_k: self.type_k,
            type_v: self.type_v,
            ..Default::default()
        };

        let model = self.model.clone();
        let context = task::spawn_blocking(move || Context::new(model.clone(), params))
            .await
            .map_err(|e| MullamaError::ContextError(format!("Async task failed: {}", e)))?;

        match context {
            Ok(ctx) => Ok(crate::async_support::AsyncContext::new(ctx, self.model)),
            Err(e) => Err(e),
        }
    }
}

/// Builder for creating samplers with fluent API
#[derive(Debug, Clone)]
pub struct SamplerBuilder {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
    penalty_last_n: i32,
    seed: u32,
}

impl SamplerBuilder {
    /// Create a new sampler builder
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            penalty_repeat: 1.1,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 64,
            seed: 0,
        }
    }

    /// Set temperature for sampling
    ///
    /// # Arguments
    ///
    /// * `temp` - Temperature value (0.0 = deterministic, higher = more random)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .temperature(0.7);
    /// ```
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k sampling
    ///
    /// # Arguments
    ///
    /// * `k` - Number of top tokens to consider (0 = disabled)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .top_k(50);
    /// ```
    pub fn top_k(mut self, k: i32) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p (nucleus) sampling
    ///
    /// # Arguments
    ///
    /// * `p` - Cumulative probability threshold (0.0-1.0)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .nucleus(0.9); // Top-p sampling with p=0.9
    /// ```
    pub fn nucleus(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set minimum probability threshold
    ///
    /// # Arguments
    ///
    /// * `min_p` - Minimum probability for token consideration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .min_probability(0.02);
    /// ```
    pub fn min_probability(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Configure penalties using a closure
    ///
    /// # Arguments
    ///
    /// * `config` - Closure that configures penalty settings
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .penalties(|p| p
    ///         .repetition(1.15)
    ///         .frequency(0.1)
    ///         .presence(0.1)
    ///         .lookback(128)
    ///     );
    /// ```
    pub fn penalties<F>(mut self, config: F) -> Self
    where
        F: FnOnce(PenaltyBuilder) -> PenaltyBuilder,
    {
        let penalty_builder = PenaltyBuilder {
            repeat: self.penalty_repeat,
            freq: self.penalty_freq,
            present: self.penalty_present,
            last_n: self.penalty_last_n,
        };

        let configured = config(penalty_builder);
        self.penalty_repeat = configured.repeat;
        self.penalty_freq = configured.freq;
        self.penalty_present = configured.present;
        self.penalty_last_n = configured.last_n;
        self
    }

    /// Set random seed
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed (0 = random)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::SamplerBuilder;
    ///
    /// let builder = SamplerBuilder::new()
    ///     .seed(12345);
    /// ```
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Apply a preset configuration
    ///
    /// # Arguments
    ///
    /// * `preset` - Preset configuration function
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{SamplerBuilder, presets};
    ///
    /// let builder = SamplerBuilder::new()
    ///     .preset(presets::creative_sampling);
    /// ```
    pub fn preset<F>(self, preset: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        preset(self)
    }

    /// Build the sampler
    ///
    /// # Arguments
    ///
    /// * `model` - The model to create the sampler for
    ///
    /// # Returns
    ///
    /// A `SamplerChain` ready for use
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::builder::{ModelBuilder, SamplerBuilder};
    ///
    /// # fn example() -> Result<(), mullama::MullamaError> {
    /// let model = ModelBuilder::new().path("model.gguf").build()?;
    /// let sampler = SamplerBuilder::new()
    ///     .temperature(0.8)
    ///     .top_k(50)
    ///     .build(model)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self, model: Arc<Model>) -> Result<SamplerChain, MullamaError> {
        let params = SamplerParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            penalty_repeat: self.penalty_repeat,
            penalty_freq: self.penalty_freq,
            penalty_present: self.penalty_present,
            penalty_last_n: self.penalty_last_n,
            seed: self.seed,
            ..Default::default()
        };

        params.build_chain(model)
    }
}

impl Default for SamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring penalties
#[derive(Debug, Clone)]
pub struct PenaltyBuilder {
    repeat: f32,
    freq: f32,
    present: f32,
    last_n: i32,
}

impl PenaltyBuilder {
    /// Set repetition penalty
    ///
    /// # Arguments
    ///
    /// * `penalty` - Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition)
    pub fn repetition(mut self, penalty: f32) -> Self {
        self.repeat = penalty;
        self
    }

    /// Set frequency penalty
    ///
    /// # Arguments
    ///
    /// * `penalty` - Frequency penalty
    pub fn frequency(mut self, penalty: f32) -> Self {
        self.freq = penalty;
        self
    }

    /// Set presence penalty
    ///
    /// # Arguments
    ///
    /// * `penalty` - Presence penalty
    pub fn presence(mut self, penalty: f32) -> Self {
        self.present = penalty;
        self
    }

    /// Set lookback window for penalties
    ///
    /// # Arguments
    ///
    /// * `tokens` - Number of tokens to look back for penalty calculation
    pub fn lookback(mut self, tokens: i32) -> Self {
        self.last_n = tokens;
        self
    }
}

/// Preset configurations for builders
pub mod presets {
    use super::*;

    /// Creative writing model configuration
    pub fn creative_model(builder: ModelBuilder) -> ModelBuilder {
        builder
            .gpu_layers(24)
            .context_size(4096)
            .memory_mapping(true)
    }

    /// Performance optimized model configuration
    pub fn performance_optimized(builder: ModelBuilder) -> ModelBuilder {
        builder
            .gpu_layers(99) // Offload as much as possible
            .memory_mapping(true)
            .memory_locking(false)
            .tensor_validation(false) // Skip validation for speed
    }

    /// Memory optimized model configuration
    pub fn memory_optimized(builder: ModelBuilder) -> ModelBuilder {
        builder
            .gpu_layers(0) // Use CPU only
            .memory_mapping(true)
            .memory_locking(false)
    }

    /// Creative sampling configuration
    pub fn creative_sampling(builder: SamplerBuilder) -> SamplerBuilder {
        builder
            .temperature(0.9)
            .top_k(60)
            .nucleus(0.95)
            .penalties(|p| p.repetition(1.15))
    }

    /// Precise sampling configuration
    pub fn precise_sampling(builder: SamplerBuilder) -> SamplerBuilder {
        builder
            .temperature(0.2)
            .top_k(10)
            .nucleus(0.85)
            .penalties(|p| p.repetition(1.05))
    }

    /// Balanced sampling configuration
    pub fn balanced_sampling(builder: SamplerBuilder) -> SamplerBuilder {
        builder
            .temperature(0.7)
            .top_k(40)
            .nucleus(0.9)
            .penalties(|p| p.repetition(1.1).frequency(0.1).presence(0.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_builder() {
        let builder = ModelBuilder::new()
            .path("test.gguf")
            .gpu_layers(16)
            .context_size(2048);

        assert_eq!(builder.path, Some("test.gguf".to_string()));
        assert_eq!(builder.gpu_layers, 16);
        assert_eq!(builder.context_size, Some(2048));
    }

    #[test]
    fn test_sampler_builder() {
        let builder = SamplerBuilder::new()
            .temperature(0.8)
            .top_k(50)
            .nucleus(0.95);

        assert_eq!(builder.temperature, 0.8);
        assert_eq!(builder.top_k, 50);
        assert_eq!(builder.top_p, 0.95);
    }

    #[test]
    fn test_penalty_builder() {
        let builder = SamplerBuilder::new()
            .penalties(|p| p.repetition(1.2).frequency(0.1).presence(0.1).lookback(128));

        assert_eq!(builder.penalty_repeat, 1.2);
        assert_eq!(builder.penalty_freq, 0.1);
        assert_eq!(builder.penalty_present, 0.1);
        assert_eq!(builder.penalty_last_n, 128);
    }

    #[test]
    fn test_presets() {
        let creative = SamplerBuilder::new().preset(presets::creative_sampling);

        assert!(creative.temperature > 0.8);
        assert!(creative.top_k > 50);

        let precise = SamplerBuilder::new().preset(presets::precise_sampling);

        assert!(precise.temperature < 0.3);
        assert!(precise.top_k < 20);
    }
}
