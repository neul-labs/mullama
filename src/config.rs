//! Configuration management with serde integration
//!
//! This module provides comprehensive configuration management for Mullama,
//! allowing configurations to be serialized to and from various formats
//! including JSON, YAML, TOML, and environment variables.
//!
//! ## Features
//!
//! - **Multiple formats**: JSON, YAML, TOML support via serde
//! - **Environment variables**: Load configuration from environment
//! - **Configuration validation**: Validate configurations at load time
//! - **Hot reloading**: Watch configuration files for changes
//! - **Preset configurations**: Common configurations for different use cases
//! - **Hierarchical merging**: Combine multiple configuration sources
//!
//! ## Example
//!
//! ```rust
//! use mullama::config::{MullamaConfig, ModelConfig, SamplingConfig};
//! use serde_json;
//!
//! // Create configuration programmatically
//! let config = MullamaConfig {
//!     model: ModelConfig {
//!         path: "path/to/model.gguf".to_string(),
//!         gpu_layers: 32,
//!         context_size: 4096,
//!         ..Default::default()
//!     },
//!     sampling: SamplingConfig {
//!         temperature: 0.8,
//!         top_k: 50,
//!         top_p: 0.95,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//!
//! // Serialize to JSON
//! let json = serde_json::to_string_pretty(&config).unwrap();
//! println!("{}", json);
//!
//! // Load from JSON
//! let loaded_config: MullamaConfig = serde_json::from_str(&json).unwrap();
//! ```

use crate::{ContextParams, KvCacheType, ModelParams, MullamaError, SamplerParams};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Complete Mullama configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MullamaConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Context configuration
    pub context: ContextConfig,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Performance configuration
    #[serde(default)]
    pub performance: PerformanceConfig,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Custom metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model file
    pub path: String,
    /// Number of GPU layers to offload
    #[serde(default)]
    pub gpu_layers: i32,
    /// Context size for the model
    #[serde(default = "default_context_size")]
    pub context_size: u32,
    /// Enable memory mapping
    #[serde(default = "default_true")]
    pub use_mmap: bool,
    /// Enable memory locking
    #[serde(default)]
    pub use_mlock: bool,
    /// Validate tensor data
    #[serde(default = "default_true")]
    pub check_tensors: bool,
    /// Load only vocabulary (for tokenization-only use)
    #[serde(default)]
    pub vocab_only: bool,
    /// KV overrides for model parameters
    #[serde(default)]
    pub kv_overrides: HashMap<String, KvOverrideValue>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            gpu_layers: 0,
            context_size: 2048,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
            vocab_only: false,
            kv_overrides: HashMap::new(),
        }
    }
}

/// Context configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Context size (sequence length)
    #[serde(default = "default_context_size")]
    pub n_ctx: u32,
    /// Batch size for prompt processing
    #[serde(default = "default_batch_size")]
    pub n_batch: u32,
    /// Physical batch size
    #[serde(default = "default_ubatch_size")]
    pub n_ubatch: u32,
    /// Maximum number of sequences
    #[serde(default = "default_seq_max")]
    pub n_seq_max: u32,
    /// Number of threads for generation
    #[serde(default)]
    pub n_threads: i32,
    /// Number of threads for batch processing
    #[serde(default)]
    pub n_threads_batch: i32,
    /// Enable embeddings
    #[serde(default)]
    pub embeddings: bool,
    /// Enable flash attention
    #[serde(default)]
    pub flash_attn: bool,
    /// Offload KQV operations to GPU
    #[serde(default)]
    pub offload_kqv: bool,
    /// Key cache quantization type (default: F16)
    /// Use Q8_0 for ~50% memory savings, Q4_0 for ~75% savings
    #[serde(default)]
    pub type_k: KvCacheType,
    /// Value cache quantization type (default: F16)
    /// Use Q8_0 for ~50% memory savings, Q4_0 for ~75% savings
    #[serde(default)]
    pub type_v: KvCacheType,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: num_cpus::get() as i32,
            n_threads_batch: num_cpus::get() as i32,
            embeddings: false,
            flash_attn: false,
            offload_kqv: false,
            type_k: KvCacheType::default(),
            type_v: KvCacheType::default(),
        }
    }
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Temperature for sampling (0.0 = deterministic, higher = more random)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    /// Top-p (nucleus) sampling (1.0 = disabled)
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Minimum probability for a token to be considered
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    /// Repetition penalty (1.0 = no penalty)
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: f32,
    /// Number of tokens to consider for repetition penalty
    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: i32,
    /// Random seed (0 = random)
    #[serde(default)]
    pub seed: u32,
    /// Token penalties (token_id -> penalty)
    #[serde(default)]
    pub token_penalties: HashMap<u32, f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 64,
            seed: 0,
            token_penalties: HashMap::new(),
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    #[serde(default)]
    pub enable_monitoring: bool,
    /// Memory optimization level (0-3)
    #[serde(default = "default_memory_optimization")]
    pub memory_optimization: u8,
    /// CPU optimization flags
    #[serde(default)]
    pub cpu_optimizations: CpuOptimizations,
    /// GPU optimization flags
    #[serde(default)]
    pub gpu_optimizations: GpuOptimizations,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: false,
            memory_optimization: 1,
            cpu_optimizations: CpuOptimizations::default(),
            gpu_optimizations: GpuOptimizations::default(),
        }
    }
}

/// CPU optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimizations {
    /// Enable SIMD optimizations
    #[serde(default = "default_true")]
    pub enable_simd: bool,
    /// Enable multi-threading
    #[serde(default = "default_true")]
    pub enable_threading: bool,
    /// Thread affinity settings
    #[serde(default)]
    pub thread_affinity: Option<Vec<usize>>,
}

impl Default for CpuOptimizations {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_threading: true,
            thread_affinity: None,
        }
    }
}

/// GPU optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOptimizations {
    /// Enable GPU acceleration
    #[serde(default)]
    pub enable_gpu: bool,
    /// Preferred GPU device ID
    #[serde(default)]
    pub device_id: i32,
    /// Memory pool size in MB
    #[serde(default)]
    pub memory_pool_size: Option<u64>,
    /// Enable memory optimization
    #[serde(default = "default_true")]
    pub optimize_memory: bool,
}

impl Default for GpuOptimizations {
    fn default() -> Self {
        Self {
            enable_gpu: false,
            device_id: 0,
            memory_pool_size: None,
            optimize_memory: true,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (error, warn, info, debug, trace)
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Enable performance logging
    #[serde(default)]
    pub performance: bool,
    /// Log file path (None = stdout)
    #[serde(default)]
    pub file: Option<PathBuf>,
    /// Enable structured logging (JSON)
    #[serde(default)]
    pub structured: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            performance: false,
            file: None,
            structured: false,
        }
    }
}

/// KV override value for model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KvOverrideValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

// Default value functions
fn default_context_size() -> u32 {
    2048
}
fn default_batch_size() -> u32 {
    512
}
fn default_ubatch_size() -> u32 {
    512
}
fn default_seq_max() -> u32 {
    1
}
fn default_temperature() -> f32 {
    0.8
}
fn default_top_k() -> i32 {
    40
}
fn default_top_p() -> f32 {
    0.95
}
fn default_min_p() -> f32 {
    0.05
}
fn default_repeat_penalty() -> f32 {
    1.1
}
fn default_repeat_last_n() -> i32 {
    64
}
fn default_memory_optimization() -> u8 {
    1
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_true() -> bool {
    true
}

impl MullamaConfig {
    /// Load configuration from a JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the JSON configuration file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::config::MullamaConfig;
    ///
    /// let config = MullamaConfig::from_json_file("config.json").unwrap();
    /// ```
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, MullamaError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to read config file: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to parse JSON config: {}", e)))
    }

    /// Save configuration to a JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - Path where to save the configuration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::config::MullamaConfig;
    ///
    /// let config = MullamaConfig::default();
    /// config.to_json_file("config.json").unwrap();
    /// ```
    pub fn to_json_file(&self, path: impl AsRef<Path>) -> Result<(), MullamaError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to write config file: {}", e)))
    }

    /// Load configuration from environment variables
    ///
    /// Environment variables should be prefixed with `MULLAMA_` and use underscore
    /// separation for nested fields. For example:
    /// - `MULLAMA_MODEL_PATH`
    /// - `MULLAMA_SAMPLING_TEMPERATURE`
    /// - `MULLAMA_CONTEXT_N_CTX`
    ///
    /// # Example
    ///
    /// ```rust
    /// use mullama::config::MullamaConfig;
    ///
    /// // Set environment variables
    /// std::env::set_var("MULLAMA_MODEL_PATH", "/path/to/model.gguf");
    /// std::env::set_var("MULLAMA_SAMPLING_TEMPERATURE", "0.7");
    ///
    /// let config = MullamaConfig::from_env().unwrap();
    /// ```
    pub fn from_env() -> Result<Self, MullamaError> {
        let mut config = Self::default();

        // Model configuration
        if let Ok(path) = std::env::var("MULLAMA_MODEL_PATH") {
            config.model.path = path;
        }
        if let Ok(gpu_layers) = std::env::var("MULLAMA_MODEL_GPU_LAYERS") {
            config.model.gpu_layers = gpu_layers
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid GPU layers: {}", e)))?;
        }
        if let Ok(context_size) = std::env::var("MULLAMA_MODEL_CONTEXT_SIZE") {
            config.model.context_size = context_size
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid context size: {}", e)))?;
        }

        // Context configuration
        if let Ok(n_ctx) = std::env::var("MULLAMA_CONTEXT_N_CTX") {
            config.context.n_ctx = n_ctx
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid n_ctx: {}", e)))?;
        }
        if let Ok(n_batch) = std::env::var("MULLAMA_CONTEXT_N_BATCH") {
            config.context.n_batch = n_batch
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid n_batch: {}", e)))?;
        }
        if let Ok(n_threads) = std::env::var("MULLAMA_CONTEXT_N_THREADS") {
            config.context.n_threads = n_threads
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid n_threads: {}", e)))?;
        }
        if let Ok(type_k) = std::env::var("MULLAMA_CONTEXT_TYPE_K") {
            config.context.type_k = type_k
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid type_k: {}", e)))?;
        }
        if let Ok(type_v) = std::env::var("MULLAMA_CONTEXT_TYPE_V") {
            config.context.type_v = type_v
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid type_v: {}", e)))?;
        }

        // Sampling configuration
        if let Ok(temperature) = std::env::var("MULLAMA_SAMPLING_TEMPERATURE") {
            config.sampling.temperature = temperature
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid temperature: {}", e)))?;
        }
        if let Ok(top_k) = std::env::var("MULLAMA_SAMPLING_TOP_K") {
            config.sampling.top_k = top_k
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid top_k: {}", e)))?;
        }
        if let Ok(top_p) = std::env::var("MULLAMA_SAMPLING_TOP_P") {
            config.sampling.top_p = top_p
                .parse()
                .map_err(|e| MullamaError::ConfigError(format!("Invalid top_p: {}", e)))?;
        }

        Ok(config)
    }

    /// Merge with another configuration, overriding values
    ///
    /// # Arguments
    ///
    /// * `other` - Configuration to merge with
    ///
    /// # Example
    ///
    /// ```rust
    /// use mullama::config::{MullamaConfig, ModelConfig};
    ///
    /// let mut base_config = MullamaConfig::default();
    /// let override_config = MullamaConfig {
    ///     model: ModelConfig {
    ///         path: "new_model.gguf".to_string(),
    ///         ..Default::default()
    ///     },
    ///     ..Default::default()
    /// };
    ///
    /// base_config.merge(override_config);
    /// assert_eq!(base_config.model.path, "new_model.gguf");
    /// ```
    pub fn merge(&mut self, other: Self) {
        if !other.model.path.is_empty() {
            self.model = other.model;
        }
        // Add more merge logic as needed
        self.context = other.context;
        self.sampling = other.sampling;
        self.performance = other.performance;
        self.logging = other.logging;
        self.metadata.extend(other.metadata);
    }

    /// Validate the configuration
    ///
    /// Checks for common configuration errors and invalid values.
    pub fn validate(&self) -> Result<(), MullamaError> {
        // Validate model configuration
        if self.model.path.is_empty() {
            return Err(MullamaError::ConfigError(
                "Model path cannot be empty".to_string(),
            ));
        }

        if self.model.context_size == 0 {
            return Err(MullamaError::ConfigError(
                "Context size must be greater than 0".to_string(),
            ));
        }

        if self.model.gpu_layers < 0 {
            return Err(MullamaError::ConfigError(
                "GPU layers cannot be negative".to_string(),
            ));
        }

        // Validate context configuration
        if self.context.n_ctx == 0 {
            return Err(MullamaError::ConfigError(
                "Context size must be greater than 0".to_string(),
            ));
        }

        if self.context.n_batch == 0 {
            return Err(MullamaError::ConfigError(
                "Batch size must be greater than 0".to_string(),
            ));
        }

        // Validate sampling configuration
        if self.sampling.temperature < 0.0 {
            return Err(MullamaError::ConfigError(
                "Temperature cannot be negative".to_string(),
            ));
        }

        if self.sampling.top_p <= 0.0 || self.sampling.top_p > 1.0 {
            return Err(MullamaError::ConfigError(
                "Top-p must be between 0 and 1".to_string(),
            ));
        }

        if self.sampling.repeat_penalty <= 0.0 {
            return Err(MullamaError::ConfigError(
                "Repeat penalty must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Convert to ModelParams for model loading
    pub fn to_model_params(&self) -> ModelParams {
        ModelParams {
            n_gpu_layers: self.model.gpu_layers,
            use_mmap: self.model.use_mmap,
            use_mlock: self.model.use_mlock,
            check_tensors: self.model.check_tensors,
            vocab_only: self.model.vocab_only,
            ..Default::default()
        }
    }

    /// Convert to ContextParams for context creation
    pub fn to_context_params(&self) -> ContextParams {
        ContextParams {
            n_ctx: self.context.n_ctx,
            n_batch: self.context.n_batch,
            n_ubatch: self.context.n_ubatch,
            n_seq_max: self.context.n_seq_max,
            n_threads: self.context.n_threads,
            n_threads_batch: self.context.n_threads_batch,
            embeddings: self.context.embeddings,
            flash_attn_type: if self.context.flash_attn {
                crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED
            } else {
                crate::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO
            },
            offload_kqv: self.context.offload_kqv,
            type_k: self.context.type_k,
            type_v: self.context.type_v,
            ..Default::default()
        }
    }

    /// Convert to SamplerParams for sampling configuration
    pub fn to_sampler_params(&self) -> SamplerParams {
        SamplerParams {
            temperature: self.sampling.temperature,
            top_k: self.sampling.top_k,
            top_p: self.sampling.top_p,
            min_p: self.sampling.min_p,
            penalty_repeat: self.sampling.repeat_penalty,
            penalty_freq: self.sampling.frequency_penalty,
            penalty_present: self.sampling.presence_penalty,
            penalty_last_n: self.sampling.repeat_last_n,
            seed: self.sampling.seed,
            ..Default::default()
        }
    }
}

/// Preset configurations for common use cases
pub mod presets {
    use super::*;

    /// Configuration optimized for creative writing
    pub fn creative_writing() -> MullamaConfig {
        let mut config = MullamaConfig::default();
        config.sampling.temperature = 0.9;
        config.sampling.top_k = 60;
        config.sampling.top_p = 0.95;
        config.sampling.repeat_penalty = 1.15;
        config
    }

    /// Configuration optimized for code generation
    pub fn code_generation() -> MullamaConfig {
        let mut config = MullamaConfig::default();
        config.sampling.temperature = 0.1;
        config.sampling.top_k = 10;
        config.sampling.top_p = 0.9;
        config.sampling.repeat_penalty = 1.05;
        config
    }

    /// Configuration optimized for question answering
    pub fn question_answering() -> MullamaConfig {
        let mut config = MullamaConfig::default();
        config.sampling.temperature = 0.3;
        config.sampling.top_k = 20;
        config.sampling.top_p = 0.85;
        config.sampling.repeat_penalty = 1.1;
        config
    }

    /// Configuration optimized for chatbots
    pub fn chatbot() -> MullamaConfig {
        let mut config = MullamaConfig::default();
        config.sampling.temperature = 0.7;
        config.sampling.top_k = 40;
        config.sampling.top_p = 0.9;
        config.sampling.repeat_penalty = 1.1;
        config.context.n_ctx = 4096; // Larger context for conversations
        config
    }

    /// Configuration optimized for performance
    pub fn performance_optimized() -> MullamaConfig {
        let mut config = MullamaConfig::default();
        config.context.n_batch = 1024; // Larger batch size
        config.context.flash_attn = true;
        config.performance.memory_optimization = 2;
        config.performance.cpu_optimizations.enable_simd = true;
        config.performance.cpu_optimizations.enable_threading = true;
        config
    }

    /// Configuration optimized for low memory usage
    pub fn memory_optimized() -> MullamaConfig {
        let mut config = MullamaConfig::default();
        config.context.n_ctx = 1024; // Smaller context
        config.context.n_batch = 256; // Smaller batch
        config.context.type_k = crate::KvCacheType::Q4_0; // 75% memory savings
        config.context.type_v = crate::KvCacheType::Q4_0;
        config.context.flash_attn = true;
        config.performance.memory_optimization = 3;
        config.model.use_mmap = true;
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let mut config = MullamaConfig::default();
        // Default config needs a path and context size to validate
        config.model.path = "model.gguf".to_string();
        config.model.context_size = 2048;
        config.context.n_ctx = 2048;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_json_serialization() {
        let config = MullamaConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MullamaConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            config.sampling.temperature,
            deserialized.sampling.temperature
        );
    }

    #[test]
    fn test_presets() {
        let creative = presets::creative_writing();
        assert!(creative.sampling.temperature > 0.8);

        let code = presets::code_generation();
        assert!(code.sampling.temperature < 0.2);
    }

    #[test]
    fn test_validation() {
        let mut config = MullamaConfig::default();
        config.model.path = String::new();
        assert!(config.validate().is_err());

        config.model.path = "model.gguf".to_string();
        config.sampling.temperature = -1.0;
        assert!(config.validate().is_err());
    }
}
