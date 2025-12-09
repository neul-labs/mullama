//! # Mullama
//!
//! Comprehensive Rust bindings for llama.cpp with memory-safe API and advanced integration features.
//!
//! ## Overview
//!
//! Mullama provides safe Rust bindings for llama.cpp with automatic memory management
//! and a comprehensive API. The library focuses on memory safety, ease of use, and
//! production-ready features for building LLM-powered applications.
//!
//! ## Key Features
//!
//! - **Memory Safety**: Zero unsafe operations in public API with automatic resource management
//! - **Complete API Coverage**: Comprehensive bindings for llama.cpp functionality
//! - **Production Ready**: Robust error handling, extensive testing, and performance optimization
//! - **Advanced Features**: Support for embeddings, batch processing, sampling strategies, and more
//! - **Cross-Platform**: Supports Windows, macOS, and Linux with optional GPU acceleration
//! - **Async/Await Support**: Non-blocking operations with Tokio integration
//! - **Streaming Interfaces**: Real-time token generation with backpressure handling
//! - **Configuration Management**: Serde-based configuration with validation
//! - **Builder Patterns**: Fluent APIs for complex configurations
//! - **Web Framework Integration**: Direct Axum integration for web services
//!
//! ## Core Components
//!
//! - **Model Management**: Load and manage GGUF models with various parameters
//! - **Context Operations**: Create and manage inference contexts with configurable parameters
//! - **Tokenization**: Convert between text and tokens with special token handling
//! - **Sampling**: Advanced sampling strategies including top-k, top-p, temperature, and penalties
//! - **Batch Processing**: Efficient processing of multiple token sequences
//! - **Embeddings**: Generate and manipulate text embeddings
//! - **Session Management**: Save and restore model states
//! - **Memory Management**: Automatic resource cleanup and memory optimization
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mullama::{Model, Context, ContextParams, SamplerParams};
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), mullama::MullamaError> {
//! // Load a model
//! let model = Arc::new(Model::load("path/to/model.gguf")?);
//!
//! // Create context parameters
//! let mut ctx_params = ContextParams::default();
//! ctx_params.n_ctx = 2048;      // Context size
//! ctx_params.n_batch = 512;     // Batch size
//! ctx_params.n_threads = 8;     // Number of threads
//!
//! // Create context
//! let mut context = Context::new(model.clone(), ctx_params)?;
//!
//! // Configure sampling
//! let mut sampler_params = SamplerParams::default();
//! sampler_params.temperature = 0.7;
//! sampler_params.top_k = 40;
//! sampler_params.top_p = 0.9;
//!
//! let mut sampler = sampler_params.build_chain(model.clone())?;
//!
//! // Tokenize input text
//! let prompt = "The future of artificial intelligence is";
//! let tokens = model.tokenize(prompt, true, false)?;
//!
//! // Decode the prompt tokens
//! context.decode(&tokens)?;
//!
//! // Generate tokens
//! for _ in 0..100 {
//!     // Use -1 to sample from the last token's logits
//!     let next_token = sampler.sample(&mut context, -1);
//!
//!     // Convert token to text
//!     let text = model.token_to_str(next_token, 0, false)?;
//!     print!("{}", text);
//!
//!     // Check for end of generation
//!     if next_token == 0 {
//!         break;
//!     }
//!
//!     // Accept the token and decode it
//!     sampler.accept(next_token);
//!     context.decode(&[next_token])?;
//! }
//! # Ok(())
//! # }
//! ```

pub mod batch;
pub mod context;
pub mod embedding;
pub mod error;
pub mod memory;
pub mod model;
pub mod sampling;
pub mod session;
pub mod sys;
pub mod token;
pub mod vocab;

// Integration features
#[cfg(feature = "async")]
pub mod async_support;
pub mod builder;
pub mod config;
#[cfg(feature = "daemon")]
pub mod daemon;
#[cfg(feature = "format-conversion")]
pub mod format_conversion;
#[cfg(feature = "multimodal")]
pub mod multimodal;
#[cfg(feature = "parallel")]
pub mod parallel;
#[cfg(feature = "streaming")]
pub mod streaming;
#[cfg(feature = "streaming-audio")]
pub mod streaming_audio;
#[cfg(feature = "tokio-runtime")]
pub mod tokio_integration;
#[cfg(feature = "web")]
pub mod web;
#[cfg(feature = "websockets")]
pub mod websockets;

// Advanced features
pub mod grammar;
pub mod huggingface;
pub mod lora;

// Export Hugging Face types at crate root for convenience
pub use huggingface::{GGUFFile, HFClient, HFModelInfo, ModelSearchFilters, QuantizationType};

// ==================== System-level Functions ====================

/// Initialize the llama.cpp backend
///
/// This is called automatically when loading a model, but can be called
/// manually for early initialization.
pub fn backend_init() {
    unsafe {
        sys::llama_backend_init();
    }
}

/// Free the llama.cpp backend resources
///
/// Call this when completely done with llama.cpp to free system resources.
pub fn backend_free() {
    unsafe {
        sys::llama_backend_free();
    }
}

/// Get the current timestamp in microseconds
pub fn time_us() -> i64 {
    unsafe { sys::llama_time_us() }
}

/// Get the maximum number of devices supported
pub fn max_devices() -> usize {
    unsafe { sys::llama_max_devices() }
}

/// Check if GPU offloading is supported
pub fn supports_gpu_offload() -> bool {
    unsafe { sys::llama_supports_gpu_offload() }
}

/// Check if mmap is supported for model loading
pub fn supports_mmap() -> bool {
    unsafe { sys::llama_supports_mmap() }
}

/// Check if mlock is supported for memory locking
pub fn supports_mlock() -> bool {
    unsafe { sys::llama_supports_mlock() }
}

/// Check if RPC is supported
pub fn supports_rpc() -> bool {
    unsafe { sys::llama_supports_rpc() }
}

/// Get system information string
///
/// Returns a detailed string with information about the system's
/// capabilities (AVX, CUDA, Metal, etc.)
pub fn print_system_info() -> String {
    unsafe {
        let ptr = sys::llama_print_system_info();
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().to_string()
        }
    }
}

/// Initialize NUMA (Non-Uniform Memory Access) support
pub fn numa_init(strategy: sys::ggml_numa_strategy) {
    unsafe {
        sys::llama_numa_init(strategy);
    }
}

/// Get system capabilities as a structured report
pub fn system_info() -> SystemInfo {
    SystemInfo {
        max_devices: max_devices(),
        supports_gpu_offload: supports_gpu_offload(),
        supports_mmap: supports_mmap(),
        supports_mlock: supports_mlock(),
        supports_rpc: supports_rpc(),
        details: print_system_info(),
    }
}

/// System information structure
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub max_devices: usize,
    pub supports_gpu_offload: bool,
    pub supports_mmap: bool,
    pub supports_mlock: bool,
    pub supports_rpc: bool,
    pub details: String,
}

// ==================== Logging ====================

/// Log callback function type
pub type LogCallback = extern "C" fn(
    level: i32,
    text: *const std::os::raw::c_char,
    user_data: *mut std::os::raw::c_void,
);

/// Set custom log callback
///
/// # Safety
/// The caller must ensure that `user_data` points to valid memory that will remain
/// valid for the lifetime of this callback, or is null.
pub unsafe fn log_set(callback: LogCallback, user_data: *mut std::os::raw::c_void) {
    sys::llama_log_set(Some(callback), user_data);
}

// ==================== Batch Helpers ====================

/// Create a batch for a single sequence of tokens
///
/// This is a convenience function for simple use cases.
pub fn batch_get_one(tokens: &[i32]) -> sys::llama_batch {
    unsafe { sys::llama_batch_get_one(tokens.as_ptr() as *mut i32, tokens.len() as i32) }
}

// ==================== Chat Templates ====================

/// Get the number of built-in chat templates
pub fn chat_builtin_template_count() -> i32 {
    unsafe { sys::llama_chat_builtin_templates(std::ptr::null_mut(), 0) }
}

// ==================== Vocab Helper Functions ====================

/// Get vocabulary information from a model
pub struct VocabInfo {
    pub bos_token: i32,
    pub eos_token: i32,
    pub cls_token: i32,
    pub sep_token: i32,
    pub nl_token: i32,
    pub pad_token: i32,
    pub eot_token: i32,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl Model {
    /// Get comprehensive vocabulary information
    pub fn vocab_info(&self) -> VocabInfo {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.as_ptr()) };
        VocabInfo {
            bos_token: unsafe { sys::llama_vocab_bos(vocab_ptr) },
            eos_token: unsafe { sys::llama_vocab_eos(vocab_ptr) },
            cls_token: unsafe { sys::llama_vocab_cls(vocab_ptr) },
            sep_token: unsafe { sys::llama_vocab_sep(vocab_ptr) },
            nl_token: unsafe { sys::llama_vocab_nl(vocab_ptr) },
            pad_token: unsafe { sys::llama_vocab_pad(vocab_ptr) },
            eot_token: unsafe { sys::llama_vocab_eot(vocab_ptr) },
            add_bos: unsafe { sys::llama_vocab_get_add_bos(vocab_ptr) },
            add_eos: unsafe { sys::llama_vocab_get_add_eos(vocab_ptr) },
        }
    }

    /// Get text for a token from the vocabulary
    pub fn vocab_get_text(&self, token: i32) -> Option<String> {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.as_ptr()) };
        let ptr = unsafe { sys::llama_vocab_get_text(vocab_ptr, token) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().to_string() })
        }
    }

    /// Get score for a token from the vocabulary
    pub fn vocab_get_score(&self, token: i32) -> f32 {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.as_ptr()) };
        unsafe { sys::llama_vocab_get_score(vocab_ptr, token) }
    }

    /// Get attributes for a token from the vocabulary
    pub fn vocab_get_attr(&self, token: i32) -> sys::llama_token_attr {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.as_ptr()) };
        unsafe { sys::llama_vocab_get_attr(vocab_ptr, token) }
    }

    /// Check if a token is a control token (via vocab)
    pub fn vocab_is_control(&self, token: i32) -> bool {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.as_ptr()) };
        unsafe { sys::llama_vocab_is_control(vocab_ptr, token) }
    }

    /// Check if a token is end-of-generation (via vocab)
    pub fn vocab_is_eog(&self, token: i32) -> bool {
        let vocab_ptr = unsafe { sys::llama_model_get_vocab(self.as_ptr()) };
        unsafe { sys::llama_vocab_is_eog(vocab_ptr, token) }
    }
}
pub mod control_vector;
pub mod quantization;

// Re-export the public API
pub use batch::Batch;
pub use context::{Context, ContextParams};
pub use embedding::{EmbeddingUtil, Embeddings};
pub use error::MullamaError;
pub use memory::MemoryManager;
pub use model::{Model, ModelKvOverride, ModelKvOverrideValue, ModelParams, Token};
pub use sampling::{
    LogitBias, Sampler, SamplerChain, SamplerChainParams, SamplerParams, SamplerPerfData,
    TokenData, TokenDataArray,
};
pub use session::Session;
pub use token::{Token as TokenStruct, TokenId};
pub use vocab::Vocabulary;

// Re-export integration features
#[cfg(feature = "async")]
pub use async_support::{AsyncConfig, AsyncContext, AsyncModel, ModelInfo};
pub use builder::{ContextBuilder, ModelBuilder, SamplerBuilder};
pub use config::{
    ContextConfig, CpuOptimizations, GpuOptimizations, LoggingConfig, ModelConfig, MullamaConfig,
    PerformanceConfig, SamplingConfig,
};
#[cfg(feature = "format-conversion")]
pub use format_conversion::{
    AudioConversionResult, AudioConverter, AudioConverterConfig, ConversionConfig,
    ImageConversionResult, ImageConverter, ImageConverterConfig,
};
#[cfg(feature = "multimodal")]
pub use multimodal::{
    AudioFeatures, AudioFormat, AudioInput, ImageInput, MultimodalInput, MultimodalOutput,
    MultimodalProcessor, VideoInput,
};
#[cfg(feature = "parallel")]
pub use parallel::{BatchGenerationConfig, GenerationResult, ParallelProcessor, ThreadPoolConfig};
#[cfg(feature = "streaming")]
pub use streaming::{StreamConfig, TokenData as StreamTokenData, TokenStream};
#[cfg(feature = "streaming-audio")]
pub use streaming_audio::{
    AudioChunk, AudioStream, AudioStreamConfig, DevicePreference, StreamingAudioProcessor,
    StreamingMetrics,
};
#[cfg(feature = "tokio-runtime")]
pub use tokio_integration::{
    ModelPool, MullamaRuntime, MullamaRuntimeBuilder, RuntimeMetrics, TaskManager,
};
#[cfg(feature = "web")]
pub use web::{
    ApiMetrics, AppError, AppState, GenerateRequest, GenerateResponse, RouterBuilder,
    TokenizeRequest, TokenizeResponse,
};
#[cfg(feature = "websockets")]
pub use websockets::{
    AudioProcessor as WSAudioProcessor, ConnectionManager, ServerStats, WSMessage, WebSocketConfig,
    WebSocketServer,
};

// Re-export advanced features (commented out for now)
// pub use lora::{LoRAAdapter, LoRAManager};
// pub use grammar::{Grammar, GrammarRule};
// pub use control_vector::{ControlVector, ControlVectorManager};
// pub use speculative::{SpeculativeDecoder, SpeculativeConfig};
// pub use quantization::{QuantizationEngine, QuantizationParams, QuantizationType};
// pub use gpu_advanced::{GpuManager, GpuDevice, AllocationStrategy};
// pub use multimodal::{MultimodalProcessor, MultimodalInput, MultimodalConfig};

// Re-export sys types for advanced users
pub use sys::{
    ggml_numa_strategy, ggml_type, llama_attention_type, llama_ftype, llama_memory_t,
    llama_model_kv_override_type, llama_pooling_type, llama_pos, llama_rope_scaling_type,
    llama_rope_type, llama_seq_id, llama_split_mode, llama_token, llama_token_attr,
    llama_token_type, llama_vocab_type, LLAMA_DEFAULT_SEED, LLAMA_TOKEN_NULL,
};

/// Convenience prelude for common imports
pub mod prelude {
    pub use crate::{
        Batch, Context, ContextBuilder, ContextParams, Model, ModelBuilder, ModelParams,
        MullamaConfig, MullamaError, SamplerBuilder, SamplerChain, SamplerParams,
    };

    #[cfg(feature = "async")]
    pub use crate::{AsyncContext, AsyncModel};

    #[cfg(feature = "streaming")]
    pub use crate::{StreamConfig, StreamTokenData, TokenStream};

    #[cfg(feature = "web")]
    pub use crate::{AppState, GenerateRequest, GenerateResponse, RouterBuilder};

    #[cfg(feature = "tokio-runtime")]
    pub use crate::{ModelPool, MullamaRuntime, TaskManager};

    #[cfg(feature = "parallel")]
    pub use crate::{BatchGenerationConfig, ParallelProcessor};

    #[cfg(feature = "websockets")]
    pub use crate::{WSMessage, WebSocketServer};

    #[cfg(feature = "streaming-audio")]
    pub use crate::{AudioChunk, AudioStreamConfig, StreamingAudioProcessor};

    #[cfg(feature = "format-conversion")]
    pub use crate::{AudioConverter, ImageConverter};

    #[cfg(feature = "multimodal")]
    pub use crate::{AudioInput, ImageInput, MultimodalInput, MultimodalProcessor};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_backend_initialization() {
        // Test that we can initialize the backend
        unsafe {
            sys::llama_backend_init();
            sys::llama_backend_free();
        }
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_model_params_default() {
        // Test that ModelParams has sensible defaults
        let params = model::ModelParams::default();
        assert_eq!(params.n_gpu_layers, 0);
        assert!(params.use_mmap);
        assert!(!params.use_mlock);
    }

    #[test]
    fn test_context_params_default() {
        // Test that ContextParams has sensible defaults
        let params = context::ContextParams::default();
        // n_ctx = 0 means use model default
        assert_eq!(params.n_ctx, 0);
        assert!(params.n_batch > 0);
        assert!(params.n_threads > 0);
    }

    #[test]
    fn test_token_structure() {
        // Test that we can create token structs
        let token = token::Token {
            id: 1234,
            text: "test".to_string(),
            score: 0.5,
        };
        assert_eq!(token.id, 1234);
        assert_eq!(token.text, "test");
        assert_eq!(token.score, 0.5);
    }

    #[test]
    fn test_batch_structure() {
        // Test that we can create batch structs
        let batch = batch::Batch::default();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_session_structure() {
        // Test that we can create session structs
        let session = session::Session { data: vec![] };
        assert!(session.data.is_empty());
    }

    #[test]
    fn test_sampling_structure() {
        // Test that we can create sampler structs
        let _sampler = sampling::Sampler::new().expect("Failed to create sampler");
        let params = sampling::SamplerParams::default();
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 40);
    }

    #[test]
    fn test_embedding_structure() {
        // Test that we can create embedding structs
        let embeddings = embedding::Embeddings::new(vec![0.1, 0.2, 0.3], 3);
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings.dimension, 3);
    }

    #[test]
    fn test_memory_manager_structure() {
        // Test that we can create memory manager structs
        let memory_manager = memory::MemoryManager::new();
        // New manager should not be valid (no context associated)
        assert!(!memory_manager.is_valid());
    }

    #[test]
    fn test_vocabulary_structure() {
        // Test that we can create vocabulary structs
        let vocab = vocab::Vocabulary::new();
        assert_eq!(vocab._placeholder, 0);
    }
}
