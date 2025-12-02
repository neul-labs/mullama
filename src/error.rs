use thiserror::Error;

#[derive(Error, Debug)]
pub enum MullamaError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Failed to create context: {0}")]
    ContextError(String),

    #[error("Tokenization failed: {0}")]
    TokenizationError(String),

    #[error("Generation failed: {0}")]
    GenerationError(String),

    #[error("Session error: {0}")]
    SessionError(String),

    #[error("Sampling error: {0}")]
    SamplingError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("FFI error: {0}")]
    FfiError(String),

    #[error("LoRA error: {0}")]
    LoRAError(String),

    #[error("Grammar error: {0}")]
    GrammarError(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("Feature not supported: {0}")]
    NotSupported(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Control vector error: {0}")]
    ControlVectorError(String),

    #[error("Speculative decoding error: {0}")]
    SpeculativeError(String),

    #[error("Multimodal error: {0}")]
    MultimodalError(String),

    #[error("Audio error: {0}")]
    AudioError(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("Format conversion error: {0}")]
    FormatConversionError(String),

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    #[error("Hugging Face error: {0}")]
    HuggingFaceError(String),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Daemon error: {0}")]
    DaemonError(String),
}
