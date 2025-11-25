//! Simple test to verify basic compilation
//! This test only checks basic library functionality without external dependencies

use mullama::*;

#[test]
fn test_basic_structures() {
    // Test that basic structures can be created
    let model_params = ModelParams::default();
    assert!(!model_params.vocab_only);

    let context_params = ContextParams::default();
    assert!(context_params.n_threads > 0); // Should be set to number of CPU cores

    let sampler_params = SamplerParams::default();
    assert_eq!(sampler_params.temperature, 0.8);
}

#[test]
fn test_error_types() {
    // Test that error types work
    let error = MullamaError::ModelLoadError("test error".to_string());
    assert!(error.to_string().contains("test error"));
}

#[test]
fn test_token_structure() {
    // Test token creation - using the complete Token struct from model.rs
    let token = Token {
        id: 123,
        text: "test".to_string(),
        score: 0.5,
        attr: mullama::sys::llama_token_attr::LLAMA_TOKEN_ATTR_NORMAL,
    };
    assert_eq!(token.id, 123);
    assert_eq!(token.text, "test");
    assert_eq!(token.score, 0.5);
}

#[test]
fn test_batch_creation() {
    // Test batch structure
    let batch = Batch::default();
    assert!(batch.is_empty());
}

#[test]
fn test_session_creation() {
    // Test session structure
    let session = Session {
        data: vec![1, 2, 3],
    };
    assert_eq!(session.data.len(), 3);
}

#[test]
fn test_memory_manager() {
    // Test memory manager
    let memory_manager = MemoryManager::new();
    // New manager should not be valid (no context associated)
    assert!(!memory_manager.is_valid());
}

#[test]
fn test_vocabulary() {
    // Test vocabulary
    let vocab = Vocabulary::new();
    assert_eq!(vocab._placeholder, 0);
}