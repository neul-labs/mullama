//! Advanced text generation example showcasing the full API
//!
//! This example demonstrates:
//! - Advanced model parameters
//! - Context configuration options
//! - Sampling chain composition
//! - Performance monitoring concepts

use mullama::{
    Batch, Context, ContextParams, Model, ModelParams, MullamaError, SamplerChain,
    SamplerChainParams, SamplerParams,
};
use std::sync::Arc;

fn main() -> Result<(), MullamaError> {
    println!("Mullama Advanced Generation Example");
    println!("Showcasing comprehensive API usage\n");

    // Advanced model parameters demonstration
    println!("Advanced Model Parameters:");
    let model_params = ModelParams {
        n_gpu_layers: 32,    // Offload layers to GPU if available
        use_mmap: true,      // Enable memory mapping
        use_mlock: false,    // Disable memory locking
        check_tensors: true, // Validate tensors
        vocab_only: false,   // Load full model
        ..Default::default()
    };

    println!("   - GPU layers: {}", model_params.n_gpu_layers);
    println!("   - Memory mapping: {}", model_params.use_mmap);
    println!("   - Memory locking: {}", model_params.use_mlock);
    println!("   - Tensor validation: {}", model_params.check_tensors);

    // Advanced context parameters
    println!("\nAdvanced Context Parameters:");
    let ctx_params = ContextParams {
        n_ctx: 4096,        // Large context size
        n_batch: 512,       // Batch size
        n_ubatch: 256,      // Physical batch size
        n_seq_max: 4,       // Support multiple sequences
        n_threads: 8,       // Generation threads
        n_threads_batch: 8, // Batch processing threads
        embeddings: true,   // Enable embeddings
        flash_attn_type: mullama::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED,
        offload_kqv: true, // Offload KV operations to GPU
        ..Default::default()
    };

    println!("   - Context size: {}", ctx_params.n_ctx);
    println!("   - Batch size: {}", ctx_params.n_batch);
    println!("   - Physical batch size: {}", ctx_params.n_ubatch);
    println!("   - Max sequences: {}", ctx_params.n_seq_max);
    println!("   - Generation threads: {}", ctx_params.n_threads);
    println!("   - Batch threads: {}", ctx_params.n_threads_batch);
    println!("   - Embeddings enabled: {}", ctx_params.embeddings);
    println!("   - Flash attention: {:?}", ctx_params.flash_attn_type);

    // Advanced sampling parameters
    println!("\nAdvanced Sampling Parameters:");
    let sampler_params = SamplerParams {
        temperature: 0.7,     // Controlled randomness
        top_k: 40,            // Top-k sampling
        top_p: 0.9,           // Nucleus sampling
        min_p: 0.1,           // Minimum probability
        penalty_repeat: 1.05, // Repetition penalty
        penalty_freq: 0.1,    // Frequency penalty
        penalty_present: 0.1, // Presence penalty
        penalty_last_n: 128,  // Penalty lookback
        ..Default::default()
    };

    println!("   - Temperature: {}", sampler_params.temperature);
    println!("   - Top-k: {}", sampler_params.top_k);
    println!("   - Top-p: {}", sampler_params.top_p);
    println!("   - Min-p: {}", sampler_params.min_p);
    println!("   - Repetition penalty: {}", sampler_params.penalty_repeat);
    println!("   - Frequency penalty: {}", sampler_params.penalty_freq);
    println!("   - Presence penalty: {}", sampler_params.penalty_present);
    println!("   - Penalty lookback: {}", sampler_params.penalty_last_n);

    // Demonstrate API patterns without requiring actual model
    println!("\nAPI Usage Patterns:");
    demonstrate_api_patterns()?;

    println!("\nAdvanced generation concepts demonstrated!");
    Ok(())
}

fn demonstrate_api_patterns() -> Result<(), MullamaError> {
    println!("   Loading Models:");
    println!("     - Model::load(path) - Simple loading");
    println!("     - Model::load_with_params(path, params) - Advanced loading");

    println!("   Context Creation:");
    println!("     - Context::new(model, params) - Create context");
    println!("     - Configure threads, batch size, sequences");

    println!("   Tokenization:");
    println!("     - model.tokenize(text, add_bos, special) - Convert text to tokens");
    println!("     - model.token_to_str(token, lstrip, special) - Convert token to text");

    println!("   Sampling Chains:");
    println!("     - SamplerParams::build_chain(model) - Create sampler chain");
    println!("     - Multiple sampler types: top-k, top-p, temperature, penalties");

    println!("   Batch Processing:");
    println!("     - Batch::new(max_tokens, embd, max_seq) - Create batch");
    println!("     - Batch::from_tokens(tokens) - Batch from token array");
    println!("     - context.decode(batch) - Process batch");

    println!("   Advanced Features:");
    println!("     - KV cache management");
    println!("     - Multi-sequence support");
    println!("     - State save/restore");
    println!("     - Performance monitoring");
    println!("     - GPU acceleration");
    println!("     - Memory optimization");

    Ok(())
}
