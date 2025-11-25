//! Documentation and API consistency tests for Mullama
//!
//! These tests validate that the public API is well-documented, consistent,
//! and follows Rust conventions. They also test documentation examples.

use mullama::*;

#[cfg(test)]
mod api_documentation_tests {
    use super::*;

    #[test]
    fn test_public_api_exports() {
        // Test that all expected types are exported from the root

        // Core types should be available
        let _model_params: ModelParams = ModelParams::default();
        let _context_params: ContextParams = ContextParams::default();
        let _sampler_params: SamplerParams = SamplerParams::default();
        let _sampler_chain_params: SamplerChainParams = SamplerChainParams::default();

        // Structures should be available
        let _batch: Batch = Batch::from_tokens(&[]);
        let _session: Session = Session { data: vec![] };
        let _embeddings: Embeddings = Embeddings::new(vec![], 0);
        let _memory_manager: MemoryManager = MemoryManager::new();
        let _vocabulary: Vocabulary = Vocabulary::new();

        // Sampling types should be available
        let _sampler: Sampler = Sampler::new();
        let _token_data: TokenData = TokenData { id: 0, logit: 0.0, p: 0.0 };
        let _logit_bias: LogitBias = LogitBias { token: 0, bias: 0.0 };
        let _token_data_array: TokenDataArray = TokenDataArray::new(vec![]);
        let _sampler_perf: SamplerPerfData = SamplerPerfData { t_sample_ms: 0.0, n_sample: 0 };

        // Error types should be available
        let _error: MullamaError = MullamaError::ModelLoadError("test".to_string());

        // KV override types should be available
        let _kv_override: ModelKvOverride = ModelKvOverride {
            key: "test".to_string(),
            value: ModelKvOverrideValue::Int(42),
        };
    }

    #[test]
    fn test_prelude_exports() {
        // Test that prelude module exports work correctly
        use mullama::prelude::*;

        let _model_params = ModelParams::default();
        let _context_params = ContextParams::default();
        let _sampler_params = SamplerParams::default();
        let _batch = Batch::from_tokens(&[]);

        // Error type should be available
        let _error = MullamaError::GenerationError("test".to_string());
    }

    #[test]
    fn test_sys_module_exports() {
        // Test that sys module exports work correctly
        use mullama::sys::*;

        // Constants should be available
        let _default_seed = LLAMA_DEFAULT_SEED;
        let _token_null = LLAMA_TOKEN_NULL;

        // Types should be available
        let _vocab_type: llama_vocab_type = llama_vocab_type::LLAMA_VOCAB_TYPE_BPE;
        let _rope_type: llama_rope_type = llama_rope_type::LLAMA_ROPE_TYPE_NORM;
        let _ftype: llama_ftype = llama_ftype::LLAMA_FTYPE_MOSTLY_F16;

        // Type aliases should work
        let _token: llama_token = 0;
        let _pos: llama_pos = 0;
        let _seq_id: llama_seq_id = 0;
    }

    #[test]
    fn test_re_exports_consistency() {
        // Ensure re-exported types work the same as direct imports

        // Token type consistency
        let token_id_1: TokenId = 123;
        let token_id_2: crate::token::TokenId = 123;
        assert_eq!(token_id_1, token_id_2);

        // Model type consistency - test that Token re-export works
        // Note: There's both Model::Token and token::Token, test the alias
        let _token_struct: crate::token::Token = crate::token::Token {
            id: 1,
            text: "test".to_string(),
            score: 0.5,
        };
    }
}

#[cfg(test)]
mod documentation_example_tests {
    use super::*;

    #[test]
    fn test_basic_usage_pattern() {
        // Test the documented basic usage pattern (without actual model loading)

        // This would be the documented pattern:
        // let model = Model::load("path/to/model.gguf")?;
        // let mut ctx = model.create_context(ContextParams::default())?;
        // let tokens = model.tokenize("Hello, world!", true, false)?;
        // let result = ctx.generate(&tokens, 100)?;

        // Test that the types and methods exist
        let model_params = ModelParams::default();
        let context_params = ContextParams::default();

        // Test parameter customization as documented
        let custom_context = ContextParams {
            n_ctx: 2048,
            n_batch: 512,
            ..ContextParams::default()
        };

        assert_eq!(custom_context.n_ctx, 2048);
        assert_eq!(custom_context.n_batch, 512);
    }

    #[test]
    fn test_sampling_usage_pattern() {
        // Test documented sampling patterns

        let sampler_params = SamplerParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            ..SamplerParams::default()
        };

        assert_eq!(sampler_params.temperature, 0.7);
        assert_eq!(sampler_params.top_k, 40);
        assert_eq!(sampler_params.top_p, 0.9);

        // Test sampler chain creation
        let chain_params = SamplerChainParams::default();
        let _chain = SamplerChain::new(chain_params);
    }

    #[test]
    fn test_batch_processing_pattern() {
        // Test documented batch processing pattern

        let tokens = vec![1, 2, 3, 4, 5];
        let batch = Batch::from_tokens(&tokens);

        assert!(!batch.is_empty());

        // Test batch operations
        let _llama_batch = batch.get_llama_batch();
    }

    #[test]
    fn test_error_handling_pattern() {
        // Test documented error handling patterns

        fn example_function() -> Result<(), MullamaError> {
            // Simulate various error types
            if false {
                return Err(MullamaError::ModelLoadError("Model not found".to_string()));
            }
            if false {
                return Err(MullamaError::ContextError("Context creation failed".to_string()));
            }
            if false {
                return Err(MullamaError::GenerationError("Generation failed".to_string()));
            }
            Ok(())
        }

        let result = example_function();
        assert!(result.is_ok());

        // Test error display
        let error = MullamaError::TokenizationError("Tokenization failed".to_string());
        let error_string = format!("{}", error);
        assert!(error_string.contains("Tokenization failed"));
    }

    #[test]
    fn test_kv_override_usage_pattern() {
        // Test documented KV override patterns

        let int_override = ModelKvOverride {
            key: "max_seq_len".to_string(),
            value: ModelKvOverrideValue::Int(4096),
        };

        let float_override = ModelKvOverride {
            key: "rope_freq_base".to_string(),
            value: ModelKvOverrideValue::Float(10000.0),
        };

        let bool_override = ModelKvOverride {
            key: "use_parallel_residual".to_string(),
            value: ModelKvOverrideValue::Bool(true),
        };

        let str_override = ModelKvOverride {
            key: "model_type".to_string(),
            value: ModelKvOverrideValue::Str("llama".to_string()),
        };

        // Test that overrides can be added to model params
        let mut model_params = ModelParams::default();
        model_params.kv_overrides.push(int_override);
        model_params.kv_overrides.push(float_override);
        model_params.kv_overrides.push(bool_override);
        model_params.kv_overrides.push(str_override);

        assert_eq!(model_params.kv_overrides.len(), 4);
    }
}

#[cfg(test)]
mod api_consistency_tests {
    use super::*;

    #[test]
    fn test_default_implementations() {
        // Test that all parameter types implement Default consistently

        let model_params = ModelParams::default();
        let context_params = ContextParams::default();
        let sampler_params = SamplerParams::default();
        let sampler_chain_params = SamplerChainParams::default();

        // Test that defaults are sensible
        assert_eq!(model_params.n_gpu_layers, 0); // Safe default
        assert!(context_params.n_threads > 0); // Should detect system threads
        assert!(sampler_params.temperature > 0.0); // Valid temperature
        assert!(!sampler_chain_params.no_perf); // Performance enabled by default
    }

    #[test]
    fn test_clone_implementations() {
        // Test that all parameter types implement Clone consistently

        let model_params = ModelParams::default();
        let context_params = ContextParams::default();
        let sampler_params = SamplerParams::default();
        let sampler_chain_params = SamplerChainParams::default();

        let _model_clone = model_params.clone();
        let _context_clone = context_params.clone();
        let _sampler_clone = sampler_params.clone();
        let _chain_clone = sampler_chain_params.clone();

        // Clones should be equal
        assert_eq!(model_params.n_gpu_layers, _model_clone.n_gpu_layers);
        assert_eq!(context_params.n_ctx, _context_clone.n_ctx);
        assert_eq!(sampler_params.temperature, _sampler_clone.temperature);
        assert_eq!(sampler_chain_params.no_perf, _chain_clone.no_perf);
    }

    #[test]
    fn test_debug_implementations() {
        // Test that types implement Debug for debugging

        let model_params = ModelParams::default();
        let context_params = ContextParams::default();
        let sampler_params = SamplerParams::default();

        let model_debug = format!("{:?}", model_params);
        let context_debug = format!("{:?}", context_params);
        let sampler_debug = format!("{:?}", sampler_params);

        // Debug output should contain type names
        assert!(model_debug.contains("ModelParams"));
        assert!(context_debug.contains("ContextParams"));
        assert!(sampler_debug.contains("SamplerParams"));
    }

    #[test]
    fn test_error_trait_implementations() {
        // Test that MullamaError implements std::error::Error properly

        let error = MullamaError::ModelLoadError("Test error".to_string());

        // Should implement Error trait
        let _error_trait: &dyn std::error::Error = &error;

        // Should implement Display
        let display_string = format!("{}", error);
        assert!(display_string.contains("Test error"));

        // Should implement Debug
        let debug_string = format!("{:?}", error);
        assert!(debug_string.contains("ModelLoadError"));
        assert!(debug_string.contains("Test error"));
    }

    #[test]
    fn test_send_sync_implementations() {
        // Test that types are Send + Sync where appropriate

        fn assert_send_sync<T: Send + Sync>() {}

        // Parameter types should be Send + Sync
        assert_send_sync::<ModelParams>();
        assert_send_sync::<ContextParams>();
        assert_send_sync::<SamplerParams>();
        assert_send_sync::<SamplerChainParams>();

        // Data types should be Send + Sync
        assert_send_sync::<Session>();
        assert_send_sync::<TokenData>();
        assert_send_sync::<LogitBias>();
        assert_send_sync::<ModelKvOverride>();
        assert_send_sync::<MullamaError>();
    }
}

#[cfg(test)]
mod feature_completeness_tests {
    use super::*;

    #[test]
    fn test_sampling_feature_completeness() {
        // Test that all documented sampling features are available

        let params = SamplerParams {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            penalty_repeat: 1.1,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 64,
            penalize_nl: true,
            ignore_eos: false,
            seed: sys::LLAMA_DEFAULT_SEED,
        };

        // All sampling parameters should be settable
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.min_p, 0.05);
        assert_eq!(params.typical_p, 1.0);
        assert_eq!(params.penalty_repeat, 1.1);
        assert_eq!(params.penalty_freq, 0.0);
        assert_eq!(params.penalty_present, 0.0);
        assert_eq!(params.penalty_last_n, 64);
        assert_eq!(params.penalize_nl, true);
        assert_eq!(params.ignore_eos, false);
        assert_eq!(params.seed, sys::LLAMA_DEFAULT_SEED);
    }

    #[test]
    fn test_model_feature_completeness() {
        // Test that all documented model features are available

        let params = ModelParams {
            n_gpu_layers: 35,
            split_mode: sys::llama_split_mode::LLAMA_SPLIT_MODE_LAYER,
            main_gpu: 0,
            tensor_split: vec![0.6, 0.4],
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
            use_extra_bufts: false,
            kv_overrides: vec![
                ModelKvOverride {
                    key: "context_length".to_string(),
                    value: ModelKvOverrideValue::Int(4096),
                }
            ],
            progress_callback: None,
        };

        // All model parameters should be settable
        assert_eq!(params.n_gpu_layers, 35);
        assert_eq!(params.main_gpu, 0);
        assert_eq!(params.tensor_split, vec![0.6, 0.4]);
        assert_eq!(params.vocab_only, false);
        assert_eq!(params.use_mmap, true);
        assert_eq!(params.use_mlock, false);
        assert_eq!(params.check_tensors, true);
        assert_eq!(params.use_extra_bufts, false);
        assert_eq!(params.kv_overrides.len(), 1);
    }

    #[test]
    fn test_context_feature_completeness() {
        // Test that all documented context features are available

        let params = ContextParams {
            n_ctx: 4096,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: 8,
            n_threads_batch: 8,
            rope_scaling_type: sys::llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_LINEAR,
            pooling_type: sys::llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
            attention_type: sys::llama_attention_type::LLAMA_ATTENTION_TYPE_CAUSAL,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            defrag_thold: -1.0,
            embeddings: false,
            flash_attn: false,
            offload_kqv: true,
            no_perf: false,
            op_offload: false,
            swa_full: true,
            kv_unified: false,
        };

        // All context parameters should be settable
        assert_eq!(params.n_ctx, 4096);
        assert_eq!(params.n_batch, 512);
        assert_eq!(params.n_ubatch, 512);
        assert_eq!(params.n_seq_max, 1);
        assert_eq!(params.n_threads, 8);
        assert_eq!(params.n_threads_batch, 8);
        assert_eq!(params.rope_freq_base, 10000.0);
        assert_eq!(params.rope_freq_scale, 1.0);
        assert_eq!(params.yarn_ext_factor, -1.0);
        assert_eq!(params.yarn_attn_factor, 1.0);
        assert_eq!(params.yarn_beta_fast, 32.0);
        assert_eq!(params.yarn_beta_slow, 1.0);
        assert_eq!(params.yarn_orig_ctx, 0);
        assert_eq!(params.defrag_thold, -1.0);
        assert_eq!(params.embeddings, false);
        assert_eq!(params.flash_attn, false);
        assert_eq!(params.offload_kqv, true);
        assert_eq!(params.swa_full, true);
        assert_eq!(params.kv_unified, false);
    }

    #[test]
    fn test_enum_completeness() {
        // Test that all documented enums are available and usable

        use sys::*;

        // Vocabulary types
        let _vocab_none = llama_vocab_type::LLAMA_VOCAB_TYPE_NONE;
        let _vocab_spm = llama_vocab_type::LLAMA_VOCAB_TYPE_SPM;
        let _vocab_bpe = llama_vocab_type::LLAMA_VOCAB_TYPE_BPE;
        let _vocab_wpm = llama_vocab_type::LLAMA_VOCAB_TYPE_WPM;
        let _vocab_ugm = llama_vocab_type::LLAMA_VOCAB_TYPE_UGM;
        let _vocab_rwkv = llama_vocab_type::LLAMA_VOCAB_TYPE_RWKV;

        // RoPE types
        let _rope_none = llama_rope_type::LLAMA_ROPE_TYPE_NONE;
        let _rope_norm = llama_rope_type::LLAMA_ROPE_TYPE_NORM;
        let _rope_neox = llama_rope_type::LLAMA_ROPE_TYPE_NEOX;

        // File types
        let _ftype_f32 = llama_ftype::LLAMA_FTYPE_ALL_F32;
        let _ftype_f16 = llama_ftype::LLAMA_FTYPE_MOSTLY_F16;
        let _ftype_q4_0 = llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_0;
        let _ftype_q4_1 = llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1;
        let _ftype_q5_0 = llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_0;
        let _ftype_q5_1 = llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_1;
        let _ftype_q8_0 = llama_ftype::LLAMA_FTYPE_MOSTLY_Q8_0;

        // Split modes
        let _split_none = llama_split_mode::LLAMA_SPLIT_MODE_NONE;
        let _split_layer = llama_split_mode::LLAMA_SPLIT_MODE_LAYER;
        let _split_row = llama_split_mode::LLAMA_SPLIT_MODE_ROW;

        // RoPE scaling types
        let _scaling_unspecified = llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
        let _scaling_none = llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_NONE;
        let _scaling_linear = llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_LINEAR;
        let _scaling_yarn = llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_YARN;
        let _scaling_longrope = llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_LONGROPE;

        // All enums should be accessible
    }
}

#[cfg(test)]
mod version_compatibility_tests {
    use super::*;
    use sys::*;

    #[test]
    fn test_version_constants() {
        // Test that version constants are defined
        assert_eq!(LLAMA_SESSION_VERSION, 9);
        assert_eq!(LLAMA_STATE_SEQ_VERSION, 2);

        // Test other important constants
        assert_eq!(LLAMA_TOKEN_NULL, -1);
        assert_eq!(LLAMA_DEFAULT_SEED, 0xFFFFFFFF);
    }

    #[test]
    fn test_abi_compatibility_indicators() {
        // Test struct sizes to catch ABI changes
        use std::mem::size_of;

        // Basic types should have expected sizes
        assert_eq!(size_of::<llama_token>(), 4);
        assert_eq!(size_of::<llama_pos>(), 4);
        assert_eq!(size_of::<llama_seq_id>(), 4);

        // Enum sizes should be consistent
        assert_eq!(size_of::<llama_vocab_type>(), 4);
        assert_eq!(size_of::<llama_rope_type>(), 4);
        assert_eq!(size_of::<llama_ftype>(), 4);
    }

    #[test]
    fn test_feature_flags() {
        // Test that feature detection works
        unsafe {
            let _supports_mmap = llama_supports_mmap();
            let _supports_mlock = llama_supports_mlock();
            let _supports_gpu = llama_supports_gpu_offload();
            let _supports_rpc = llama_supports_rpc();
        }

        // These should return consistent boolean values
    }
}