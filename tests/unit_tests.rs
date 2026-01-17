//! Comprehensive unit tests for Mullama
//!
//! These tests validate the complete implementation without requiring actual model files.
//! They test data structures, parameter validation, type safety, and API consistency.

use mullama::*;
use std::sync::Arc;

#[cfg(test)]
mod ffi_tests {
    use super::*;
    use mullama::sys;

    #[test]
    fn test_backend_initialization() {
        // Test that backend can be initialized and freed safely
        unsafe {
            sys::llama_backend_init();
            sys::llama_backend_free();
        }
        // Should not crash
    }

    #[test]
    fn test_system_capabilities() {
        // Test system capability queries
        unsafe {
            let max_devices = sys::llama_max_devices();
            let max_sequences = sys::llama_max_parallel_sequences();
            let supports_mmap = sys::llama_supports_mmap();
            let supports_mlock = sys::llama_supports_mlock();
            let supports_gpu = sys::llama_supports_gpu_offload();
            let supports_rpc = sys::llama_supports_rpc();

            // These should return reasonable values
            assert!(max_devices >= 1);
            assert!(max_sequences >= 1);
            // Boolean values are inherently valid
            println!("System capabilities - devices: {}, sequences: {}, mmap: {}, mlock: {}, gpu: {}, rpc: {}",
                    max_devices, max_sequences, supports_mmap, supports_mlock, supports_gpu, supports_rpc);
        }
    }

    #[test]
    fn test_constants() {
        // Test that constants are defined correctly
        assert_eq!(sys::LLAMA_TOKEN_NULL, -1);
        assert_eq!(sys::LLAMA_DEFAULT_SEED, 0xFFFFFFFF);
        assert_eq!(sys::LLAMA_SESSION_VERSION, 9);
        assert_eq!(sys::LLAMA_STATE_SEQ_VERSION, 2);
    }

    #[test]
    fn test_enum_values() {
        // Test that enums have expected values
        assert_eq!(sys::llama_vocab_type::LLAMA_VOCAB_TYPE_NONE as i32, 0);
        assert_eq!(sys::llama_vocab_type::LLAMA_VOCAB_TYPE_SPM as i32, 1);
        assert_eq!(sys::llama_vocab_type::LLAMA_VOCAB_TYPE_BPE as i32, 2);

        assert_eq!(sys::llama_rope_type::LLAMA_ROPE_TYPE_NONE as i32, -1);
        assert_eq!(sys::llama_rope_type::LLAMA_ROPE_TYPE_NORM as i32, 0);

        assert_eq!(sys::llama_ftype::LLAMA_FTYPE_ALL_F32 as i32, 0);
        assert_eq!(sys::llama_ftype::LLAMA_FTYPE_MOSTLY_F16 as i32, 1);
    }

    #[test]
    fn test_type_sizes() {
        // Ensure types have expected sizes
        assert_eq!(std::mem::size_of::<sys::llama_token>(), 4);
        assert_eq!(std::mem::size_of::<sys::llama_pos>(), 4);
        assert_eq!(std::mem::size_of::<sys::llama_seq_id>(), 4);

        // Test struct sizes are reasonable
        let model_params_size = std::mem::size_of::<sys::llama_model_params>();
        let context_params_size = std::mem::size_of::<sys::llama_context_params>();
        let batch_size = std::mem::size_of::<sys::llama_batch>();

        assert!(model_params_size > 20); // Should have multiple fields
        assert!(context_params_size > 50); // Should have many fields
        assert!(batch_size > 20); // Should have multiple pointers

        println!(
            "Struct sizes - model_params: {}, context_params: {}, batch: {}",
            model_params_size, context_params_size, batch_size
        );
    }
}

#[cfg(test)]
mod parameter_tests {
    use super::*;

    #[test]
    fn test_model_params_default() {
        let params = ModelParams::default();
        assert_eq!(params.n_gpu_layers, 0);
        assert_eq!(params.use_mmap, true);
        assert_eq!(params.use_mlock, false);
        assert_eq!(params.check_tensors, true);
        assert_eq!(params.vocab_only, false);
        assert_eq!(params.use_extra_bufts, false);
        assert!(params.tensor_split.is_empty());
        assert!(params.kv_overrides.is_empty());
    }

    #[test]
    fn test_context_params_default() {
        let params = ContextParams::default();
        assert_eq!(params.n_ctx, 0); // Use model default
        assert_eq!(params.n_batch, 2048);
        assert_eq!(params.n_ubatch, 512);
        assert_eq!(params.n_seq_max, 1);
        assert!(params.n_threads > 0);
        assert!(params.n_threads_batch > 0);
        assert_eq!(params.embeddings, false);
        assert_eq!(
            params.flash_attn_type,
            mullama::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO
        );
        assert_eq!(params.offload_kqv, true);
        assert_eq!(params.swa_full, true);
        assert_eq!(params.kv_unified, false);
    }

    #[test]
    fn test_sampler_params_default() {
        let params = SamplerParams::default();
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
    fn test_sampler_chain_params() {
        let params = SamplerChainParams::default();
        assert_eq!(params.no_perf, false);

        let custom_params = SamplerChainParams { no_perf: true };
        assert_eq!(custom_params.no_perf, true);
    }

    #[test]
    fn test_model_kv_override() {
        let override_int = ModelKvOverride {
            key: "test_key".to_string(),
            value: ModelKvOverrideValue::Int(42),
        };

        let override_float = ModelKvOverride {
            key: "test_key".to_string(),
            value: ModelKvOverrideValue::Float(3.14),
        };

        let override_bool = ModelKvOverride {
            key: "test_key".to_string(),
            value: ModelKvOverrideValue::Bool(true),
        };

        let override_str = ModelKvOverride {
            key: "test_key".to_string(),
            value: ModelKvOverrideValue::Str("test_value".to_string()),
        };

        // Should not panic when created
        assert_eq!(override_int.key, "test_key");
        assert_eq!(override_float.key, "test_key");
        assert_eq!(override_bool.key, "test_key");
        assert_eq!(override_str.key, "test_key");
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::*;

    #[test]
    fn test_logit_bias_creation() {
        let bias = LogitBias {
            token: 123,
            bias: 0.5,
        };
        assert_eq!(bias.token, 123);
        assert_eq!(bias.bias, 0.5);
    }

    #[test]
    fn test_token_data_creation() {
        let token_data = TokenData {
            id: 456,
            logit: 2.5,
            p: 0.8,
        };
        assert_eq!(token_data.id, 456);
        assert_eq!(token_data.logit, 2.5);
        assert_eq!(token_data.p, 0.8);
    }

    #[test]
    fn test_token_data_array_creation() {
        let candidates = vec![
            TokenData {
                id: 1,
                logit: 1.0,
                p: 0.5,
            },
            TokenData {
                id: 2,
                logit: 2.0,
                p: 0.3,
            },
            TokenData {
                id: 3,
                logit: 0.5,
                p: 0.2,
            },
        ];

        let array = TokenDataArray::new(candidates.clone());
        assert_eq!(array.len(), 3);
        assert!(!array.is_empty());
        assert_eq!(array.selected(), None);
        assert!(!array.is_sorted());

        // Test candidates access
        let retrieved_candidates = array.candidates();
        assert_eq!(retrieved_candidates.len(), 3);
    }

    #[test]
    fn test_sampler_perf_data() {
        let perf = SamplerPerfData {
            t_sample_ms: 15.5,
            n_sample: 100,
        };
        assert_eq!(perf.t_sample_ms, 15.5);
        assert_eq!(perf.n_sample, 100);
    }

    #[test]
    fn test_empty_token_data_array() {
        let array = TokenDataArray::new(vec![]);
        assert_eq!(array.len(), 0);
        assert!(array.is_empty());
        assert_eq!(array.selected(), None);
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let model_error = MullamaError::ModelLoadError("Test error".to_string());
        let context_error = MullamaError::ContextError("Test error".to_string());
        let generation_error = MullamaError::GenerationError("Test error".to_string());
        let tokenization_error = MullamaError::TokenizationError("Test error".to_string());
        let sampling_error = MullamaError::SamplingError("Test error".to_string());
        let session_error = MullamaError::SessionError("Test error".to_string());

        // Test that errors can be created and displayed
        println!("Model error: {}", model_error);
        println!("Context error: {}", context_error);
        println!("Generation error: {}", generation_error);
        println!("Tokenization error: {}", tokenization_error);
        println!("Sampling error: {}", sampling_error);
        println!("Session error: {}", session_error);

        // Test error trait implementation
        assert!(std::error::Error::source(&model_error).is_none());
    }

    #[test]
    fn test_error_display() {
        let error = MullamaError::ModelLoadError("File not found".to_string());
        let error_string = format!("{}", error);
        assert!(error_string.contains("File not found"));
    }

    #[test]
    fn test_error_debug() {
        let error = MullamaError::ContextError("Invalid parameters".to_string());
        let debug_string = format!("{:?}", error);
        assert!(debug_string.contains("ContextError"));
        assert!(debug_string.contains("Invalid parameters"));
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    #[test]
    fn test_batch_from_tokens() {
        let tokens = vec![1, 2, 3, 4, 5];
        let batch = Batch::from_tokens(&tokens);

        // Test that batch was created successfully
        // We can't test internal structure without model, but creation should work
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_empty_batch() {
        let batch = Batch::from_tokens(&[]);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_operations() {
        let tokens = vec![10, 20, 30];
        let batch = Batch::from_tokens(&tokens);

        // Test that we can call methods without panicking
        let _llama_batch = batch.get_llama_batch();
        // Should not crash
    }
}

#[cfg(test)]
mod type_safety_tests {
    use super::*;

    #[test]
    fn test_token_id_type() {
        let token: TokenId = 12345;
        assert_eq!(token, 12345);

        // Test conversion safety
        let sys_token = token as sys::llama_token;
        let back_token = sys_token as TokenId;
        assert_eq!(token, back_token);
    }

    #[test]
    fn test_type_aliases() {
        // Test that type aliases work correctly
        let pos: sys::llama_pos = 100;
        let seq_id: sys::llama_seq_id = 5;
        let token: sys::llama_token = 999;

        assert_eq!(pos, 100);
        assert_eq!(seq_id, 5);
        assert_eq!(token, 999);
    }

    #[test]
    fn test_enum_conversions() {
        // Test enum type safety
        let vocab_type = sys::llama_vocab_type::LLAMA_VOCAB_TYPE_BPE;
        let rope_type = sys::llama_rope_type::LLAMA_ROPE_TYPE_NORM;
        let ftype = sys::llama_ftype::LLAMA_FTYPE_MOSTLY_F16;

        // Should be able to convert to/from integers safely
        assert_eq!(vocab_type as i32, 2);
        assert_eq!(rope_type as i32, 0);
        assert_eq!(ftype as i32, 1);
    }

    #[test]
    fn test_pointer_types() {
        // Test that pointer types are properly defined
        let model_ptr: *mut sys::llama_model = std::ptr::null_mut();
        let context_ptr: *mut sys::llama_context = std::ptr::null_mut();
        let sampler_ptr: *mut sys::llama_sampler = std::ptr::null_mut();
        let vocab_ptr: *const sys::llama_vocab = std::ptr::null();

        assert!(model_ptr.is_null());
        assert!(context_ptr.is_null());
        assert!(sampler_ptr.is_null());
        assert!(vocab_ptr.is_null());
    }
}

#[cfg(test)]
mod api_consistency_tests {
    use super::*;

    #[test]
    fn test_parameter_consistency() {
        // Test that default parameters are consistent across types
        let model_params = ModelParams::default();
        let context_params = ContextParams::default();
        let sampler_params = SamplerParams::default();

        // GPU layers should be disabled by default for compatibility
        assert_eq!(model_params.n_gpu_layers, 0);

        // Context should use reasonable defaults
        assert!(context_params.n_batch > 0);
        assert!(context_params.n_ubatch > 0);
        assert!(context_params.n_threads > 0);

        // Sampler should have sensible defaults
        assert!(sampler_params.temperature > 0.0);
        assert!(sampler_params.top_k > 0);
        assert!(sampler_params.top_p > 0.0 && sampler_params.top_p <= 1.0);
    }

    #[test]
    fn test_thread_count_consistency() {
        let params = ContextParams::default();

        // Thread counts should be reasonable
        assert!(params.n_threads >= 1);
        assert!(params.n_threads_batch >= 1);
        assert!(params.n_threads <= 256); // Reasonable upper bound
        assert!(params.n_threads_batch <= 256);
    }

    #[test]
    fn test_batch_size_consistency() {
        let params = ContextParams::default();

        // Batch sizes should be reasonable
        assert!(params.n_batch >= params.n_ubatch);
        assert!(params.n_ubatch > 0);
        assert!(params.n_batch <= 32768); // Reasonable upper bound
    }

    #[test]
    fn test_sampling_parameter_ranges() {
        let params = SamplerParams::default();

        // Test parameter ranges are sensible
        assert!(params.temperature >= 0.0);
        assert!(params.top_p > 0.0 && params.top_p <= 1.0);
        assert!(params.min_p >= 0.0 && params.min_p <= 1.0);
        assert!(params.typical_p >= 0.0);
        assert!(params.penalty_repeat >= 0.0);
        assert!(params.penalty_last_n >= 0);
    }
}

#[cfg(test)]
mod documentation_tests {
    use super::*;

    #[test]
    fn test_prelude_module() {
        // Test that prelude exports work
        use mullama::prelude::*;

        let _model_params = ModelParams::default();
        let _context_params = ContextParams::default();
        let _sampler_params = SamplerParams::default();

        // Should compile without issues
    }

    #[test]
    fn test_sys_exports() {
        // Test that sys types are exported
        let _vocab_type: llama_vocab_type = llama_vocab_type::LLAMA_VOCAB_TYPE_BPE;
        let _rope_type: llama_rope_type = llama_rope_type::LLAMA_ROPE_TYPE_NORM;
        let _token_null: llama_token = LLAMA_TOKEN_NULL;
        let _default_seed: u32 = LLAMA_DEFAULT_SEED;

        // Should compile without issues
    }

    #[test]
    fn test_error_handling_patterns() {
        // Test common error handling patterns
        fn example_function() -> Result<(), MullamaError> {
            // Simulate potential failure
            if false {
                Err(MullamaError::ModelLoadError("Test".to_string()))
            } else {
                Ok(())
            }
        }

        let result = example_function();
        assert!(result.is_ok());
    }
}
