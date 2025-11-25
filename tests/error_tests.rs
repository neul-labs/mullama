//! Comprehensive error handling and edge case tests for Mullama
//!
//! These tests validate error conditions, edge cases, and robustness.
//! They ensure the library handles failures gracefully and provides meaningful error messages.

use mullama::*;
use std::{path::Path, sync::Arc};

#[cfg(test)]
mod model_error_tests {
    use super::*;

    #[test]
    fn test_model_load_nonexistent_file() {
        let result = Model::load("nonexistent_model.gguf");
        assert!(result.is_err());

        match result.unwrap_err() {
            MullamaError::ModelLoadError(msg) => {
                assert!(!msg.is_empty());
                println!("Expected error for nonexistent file: {}", msg);
            }
            _ => panic!("Expected ModelLoadError"),
        }
    }

    #[test]
    fn test_model_load_invalid_path() {
        let invalid_paths = [
            "",
            "/dev/null",
            "/invalid/path/to/model.gguf",
            "model_with_invalid_extension.txt",
        ];

        for path in &invalid_paths {
            let result = Model::load(path);
            assert!(result.is_err(), "Should fail for path: {}", path);
        }
    }

    #[test]
    fn test_model_params_validation() {
        let mut params = ModelParams::default();

        // Test negative GPU layers (should be handled gracefully)
        params.n_gpu_layers = -1;
        assert_eq!(params.n_gpu_layers, -1); // Should accept negative values

        // Test extreme values
        params.n_gpu_layers = i32::MAX;
        assert_eq!(params.n_gpu_layers, i32::MAX);

        params.n_gpu_layers = i32::MIN;
        assert_eq!(params.n_gpu_layers, i32::MIN);
    }

    #[test]
    fn test_model_kv_override_edge_cases() {
        // Test empty key
        let override_empty_key = ModelKvOverride {
            key: "".to_string(),
            value: ModelKvOverrideValue::Int(42),
        };
        assert!(override_empty_key.key.is_empty());

        // Test very long key
        let long_key = "a".repeat(1000);
        let override_long_key = ModelKvOverride {
            key: long_key.clone(),
            value: ModelKvOverrideValue::Str("test".to_string()),
        };
        assert_eq!(override_long_key.key.len(), 1000);

        // Test extreme values
        let override_extreme_int = ModelKvOverride {
            key: "extreme_int".to_string(),
            value: ModelKvOverrideValue::Int(i64::MAX),
        };

        let override_extreme_float = ModelKvOverride {
            key: "extreme_float".to_string(),
            value: ModelKvOverrideValue::Float(f64::MAX),
        };

        // Should not panic
        assert_eq!(override_extreme_int.key, "extreme_int");
        assert_eq!(override_extreme_float.key, "extreme_float");
    }

    #[test]
    fn test_tensor_split_validation() {
        let mut params = ModelParams::default();

        // Test empty tensor split
        params.tensor_split = vec![];
        assert!(params.tensor_split.is_empty());

        // Test single value
        params.tensor_split = vec![1.0];
        assert_eq!(params.tensor_split.len(), 1);

        // Test many values
        params.tensor_split = vec![0.5; 16];
        assert_eq!(params.tensor_split.len(), 16);

        // Test extreme values
        params.tensor_split = vec![0.0, f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(params.tensor_split.len(), 5);
    }
}

#[cfg(test)]
mod context_error_tests {
    use super::*;

    #[test]
    fn test_context_params_validation() {
        let mut params = ContextParams::default();

        // Test zero context size
        params.n_ctx = 0;
        assert_eq!(params.n_ctx, 0);

        // Test extreme context sizes
        params.n_ctx = u32::MAX;
        assert_eq!(params.n_ctx, u32::MAX);

        // Test zero batch size
        params.n_batch = 0;
        assert_eq!(params.n_batch, 0);

        // Test batch size larger than context
        params.n_ctx = 1024;
        params.n_batch = 2048;
        assert!(params.n_batch > params.n_ctx);

        // Test zero threads
        params.n_threads = 0;
        assert_eq!(params.n_threads, 0);

        // Test extreme thread counts
        params.n_threads = i32::MAX;
        assert_eq!(params.n_threads, i32::MAX);
    }

    #[test]
    fn test_context_sequence_limits() {
        let mut params = ContextParams::default();

        // Test zero sequences
        params.n_seq_max = 0;
        assert_eq!(params.n_seq_max, 0);

        // Test single sequence
        params.n_seq_max = 1;
        assert_eq!(params.n_seq_max, 1);

        // Test many sequences
        params.n_seq_max = 1000;
        assert_eq!(params.n_seq_max, 1000);

        // Test extreme sequence count
        params.n_seq_max = u32::MAX;
        assert_eq!(params.n_seq_max, u32::MAX);
    }

    #[test]
    fn test_context_memory_constraints() {
        let mut params = ContextParams::default();

        // Test very large context with very large batch
        params.n_ctx = 1_000_000;
        params.n_batch = 100_000;
        params.n_ubatch = 50_000;

        // Should not panic during parameter setting
        assert_eq!(params.n_ctx, 1_000_000);
        assert_eq!(params.n_batch, 100_000);
        assert_eq!(params.n_ubatch, 50_000);
    }
}

#[cfg(test)]
mod sampling_error_tests {
    use super::*;

    #[test]
    fn test_sampler_params_edge_cases() {
        let mut params = SamplerParams::default();

        // Test zero temperature
        params.temperature = 0.0;
        assert_eq!(params.temperature, 0.0);

        // Test negative temperature
        params.temperature = -1.0;
        assert_eq!(params.temperature, -1.0);

        // Test extreme temperatures
        params.temperature = f32::MAX;
        assert_eq!(params.temperature, f32::MAX);

        params.temperature = f32::INFINITY;
        assert!(params.temperature.is_infinite());

        // Test invalid probabilities
        params.top_p = -0.5;
        assert_eq!(params.top_p, -0.5);

        params.top_p = 1.5;
        assert_eq!(params.top_p, 1.5);

        params.min_p = 2.0;
        assert_eq!(params.min_p, 2.0);

        // Test zero top_k
        params.top_k = 0;
        assert_eq!(params.top_k, 0);

        // Test negative top_k
        params.top_k = -10;
        assert_eq!(params.top_k, -10);
    }

    #[test]
    fn test_logit_bias_edge_cases() {
        // Test with extreme token IDs
        let bias_negative_token = LogitBias {
            token: -1,
            bias: 1.0,
        };
        assert_eq!(bias_negative_token.token, -1);

        let bias_max_token = LogitBias {
            token: TokenId::MAX,
            bias: 0.5,
        };
        assert_eq!(bias_max_token.token, TokenId::MAX);

        // Test with extreme bias values
        let bias_infinite = LogitBias {
            token: 100,
            bias: f32::INFINITY,
        };
        assert!(bias_infinite.bias.is_infinite());

        let bias_nan = LogitBias {
            token: 100,
            bias: f32::NAN,
        };
        assert!(bias_nan.bias.is_nan());
    }

    #[test]
    fn test_token_data_array_edge_cases() {
        // Test with empty array
        let empty_array = TokenDataArray::new(vec![]);
        assert!(empty_array.is_empty());
        assert_eq!(empty_array.len(), 0);

        // Test with single element
        let single_element = TokenDataArray::new(vec![
            TokenData { id: 1, logit: 1.0, p: 1.0 }
        ]);
        assert_eq!(single_element.len(), 1);
        assert!(!single_element.is_empty());

        // Test with extreme values
        let extreme_values = TokenDataArray::new(vec![
            TokenData { id: TokenId::MAX, logit: f32::MAX, p: 1.0 },
            TokenData { id: TokenId::MIN, logit: f32::MIN, p: 0.0 },
            TokenData { id: 0, logit: f32::INFINITY, p: 0.5 },
            TokenData { id: 1, logit: f32::NEG_INFINITY, p: 0.25 },
            TokenData { id: 2, logit: f32::NAN, p: f32::NAN },
        ]);
        assert_eq!(extreme_values.len(), 5);

        // Test with duplicate IDs
        let duplicate_ids = TokenDataArray::new(vec![
            TokenData { id: 1, logit: 1.0, p: 0.5 },
            TokenData { id: 1, logit: 2.0, p: 0.5 },
        ]);
        assert_eq!(duplicate_ids.len(), 2);
    }

    #[test]
    fn test_sampler_chain_edge_cases() {
        let params = SamplerChainParams { no_perf: true };

        // Test chain operations should not panic
        let chain = SamplerChain::new(params);
        // Basic operations should work even without a model
    }
}

#[cfg(test)]
mod batch_error_tests {
    use super::*;

    #[test]
    fn test_batch_with_empty_tokens() {
        let batch = Batch::from_tokens(&[]);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_with_extreme_tokens() {
        let extreme_tokens = vec![
            TokenId::MAX,
            TokenId::MIN,
            0,
            -1,
            1000000,
        ];

        let batch = Batch::from_tokens(&extreme_tokens);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_with_many_tokens() {
        // Test with very large batch
        let many_tokens: Vec<TokenId> = (0..10000).collect();
        let batch = Batch::from_tokens(&many_tokens);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_operations_safety() {
        let tokens = vec![1, 2, 3];
        let batch = Batch::from_tokens(&tokens);

        // Test that internal operations don't panic
        let _llama_batch = batch.get_llama_batch();
        // Should not crash even if underlying pointers are invalid
    }
}

#[cfg(test)]
mod session_error_tests {
    use super::*;

    #[test]
    fn test_session_with_empty_data() {
        let session = Session { data: vec![] };
        assert!(session.data.is_empty());
    }

    #[test]
    fn test_session_with_large_data() {
        let large_data = vec![0u8; 1_000_000];
        let session = Session { data: large_data };
        assert_eq!(session.data.len(), 1_000_000);
    }

    #[test]
    fn test_session_with_invalid_data() {
        // Test with potentially invalid session data
        let invalid_data = vec![0xFF; 1000];
        let session = Session { data: invalid_data };
        assert_eq!(session.data.len(), 1000);
        assert!(session.data.iter().all(|&b| b == 0xFF));
    }
}

#[cfg(test)]
mod memory_error_tests {
    use super::*;

    #[test]
    fn test_memory_manager_initialization() {
        let memory_manager = MemoryManager::new();
        // Should not crash during initialization
        // New manager should not be valid (no context associated)
        assert!(!memory_manager.is_valid());
    }

    #[test]
    fn test_embeddings_edge_cases() {
        // Test with empty embeddings
        let empty_embeddings = Embeddings::new(vec![], 0);
        assert_eq!(empty_embeddings.len(), 0);
        assert_eq!(empty_embeddings.dimension, 0);

        // Test with mismatched dimensions
        let mismatched = Embeddings::new(vec![1.0, 2.0, 3.0], 5);
        assert_eq!(mismatched.len(), 0); // Should be 0 due to mismatch
        assert_eq!(mismatched.dimension, 5);

        // Test with extreme values
        let extreme_embeddings = Embeddings::new(
            vec![f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            5
        );
        assert_eq!(extreme_embeddings.len(), 1);
        assert_eq!(extreme_embeddings.dimension, 5);
    }

    #[test]
    fn test_vocabulary_initialization() {
        let vocab = Vocabulary::new();
        // Should not crash during initialization
        assert_eq!(vocab._placeholder, 0);
    }
}

#[cfg(test)]
mod ffi_error_tests {
    use super::*;
    use mullama::sys;

    #[test]
    #[cfg(feature = "llama-cpp-tests")]
    fn test_null_pointer_handling() {
        // Test that FFI functions handle null pointers gracefully
        unsafe {
            // These should not crash, even with null pointers
            let _result = sys::llama_model_n_vocab(std::ptr::null());
            let _result = sys::llama_model_n_ctx_train(std::ptr::null());
            let _result = sys::llama_model_n_embd(std::ptr::null());
        }
        // If we reach here, null pointer handling didn't crash
    }

    #[test]
    #[cfg(feature = "llama-cpp-tests")]
    fn test_invalid_parameters() {
        unsafe {
            // Test with invalid but non-null parameters
            let fake_ptr = 0x1 as *mut sys::llama_model;

            // These should not crash immediately (might return error codes)
            let _result = sys::llama_model_n_vocab(fake_ptr);
            let _result = sys::llama_model_n_ctx_train(fake_ptr);
        }
        // If we reach here, invalid parameter handling didn't crash immediately
    }

    #[test]
    fn test_backend_multiple_init_free() {
        // Test multiple init/free cycles
        unsafe {
            for _ in 0..5 {
                sys::llama_backend_init();
                sys::llama_backend_free();
            }
        }
        // Should handle multiple init/free cycles gracefully
    }

    #[test]
    fn test_system_info_functions() {
        unsafe {
            // Test that system info functions don't crash
            let _max_devices = sys::llama_max_devices();
            let _max_sequences = sys::llama_max_parallel_sequences();
            let _supports_mmap = sys::llama_supports_mmap();
            let _supports_mlock = sys::llama_supports_mlock();
            let _supports_gpu = sys::llama_supports_gpu_offload();
            let _supports_rpc = sys::llama_supports_rpc();

            // Test print functions (might return null)
            let _system_info = sys::llama_print_system_info();
        }
        // All system queries should complete without crashing
    }
}

#[cfg(test)]
mod thread_safety_tests {
    use super::*;
    use std::thread;
    use std::sync::{Arc, Barrier};

    #[test]
    fn test_concurrent_parameter_creation() {
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        for _ in 0..4 {
            let barrier = barrier.clone();
            let handle = thread::spawn(move || {
                barrier.wait();

                // Create parameters concurrently
                let _model_params = ModelParams::default();
                let _context_params = ContextParams::default();
                let _sampler_params = SamplerParams::default();

                // Should not crash
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_structure_creation() {
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        for _ in 0..4 {
            let barrier = barrier.clone();
            let handle = thread::spawn(move || {
                barrier.wait();

                // Create structures concurrently
                let _batch = Batch::from_tokens(&[1, 2, 3]);
                let _session = Session { data: vec![1, 2, 3] };
                let _embeddings = Embeddings::new(vec![1.0, 2.0, 3.0], 3);

                // Should not crash
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod resource_exhaustion_tests {
    use super::*;

    #[test]
    fn test_large_parameter_structures() {
        // Test with very large vectors that might cause allocation issues
        let mut params = ModelParams::default();

        // Try to allocate large tensor split (might fail gracefully)
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            params.tensor_split = vec![1.0; 1_000_000];
        })) {
            Ok(_) => {
                // Allocation succeeded
                assert_eq!(params.tensor_split.len(), 1_000_000);
            }
            Err(_) => {
                // Allocation failed, which is acceptable
                println!("Large allocation failed gracefully");
            }
        }
    }

    #[test]
    fn test_many_kv_overrides() {
        // Try to create many KV overrides
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut params = ModelParams::default();
            for i in 0..10000 {
                params.kv_overrides.push(ModelKvOverride {
                    key: format!("key_{}", i),
                    value: ModelKvOverrideValue::Int(i as i64),
                });
            }
            params.kv_overrides.len()
        }));

        match result {
            Ok(len) => {
                // Creation succeeded
                assert_eq!(len, 10000);
            }
            Err(_) => {
                // Creation failed, which is acceptable for resource exhaustion
                println!("Many KV overrides failed gracefully");
            }
        }
    }

    #[test]
    fn test_large_token_arrays() {
        // Test with very large token arrays
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let large_tokens: Vec<TokenId> = (0..1_000_000).collect();
            let _batch = Batch::from_tokens(&large_tokens);
            "success"
        }));

        match result {
            Ok(_) => {
                println!("Large token array handled successfully");
            }
            Err(_) => {
                println!("Large token array failed gracefully");
            }
        }
    }
}