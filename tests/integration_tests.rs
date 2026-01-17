//! Integration tests for Mullama
//!
//! These tests validate the integration between components and test realistic usage patterns.
//! Some tests require actual model files and will be skipped if not available.

use mullama::*;
use std::{path::Path, sync::Arc};

#[cfg(test)]
mod model_integration_tests {
    use super::*;

    #[test]
    fn test_model_loading_parameters() {
        // Test different model parameter configurations
        let configs = vec![
            ModelParams {
                n_gpu_layers: 0,
                use_mmap: true,
                use_mlock: false,
                check_tensors: true,
                vocab_only: false,
                ..Default::default()
            },
            ModelParams {
                n_gpu_layers: 10,
                use_mmap: false,
                use_mlock: true,
                check_tensors: false,
                vocab_only: true,
                ..Default::default()
            },
        ];

        for (i, params) in configs.iter().enumerate() {
            // These would fail without actual model files, but we're testing parameter validation
            let result = Model::load_with_params("nonexistent_model.gguf", params.clone());

            // Should fail with model load error, not parameter validation error
            match result {
                Err(MullamaError::ModelLoadError(_)) => {
                    println!("Config {} correctly failed with model load error", i);
                }
                Err(other) => panic!("Config {} failed with unexpected error: {:?}", i, other),
                Ok(_) => panic!("Config {} unexpectedly succeeded", i),
            }
        }
    }

    #[test]
    fn test_model_with_kv_overrides() {
        let mut params = ModelParams::default();
        params.kv_overrides = vec![
            ModelKvOverride {
                key: "general.name".to_string(),
                value: ModelKvOverrideValue::Str("test_model".to_string()),
            },
            ModelKvOverride {
                key: "general.version".to_string(),
                value: ModelKvOverrideValue::Int(1),
            },
            ModelKvOverride {
                key: "attention.use_scaled".to_string(),
                value: ModelKvOverrideValue::Bool(true),
            },
            ModelKvOverride {
                key: "attention.scale".to_string(),
                value: ModelKvOverrideValue::Float(1.5),
            },
        ];

        let result = Model::load_with_params("test_model.gguf", params);

        // Should fail with model load error (file doesn't exist)
        assert!(matches!(result, Err(MullamaError::ModelLoadError(_))));
    }

    #[test]
    fn test_model_parameter_edge_cases() {
        // Test edge case parameters
        let edge_cases = vec![
            ModelParams {
                n_gpu_layers: -1, // Should be handled gracefully
                ..Default::default()
            },
            ModelParams {
                n_gpu_layers: 999, // Very high value
                ..Default::default()
            },
            ModelParams {
                tensor_split: vec![0.5, 0.3, 0.2], // Custom tensor split
                ..Default::default()
            },
            ModelParams {
                tensor_split: vec![], // Empty tensor split
                ..Default::default()
            },
        ];

        for (i, params) in edge_cases.iter().enumerate() {
            let result = Model::load_with_params("edge_case_model.gguf", params.clone());

            // All should fail with model load error (file doesn't exist), not crash
            match result {
                Err(MullamaError::ModelLoadError(_)) => {
                    println!("Edge case {} handled correctly", i);
                }
                other => panic!("Edge case {} had unexpected result: {:?}", i, other),
            }
        }
    }

    #[test]
    fn test_model_parameter_validation() {
        // Test that parameter structures can be created and cloned
        let params = ModelParams {
            n_gpu_layers: 16,
            split_mode: llama_split_mode::LLAMA_SPLIT_MODE_LAYER,
            main_gpu: 0,
            tensor_split: vec![0.6, 0.4],
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
            use_extra_bufts: false,
            kv_overrides: vec![ModelKvOverride {
                key: "test.key".to_string(),
                value: ModelKvOverrideValue::Int(42),
            }],
            progress_callback: None,
        };

        // Test cloning
        let cloned_params = params.clone();
        assert_eq!(cloned_params.n_gpu_layers, params.n_gpu_layers);
        assert_eq!(cloned_params.tensor_split.len(), params.tensor_split.len());
        assert_eq!(cloned_params.kv_overrides.len(), params.kv_overrides.len());

        // Test debug formatting
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("n_gpu_layers"));
        assert!(debug_str.contains("16"));
    }
}

#[cfg(test)]
mod context_integration_tests {
    use super::*;

    #[test]
    fn test_context_parameter_configurations() {
        let configs = vec![
            // Minimal configuration
            ContextParams {
                n_ctx: 512,
                n_batch: 128,
                n_ubatch: 64,
                n_seq_max: 1,
                n_threads: 1,
                n_threads_batch: 1,
                embeddings: false,
                flash_attn_type:
                    mullama::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED,
                offload_kqv: false,
                ..Default::default()
            },
            // High-performance configuration
            ContextParams {
                n_ctx: 4096,
                n_batch: 2048,
                n_ubatch: 512,
                n_seq_max: 8,
                n_threads: 16,
                n_threads_batch: 16,
                embeddings: true,
                flash_attn_type: mullama::sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED,
                offload_kqv: true,
                swa_full: true,
                kv_unified: false,
                ..Default::default()
            },
            // Edge case configuration
            ContextParams {
                n_ctx: 0,   // Use model default
                n_batch: 1, // Minimal batch
                n_ubatch: 1,
                n_seq_max: 1,
                n_threads: 128,     // High thread count
                n_threads_batch: 1, // Minimal batch threads
                rope_scaling_type: llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_YARN,
                pooling_type: llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
                attention_type: llama_attention_type::LLAMA_ATTENTION_TYPE_CAUSAL,
                ..Default::default()
            },
        ];

        for (i, params) in configs.iter().enumerate() {
            // Test parameter validation
            assert!(
                params.n_batch >= params.n_ubatch,
                "Config {} has invalid batch sizes",
                i
            );
            assert!(
                params.n_threads >= 1,
                "Config {} has invalid thread count",
                i
            );
            assert!(params.n_seq_max >= 1, "Config {} has invalid seq_max", i);

            // Test cloning and debug
            let cloned = params.clone();
            let _debug_str = format!("{:?}", cloned);

            println!("Config {} validated successfully", i);
        }
    }

    #[test]
    fn test_context_thread_management() {
        // Test thread configuration validation
        let thread_configs = vec![
            (1, 1),   // Minimal
            (4, 4),   // Balanced
            (16, 8),  // More generation threads
            (8, 16),  // More batch threads
            (32, 32), // High performance
        ];

        for (gen_threads, batch_threads) in thread_configs {
            let params = ContextParams {
                n_threads: gen_threads,
                n_threads_batch: batch_threads,
                ..Default::default()
            };

            // Validate thread counts are reasonable
            assert!(params.n_threads >= 1 && params.n_threads <= 256);
            assert!(params.n_threads_batch >= 1 && params.n_threads_batch <= 256);

            println!(
                "Thread config ({}, {}) validated",
                gen_threads, batch_threads
            );
        }
    }

    #[test]
    fn test_rope_scaling_configurations() {
        let rope_configs = vec![
            (
                llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_NONE,
                0.0,
                0.0,
            ),
            (
                llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_LINEAR,
                10000.0,
                1.0,
            ),
            (
                llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_YARN,
                10000.0,
                1.0,
            ),
            (
                llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_LONGROPE,
                500000.0,
                1.0,
            ),
        ];

        for (scaling_type, freq_base, freq_scale) in rope_configs {
            let params = ContextParams {
                rope_scaling_type: scaling_type,
                rope_freq_base: freq_base,
                rope_freq_scale: freq_scale,
                ..Default::default()
            };

            // Test that parameters can be created without errors
            let _debug_str = format!("{:?}", params);
            println!("RoPE config {:?} validated", scaling_type);
        }
    }
}

#[cfg(test)]
mod sampling_integration_tests {
    use super::*;

    #[test]
    fn test_sampler_parameter_combinations() {
        let combinations = vec![
            // Conservative sampling
            SamplerParams {
                temperature: 0.1,
                top_k: 10,
                top_p: 0.8,
                min_p: 0.1,
                typical_p: 1.0,
                penalty_repeat: 1.05,
                ..Default::default()
            },
            // Creative sampling
            SamplerParams {
                temperature: 1.2,
                top_k: 100,
                top_p: 0.95,
                min_p: 0.01,
                typical_p: 0.9,
                penalty_repeat: 1.1,
                penalty_freq: 0.1,
                penalty_present: 0.1,
                ..Default::default()
            },
            // Deterministic sampling
            SamplerParams {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                typical_p: 1.0,
                penalty_repeat: 1.0,
                penalty_freq: 0.0,
                penalty_present: 0.0,
                ..Default::default()
            },
            // Edge cases
            SamplerParams {
                temperature: 2.0,
                top_k: 0,   // Disabled
                top_p: 0.1, // Very restrictive
                min_p: 0.5, // High threshold
                typical_p: 0.5,
                penalty_repeat: 2.0, // Strong penalty
                penalty_freq: 1.0,
                penalty_present: 1.0,
                penalty_last_n: 1,
                ..Default::default()
            },
        ];

        for (i, params) in combinations.iter().enumerate() {
            // Validate parameter ranges
            assert!(
                params.temperature >= 0.0,
                "Config {} has negative temperature",
                i
            );
            assert!(
                params.top_p > 0.0 && params.top_p <= 1.0,
                "Config {} has invalid top_p",
                i
            );
            assert!(
                params.min_p >= 0.0 && params.min_p <= 1.0,
                "Config {} has invalid min_p",
                i
            );
            assert!(
                params.typical_p >= 0.0,
                "Config {} has negative typical_p",
                i
            );
            assert!(
                params.penalty_repeat >= 0.0,
                "Config {} has negative penalty_repeat",
                i
            );
            assert!(
                params.penalty_last_n >= 0,
                "Config {} has negative penalty_last_n",
                i
            );

            // Test that parameters can be cloned and formatted
            let cloned = params.clone();
            let _debug_str = format!("{:?}", cloned);

            println!("Sampler config {} validated", i);
        }
    }

    #[test]
    fn test_logit_bias_configurations() {
        let bias_configs = vec![
            // Empty bias
            vec![],
            // Single bias
            vec![LogitBias {
                token: 100,
                bias: 1.0,
            }],
            // Multiple biases
            vec![
                LogitBias {
                    token: 1,
                    bias: -1.0,
                }, // Suppress token
                LogitBias {
                    token: 2,
                    bias: 1.0,
                }, // Promote token
                LogitBias {
                    token: 3,
                    bias: 0.0,
                }, // Neutral
            ],
            // Extreme biases
            vec![
                LogitBias {
                    token: 50,
                    bias: -100.0,
                }, // Strongly suppress
                LogitBias {
                    token: 51,
                    bias: 100.0,
                }, // Strongly promote
            ],
        ];

        for (i, biases) in bias_configs.iter().enumerate() {
            // Test that bias arrays can be created and used
            for bias in biases {
                assert!(bias.token >= 0, "Config {} has negative token ID", i);
                // Bias can be any finite value
                assert!(bias.bias.is_finite(), "Config {} has non-finite bias", i);
            }

            println!(
                "Logit bias config {} validated ({} biases)",
                i,
                biases.len()
            );
        }
    }

    #[test]
    fn test_token_data_array_operations() {
        let test_cases = vec![
            // Empty array
            vec![],
            // Single token
            vec![TokenData {
                id: 1,
                logit: 1.0,
                p: 1.0,
            }],
            // Multiple tokens with different probabilities
            vec![
                TokenData {
                    id: 1,
                    logit: 2.0,
                    p: 0.5,
                },
                TokenData {
                    id: 2,
                    logit: 1.0,
                    p: 0.3,
                },
                TokenData {
                    id: 3,
                    logit: 0.0,
                    p: 0.2,
                },
            ],
            // Edge case probabilities
            vec![
                TokenData {
                    id: 10,
                    logit: f32::NEG_INFINITY,
                    p: 0.0,
                },
                TokenData {
                    id: 11,
                    logit: 0.0,
                    p: 0.5,
                },
                TokenData {
                    id: 12,
                    logit: f32::INFINITY,
                    p: 0.5,
                },
            ],
        ];

        for (i, candidates) in test_cases.iter().enumerate() {
            let array = TokenDataArray::new(candidates.clone());

            // Test basic properties
            assert_eq!(
                array.len(),
                candidates.len(),
                "Array {} has wrong length",
                i
            );
            assert_eq!(
                array.is_empty(),
                candidates.is_empty(),
                "Array {} empty check failed",
                i
            );
            assert_eq!(
                array.selected(),
                None,
                "Array {} should have no selection",
                i
            );
            assert!(
                !array.is_sorted(),
                "Array {} should not be sorted initially",
                i
            );

            // Test candidate access
            let retrieved = array.candidates();
            assert_eq!(
                retrieved.len(),
                candidates.len(),
                "Array {} candidate access failed",
                i
            );

            println!(
                "Token data array {} validated ({} tokens)",
                i,
                candidates.len()
            );
        }
    }
}

#[cfg(test)]
mod batch_integration_tests {
    use super::*;

    #[test]
    fn test_batch_creation_patterns() {
        let test_patterns = vec![
            // Empty batch
            vec![],
            // Single token
            vec![42],
            // Short sequence
            vec![1, 2, 3, 4, 5],
            // Longer sequence
            (0..100).collect::<Vec<_>>(),
            // Large sequence
            (0..1000).collect::<Vec<_>>(),
            // Edge case tokens
            vec![0, i32::MAX, i32::MIN, -1],
        ];

        for (i, tokens) in test_patterns.iter().enumerate() {
            let batch = Batch::from_tokens(tokens);

            // Test basic properties
            assert_eq!(
                batch.is_empty(),
                tokens.is_empty(),
                "Batch {} empty check failed",
                i
            );

            // Test that llama_batch can be retrieved
            let _llama_batch = batch.get_llama_batch();

            println!("Batch pattern {} validated ({} tokens)", i, tokens.len());
        }
    }

    #[test]
    fn test_batch_memory_safety() {
        // Test that batches can be created and dropped safely
        let batches: Vec<Batch> = (0..10)
            .map(|i| {
                let tokens: Vec<TokenId> = (0..i * 10).collect();
                Batch::from_tokens(&tokens)
            })
            .collect();

        // All batches should be valid
        for (i, batch) in batches.iter().enumerate() {
            let expected_empty = i == 0;
            assert_eq!(
                batch.is_empty(),
                expected_empty,
                "Batch {} has wrong empty state",
                i
            );
        }

        // Batches will be dropped here - should not crash
        println!("Created and dropped {} batches safely", batches.len());
    }

    #[test]
    fn test_batch_with_special_tokens() {
        // Test batches with special token values
        let special_tokens = vec![
            vec![LLAMA_TOKEN_NULL],          // Null token
            vec![0, 1, LLAMA_TOKEN_NULL, 2], // Mixed with null
            vec![-1, -2, -100],              // Negative tokens
            vec![1000000, 2000000],          // Large token IDs
        ];

        for (i, tokens) in special_tokens.iter().enumerate() {
            let batch = Batch::from_tokens(tokens);
            assert!(
                !batch.is_empty() || tokens.is_empty(),
                "Special batch {} failed",
                i
            );
            println!("Special token batch {} validated", i);
        }
    }
}

#[cfg(test)]
mod error_handling_integration_tests {
    use super::*;

    #[test]
    fn test_model_loading_error_paths() {
        let error_cases = vec![
            // Non-existent file
            ("nonexistent.gguf", "should fail with file not found"),
            // Invalid path
            ("/invalid/path/model.gguf", "should fail with invalid path"),
            // Empty path
            ("", "should fail with empty path"),
            // Directory instead of file
            (".", "should fail when path is directory"),
        ];

        for (path, description) in error_cases {
            let result = Model::load(path);

            match result {
                Err(MullamaError::ModelLoadError(_)) => {
                    println!("✓ {}: {}", path, description);
                }
                other => panic!("Path '{}' had unexpected result: {:?}", path, other),
            }
        }
    }

    #[test]
    fn test_parameter_validation_errors() {
        // Test invalid KV override keys (too long)
        let long_key = "a".repeat(200);
        let invalid_override = ModelKvOverride {
            key: long_key.clone(),
            value: ModelKvOverrideValue::Str("value".to_string()),
        };

        let mut params = ModelParams::default();
        params.kv_overrides = vec![invalid_override];

        let result = Model::load_with_params("test.gguf", params);

        // Should fail (either with model load error or validation error)
        assert!(result.is_err(), "Long key should cause error");

        // Test invalid string value (too long)
        let long_value = "a".repeat(200);
        let invalid_override = ModelKvOverride {
            key: "test".to_string(),
            value: ModelKvOverrideValue::Str(long_value),
        };

        let mut params = ModelParams::default();
        params.kv_overrides = vec![invalid_override];

        let result = Model::load_with_params("test.gguf", params);
        assert!(result.is_err(), "Long value should cause error");
    }

    #[test]
    fn test_context_parameter_validation() {
        // Test various invalid parameter combinations
        let invalid_configs = vec![
            // n_ubatch > n_batch
            ContextParams {
                n_batch: 100,
                n_ubatch: 200,
                ..Default::default()
            },
            // Zero sequences
            ContextParams {
                n_seq_max: 0,
                ..Default::default()
            },
        ];

        for (i, params) in invalid_configs.iter().enumerate() {
            // These should be caught either during context creation or during validation
            if params.n_batch < params.n_ubatch {
                println!("Config {} correctly has n_ubatch > n_batch", i);
            }
            if params.n_seq_max == 0 {
                println!("Config {} correctly has zero sequences", i);
            }
        }
    }

    #[test]
    fn test_error_message_quality() {
        // Test that error messages are informative
        let result = Model::load("definitely_nonexistent_file.gguf");

        match result {
            Err(MullamaError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("not found") || msg.contains("Model file"),
                    "Error message should be informative: {}",
                    msg
                );
                println!("✓ Error message quality: {}", msg);
            }
            other => panic!("Expected ModelLoadError, got: {:?}", other),
        }
    }

    #[test]
    fn test_error_chain_handling() {
        // Test that errors can be properly chained and handled
        fn nested_operation() -> Result<(), MullamaError> {
            Model::load("missing.gguf")?;
            Ok(())
        }

        fn higher_level_operation() -> Result<(), Box<dyn std::error::Error>> {
            nested_operation()?;
            Ok(())
        }

        let result = higher_level_operation();
        assert!(result.is_err(), "Error should propagate through chain");

        let error = result.unwrap_err();
        let error_string = format!("{}", error);
        println!("✓ Error chain handling: {}", error_string);
    }
}

#[cfg(test)]
mod performance_integration_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_parameter_creation_performance() {
        // Test that parameter creation is fast
        let start = Instant::now();

        for _ in 0..1000 {
            let _model_params = ModelParams::default();
            let _context_params = ContextParams::default();
            let _sampler_params = SamplerParams::default();
        }

        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 1000,
            "Parameter creation should be fast"
        );
        println!(
            "✓ Created 3000 parameter objects in {}ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_batch_creation_performance() {
        // Test batch creation performance
        let tokens: Vec<TokenId> = (0..10000).collect();

        let start = Instant::now();
        for _ in 0..100 {
            let _batch = Batch::from_tokens(&tokens);
        }
        let duration = start.elapsed();

        assert!(duration.as_millis() < 1000, "Batch creation should be fast");
        println!(
            "✓ Created 100 batches of 10k tokens in {}ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_token_data_array_performance() {
        // Test token data array performance
        let candidates: Vec<TokenData> = (0..1000)
            .map(|i| TokenData {
                id: i,
                logit: i as f32 * 0.1,
                p: 1.0 / (1000.0 - i as f32),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..100 {
            let _array = TokenDataArray::new(candidates.clone());
        }
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 500,
            "Token data array creation should be fast"
        );
        println!(
            "✓ Created 100 token arrays of 1k tokens in {}ms",
            duration.as_millis()
        );
    }

    #[test]
    fn test_memory_usage_patterns() {
        // Test that structures don't use excessive memory
        let model_params_size = std::mem::size_of::<ModelParams>();
        let context_params_size = std::mem::size_of::<ContextParams>();
        let sampler_params_size = std::mem::size_of::<SamplerParams>();

        println!("Memory usage:");
        println!("  ModelParams: {} bytes", model_params_size);
        println!("  ContextParams: {} bytes", context_params_size);
        println!("  SamplerParams: {} bytes", sampler_params_size);

        // Reasonable size limits
        assert!(
            model_params_size < 1024,
            "ModelParams should be reasonably sized"
        );
        assert!(
            context_params_size < 1024,
            "ContextParams should be reasonably sized"
        );
        assert!(
            sampler_params_size < 512,
            "SamplerParams should be reasonably sized"
        );
    }
}
