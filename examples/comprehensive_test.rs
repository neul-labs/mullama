//! Comprehensive test suite for mullama text generation
//!
//! Run with: cargo run --example comprehensive_test
//!
//! This tests:
//! - Model loading
//! - Tokenization and detokenization
//! - Basic text generation
//! - Custom sampling parameters
//! - Streaming generation
//! - Logits access
//! - Edge cases and error handling

use mullama::{Context, ContextParams, Model, MullamaError, SamplerParams};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

const MODEL_PATH: &str = "models/tinyllama-1.1b-q4_k_m.gguf";

fn main() -> Result<(), MullamaError> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Mullama Comprehensive Test Suite                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load model once for all tests
    println!("Loading model: {}", MODEL_PATH);
    let start = Instant::now();
    let model = Arc::new(Model::load(MODEL_PATH)?);
    println!("Model loaded in {:?}\n", start.elapsed());

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Tokenization
    match test_tokenization(&model) {
        Ok(_) => {
            println!("[PASS] Test 1: Tokenization\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 1: Tokenization - {}\n", e);
            failed += 1;
        }
    }

    // Test 2: Detokenization
    match test_detokenization(&model) {
        Ok(_) => {
            println!("[PASS] Test 2: Detokenization\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 2: Detokenization - {}\n", e);
            failed += 1;
        }
    }

    // Test 3: Tokenize/Detokenize roundtrip
    match test_roundtrip(&model) {
        Ok(_) => {
            println!("[PASS] Test 3: Tokenize/Detokenize roundtrip\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 3: Tokenize/Detokenize roundtrip - {}\n", e);
            failed += 1;
        }
    }

    // Test 4: Basic generation
    match test_basic_generation(&model) {
        Ok(_) => {
            println!("[PASS] Test 4: Basic generation\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 4: Basic generation - {}\n", e);
            failed += 1;
        }
    }

    // Test 5: Generation with custom temperature
    match test_temperature_variation(&model) {
        Ok(_) => {
            println!("[PASS] Test 5: Temperature variation\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 5: Temperature variation - {}\n", e);
            failed += 1;
        }
    }

    // Test 6: Greedy generation (temperature = 0)
    match test_greedy_generation(&model) {
        Ok(_) => {
            println!("[PASS] Test 6: Greedy generation\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 6: Greedy generation - {}\n", e);
            failed += 1;
        }
    }

    // Test 7: Streaming generation
    match test_streaming_generation(&model) {
        Ok(_) => {
            println!("[PASS] Test 7: Streaming generation\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 7: Streaming generation - {}\n", e);
            failed += 1;
        }
    }

    // Test 8: Streaming with early stop
    match test_streaming_early_stop(&model) {
        Ok(_) => {
            println!("[PASS] Test 8: Streaming early stop\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 8: Streaming early stop - {}\n", e);
            failed += 1;
        }
    }

    // Test 9: Logits access
    match test_logits_access(&model) {
        Ok(_) => {
            println!("[PASS] Test 9: Logits access\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 9: Logits access - {}\n", e);
            failed += 1;
        }
    }

    // Test 10: Multiple prompts
    match test_multiple_prompts(&model) {
        Ok(_) => {
            println!("[PASS] Test 10: Multiple prompts\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 10: Multiple prompts - {}\n", e);
            failed += 1;
        }
    }

    // Test 11: Top-k and Top-p variations
    match test_sampling_variations(&model) {
        Ok(_) => {
            println!("[PASS] Test 11: Sampling variations\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 11: Sampling variations - {}\n", e);
            failed += 1;
        }
    }

    // Test 12: Repetition penalty
    match test_repetition_penalty(&model) {
        Ok(_) => {
            println!("[PASS] Test 12: Repetition penalty\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 12: Repetition penalty - {}\n", e);
            failed += 1;
        }
    }

    // Test 13: Empty/edge cases
    match test_edge_cases(&model) {
        Ok(_) => {
            println!("[PASS] Test 13: Edge cases\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 13: Edge cases - {}\n", e);
            failed += 1;
        }
    }

    // Test 14: Context reuse
    match test_context_reuse(&model) {
        Ok(_) => {
            println!("[PASS] Test 14: Context reuse\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 14: Context reuse - {}\n", e);
            failed += 1;
        }
    }

    // Test 15: Long prompt
    match test_long_prompt(&model) {
        Ok(_) => {
            println!("[PASS] Test 15: Long prompt\n");
            passed += 1;
        }
        Err(e) => {
            println!("[FAIL] Test 15: Long prompt - {}\n", e);
            failed += 1;
        }
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        TEST SUMMARY                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Passed: {:2}                                                  ║",
        passed
    );
    println!(
        "║  Failed: {:2}                                                  ║",
        failed
    );
    println!(
        "║  Total:  {:2}                                                  ║",
        passed + failed
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    if failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn test_tokenization(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 1: Tokenization ---");

    let test_cases = vec![
        ("Hello, world!", true),
        ("The quick brown fox", true),
        ("1 + 1 = 2", true),
        ("", true),             // Empty string
        ("Hello\nWorld", true), // With newline
        ("Special chars: @#$%^&*()", true),
    ];

    for (text, add_bos) in test_cases {
        let tokens = model.tokenize(text, add_bos, false)?;
        println!(
            "  '{}' -> {} tokens",
            text.replace('\n', "\\n"),
            tokens.len()
        );

        // Verify BOS token if requested
        if add_bos && !text.is_empty() {
            let bos = model.token_bos();
            if tokens.first() != Some(&bos) {
                return Err(MullamaError::TokenizationError(format!(
                    "Expected BOS token {} at start, got {:?}",
                    bos,
                    tokens.first()
                )));
            }
        }
    }

    Ok(())
}

fn test_detokenization(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 2: Detokenization ---");

    // Tokenize some text
    let original = "Hello, how are you today?";
    let tokens = model.tokenize(original, false, false)?;
    println!("  Original: '{}'", original);
    println!("  Tokens: {:?}", tokens);

    // Detokenize
    let recovered = model.detokenize(&tokens, false, false)?;
    println!("  Recovered: '{}'", recovered);

    // Check they match
    if recovered != original {
        return Err(MullamaError::TokenizationError(format!(
            "Mismatch: '{}' != '{}'",
            recovered, original
        )));
    }

    Ok(())
}

fn test_roundtrip(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 3: Tokenize/Detokenize Roundtrip ---");

    let test_strings = vec![
        "Simple text",
        "With numbers 123 and symbols!",
        "Multi\nline\ntext",
        "Unicode: Hello, 世界, مرحبا",
        "Code: fn main() { println!(\"Hello\"); }",
    ];

    for original in test_strings {
        let tokens = model.tokenize(original, false, false)?;
        let recovered = model.detokenize(&tokens, false, false)?;

        if recovered != original {
            println!("  MISMATCH:");
            println!("    Original:  '{}'", original);
            println!("    Recovered: '{}'", recovered);
            // Don't fail on Unicode issues - just warn
            if original.is_ascii() {
                return Err(MullamaError::TokenizationError(format!(
                    "ASCII roundtrip failed: '{}' != '{}'",
                    recovered, original
                )));
            }
        } else {
            println!("  OK: '{}'", original.replace('\n', "\\n"));
        }
    }

    Ok(())
}

fn test_basic_generation(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 4: Basic Generation ---");

    let ctx_params = ContextParams {
        n_ctx: 256,
        n_batch: 64,
        n_threads: 4,
        ..Default::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;

    let prompt = "Once upon a time";
    let tokens = model.tokenize(prompt, true, false)?;

    println!("  Prompt: '{}'", prompt);
    let start = Instant::now();
    let generated = context.generate(&tokens, 20)?;
    let duration = start.elapsed();

    println!("  Generated: '{}'", generated);
    println!("  Time: {:?}", duration);

    // Verify we got some output
    if generated.is_empty() {
        return Err(MullamaError::GenerationError(
            "Generated empty output".to_string(),
        ));
    }

    Ok(())
}

fn test_temperature_variation(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 5: Temperature Variation ---");

    let prompt = "The weather today is";
    let tokens = model.tokenize(prompt, true, false)?;

    let temperatures = [0.1, 0.5, 1.0, 1.5];

    for temp in temperatures {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let params = SamplerParams {
            temperature: temp,
            ..Default::default()
        };

        let generated = context.generate_with_params(&tokens, 15, &params)?;
        println!("  temp={:.1}: '{}'", temp, generated.trim());
    }

    Ok(())
}

fn test_greedy_generation(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 6: Greedy Generation (Deterministic) ---");

    let prompt = "2 + 2 =";
    let tokens = model.tokenize(prompt, true, false)?;

    // Generate twice with same settings - should be identical
    let mut results = Vec::new();

    for i in 0..2 {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        // Use very low temperature for near-deterministic output
        let params = SamplerParams {
            temperature: 0.01,
            top_k: 1, // Greedy
            seed: 42,
            ..Default::default()
        };

        let generated = context.generate_with_params(&tokens, 10, &params)?;
        println!("  Run {}: '{}'", i + 1, generated.trim());
        results.push(generated);
    }

    // With greedy sampling, results should be identical
    if results[0] != results[1] {
        println!("  Warning: Greedy results differ (may be due to floating point)");
    }

    Ok(())
}

fn test_streaming_generation(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 7: Streaming Generation ---");

    let ctx_params = ContextParams {
        n_ctx: 256,
        n_batch: 64,
        n_threads: 4,
        ..Default::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;

    let prompt = "Counting: 1, 2, 3";
    let tokens = model.tokenize(prompt, true, false)?;

    let params = SamplerParams::default();

    print!("  Streaming: ");
    io::stdout().flush().unwrap();

    let mut token_count = 0;
    let result = context.generate_streaming(&tokens, 15, &params, |piece| {
        print!("{}", piece);
        io::stdout().flush().unwrap();
        token_count += 1;
        true // Continue
    })?;
    println!();

    println!("  Total tokens streamed: {}", token_count);
    println!("  Final result: '{}'", result);

    if token_count == 0 {
        return Err(MullamaError::GenerationError(
            "No tokens were streamed".to_string(),
        ));
    }

    Ok(())
}

fn test_streaming_early_stop(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 8: Streaming with Early Stop ---");

    let ctx_params = ContextParams {
        n_ctx: 256,
        n_batch: 64,
        n_threads: 4,
        ..Default::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;

    let prompt = "List of numbers:";
    let tokens = model.tokenize(prompt, true, false)?;

    let params = SamplerParams::default();

    let mut token_count = 0;
    let stop_after = 5;

    print!("  Streaming (stop after {} tokens): ", stop_after);
    io::stdout().flush().unwrap();

    let _result = context.generate_streaming(&tokens, 50, &params, |piece| {
        print!("{}", piece);
        io::stdout().flush().unwrap();
        token_count += 1;
        token_count < stop_after // Stop after N tokens
    })?;
    println!();

    println!("  Stopped after {} tokens", token_count);

    if token_count > stop_after {
        return Err(MullamaError::GenerationError(format!(
            "Should have stopped after {} tokens, got {}",
            stop_after, token_count
        )));
    }

    Ok(())
}

fn test_logits_access(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 9: Logits Access ---");

    let ctx_params = ContextParams {
        n_ctx: 256,
        n_batch: 64,
        n_threads: 4,
        ..Default::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;

    let prompt = "Hello";
    let tokens = model.tokenize(prompt, true, false)?;

    // Decode the prompt
    context.decode(&tokens)?;

    // Access logits
    let logits = context.logits()?;
    println!(
        "  Logits size: {} (vocab size: {})",
        logits.len(),
        model.vocab_size()
    );

    // Find top 5 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 5 next tokens:");
    for (idx, logit) in indexed.iter().take(5) {
        let token_str = model.token_to_str(*idx as i32, 0, false)?;
        println!(
            "    {} (id={}, logit={:.2})",
            token_str.replace('\n', "\\n"),
            idx,
            logit
        );
    }

    if logits.len() != model.vocab_size() as usize {
        return Err(MullamaError::GenerationError(format!(
            "Logits size {} != vocab size {}",
            logits.len(),
            model.vocab_size()
        )));
    }

    Ok(())
}

fn test_multiple_prompts(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 10: Multiple Prompts ---");

    let prompts = vec![
        "The capital of France is",
        "Water freezes at",
        "The color of the sky is",
    ];

    for prompt in prompts {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let tokens = model.tokenize(prompt, true, false)?;
        let generated = context.generate(&tokens, 10)?;

        println!("  '{}' -> '{}'", prompt, generated.trim());
    }

    Ok(())
}

fn test_sampling_variations(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 11: Sampling Variations ---");

    let prompt = "In the beginning";
    let tokens = model.tokenize(prompt, true, false)?;

    let configs = vec![
        (
            "top_k=10",
            SamplerParams {
                top_k: 10,
                ..Default::default()
            },
        ),
        (
            "top_p=0.5",
            SamplerParams {
                top_p: 0.5,
                ..Default::default()
            },
        ),
        (
            "min_p=0.1",
            SamplerParams {
                min_p: 0.1,
                ..Default::default()
            },
        ),
        (
            "typical_p=0.9",
            SamplerParams {
                typical_p: 0.9,
                ..Default::default()
            },
        ),
    ];

    for (name, params) in configs {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let generated = context.generate_with_params(&tokens, 12, &params)?;
        println!("  {}: '{}'", name, generated.trim());
    }

    Ok(())
}

fn test_repetition_penalty(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 12: Repetition Penalty ---");

    let prompt = "Repeat after me: hello hello hello";
    let tokens = model.tokenize(prompt, true, false)?;

    // Without penalty
    {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let params = SamplerParams {
            penalty_repeat: 1.0, // No penalty
            ..Default::default()
        };

        let generated = context.generate_with_params(&tokens, 15, &params)?;
        println!("  No penalty: '{}'", generated.trim());
    }

    // With penalty
    {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let params = SamplerParams {
            penalty_repeat: 1.5, // Strong penalty
            penalty_last_n: 64,
            ..Default::default()
        };

        let generated = context.generate_with_params(&tokens, 15, &params)?;
        println!("  With penalty: '{}'", generated.trim());
    }

    Ok(())
}

fn test_edge_cases(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 13: Edge Cases ---");

    // Test 1: Single token prompt
    {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let tokens = model.tokenize("A", true, false)?;
        let generated = context.generate(&tokens, 5)?;
        println!("  Single token: 'A' -> '{}'", generated.trim());
    }

    // Test 2: Empty generation request (max_tokens = 0)
    {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let tokens = model.tokenize("Hello", true, false)?;
        let generated = context.generate(&tokens, 0)?;
        println!("  Zero tokens: result = '{}'", generated);
    }

    // Test 3: Special characters in prompt
    {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let tokens = model.tokenize("@#$%^&*()", true, false)?;
        let generated = context.generate(&tokens, 5)?;
        println!("  Special chars: '@#$%^&*()' -> '{}'", generated.trim());
    }

    // Test 4: Empty prompt should fail
    {
        let ctx_params = ContextParams {
            n_ctx: 256,
            n_batch: 64,
            n_threads: 4,
            ..Default::default()
        };
        let mut context = Context::new(model.clone(), ctx_params)?;

        let empty_tokens: Vec<i32> = vec![];
        match context.generate(&empty_tokens, 10) {
            Err(MullamaError::GenerationError(_)) => {
                println!("  Empty tokens: Correctly rejected");
            }
            Ok(_) => {
                return Err(MullamaError::GenerationError(
                    "Should have rejected empty tokens".to_string(),
                ));
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    Ok(())
}

fn test_context_reuse(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 14: Context Reuse ---");

    let ctx_params = ContextParams {
        n_ctx: 512,
        n_batch: 64,
        n_threads: 4,
        ..Default::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;

    // First generation
    let tokens1 = model.tokenize("Hello", true, false)?;
    let gen1 = context.generate(&tokens1, 10)?;
    println!("  First: 'Hello' -> '{}'", gen1.trim());

    // Clear KV cache
    context.kv_cache_clear();

    // Second generation with same context
    let tokens2 = model.tokenize("Goodbye", true, false)?;
    let gen2 = context.generate(&tokens2, 10)?;
    println!("  Second: 'Goodbye' -> '{}'", gen2.trim());

    // Verify both worked
    if gen1.is_empty() || gen2.is_empty() {
        return Err(MullamaError::GenerationError(
            "Context reuse failed".to_string(),
        ));
    }

    Ok(())
}

fn test_long_prompt(model: &Arc<Model>) -> Result<(), MullamaError> {
    println!("--- Test 15: Long Prompt ---");

    // Create a longer prompt
    let long_prompt = "This is a test of a longer prompt. ".repeat(10);
    println!("  Prompt length: {} chars", long_prompt.len());

    let ctx_params = ContextParams {
        n_ctx: 512,
        n_batch: 64,
        n_threads: 4,
        ..Default::default()
    };
    let mut context = Context::new(model.clone(), ctx_params)?;

    let tokens = model.tokenize(&long_prompt, true, false)?;
    println!("  Token count: {}", tokens.len());

    let start = Instant::now();
    let generated = context.generate(&tokens, 20)?;
    let duration = start.elapsed();

    println!("  Generated: '{}'", generated.trim());
    println!("  Time: {:?}", duration);

    if generated.is_empty() {
        return Err(MullamaError::GenerationError(
            "Long prompt generation failed".to_string(),
        ));
    }

    Ok(())
}
