//! Speculative decoding for accelerated text generation
//!
//! Speculative decoding uses a smaller, faster "draft" model to generate candidate tokens,
//! which are then validated by a larger "target" model. This can significantly speed up
//! generation while maintaining the quality of the larger model.

use crate::error::MullamaError;
use crate::{Model, Context};
use crate::token::TokenId;
use crate::context::ContextParams;
use std::sync::Arc;
use std::collections::VecDeque;

/// Type alias for token
pub type Token = TokenId;

/// Speculative decoding engine that orchestrates draft and target models
#[derive(Debug)]
pub struct SpeculativeDecoder {
    /// The main (target) model that provides final quality
    target_model: Arc<Model>,
    /// The draft model used for fast candidate generation
    draft_model: Arc<Model>,
    /// Context for the target model
    target_context: Context,
    /// Context for the draft model
    draft_context: Context,
    /// Configuration parameters
    config: SpeculativeConfig,
    /// Performance statistics
    stats: SpeculativeStats,
}

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to generate speculatively (lookahead window)
    pub lookahead_tokens: usize,
    /// Minimum acceptance threshold for draft tokens
    pub acceptance_threshold: f32,
    /// Maximum number of rejections before fallback
    pub max_rejections: usize,
    /// Temperature for draft model (can be different from target)
    pub draft_temperature: f32,
    /// Temperature for target model validation
    pub target_temperature: f32,
    /// Whether to use dynamic lookahead adjustment
    pub dynamic_lookahead: bool,
    /// Batch size for speculative generation
    pub batch_size: usize,
}

/// Statistics for speculative decoding performance
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated
    pub total_tokens: usize,
    /// Tokens accepted from draft model
    pub accepted_tokens: usize,
    /// Tokens rejected and regenerated
    pub rejected_tokens: usize,
    /// Average lookahead window size used
    pub avg_lookahead: f32,
    /// Time spent in draft generation (nanoseconds)
    pub draft_time_ns: u64,
    /// Time spent in target validation (nanoseconds)
    pub target_time_ns: u64,
    /// Number of speculation rounds
    pub speculation_rounds: usize,
}

/// Result of a speculative decoding step
#[derive(Debug)]
pub struct SpeculativeResult {
    /// Generated tokens
    pub tokens: Vec<Token>,
    /// Whether generation should continue
    pub should_continue: bool,
    /// Updated statistics
    pub stats: SpeculativeStats,
}

/// A candidate token with its probability
#[derive(Debug, Clone)]
pub struct CandidateToken {
    pub token: Token,
    pub log_prob: f32,
    pub probability: f32,
}

/// Draft proposal containing candidate tokens
#[derive(Debug)]
pub struct DraftProposal {
    pub candidates: Vec<CandidateToken>,
    pub draft_context_state: Vec<u8>, // Serialized context state
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    ///
    /// # Arguments
    /// * `target_model` - The high-quality model for final generation
    /// * `draft_model` - The fast model for candidate generation
    /// * `config` - Configuration parameters
    ///
    /// # Example
    /// ```rust
    /// use mullama::speculative::{SpeculativeDecoder, SpeculativeConfig};
    /// use mullama::{Model, ContextParams};
    ///
    /// let target_model = Model::from_file("large_model.gguf")?;
    /// let draft_model = Model::from_file("small_model.gguf")?;
    ///
    /// let config = SpeculativeConfig::default()
    ///     .with_lookahead_tokens(4)
    ///     .with_acceptance_threshold(0.8);
    ///
    /// let decoder = SpeculativeDecoder::new(target_model, draft_model, config)?;
    /// ```
    pub fn new(
        target_model: Arc<Model>,
        draft_model: Arc<Model>,
        config: SpeculativeConfig,
    ) -> Result<Self, MullamaError> {
        // Validate model compatibility
        Self::validate_models(&target_model, &draft_model)?;

        // Create contexts
        let target_context = Context::new(target_model.clone(), ContextParams::default())?;
        let draft_context = Context::new(draft_model.clone(), ContextParams::default())?;

        Ok(Self {
            target_model,
            draft_model,
            target_context,
            draft_context,
            config,
            stats: SpeculativeStats::default(),
        })
    }

    /// Generate tokens using speculative decoding
    ///
    /// # Arguments
    /// * `prompt_tokens` - Initial prompt tokens
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// Vector of generated tokens
    pub fn generate(
        &mut self,
        prompt_tokens: &[Token],
        max_tokens: usize,
    ) -> Result<Vec<Token>, MullamaError> {
        // Initialize contexts with prompt
        self.initialize_contexts(prompt_tokens)?;

        let mut generated_tokens = Vec::new();

        while generated_tokens.len() < max_tokens {
            let result = self.speculative_step()?;

            generated_tokens.extend(result.tokens);

            if !result.should_continue {
                break;
            }

            // Dynamic lookahead adjustment
            if self.config.dynamic_lookahead {
                self.adjust_lookahead();
            }
        }

        Ok(generated_tokens)
    }

    /// Perform a single speculative decoding step
    pub fn speculative_step(&mut self) -> Result<SpeculativeResult, MullamaError> {
        let start_time = std::time::Instant::now();

        // Phase 1: Generate candidates with draft model
        let draft_start = std::time::Instant::now();
        let proposal = self.generate_draft_proposal()?;
        self.stats.draft_time_ns += draft_start.elapsed().as_nanos() as u64;

        // Phase 2: Validate candidates with target model
        let target_start = std::time::Instant::now();
        let (accepted_tokens, should_continue) = self.validate_with_target(&proposal)?;
        self.stats.target_time_ns += target_start.elapsed().as_nanos() as u64;

        // Update statistics
        self.stats.speculation_rounds += 1;
        self.stats.total_tokens += accepted_tokens.len();
        self.stats.accepted_tokens += accepted_tokens.len();

        if accepted_tokens.len() < proposal.candidates.len() {
            self.stats.rejected_tokens += proposal.candidates.len() - accepted_tokens.len();
        }

        // Update contexts with accepted tokens
        self.update_contexts(&accepted_tokens)?;

        Ok(SpeculativeResult {
            tokens: accepted_tokens,
            should_continue,
            stats: self.stats.clone(),
        })
    }

    /// Generate candidate tokens using the draft model
    fn generate_draft_proposal(&mut self) -> Result<DraftProposal, MullamaError> {
        let mut candidates = Vec::new();

        for _ in 0..self.config.lookahead_tokens {
            // Get logits from draft model (for the last token position)
            let logits = self.draft_context.get_logits();

            if logits.is_empty() {
                return Err(MullamaError::GenerationError("Empty logits from draft model".to_string()));
            }

            // Apply temperature
            let scaled_logits = self.apply_temperature(logits, self.config.draft_temperature);

            // Sample token
            let token = self.sample_from_logits(&scaled_logits)?;
            let log_prob = scaled_logits[token as usize];
            let probability = log_prob.exp();

            candidates.push(CandidateToken {
                token,
                log_prob,
                probability,
            });

            // Evaluate token in draft context for next iteration
            self.draft_context.decode(&[token])?;

            // Early stopping if end token
            if self.draft_model.token_is_eog(token) {
                break;
            }
        }

        // Save draft context state for potential rollback
        let draft_context_state = self.draft_context.save_state();

        Ok(DraftProposal {
            candidates,
            draft_context_state,
        })
    }

    /// Validate draft candidates with the target model
    fn validate_with_target(
        &mut self,
        proposal: &DraftProposal,
    ) -> Result<(Vec<Token>, bool), MullamaError> {
        let mut accepted_tokens = Vec::new();
        let mut should_continue = true;

        for (i, candidate) in proposal.candidates.iter().enumerate() {
            // Get target model's probability for this token
            let target_logits = self.target_context.get_logits();
            if target_logits.is_empty() {
                return Err(MullamaError::GenerationError("Empty logits from target model".to_string()));
            }

            let target_scaled_logits = self.apply_temperature(target_logits, self.config.target_temperature);
            let target_prob = target_scaled_logits[candidate.token as usize].exp();

            // Accept/reject based on probability ratio
            let acceptance_ratio = target_prob / candidate.probability;
            let random_value: f32 = self.simple_random();

            if random_value < acceptance_ratio.min(1.0) && acceptance_ratio >= self.config.acceptance_threshold {
                // Accept the token
                accepted_tokens.push(candidate.token);
                self.target_context.decode(&[candidate.token])?;

                // Check for end token
                if self.target_model.token_is_eog(candidate.token) {
                    should_continue = false;
                    break;
                }
            } else {
                // Reject token and resample from target model
                let corrected_logits = self.correct_distribution(&target_scaled_logits, &proposal.candidates[..=i]);
                let corrected_token = self.sample_from_logits(&corrected_logits)?;

                accepted_tokens.push(corrected_token);
                self.target_context.decode(&[corrected_token])?;

                // Check for end token
                if self.target_model.token_is_eog(corrected_token) {
                    should_continue = false;
                }

                // Stop speculation after first rejection
                break;
            }
        }

        Ok((accepted_tokens, should_continue))
    }

    /// Apply temperature scaling to logits
    fn apply_temperature(&self, logits: &[f32], temperature: f32) -> Vec<f32> {
        if temperature == 1.0 {
            return logits.to_vec();
        }

        logits.iter().map(|&logit| logit / temperature).collect()
    }

    /// Sample a token from logits using multinomial sampling
    fn sample_from_logits(&self, logits: &[f32]) -> Result<Token, MullamaError> {
        // Convert logits to probabilities
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        let probabilities: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

        // Sample using cumulative distribution
        let random_value: f32 = self.simple_random();
        let mut cumulative = 0.0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(i as Token);
            }
        }

        // Fallback to last token
        Ok((probabilities.len() - 1) as Token)
    }

    /// Simple random number generator (0.0 to 1.0)
    /// Uses a basic LCG for simplicity - in production, consider a better RNG
    fn simple_random(&self) -> f32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        ((seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff) as f32 / 0x7fffffff as f32
    }

    /// Correct the probability distribution after rejection
    fn correct_distribution(
        &self,
        target_logits: &[f32],
        rejected_candidates: &[CandidateToken],
    ) -> Vec<f32> {
        let mut corrected_logits = target_logits.to_vec();

        // Reduce probability of rejected tokens
        for candidate in rejected_candidates {
            if (candidate.token as usize) < corrected_logits.len() {
                corrected_logits[candidate.token as usize] -= 1.0; // Penalize rejected tokens
            }
        }

        corrected_logits
    }

    /// Initialize both contexts with prompt tokens
    fn initialize_contexts(&mut self, prompt_tokens: &[Token]) -> Result<(), MullamaError> {
        // Clear caches for fresh start
        self.target_context.kv_cache_clear();
        self.draft_context.kv_cache_clear();

        // Process prompt tokens
        self.target_context.decode(prompt_tokens)?;
        self.draft_context.decode(prompt_tokens)?;
        Ok(())
    }

    /// Update both contexts with accepted tokens
    fn update_contexts(&mut self, accepted_tokens: &[Token]) -> Result<(), MullamaError> {
        // Target context is already updated during validation
        // Update draft context to match - we need to sync state
        // Get target state and load into draft
        let target_state = self.target_context.save_state();
        let _ = self.draft_context.load_state(&target_state);

        Ok(())
    }

    /// Adjust lookahead window based on acceptance rate
    fn adjust_lookahead(&mut self) {
        if self.stats.speculation_rounds < 10 {
            return; // Need more data
        }

        let acceptance_rate = self.stats.accepted_tokens as f32 / self.stats.total_tokens as f32;

        if acceptance_rate > 0.8 {
            // High acceptance rate, try larger lookahead
            self.config.lookahead_tokens = (self.config.lookahead_tokens + 1).min(8);
        } else if acceptance_rate < 0.5 {
            // Low acceptance rate, reduce lookahead
            self.config.lookahead_tokens = (self.config.lookahead_tokens.saturating_sub(1)).max(1);
        }

        // Update average lookahead
        self.stats.avg_lookahead = (self.stats.avg_lookahead * (self.stats.speculation_rounds - 1) as f32
            + self.config.lookahead_tokens as f32) / self.stats.speculation_rounds as f32;
    }

    /// Validate that models are compatible for speculative decoding
    fn validate_models(target: &Model, draft: &Model) -> Result<(), MullamaError> {
        // Check vocabulary compatibility
        if target.vocab_size() != draft.vocab_size() {
            return Err(MullamaError::InvalidInput(
                "Target and draft models must have the same vocabulary size".to_string()
            ));
        }

        // Check that both models use the same tokenizer
        if target.vocab_type() != draft.vocab_type() {
            return Err(MullamaError::InvalidInput(
                "Target and draft models must use the same vocabulary type".to_string()
            ));
        }

        // Verify that draft model is actually smaller/faster
        if draft.n_params() >= target.n_params() {
            eprintln!("Warning: Draft model is not smaller than target model");
        }

        Ok(())
    }

    /// Get current performance statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Get current configuration
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: SpeculativeConfig) {
        self.config = config;
    }
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            lookahead_tokens: 4,
            acceptance_threshold: 0.6,
            max_rejections: 10,
            draft_temperature: 1.0,
            target_temperature: 1.0,
            dynamic_lookahead: true,
            batch_size: 1,
        }
    }
}

impl SpeculativeConfig {
    /// Builder pattern for configuration
    pub fn with_lookahead_tokens(mut self, tokens: usize) -> Self {
        self.lookahead_tokens = tokens;
        self
    }

    pub fn with_acceptance_threshold(mut self, threshold: f32) -> Self {
        self.acceptance_threshold = threshold;
        self
    }

    pub fn with_max_rejections(mut self, max: usize) -> Self {
        self.max_rejections = max;
        self
    }

    pub fn with_draft_temperature(mut self, temp: f32) -> Self {
        self.draft_temperature = temp;
        self
    }

    pub fn with_target_temperature(mut self, temp: f32) -> Self {
        self.target_temperature = temp;
        self
    }

    pub fn with_dynamic_lookahead(mut self, enabled: bool) -> Self {
        self.dynamic_lookahead = enabled;
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
}

impl SpeculativeStats {
    /// Calculate acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.total_tokens as f32
        }
    }

    /// Calculate speedup factor
    pub fn speedup_factor(&self) -> f32 {
        if self.speculation_rounds == 0 {
            1.0
        } else {
            self.total_tokens as f32 / self.speculation_rounds as f32
        }
    }

    /// Get total time in seconds
    pub fn total_time_seconds(&self) -> f64 {
        (self.draft_time_ns + self.target_time_ns) as f64 / 1_000_000_000.0
    }

    /// Get tokens per second
    pub fn tokens_per_second(&self) -> f64 {
        let total_time = self.total_time_seconds();
        if total_time > 0.0 {
            self.total_tokens as f64 / total_time
        } else {
            0.0
        }
    }
}

/// Utilities for speculative decoding
pub mod utils {
    use super::*;

    /// Find optimal lookahead window size for given models
    pub fn find_optimal_lookahead(
        target_model: Arc<Model>,
        draft_model: Arc<Model>,
        test_prompt: &[Token],
        max_lookahead: usize,
    ) -> Result<usize, MullamaError> {
        let mut best_lookahead = 1;
        let mut best_speedup = 0.0;

        for lookahead in 1..=max_lookahead {
            let config = SpeculativeConfig::default()
                .with_lookahead_tokens(lookahead)
                .with_dynamic_lookahead(false);

            let mut decoder = SpeculativeDecoder::new(
                Arc::clone(&target_model),
                Arc::clone(&draft_model),
                config,
            )?;

            // Generate a small sample
            let _tokens = decoder.generate(test_prompt, 50)?;
            let speedup = decoder.stats().speedup_factor();

            if speedup > best_speedup {
                best_speedup = speedup;
                best_lookahead = lookahead;
            }
        }

        Ok(best_lookahead)
    }

    /// Create a configuration optimized for speed
    pub fn speed_optimized_config() -> SpeculativeConfig {
        SpeculativeConfig::default()
            .with_lookahead_tokens(6)
            .with_acceptance_threshold(0.5)
            .with_draft_temperature(1.2)
            .with_dynamic_lookahead(true)
    }

    /// Create a configuration optimized for quality
    pub fn quality_optimized_config() -> SpeculativeConfig {
        SpeculativeConfig::default()
            .with_lookahead_tokens(3)
            .with_acceptance_threshold(0.8)
            .with_draft_temperature(0.8)
            .with_dynamic_lookahead(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config() {
        let config = SpeculativeConfig::default()
            .with_lookahead_tokens(6)
            .with_acceptance_threshold(0.8);

        assert_eq!(config.lookahead_tokens, 6);
        assert_eq!(config.acceptance_threshold, 0.8);
    }

    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::default();
        stats.total_tokens = 100;
        stats.accepted_tokens = 80;
        stats.speculation_rounds = 25;

        assert_eq!(stats.acceptance_rate(), 0.8);
        assert_eq!(stats.speedup_factor(), 4.0);
    }

    #[test]
    fn test_candidate_token() {
        let candidate = CandidateToken {
            token: 42,
            log_prob: -0.5,
            probability: 0.606,
        };

        assert_eq!(candidate.token, 42);
        assert!((candidate.probability - 0.606).abs() < 0.001);
    }

    #[test]
    fn test_config_presets() {
        let speed_config = utils::speed_optimized_config();
        assert_eq!(speed_config.lookahead_tokens, 6);
        assert!(speed_config.dynamic_lookahead);

        let quality_config = utils::quality_optimized_config();
        assert_eq!(quality_config.acceptance_threshold, 0.8);
        assert!(!quality_config.dynamic_lookahead);
    }
}