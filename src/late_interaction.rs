//! Late Interaction / ColBERT-style Multi-Vector Embeddings
//!
//! This module provides support for late interaction retrieval methods like ColBERT:
//! - Multi-vector embeddings (per-token instead of single pooled vector)
//! - MaxSim scoring between query and document token embeddings
//! - Compatible with any embedding model (not just ColBERT-trained models)
//!
//! # Overview
//!
//! Traditional embedding models produce a single vector per text by pooling all token
//! representations. Late interaction models like ColBERT preserve individual token
//! embeddings and compute similarity at retrieval time using MaxSim:
//!
//! ```text
//! MaxSim(Q, D) = sum_{q in Q} max_{d in D} similarity(q, d)
//! ```
//!
//! This provides more fine-grained matching at the cost of storing more vectors.
//!
//! # Example
//!
//! ```rust,no_run
//! use mullama::{Model, late_interaction::{MultiVectorGenerator, MultiVectorConfig, LateInteractionScorer}};
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), mullama::MullamaError> {
//! let model = Arc::new(Model::load("model.gguf")?);
//! let config = MultiVectorConfig::default();
//! let mut generator = MultiVectorGenerator::new(model, config)?;
//!
//! let query = generator.embed_text("What is machine learning?")?;
//! let doc = generator.embed_text("Machine learning is a branch of AI...")?;
//!
//! let score = LateInteractionScorer::max_sim(&query, &doc);
//! println!("MaxSim score: {}", score);
//! # Ok(())
//! # }
//! ```
//!
//! # Model Compatibility
//!
//! This works with **any embedding model** that supports embeddings in llama.cpp.
//! However, models specifically trained for late interaction (like ColBERT) will
//! produce better retrieval results. Recommended models:
//!
//! - LiquidAI/LFM2-ColBERT-350M-GGUF (purpose-trained ColBERT)
//! - Any GGUF embedding model (works but suboptimal for retrieval)

use crate::context::{Context, ContextParams};
use crate::error::MullamaError;
use crate::model::Model;
use crate::sys;
use crate::token::TokenId;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================================
// MultiVectorEmbedding
// ============================================================================

/// Represents multi-vector (per-token) embeddings for late interaction retrieval.
///
/// Unlike single-vector embeddings that pool all tokens into one vector,
/// `MultiVectorEmbedding` preserves individual token embeddings for fine-grained
/// similarity computation using MaxSim or similar algorithms.
///
/// # Memory Layout
///
/// Embeddings are stored as a contiguous `Vec<f32>` where each token's embedding
/// occupies `dimension` consecutive elements. This layout is cache-friendly and
/// enables efficient SIMD operations.
#[derive(Debug, Clone)]
pub struct MultiVectorEmbedding {
    /// Raw embedding data: [n_tokens * dimension] as contiguous f32 values
    data: Vec<f32>,
    /// Embedding dimension per token
    dimension: usize,
    /// Number of token embeddings
    n_tokens: usize,
    /// Optional token IDs for debugging/analysis
    token_ids: Option<Vec<TokenId>>,
    /// Whether embeddings are L2 normalized
    normalized: bool,
}

impl MultiVectorEmbedding {
    /// Create a new MultiVectorEmbedding from raw data.
    ///
    /// # Arguments
    /// * `data` - Flattened embedding data of length `n_tokens * dimension`
    /// * `dimension` - Embedding dimension per token
    /// * `token_ids` - Optional token IDs for debugging
    ///
    /// # Panics
    /// Panics if `data.len()` is not divisible by `dimension` (when dimension > 0).
    pub fn new(data: Vec<f32>, dimension: usize, token_ids: Option<Vec<TokenId>>) -> Self {
        let n_tokens = if dimension > 0 {
            debug_assert!(
                data.len() % dimension == 0,
                "Data length {} is not divisible by dimension {}",
                data.len(),
                dimension
            );
            data.len() / dimension
        } else {
            0
        };

        Self {
            data,
            dimension,
            n_tokens,
            token_ids,
            normalized: false,
        }
    }

    /// Create an empty MultiVectorEmbedding with specified dimension.
    pub fn empty(dimension: usize) -> Self {
        Self {
            data: Vec::new(),
            dimension,
            n_tokens: 0,
            token_ids: None,
            normalized: false,
        }
    }

    /// Get the embedding for a specific token index.
    ///
    /// Returns `None` if index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        if index >= self.n_tokens {
            return None;
        }
        let start = index * self.dimension;
        let end = start + self.dimension;
        Some(&self.data[start..end])
    }

    /// Get mutable access to a specific token's embedding.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [f32]> {
        if index >= self.n_tokens {
            return None;
        }
        let start = index * self.dimension;
        let end = start + self.dimension;
        Some(&mut self.data[start..end])
    }

    /// Get the number of token embeddings.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_tokens
    }

    /// Check if there are no embeddings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_tokens == 0
    }

    /// Get the embedding dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get all embeddings as a contiguous slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable access to raw embedding data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Convert embeddings to a 2D vector representation.
    ///
    /// Returns a `Vec<Vec<f32>>` where each inner vector is one token's embedding.
    pub fn to_vecs(&self) -> Vec<Vec<f32>> {
        (0..self.n_tokens)
            .filter_map(|i| self.get(i).map(|e| e.to_vec()))
            .collect()
    }

    /// Get the token IDs if available.
    pub fn token_ids(&self) -> Option<&[TokenId]> {
        self.token_ids.as_deref()
    }

    /// Check if embeddings are L2 normalized.
    #[inline]
    pub fn is_normalized(&self) -> bool {
        self.normalized
    }

    /// Normalize all token embeddings to unit length (L2 normalization).
    ///
    /// This is recommended for MaxSim scoring as it makes dot product equivalent
    /// to cosine similarity, which is more numerically stable.
    pub fn normalize(&mut self) {
        if self.normalized {
            return;
        }

        for i in 0..self.n_tokens {
            let start = i * self.dimension;
            let end = start + self.dimension;
            let slice = &mut self.data[start..end];

            let magnitude: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > f32::EPSILON {
                for x in slice.iter_mut() {
                    *x /= magnitude;
                }
            }
        }
        self.normalized = true;
    }

    /// Create a normalized copy of the embeddings.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Iterate over individual token embeddings.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        (0..self.n_tokens).map(move |i| {
            let start = i * self.dimension;
            let end = start + self.dimension;
            &self.data[start..end]
        })
    }

    /// Get the total memory size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
            + self
                .token_ids
                .as_ref()
                .map(|ids| ids.len() * std::mem::size_of::<TokenId>())
                .unwrap_or(0)
    }
}

// ============================================================================
// MultiVectorConfig
// ============================================================================

/// Configuration for multi-vector (late interaction) embedding generation.
#[derive(Debug, Clone)]
pub struct MultiVectorConfig {
    /// Whether to L2 normalize each token embedding.
    ///
    /// Recommended for MaxSim scoring as it makes dot product = cosine similarity.
    /// Default: `true`
    pub normalize: bool,

    /// Whether to skip special tokens (BOS, EOS, PAD, etc.) in the output.
    ///
    /// Most retrieval models don't need special token embeddings.
    /// Default: `true`
    pub skip_special_tokens: bool,

    /// Whether to store token IDs alongside embeddings for debugging.
    ///
    /// Default: `false` (saves memory)
    pub store_token_ids: bool,

    /// Batch size for processing multiple texts (used in `embed_batch`).
    /// Default: `32`
    pub batch_size: usize,

    /// Maximum sequence length. 0 = use model default.
    /// Default: `0`
    pub max_seq_len: u32,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            skip_special_tokens: true,
            store_token_ids: false,
            batch_size: 32,
            max_seq_len: 0,
        }
    }
}

impl MultiVectorConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method: set L2 normalization.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Builder method: set special token handling.
    pub fn skip_special_tokens(mut self, skip: bool) -> Self {
        self.skip_special_tokens = skip;
        self
    }

    /// Builder method: set token ID storage.
    pub fn store_token_ids(mut self, store: bool) -> Self {
        self.store_token_ids = store;
        self
    }

    /// Builder method: set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Builder method: set max sequence length.
    pub fn max_seq_len(mut self, len: u32) -> Self {
        self.max_seq_len = len;
        self
    }
}

// ============================================================================
// MultiVectorGenerator
// ============================================================================

/// Generator for multi-vector (per-token) embeddings.
///
/// Uses `LLAMA_POOLING_TYPE_NONE` to extract individual token embeddings
/// instead of a pooled single vector. This is the core component for
/// ColBERT-style late interaction retrieval.
///
/// # Thread Safety
///
/// `MultiVectorGenerator` is **not** `Sync` because it holds a mutable `Context`.
/// Create separate generators for concurrent use, or use external synchronization.
pub struct MultiVectorGenerator {
    model: Arc<Model>,
    context: Context,
    config: MultiVectorConfig,
}

impl MultiVectorGenerator {
    /// Create a new multi-vector embedding generator.
    ///
    /// # Arguments
    /// * `model` - The model to use for generating embeddings
    /// * `config` - Configuration for embedding generation
    ///
    /// # Example
    /// ```rust,no_run
    /// use mullama::{Model, late_interaction::{MultiVectorGenerator, MultiVectorConfig}};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), mullama::MullamaError> {
    /// let model = Arc::new(Model::load("model.gguf")?);
    /// let config = MultiVectorConfig::default();
    /// let generator = MultiVectorGenerator::new(model, config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model: Arc<Model>, config: MultiVectorConfig) -> Result<Self, MullamaError> {
        // Create context with embeddings enabled and NO pooling
        let mut ctx_params = ContextParams::default();
        ctx_params.embeddings = true;
        ctx_params.pooling_type = sys::llama_pooling_type::LLAMA_POOLING_TYPE_NONE;

        if config.max_seq_len > 0 {
            ctx_params.n_ctx = config.max_seq_len;
        }

        let context = Context::new(model.clone(), ctx_params)?;

        Ok(Self {
            model,
            context,
            config,
        })
    }

    /// Generate multi-vector embeddings for text.
    ///
    /// Returns per-token embeddings that can be used for late interaction scoring.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use mullama::{Model, late_interaction::{MultiVectorGenerator, MultiVectorConfig}};
    /// # use std::sync::Arc;
    /// # fn main() -> Result<(), mullama::MullamaError> {
    /// # let model = Arc::new(Model::load("model.gguf")?);
    /// # let mut generator = MultiVectorGenerator::new(model, MultiVectorConfig::default())?;
    /// let mv = generator.embed_text("What is machine learning?")?;
    /// println!("Generated {} token embeddings", mv.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_text(&mut self, text: &str) -> Result<MultiVectorEmbedding, MullamaError> {
        let tokens = self.model.tokenize(text, true, false)?;
        self.embed_tokens(&tokens)
    }

    /// Generate multi-vector embeddings from pre-tokenized input.
    ///
    /// Use this when you've already tokenized the input or need fine control
    /// over tokenization.
    pub fn embed_tokens(
        &mut self,
        tokens: &[TokenId],
    ) -> Result<MultiVectorEmbedding, MullamaError> {
        if tokens.is_empty() {
            return Err(MullamaError::InvalidInput(
                "Cannot embed empty token sequence".to_string(),
            ));
        }

        // Clear KV cache for fresh generation
        self.context.kv_cache_clear();

        // Process tokens through the model
        self.context.decode(tokens)?;

        let n_embd = self.model.n_embd() as usize;
        let mut embeddings_data = Vec::with_capacity(tokens.len() * n_embd);
        let mut output_token_ids = if self.config.store_token_ids {
            Some(Vec::with_capacity(tokens.len()))
        } else {
            None
        };

        // Extract per-token embeddings using get_embeddings_ith
        for (i, &token) in tokens.iter().enumerate() {
            // Optionally skip special tokens
            if self.config.skip_special_tokens && self.is_special_token(token) {
                continue;
            }

            if let Some(emb) = self.context.get_embeddings_ith(i as i32) {
                embeddings_data.extend_from_slice(emb);
                if let Some(ref mut ids) = output_token_ids {
                    ids.push(token);
                }
            } else {
                return Err(MullamaError::EmbeddingError(format!(
                    "Failed to get embedding for token at index {}",
                    i
                )));
            }
        }

        let mut mv = MultiVectorEmbedding::new(embeddings_data, n_embd, output_token_ids);

        if self.config.normalize {
            mv.normalize();
        }

        Ok(mv)
    }

    /// Generate multi-vector embeddings for multiple texts.
    ///
    /// Processes texts sequentially. For parallel processing with rayon,
    /// use multiple generators or the parallel scoring functions.
    pub fn embed_batch(
        &mut self,
        texts: &[&str],
    ) -> Result<Vec<MultiVectorEmbedding>, MullamaError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_text(text)?);
        }
        Ok(results)
    }

    /// Get the embedding dimension.
    #[inline]
    pub fn embedding_dim(&self) -> usize {
        self.model.n_embd() as usize
    }

    /// Get the underlying model.
    #[inline]
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &MultiVectorConfig {
        &self.config
    }

    /// Check if a token is a special token (BOS, EOS, PAD, EOT, or control token).
    fn is_special_token(&self, token: TokenId) -> bool {
        token == self.model.token_bos()
            || token == self.model.token_eos()
            || token == self.model.token_pad()
            || token == self.model.token_eot()
            || self.model.token_is_control(token)
    }
}

// ============================================================================
// LateInteractionScorer
// ============================================================================

/// Scoring functions for late interaction retrieval.
///
/// Implements MaxSim and related scoring algorithms for comparing
/// multi-vector embeddings. These are the core retrieval functions
/// for ColBERT-style late interaction.
///
/// # Algorithm
///
/// MaxSim computes similarity by:
/// 1. For each query token, find the maximum similarity with any document token
/// 2. Sum these maximum similarities
///
/// ```text
/// MaxSim(Q, D) = sum_{q in Q} max_{d in D} sim(q, d)
/// ```
///
/// This allows fine-grained token-level matching while maintaining efficiency.
pub struct LateInteractionScorer;

impl LateInteractionScorer {
    /// Compute MaxSim score between query and document embeddings.
    ///
    /// For each query token, finds the maximum similarity with any document token,
    /// then sums these maximum similarities.
    ///
    /// # Arguments
    /// * `query` - Query multi-vector embedding
    /// * `document` - Document multi-vector embedding
    ///
    /// # Returns
    /// MaxSim score (higher = more similar). For normalized embeddings, the score
    /// is in range `[0, query.len()]`.
    ///
    /// # Example
    /// ```rust
    /// use mullama::late_interaction::{MultiVectorEmbedding, LateInteractionScorer};
    ///
    /// let query = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 1.0], 2, None);
    /// let doc = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 1.0], 2, None);
    ///
    /// let score = LateInteractionScorer::max_sim(&query, &doc);
    /// assert!(score > 0.0);
    /// ```
    pub fn max_sim(query: &MultiVectorEmbedding, document: &MultiVectorEmbedding) -> f32 {
        if query.is_empty() || document.is_empty() {
            return 0.0;
        }

        if query.dimension() != document.dimension() {
            return 0.0; // Dimension mismatch
        }

        let mut total_score = 0.0;

        // For each query token embedding
        for q_emb in query.iter() {
            let mut max_sim = f32::NEG_INFINITY;

            // Find max similarity with any document token
            for d_emb in document.iter() {
                let sim = Self::dot_product(q_emb, d_emb);
                if sim > max_sim {
                    max_sim = sim;
                }
            }

            if max_sim > f32::NEG_INFINITY {
                total_score += max_sim;
            }
        }

        total_score
    }

    /// Compute normalized MaxSim score.
    ///
    /// Divides the MaxSim score by the number of query tokens to get
    /// an average similarity per query token. This is useful when
    /// comparing scores across queries of different lengths.
    ///
    /// # Returns
    /// Normalized score in range `[0, 1]` for normalized embeddings.
    pub fn max_sim_normalized(
        query: &MultiVectorEmbedding,
        document: &MultiVectorEmbedding,
    ) -> f32 {
        if query.is_empty() {
            return 0.0;
        }
        Self::max_sim(query, document) / query.len() as f32
    }

    /// Compute symmetric MaxSim score.
    ///
    /// Averages the MaxSim in both directions: Q->D and D->Q.
    /// More robust for comparing documents of different lengths.
    pub fn max_sim_symmetric(a: &MultiVectorEmbedding, b: &MultiVectorEmbedding) -> f32 {
        let ab = Self::max_sim(a, b);
        let ba = Self::max_sim(b, a);
        (ab + ba) / 2.0
    }

    /// Find top-k documents by MaxSim score.
    ///
    /// # Arguments
    /// * `query` - Query multi-vector embedding
    /// * `documents` - List of document embeddings
    /// * `k` - Number of top results to return
    ///
    /// # Returns
    /// Vector of `(document_index, score)` pairs, sorted descending by score.
    pub fn find_top_k(
        query: &MultiVectorEmbedding,
        documents: &[MultiVectorEmbedding],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, Self::max_sim(query, doc)))
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Score all documents and return sorted results.
    ///
    /// Like `find_top_k` but returns all documents, not just top-k.
    pub fn rank_documents(
        query: &MultiVectorEmbedding,
        documents: &[MultiVectorEmbedding],
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, Self::max_sim(query, doc)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Compute per-token similarity matrix between query and document.
    ///
    /// Useful for visualization and analysis of which tokens match.
    ///
    /// # Returns
    /// A matrix of shape `[query.len(), document.len()]` where `matrix[i][j]`
    /// is the similarity between query token `i` and document token `j`.
    pub fn similarity_matrix(
        query: &MultiVectorEmbedding,
        document: &MultiVectorEmbedding,
    ) -> Vec<Vec<f32>> {
        let mut matrix = Vec::with_capacity(query.len());

        for q_emb in query.iter() {
            let mut row = Vec::with_capacity(document.len());
            for d_emb in document.iter() {
                row.push(Self::dot_product(q_emb, d_emb));
            }
            matrix.push(row);
        }

        matrix
    }

    /// Get the indices of best-matching document tokens for each query token.
    ///
    /// # Returns
    /// A vector where `result[i] = (doc_token_idx, similarity)` for the best
    /// matching document token of query token `i`.
    pub fn best_matches(
        query: &MultiVectorEmbedding,
        document: &MultiVectorEmbedding,
    ) -> Vec<(usize, f32)> {
        let mut matches = Vec::with_capacity(query.len());

        for q_emb in query.iter() {
            let mut best_idx = 0;
            let mut best_sim = f32::NEG_INFINITY;

            for (d_idx, d_emb) in document.iter().enumerate() {
                let sim = Self::dot_product(q_emb, d_emb);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = d_idx;
                }
            }

            matches.push((best_idx, best_sim));
        }

        matches
    }

    /// Compute multiple query scores against multiple documents.
    ///
    /// # Returns
    /// A 2D vector where `result[q][d]` is the MaxSim score between
    /// query `q` and document `d`.
    pub fn batch_score(
        queries: &[MultiVectorEmbedding],
        documents: &[MultiVectorEmbedding],
    ) -> Vec<Vec<f32>> {
        queries
            .iter()
            .map(|q| documents.iter().map(|d| Self::max_sim(q, d)).collect())
            .collect()
    }

    /// Compute dot product between two embedding vectors.
    #[inline]
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute cosine similarity between two embedding vectors.
    ///
    /// Note: If embeddings are pre-normalized, use `dot_product` instead for efficiency.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }
}

// ============================================================================
// Parallel Processing (with rayon feature)
// ============================================================================

#[cfg(feature = "parallel")]
impl LateInteractionScorer {
    /// Parallel top-k search across documents.
    ///
    /// Uses rayon for parallel scoring, which is significantly faster
    /// for large document collections.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use mullama::late_interaction::{MultiVectorEmbedding, LateInteractionScorer};
    /// # let query = MultiVectorEmbedding::empty(128);
    /// # let documents: Vec<MultiVectorEmbedding> = vec![];
    /// let top_k = LateInteractionScorer::find_top_k_parallel(&query, &documents, 10);
    /// ```
    pub fn find_top_k_parallel(
        query: &MultiVectorEmbedding,
        documents: &[MultiVectorEmbedding],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = documents
            .par_iter()
            .enumerate()
            .map(|(i, doc)| (i, Self::max_sim(query, doc)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Parallel scoring of multiple queries against multiple documents.
    ///
    /// Returns a 2D vector where `result[q][d]` is the MaxSim score.
    /// Queries are processed in parallel.
    pub fn batch_score_parallel(
        queries: &[MultiVectorEmbedding],
        documents: &[MultiVectorEmbedding],
    ) -> Vec<Vec<f32>> {
        queries
            .par_iter()
            .map(|q| documents.iter().map(|d| Self::max_sim(q, d)).collect())
            .collect()
    }

    /// Parallel document ranking.
    ///
    /// Scores all documents in parallel and returns sorted results.
    pub fn rank_documents_parallel(
        query: &MultiVectorEmbedding,
        documents: &[MultiVectorEmbedding],
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = documents
            .par_iter()
            .enumerate()
            .map(|(i, doc)| (i, Self::max_sim(query, doc)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_vector_embedding_creation() {
        // 2 tokens, dimension 3
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mv = MultiVectorEmbedding::new(data, 3, None);

        assert_eq!(mv.len(), 2);
        assert_eq!(mv.dimension(), 3);
        assert!(!mv.is_empty());
        assert_eq!(mv.get(0), Some(&[1.0, 0.0, 0.0][..]));
        assert_eq!(mv.get(1), Some(&[0.0, 1.0, 0.0][..]));
        assert_eq!(mv.get(2), None);
    }

    #[test]
    fn test_multi_vector_empty() {
        let mv = MultiVectorEmbedding::empty(128);
        assert!(mv.is_empty());
        assert_eq!(mv.len(), 0);
        assert_eq!(mv.dimension(), 128);
    }

    #[test]
    fn test_multi_vector_normalization() {
        let data = vec![3.0, 4.0, 0.0]; // magnitude = 5
        let mut mv = MultiVectorEmbedding::new(data, 3, None);

        assert!(!mv.is_normalized());
        mv.normalize();

        let emb = mv.get(0).unwrap();
        assert!((emb[0] - 0.6).abs() < 0.001);
        assert!((emb[1] - 0.8).abs() < 0.001);
        assert!(mv.is_normalized());

        // Normalizing again should be no-op
        mv.normalize();
        let emb = mv.get(0).unwrap();
        assert!((emb[0] - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_multi_vector_to_vecs() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mv = MultiVectorEmbedding::new(data, 2, None);

        let vecs = mv.to_vecs();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![1.0, 2.0]);
        assert_eq!(vecs[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_multi_vector_with_token_ids() {
        let data = vec![1.0, 0.0, 0.0, 1.0];
        let token_ids = Some(vec![100, 200]);
        let mv = MultiVectorEmbedding::new(data, 2, token_ids);

        assert_eq!(mv.token_ids(), Some(&[100, 200][..]));
    }

    #[test]
    fn test_max_sim_identical() {
        // Identical embeddings should have high MaxSim
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mv1 = MultiVectorEmbedding::new(data.clone(), 3, None);
        let mv2 = MultiVectorEmbedding::new(data, 3, None);

        let score = LateInteractionScorer::max_sim(&mv1, &mv2);
        assert!((score - 2.0).abs() < 0.001); // 1.0 + 1.0 for two tokens
    }

    #[test]
    fn test_max_sim_orthogonal() {
        // Orthogonal embeddings should have zero MaxSim
        let q_data = vec![1.0, 0.0, 0.0]; // Token pointing in X direction
        let d_data = vec![0.0, 1.0, 0.0]; // Token pointing in Y direction

        let query = MultiVectorEmbedding::new(q_data, 3, None);
        let doc = MultiVectorEmbedding::new(d_data, 3, None);

        let score = LateInteractionScorer::max_sim(&query, &doc);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_max_sim_empty() {
        let empty = MultiVectorEmbedding::empty(3);
        let non_empty = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0], 3, None);

        assert_eq!(LateInteractionScorer::max_sim(&empty, &non_empty), 0.0);
        assert_eq!(LateInteractionScorer::max_sim(&non_empty, &empty), 0.0);
    }

    #[test]
    fn test_max_sim_dimension_mismatch() {
        let mv1 = MultiVectorEmbedding::new(vec![1.0, 0.0], 2, None);
        let mv2 = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0], 3, None);

        assert_eq!(LateInteractionScorer::max_sim(&mv1, &mv2), 0.0);
    }

    #[test]
    fn test_max_sim_normalized() {
        let q = MultiVectorEmbedding::new(vec![1.0, 0.0, 1.0, 0.0], 2, None);
        let d = MultiVectorEmbedding::new(vec![1.0, 0.0], 2, None);

        let score = LateInteractionScorer::max_sim(&q, &d);
        let norm_score = LateInteractionScorer::max_sim_normalized(&q, &d);

        assert_eq!(norm_score, score / 2.0);
    }

    #[test]
    fn test_max_sim_symmetric() {
        let a = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.5, 0.5], 2, None);
        let b = MultiVectorEmbedding::new(vec![0.0, 1.0, 0.5, 0.5], 2, None);

        let ab = LateInteractionScorer::max_sim(&a, &b);
        let ba = LateInteractionScorer::max_sim(&b, &a);
        let sym = LateInteractionScorer::max_sim_symmetric(&a, &b);

        assert!((sym - (ab + ba) / 2.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_matrix() {
        let q = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 1.0], 2, None);
        let d = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2, None);

        let matrix = LateInteractionScorer::similarity_matrix(&q, &d);

        assert_eq!(matrix.len(), 2); // 2 query tokens
        assert_eq!(matrix[0].len(), 3); // 3 doc tokens

        // q[0] = [1,0] dot d[0] = [1,0] = 1.0
        assert!((matrix[0][0] - 1.0).abs() < 0.001);
        // q[0] = [1,0] dot d[1] = [0,1] = 0.0
        assert!((matrix[0][1] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_best_matches() {
        let q = MultiVectorEmbedding::new(vec![1.0, 0.0, 0.0, 1.0], 2, None);
        let d = MultiVectorEmbedding::new(vec![0.5, 0.5, 1.0, 0.0, 0.0, 1.0], 2, None);

        let matches = LateInteractionScorer::best_matches(&q, &d);

        assert_eq!(matches.len(), 2);
        // q[0] = [1,0] best matches d[1] = [1,0]
        assert_eq!(matches[0].0, 1);
        // q[1] = [0,1] best matches d[2] = [0,1]
        assert_eq!(matches[1].0, 2);
    }

    #[test]
    fn test_find_top_k() {
        let query = MultiVectorEmbedding::new(vec![1.0, 0.0], 2, None);

        let docs = vec![
            MultiVectorEmbedding::new(vec![0.0, 1.0], 2, None), // orthogonal, score ~ 0
            MultiVectorEmbedding::new(vec![1.0, 0.0], 2, None), // identical, score = 1
            MultiVectorEmbedding::new(vec![0.5, 0.5], 2, None), // partial, score = 0.5
        ];

        let top_k = LateInteractionScorer::find_top_k(&query, &docs, 2);

        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k[0].0, 1); // Best match is doc[1] (identical)
        assert_eq!(top_k[1].0, 2); // Second best is doc[2] (partial)
    }

    #[test]
    fn test_batch_score() {
        let queries = vec![
            MultiVectorEmbedding::new(vec![1.0, 0.0], 2, None),
            MultiVectorEmbedding::new(vec![0.0, 1.0], 2, None),
        ];

        let docs = vec![
            MultiVectorEmbedding::new(vec![1.0, 0.0], 2, None),
            MultiVectorEmbedding::new(vec![0.0, 1.0], 2, None),
        ];

        let scores = LateInteractionScorer::batch_score(&queries, &docs);

        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].len(), 2);

        // q[0] matches d[0], not d[1]
        assert!(scores[0][0] > scores[0][1]);
        // q[1] matches d[1], not d[0]
        assert!(scores[1][1] > scores[1][0]);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        assert!((LateInteractionScorer::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = [0.0, 1.0, 0.0];
        assert!((LateInteractionScorer::cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = [-1.0, 0.0, 0.0];
        assert!((LateInteractionScorer::cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_config_builder() {
        let config = MultiVectorConfig::default()
            .normalize(false)
            .skip_special_tokens(false)
            .store_token_ids(true)
            .batch_size(64)
            .max_seq_len(512);

        assert!(!config.normalize);
        assert!(!config.skip_special_tokens);
        assert!(config.store_token_ids);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.max_seq_len, 512);
    }

    #[test]
    fn test_size_bytes() {
        let data = vec![1.0f32; 100]; // 100 floats
        let mv = MultiVectorEmbedding::new(data, 10, None);

        assert_eq!(mv.size_bytes(), 100 * 4); // 100 floats * 4 bytes

        let data2 = vec![1.0f32; 100];
        let ids = Some(vec![0i32; 10]);
        let mv2 = MultiVectorEmbedding::new(data2, 10, ids);

        assert_eq!(mv2.size_bytes(), 100 * 4 + 10 * 4); // floats + token ids
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_find_top_k() {
        let query = MultiVectorEmbedding::new(vec![1.0, 0.0], 2, None);

        let docs: Vec<MultiVectorEmbedding> = (0..100)
            .map(|i| {
                let x = (i as f32) / 100.0;
                MultiVectorEmbedding::new(vec![x, 1.0 - x], 2, None)
            })
            .collect();

        let top_k = LateInteractionScorer::find_top_k_parallel(&query, &docs, 5);

        assert_eq!(top_k.len(), 5);
        // Highest scores should be for documents with high x component
        assert!(top_k[0].0 > 90);
    }
}
