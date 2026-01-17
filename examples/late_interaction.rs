//! Late Interaction / ColBERT-style Multi-Vector Embedding Example
//!
//! This example demonstrates:
//! 1. Creating multi-vector (per-token) embeddings
//! 2. MaxSim scoring for late interaction retrieval
//! 3. Top-k document ranking
//! 4. Token-level similarity analysis
//!
//! Run with: cargo run --example late_interaction --features late-interaction
//!
//! For real usage with a model:
//! cargo run --example late_interaction --features late-interaction -- path/to/model.gguf

use mullama::late_interaction::{
    LateInteractionScorer, MultiVectorConfig, MultiVectorEmbedding, MultiVectorGenerator,
};
use mullama::{Model, MullamaError};
use std::sync::Arc;

fn main() -> Result<(), MullamaError> {
    println!("=== Late Interaction / ColBERT Example ===\n");

    // Check if a model path was provided
    let model_path = std::env::args().nth(1);

    if let Some(path) = model_path {
        // Real usage with a model
        run_with_model(&path)?;
    } else {
        // Demo mode without a model
        run_demo()?;
    }

    Ok(())
}

/// Run the example with an actual model
fn run_with_model(model_path: &str) -> Result<(), MullamaError> {
    println!("Loading model: {}\n", model_path);
    let model = Arc::new(Model::load(model_path)?);

    // Create generator with default config (normalized, skip special tokens)
    let config = MultiVectorConfig::default()
        .normalize(true)
        .skip_special_tokens(true)
        .store_token_ids(true);

    let mut generator = MultiVectorGenerator::new(model.clone(), config)?;
    println!("Embedding dimension: {}\n", generator.embedding_dim());

    // Define query and documents
    let query_text = "What is machine learning?";
    let documents = [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        "The weather today is sunny with a high of 75 degrees Fahrenheit.",
        "Deep learning uses neural networks with many layers to learn representations.",
        "Supervised learning requires labeled training data to make predictions.",
    ];

    println!("Query: \"{}\"\n", query_text);

    // Generate query embedding
    let query_mv = generator.embed_text(query_text)?;
    println!("Query: {} tokens\n", query_mv.len());

    // Generate document embeddings
    println!("Scoring {} documents:\n", documents.len());
    println!("{}", "-".repeat(70));

    let doc_mvs: Vec<_> = documents
        .iter()
        .map(|d| generator.embed_text(d))
        .collect::<Result<Vec<_>, _>>()?;

    for (i, (doc_text, doc_mv)) in documents.iter().zip(doc_mvs.iter()).enumerate() {
        let score = LateInteractionScorer::max_sim(&query_mv, doc_mv);
        let norm_score = LateInteractionScorer::max_sim_normalized(&query_mv, doc_mv);

        println!(
            "Doc {}: score={:.4}, norm={:.4}, tokens={}",
            i + 1,
            score,
            norm_score,
            doc_mv.len()
        );
        println!("   \"{}\"", truncate_str(doc_text, 60));
    }

    // Find top-k documents
    println!("\n{}", "-".repeat(70));
    println!("Top-K Retrieval Results:\n");

    let top_k = LateInteractionScorer::find_top_k(&query_mv, &doc_mvs, 3);
    for (rank, (idx, score)) in top_k.iter().enumerate() {
        println!("  Rank {}: Doc {} (score: {:.4})", rank + 1, idx + 1, score);
        println!("     \"{}\"", truncate_str(documents[*idx], 55));
    }

    // Show token-level matches for top document
    if let Some((best_idx, _)) = top_k.first() {
        println!("\n{}", "-".repeat(70));
        println!("Token Matches (Query -> Top Doc):\n");

        let matches = LateInteractionScorer::best_matches(&query_mv, &doc_mvs[*best_idx]);
        for (q_idx, (d_idx, sim)) in matches.iter().enumerate().take(10) {
            println!("  Q[{:2}] -> D[{:2}]: {:.4}", q_idx, d_idx, sim);
        }
        if matches.len() > 10 {
            println!("  ... ({} more)", matches.len() - 10);
        }
    }

    println!("\nDone!");
    Ok(())
}

/// Demo mode showing API usage without requiring a model
fn run_demo() -> Result<(), MullamaError> {
    println!("Running in demo mode (no model loaded)\n");
    println!("To use with a real model:");
    println!("  cargo run --example late_interaction --features late-interaction -- model.gguf\n");
    println!("{}", "=".repeat(70));

    // Demonstrate MultiVectorEmbedding API
    println!("\n1. Creating Multi-Vector Embeddings\n");

    // Simulate a query with 3 tokens, dimension 4
    let query_data = vec![
        1.0, 0.0, 0.0, 0.0, // token 0: "what"
        0.0, 1.0, 0.0, 0.0, // token 1: "is"
        0.5, 0.5, 0.0, 0.0, // token 2: "learning"
    ];
    let mut query = MultiVectorEmbedding::new(query_data, 4, Some(vec![100, 200, 300]));

    println!(
        "   Query: {} tokens, {} dimensions",
        query.len(),
        query.dimension()
    );
    println!("   Token IDs: {:?}", query.token_ids());
    println!("   Memory: {} bytes", query.size_bytes());

    // Normalize for proper cosine similarity
    query.normalize();
    println!("   Normalized: {}", query.is_normalized());

    // Simulate documents
    let doc1_data = vec![
        1.0, 0.0, 0.0, 0.0, // Similar to query token 0
        0.0, 0.0, 1.0, 0.0, // Different
        0.6, 0.4, 0.0, 0.0, // Similar to query token 2
    ];
    let mut doc1 = MultiVectorEmbedding::new(doc1_data, 4, None);
    doc1.normalize();

    let doc2_data = vec![
        0.0, 0.0, 0.0, 1.0, // Orthogonal to query
        0.0, 0.0, 1.0, 0.0, // Different
    ];
    let mut doc2 = MultiVectorEmbedding::new(doc2_data, 4, None);
    doc2.normalize();

    let doc3_data = vec![
        0.9, 0.1, 0.0, 0.0, // Very similar to query token 0
        0.1, 0.9, 0.0, 0.0, // Very similar to query token 1
        0.5, 0.5, 0.0, 0.0, // Same as query token 2
    ];
    let mut doc3 = MultiVectorEmbedding::new(doc3_data, 4, None);
    doc3.normalize();

    println!("\n   Created 3 documents with {} tokens each (approx)", 3);

    // Demonstrate MaxSim scoring
    println!("\n2. MaxSim Scoring\n");

    let documents = [&doc1, &doc2, &doc3];
    for (i, doc) in documents.iter().enumerate() {
        let score = LateInteractionScorer::max_sim(&query, doc);
        let norm_score = LateInteractionScorer::max_sim_normalized(&query, doc);
        println!(
            "   Doc {}: MaxSim={:.4}, Normalized={:.4}",
            i + 1,
            score,
            norm_score
        );
    }

    // Demonstrate top-k retrieval
    println!("\n3. Top-K Retrieval\n");

    let doc_vec: Vec<MultiVectorEmbedding> = vec![doc1.clone(), doc2.clone(), doc3.clone()];
    let top_k = LateInteractionScorer::find_top_k(&query, &doc_vec, 2);

    for (rank, (idx, score)) in top_k.iter().enumerate() {
        println!(
            "   Rank {}: Document {} (score: {:.4})",
            rank + 1,
            idx + 1,
            score
        );
    }

    // Demonstrate similarity matrix
    println!("\n4. Token Similarity Matrix (Query vs Doc 3)\n");

    let matrix = LateInteractionScorer::similarity_matrix(&query, &doc3);
    println!("   Matrix shape: {}x{}", matrix.len(), matrix[0].len());
    for (q_idx, row) in matrix.iter().enumerate() {
        let row_str: Vec<String> = row.iter().map(|v| format!("{:.2}", v)).collect();
        println!("   Q[{}]: [{}]", q_idx, row_str.join(", "));
    }

    // Demonstrate best matches
    println!("\n5. Best Token Matches (Query -> Doc 3)\n");

    let matches = LateInteractionScorer::best_matches(&query, &doc3);
    for (q_idx, (d_idx, sim)) in matches.iter().enumerate() {
        println!(
            "   Query token {} -> Doc token {}: similarity {:.4}",
            q_idx, d_idx, sim
        );
    }

    // Demonstrate symmetric scoring
    println!("\n6. Symmetric MaxSim\n");

    let sym_score = LateInteractionScorer::max_sim_symmetric(&query, &doc3);
    let forward = LateInteractionScorer::max_sim(&query, &doc3);
    let backward = LateInteractionScorer::max_sim(&doc3, &query);
    println!("   Forward (Q->D):  {:.4}", forward);
    println!("   Backward (D->Q): {:.4}", backward);
    println!("   Symmetric:       {:.4}", sym_score);

    // Demonstrate batch scoring
    println!("\n7. Batch Scoring\n");

    let queries = vec![query.clone()];
    let scores = LateInteractionScorer::batch_score(&queries, &doc_vec);
    println!(
        "   Query 1 vs all docs: {:?}",
        scores[0]
            .iter()
            .map(|s| format!("{:.2}", s))
            .collect::<Vec<_>>()
    );

    // Demonstrate config
    println!("\n8. Configuration Options\n");

    let config = MultiVectorConfig::default()
        .normalize(true)
        .skip_special_tokens(true)
        .store_token_ids(false)
        .batch_size(32)
        .max_seq_len(512);

    println!("   normalize: {}", config.normalize);
    println!("   skip_special_tokens: {}", config.skip_special_tokens);
    println!("   store_token_ids: {}", config.store_token_ids);
    println!("   batch_size: {}", config.batch_size);
    println!("   max_seq_len: {}", config.max_seq_len);

    println!("\n{}", "=".repeat(70));
    println!("\nDemo completed! Use a real model for actual retrieval.");
    println!("\nRecommended ColBERT models:");
    println!("  - LiquidAI/LFM2-ColBERT-350M-GGUF (purpose-trained)");
    println!("  - Any GGUF embedding model (works but suboptimal)");

    Ok(())
}

/// Truncate a string to max length, adding "..." if truncated
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
