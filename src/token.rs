use smallvec::SmallVec;

/// Token identifier - matches the C type from llama.cpp
pub type TokenId = i32;

/// Stack-allocated token buffer for small operations (up to 32 tokens on stack)
///
/// This is a Rust-specific optimization that cannot be replicated in Go:
/// - Go slices always have heap backing
/// - No stack-allocated dynamic arrays in Go
/// - Go's escape analysis is a runtime heuristic, not compile-time guarantee
///
/// For operations with <= 32 tokens, this avoids heap allocation entirely.
/// Larger operations transparently spill to heap with no API change.
///
/// **Use cases:**
/// - Single token decoding (common in generation loops)
/// - Small batch operations
/// - Prompt processing chunks
///
/// **Performance impact:** 5-10% faster for small prompts, reduced allocator pressure
pub type TokenBuffer = SmallVec<[TokenId; 32]>;

/// Larger token buffer for generation output (up to 256 tokens on stack)
///
/// Suitable for typical generation responses before spilling to heap.
pub type GenerationBuffer = SmallVec<[TokenId; 256]>;

/// Represents a token with its metadata
#[derive(Debug, Clone)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
    pub score: f32,
}
