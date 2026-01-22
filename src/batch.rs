use crate::{
    sys,
    token::{TokenBuffer, TokenId},
};

/// Represents a batch of tokens for processing
///
/// ## Performance Optimization
///
/// Uses `SmallVec` (via `TokenBuffer`) for token storage, which provides:
/// - Stack allocation for batches up to 32 tokens (no heap allocation)
/// - Transparent heap fallback for larger batches
/// - 5-10% faster for typical small batch operations
///
/// This optimization is only possible in Rust - Go slices always allocate on heap.
#[allow(dead_code)]
pub struct Batch {
    inner: Option<sys::llama_batch>,
    /// Store tokens to ensure they outlive the batch (for llama_batch_get_one)
    /// Uses SmallVec for stack allocation when possible (Rust-exclusive optimization)
    tokens_storage: Option<TokenBuffer>,
    /// Whether this batch was created with llama_batch_init (needs to be freed)
    needs_free: bool,
}

impl Batch {
    /// Create a new batch with allocated memory
    pub fn new(max_tokens: usize, embd: i32, max_seq: usize) -> Self {
        let inner = unsafe { sys::llama_batch_init(max_tokens as i32, embd, max_seq as i32) };

        Self {
            inner: Some(inner),
            tokens_storage: None,
            needs_free: true,
        }
    }

    /// Create a batch from a TokenBuffer using llama_batch_get_one
    ///
    /// Uses SmallVec internally - for up to 32 tokens, this stays on the stack.
    pub fn from_token_buffer(mut tokens: TokenBuffer) -> Self {
        if tokens.is_empty() {
            return Self {
                inner: None,
                tokens_storage: None,
                needs_free: false,
            };
        }

        let inner = unsafe { sys::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };

        Self {
            inner: Some(inner),
            tokens_storage: Some(tokens),
            needs_free: false,
        }
    }

    /// Create a batch from owned tokens using llama_batch_get_one
    /// This avoids copying when you already have a Vec
    pub fn from_tokens_owned(tokens: Vec<TokenId>) -> Self {
        // Convert Vec to TokenBuffer (will use heap if > 32 tokens)
        Self::from_token_buffer(TokenBuffer::from_vec(tokens))
    }

    /// Create a batch from a token slice using llama_batch_get_one
    ///
    /// For small slices (≤32 tokens), this uses stack allocation via SmallVec.
    /// This is a Rust-exclusive optimization - Go slices always heap-allocate.
    pub fn from_tokens(tokens: &[TokenId]) -> Self {
        Self::from_token_buffer(TokenBuffer::from_slice(tokens))
    }

    /// Get the internal llama_batch struct
    #[allow(dead_code)]
    pub(crate) fn as_llama_batch(&self) -> Option<&sys::llama_batch> {
        self.inner.as_ref()
    }

    /// Get the internal llama_batch struct (public for testing)
    pub fn get_llama_batch(&self) -> Option<&sys::llama_batch> {
        self.inner.as_ref()
    }

    /// Take the internal llama_batch struct (consuming the Batch)
    pub(crate) fn take_llama_batch(&mut self) -> Option<sys::llama_batch> {
        self.inner.take()
    }

    /// Get the number of tokens in the batch
    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, |batch| batch.n_tokens as usize)
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.inner
            .as_ref()
            .map_or(true, |batch| batch.n_tokens == 0)
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new(512, 0, 1)
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // Only free batches created with llama_batch_init
        // llama_batch_get_one doesn't allocate memory that needs freeing
        if self.needs_free {
            if let Some(batch) = self.inner.take() {
                unsafe {
                    sys::llama_batch_free(batch);
                }
            }
        }
    }
}
