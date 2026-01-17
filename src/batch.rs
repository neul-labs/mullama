use crate::{sys, token::TokenId};

/// Represents a batch of tokens for processing
#[allow(dead_code)]
pub struct Batch {
    inner: Option<sys::llama_batch>,
    /// Store tokens to ensure they outlive the batch (for llama_batch_get_one)
    tokens_storage: Option<Vec<TokenId>>,
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

    /// Create a batch from owned tokens using llama_batch_get_one
    /// This avoids copying when you already have a Vec
    pub fn from_tokens_owned(mut tokens: Vec<TokenId>) -> Self {
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

    /// Create a batch from a token slice using llama_batch_get_one
    /// This copies the tokens - use from_tokens_owned if you already have a Vec
    pub fn from_tokens(tokens: &[TokenId]) -> Self {
        Self::from_tokens_owned(tokens.to_vec())
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
