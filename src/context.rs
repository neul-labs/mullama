use crate::{
    batch::Batch, error::MullamaError, model::Model, sampling::SamplerParams, sys, token::TokenId,
};
use std::sync::Arc;

/// Parameters for creating a context
#[derive(Debug, Clone)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: sys::llama_rope_scaling_type,
    pub pooling_type: sys::llama_pooling_type,
    pub attention_type: sys::llama_attention_type,
    pub flash_attn_type: sys::llama_flash_attn_type,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 0, // Use model default
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: num_cpus::get() as i32,
            n_threads_batch: num_cpus::get() as i32,
            rope_scaling_type: sys::llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
            pooling_type: sys::llama_pooling_type::LLAMA_POOLING_TYPE_UNSPECIFIED,
            attention_type: sys::llama_attention_type::LLAMA_ATTENTION_TYPE_UNSPECIFIED,
            flash_attn_type: sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            defrag_thold: -1.0,
            embeddings: false,
            offload_kqv: true,
            no_perf: false,
            op_offload: false,
            swa_full: true,
            kv_unified: false,
        }
    }
}

/// Represents a model context for inference
pub struct Context {
    pub model: Arc<Model>,
    pub ctx_ptr: *mut sys::llama_context,
}

impl Context {
    /// Create a new context from a model
    pub fn new(model: Arc<Model>, params: ContextParams) -> Result<Self, MullamaError> {
        // Get default context parameters
        let mut llama_params = unsafe { sys::llama_context_default_params() };

        // Apply our parameters
        llama_params.n_ctx = params.n_ctx;
        llama_params.n_batch = params.n_batch;
        llama_params.n_ubatch = params.n_ubatch;
        llama_params.n_seq_max = params.n_seq_max;
        llama_params.n_threads = params.n_threads;
        llama_params.n_threads_batch = params.n_threads_batch;
        llama_params.rope_scaling_type = params.rope_scaling_type;
        llama_params.pooling_type = params.pooling_type;
        llama_params.attention_type = params.attention_type;
        llama_params.flash_attn_type = params.flash_attn_type;
        llama_params.rope_freq_base = params.rope_freq_base;
        llama_params.rope_freq_scale = params.rope_freq_scale;
        llama_params.yarn_ext_factor = params.yarn_ext_factor;
        llama_params.yarn_attn_factor = params.yarn_attn_factor;
        llama_params.yarn_beta_fast = params.yarn_beta_fast;
        llama_params.yarn_beta_slow = params.yarn_beta_slow;
        llama_params.yarn_orig_ctx = params.yarn_orig_ctx;
        llama_params.defrag_thold = params.defrag_thold;
        llama_params.embeddings = params.embeddings;
        llama_params.offload_kqv = params.offload_kqv;
        llama_params.no_perf = params.no_perf;
        llama_params.op_offload = params.op_offload;
        llama_params.swa_full = params.swa_full;
        llama_params.kv_unified = params.kv_unified;

        // Create the context
        let ctx_ptr = unsafe { sys::llama_init_from_model(model.as_ptr(), llama_params) };

        if ctx_ptr.is_null() {
            return Err(MullamaError::ContextError(
                "Failed to create context".to_string(),
            ));
        }

        Ok(Context { model, ctx_ptr })
    }

    /// Process a batch of tokens
    ///
    /// If the number of tokens exceeds the batch size (n_batch), they will be
    /// automatically processed in chunks.
    pub fn decode(&mut self, tokens: &[TokenId]) -> Result<(), MullamaError> {
        if tokens.is_empty() {
            return Ok(());
        }

        let batch_size = self.n_batch() as usize;

        // Process tokens in chunks if they exceed batch size
        for chunk in tokens.chunks(batch_size) {
            let mut batch = Batch::from_tokens(chunk);

            // Get the llama_batch and call llama_decode
            if let Some(llama_batch) = batch.take_llama_batch() {
                let result = unsafe { sys::llama_decode(self.ctx_ptr, llama_batch) };

                if result != 0 {
                    return Err(MullamaError::GenerationError(format!(
                        "Decode failed with code: {}",
                        result
                    )));
                }
            }
        }

        Ok(())
    }

    /// Decode a single token - optimized to avoid allocation
    #[inline]
    pub fn decode_single(&mut self, token: TokenId) -> Result<(), MullamaError> {
        // Use stack-allocated array and llama_batch_get_one directly
        let mut tokens = [token];
        let llama_batch = unsafe { sys::llama_batch_get_one(tokens.as_mut_ptr(), 1) };
        let result = unsafe { sys::llama_decode(self.ctx_ptr, llama_batch) };
        if result != 0 {
            return Err(MullamaError::GenerationError(format!(
                "Decode failed with code: {}",
                result
            )));
        }
        Ok(())
    }

    /// Generate text from prompt tokens using default sampling parameters
    ///
    /// This is the main generation method that:
    /// 1. Processes the prompt through the model
    /// 2. Samples tokens using a sampler chain
    /// 3. Returns the generated text
    ///
    /// For custom sampling parameters, use `generate_with_params()`.
    pub fn generate(
        &mut self,
        prompt_tokens: &[TokenId],
        max_tokens: usize,
    ) -> Result<String, MullamaError> {
        self.generate_with_params(prompt_tokens, max_tokens, &SamplerParams::default())
    }

    /// Generate text with custom sampling parameters
    ///
    /// # Arguments
    /// * `prompt_tokens` - Tokenized prompt
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `sampler_params` - Sampling parameters (temperature, top-k, top-p, etc.)
    ///
    /// # Example
    /// ```rust,ignore
    /// let params = SamplerParams {
    ///     temperature: 0.7,
    ///     top_k: 50,
    ///     top_p: 0.9,
    ///     ..Default::default()
    /// };
    /// let text = context.generate_with_params(&tokens, 100, &params)?;
    /// ```
    pub fn generate_with_params(
        &mut self,
        prompt_tokens: &[TokenId],
        max_tokens: usize,
        sampler_params: &SamplerParams,
    ) -> Result<String, MullamaError> {
        if prompt_tokens.is_empty() {
            return Err(MullamaError::GenerationError(
                "Empty prompt tokens".to_string(),
            ));
        }

        // Process the prompt
        self.decode(prompt_tokens)?;

        // Create sampler chain with the provided parameters
        let mut sampler = sampler_params.build_chain(self.model.clone())?;

        // Generation loop
        let mut generated_tokens = Vec::new();
        for _ in 0..max_tokens {
            // Sample next token (idx=-1 for last token's logits)
            let token = sampler.sample(self, -1);
            sampler.accept(token);

            // Check for end of generation
            if self.model.token_is_eog(token) {
                break;
            }

            generated_tokens.push(token);

            // Decode the new token for next iteration
            self.decode(&[token])?;
        }

        // Convert tokens to text
        self.model.detokenize(&generated_tokens, false, false)
    }

    /// Generate text with streaming callback for real-time output
    ///
    /// The callback is called for each generated token with the token's text.
    /// Return `false` from the callback to stop generation early.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Tokenized prompt
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `sampler_params` - Sampling parameters
    /// * `callback` - Called with each token's text; return false to stop
    ///
    /// # Example
    /// ```rust,ignore
    /// context.generate_streaming(&tokens, 100, &SamplerParams::default(), |text| {
    ///     print!("{}", text);
    ///     std::io::stdout().flush().unwrap();
    ///     true // Continue generation
    /// })?;
    /// ```
    pub fn generate_streaming<F>(
        &mut self,
        prompt_tokens: &[TokenId],
        max_tokens: usize,
        sampler_params: &SamplerParams,
        mut callback: F,
    ) -> Result<String, MullamaError>
    where
        F: FnMut(&str) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err(MullamaError::GenerationError(
                "Empty prompt tokens".to_string(),
            ));
        }

        // Process the prompt
        self.decode(prompt_tokens)?;

        // Create sampler chain
        let mut sampler = sampler_params.build_chain(self.model.clone())?;

        // Generation loop with streaming
        let mut generated_tokens = Vec::new();
        for _ in 0..max_tokens {
            // Sample next token
            let token = sampler.sample(self, -1);
            sampler.accept(token);

            // Check for end of generation
            if self.model.token_is_eog(token) {
                break;
            }

            generated_tokens.push(token);

            // Convert token to text and call callback
            let piece = self.model.token_to_str(token, 0, false)?;
            if !callback(&piece) {
                // User requested stop
                break;
            }

            // Decode the new token for next iteration
            self.decode(&[token])?;
        }

        // Return the full generated text
        self.model.detokenize(&generated_tokens, false, false)
    }

    /// Get logits from the last evaluation
    ///
    /// Returns a slice of logits for the last token in the batch.
    /// Use get_logits() for all tokens or get_logits_ith() for specific positions.
    pub fn logits(&self) -> Result<&[f32], MullamaError> {
        let logits = self.get_logits();
        if logits.is_empty() {
            return Err(MullamaError::GenerationError(
                "No logits available - call decode() first".to_string(),
            ));
        }
        Ok(logits)
    }

    /// Get embeddings (if enabled)
    ///
    /// Returns embeddings if the context was created with embeddings=true.
    /// Use get_embeddings_ith() or get_embeddings_seq() for specific positions.
    pub fn embeddings(&self) -> Result<Option<&[f32]>, MullamaError> {
        Ok(self.get_embeddings())
    }

    /// Get the model associated with this context
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Get the internal context pointer (for use by other modules)
    pub fn as_ptr(&self) -> *mut sys::llama_context {
        self.ctx_ptr
    }

    // ==================== Context Getters ====================

    /// Get the context size (number of tokens that can be processed)
    pub fn n_ctx(&self) -> u32 {
        unsafe { sys::llama_n_ctx(self.ctx_ptr) }
    }

    /// Get the logical batch size for processing
    pub fn n_batch(&self) -> u32 {
        unsafe { sys::llama_n_batch(self.ctx_ptr) }
    }

    /// Get the physical batch size (used for memory allocation)
    pub fn n_ubatch(&self) -> u32 {
        unsafe { sys::llama_n_ubatch(self.ctx_ptr) }
    }

    /// Get maximum number of sequences
    pub fn n_seq_max(&self) -> u32 {
        unsafe { sys::llama_n_seq_max(self.ctx_ptr) }
    }

    /// Get number of threads used for generation
    pub fn n_threads(&self) -> i32 {
        unsafe { sys::llama_n_threads(self.ctx_ptr) }
    }

    /// Get number of threads used for batch processing
    pub fn n_threads_batch(&self) -> i32 {
        unsafe { sys::llama_n_threads_batch(self.ctx_ptr) }
    }

    /// Set number of threads for generation and batch processing
    pub fn set_n_threads(&mut self, n_threads: i32, n_threads_batch: i32) {
        unsafe {
            sys::llama_set_n_threads(self.ctx_ptr, n_threads, n_threads_batch);
        }
    }

    /// Get the pooling type used by this context
    pub fn pooling_type(&self) -> sys::llama_pooling_type {
        unsafe { sys::llama_get_pooling_type(self.ctx_ptr) }
    }

    /// Set whether to compute embeddings
    pub fn set_embeddings(&mut self, enabled: bool) {
        unsafe {
            sys::llama_set_embeddings(self.ctx_ptr, enabled);
        }
    }

    /// Set causal attention mode
    pub fn set_causal_attn(&mut self, causal: bool) {
        unsafe {
            sys::llama_set_causal_attn(self.ctx_ptr, causal);
        }
    }

    /// Set warmup mode (skip computing output for speedup during warmup)
    pub fn set_warmup(&mut self, warmup: bool) {
        unsafe {
            sys::llama_set_warmup(self.ctx_ptr, warmup);
        }
    }

    /// Synchronize the context (wait for pending operations)
    pub fn synchronize(&mut self) {
        unsafe {
            sys::llama_synchronize(self.ctx_ptr);
        }
    }

    // ==================== Logits Access ====================

    /// Get logits for all tokens in the batch
    ///
    /// Returns a slice of shape [n_tokens, n_vocab] where n_tokens
    /// is the number of tokens in the last decoded batch.
    pub fn get_logits(&self) -> &[f32] {
        let n_vocab = self.model.vocab_size() as usize;
        unsafe {
            let ptr = sys::llama_get_logits(self.ctx_ptr);
            if ptr.is_null() {
                &[]
            } else {
                // Note: This returns logits for all evaluated tokens
                // The actual size depends on the batch that was decoded
                std::slice::from_raw_parts(ptr, n_vocab)
            }
        }
    }

    /// Get logits for a specific token index in the batch
    ///
    /// This is more efficient than get_logits() when you only need
    /// logits for one position (e.g., the last token for generation).
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        let n_vocab = self.model.vocab_size() as usize;
        unsafe {
            let ptr = sys::llama_get_logits_ith(self.ctx_ptr, i);
            if ptr.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, n_vocab)
            }
        }
    }

    // ==================== Embeddings Access ====================

    /// Get embeddings for all tokens (requires embeddings mode)
    ///
    /// Returns a slice of shape [n_tokens, n_embd].
    pub fn get_embeddings(&self) -> Option<&[f32]> {
        let n_embd = self.model.n_embd() as usize;
        unsafe {
            let ptr = sys::llama_get_embeddings(self.ctx_ptr);
            if ptr.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(ptr, n_embd))
            }
        }
    }

    /// Get embeddings for a specific token index
    pub fn get_embeddings_ith(&self, i: i32) -> Option<&[f32]> {
        let n_embd = self.model.n_embd() as usize;
        unsafe {
            let ptr = sys::llama_get_embeddings_ith(self.ctx_ptr, i);
            if ptr.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(ptr, n_embd))
            }
        }
    }

    /// Get embeddings for a specific sequence
    pub fn get_embeddings_seq(&self, seq_id: i32) -> Option<&[f32]> {
        let n_embd = self.model.n_embd() as usize;
        unsafe {
            let ptr = sys::llama_get_embeddings_seq(self.ctx_ptr, seq_id);
            if ptr.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(ptr, n_embd))
            }
        }
    }

    // ==================== Memory/KV Cache Operations ====================

    /// Get the memory handle for this context
    fn get_memory(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx_ptr) }
    }

    /// Clear the memory/KV cache - removes all tokens
    ///
    /// Use this to reset the context for a new conversation.
    /// If `data` is true, also clear the underlying data.
    pub fn kv_cache_clear(&mut self) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_clear(mem, false);
        }
    }

    /// Remove tokens from the memory/KV cache
    ///
    /// Removes all tokens in positions [p0, p1) for the specified sequence.
    /// Set seq_id to -1 to remove from all sequences.
    /// Returns true if successful.
    pub fn kv_cache_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_rm(mem, seq_id, p0, p1)
        }
    }

    /// Copy tokens from one sequence to another in the memory/KV cache
    ///
    /// Copies all tokens in positions [p0, p1) from seq_id_src to seq_id_dst.
    pub fn kv_cache_seq_cp(&mut self, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
        }
    }

    /// Keep only the specified sequence, removing all others
    pub fn kv_cache_seq_keep(&mut self, seq_id: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_keep(mem, seq_id);
        }
    }

    /// Shift token positions in the memory/KV cache
    ///
    /// Adds delta to all token positions in [p0, p1) for the specified sequence.
    /// Useful for context shifting when the cache is full.
    pub fn kv_cache_seq_add(&mut self, seq_id: i32, p0: i32, p1: i32, delta: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_add(mem, seq_id, p0, p1, delta);
        }
    }

    /// Divide positions by a factor (for position interpolation)
    pub fn kv_cache_seq_div(&mut self, seq_id: i32, p0: i32, p1: i32, d: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_div(mem, seq_id, p0, p1, d);
        }
    }

    /// Get the minimum position in the memory/KV cache for a sequence
    ///
    /// Returns -1 if the sequence is empty.
    pub fn kv_cache_seq_pos_min(&self, seq_id: i32) -> i32 {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_pos_min(mem, seq_id)
        }
    }

    /// Get the maximum position in the memory/KV cache for a sequence
    ///
    /// Returns -1 if the sequence is empty.
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_pos_max(mem, seq_id)
        }
    }

    /// Check if the memory/KV cache supports shifting
    pub fn kv_cache_can_shift(&self) -> bool {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_can_shift(mem)
        }
    }

    // ==================== State Management ====================

    /// Get the size of the state in bytes
    pub fn state_size(&self) -> usize {
        unsafe { sys::llama_state_get_size(self.ctx_ptr) }
    }

    /// Save the context state to a byte vector
    ///
    /// This includes the complete KV cache and RNG state.
    pub fn save_state(&self) -> Vec<u8> {
        let size = self.state_size();
        let mut buffer = vec![0u8; size];
        unsafe {
            let written = sys::llama_state_get_data(self.ctx_ptr, buffer.as_mut_ptr(), size);
            buffer.truncate(written);
        }
        buffer
    }

    /// Load context state from a byte slice
    ///
    /// Returns the number of bytes read.
    pub fn load_state(&mut self, data: &[u8]) -> Result<usize, MullamaError> {
        let read = unsafe { sys::llama_state_set_data(self.ctx_ptr, data.as_ptr(), data.len()) };
        if read == 0 && !data.is_empty() {
            Err(MullamaError::ContextError(
                "Failed to load state".to_string(),
            ))
        } else {
            Ok(read)
        }
    }

    /// Save state to a file
    pub fn save_state_file(&self, path: &str, tokens: &[i32]) -> Result<(), MullamaError> {
        let c_path = std::ffi::CString::new(path)
            .map_err(|_| MullamaError::InvalidInput("Invalid path".to_string()))?;

        let result = unsafe {
            sys::llama_state_save_file(self.ctx_ptr, c_path.as_ptr(), tokens.as_ptr(), tokens.len())
        };

        if result {
            Ok(())
        } else {
            Err(MullamaError::ContextError(
                "Failed to save state file".to_string(),
            ))
        }
    }

    /// Load state from a file
    ///
    /// Returns the tokens that were saved with the state.
    pub fn load_state_file(&mut self, path: &str) -> Result<Vec<i32>, MullamaError> {
        let c_path = std::ffi::CString::new(path)
            .map_err(|_| MullamaError::InvalidInput("Invalid path".to_string()))?;

        // Allocate buffer for tokens (estimate max based on context size)
        let max_tokens = self.n_ctx() as usize;
        let mut tokens = vec![0i32; max_tokens];
        let mut n_tokens = max_tokens;

        let result = unsafe {
            sys::llama_state_load_file(
                self.ctx_ptr,
                c_path.as_ptr(),
                tokens.as_mut_ptr(),
                max_tokens,
                &mut n_tokens,
            )
        };

        if result {
            tokens.truncate(n_tokens);
            Ok(tokens)
        } else {
            Err(MullamaError::ContextError(
                "Failed to load state file".to_string(),
            ))
        }
    }

    // ==================== Memory Operations (newer API) ====================

    /// Clear all memory (newer API, equivalent to kv_cache_clear)
    pub fn memory_clear(&mut self) {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if !memory.is_null() {
            unsafe {
                sys::llama_memory_clear(memory, true);
            }
        }
    }

    /// Remove tokens from memory for a sequence
    pub fn memory_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if memory.is_null() {
            return false;
        }
        unsafe { sys::llama_memory_seq_rm(memory, seq_id, p0, p1) }
    }

    /// Copy tokens between sequences in memory
    pub fn memory_seq_cp(&mut self, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if !memory.is_null() {
            unsafe {
                sys::llama_memory_seq_cp(memory, seq_id_src, seq_id_dst, p0, p1);
            }
        }
    }

    /// Keep only the specified sequence in memory
    pub fn memory_seq_keep(&mut self, seq_id: i32) {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if !memory.is_null() {
            unsafe {
                sys::llama_memory_seq_keep(memory, seq_id);
            }
        }
    }

    /// Add delta to positions in memory
    pub fn memory_seq_add(&mut self, seq_id: i32, p0: i32, p1: i32, delta: i32) {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if !memory.is_null() {
            unsafe {
                sys::llama_memory_seq_add(memory, seq_id, p0, p1, delta);
            }
        }
    }

    /// Divide positions in memory
    pub fn memory_seq_div(&mut self, seq_id: i32, p0: i32, p1: i32, d: i32) {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if !memory.is_null() {
            unsafe {
                sys::llama_memory_seq_div(memory, seq_id, p0, p1, d);
            }
        }
    }

    /// Get max position in memory for a sequence
    pub fn memory_seq_pos_max(&self, seq_id: i32) -> i32 {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if memory.is_null() {
            return -1;
        }
        unsafe { sys::llama_memory_seq_pos_max(memory, seq_id) }
    }

    /// Get min position in memory for a sequence
    pub fn memory_seq_pos_min(&self, seq_id: i32) -> i32 {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if memory.is_null() {
            return -1;
        }
        unsafe { sys::llama_memory_seq_pos_min(memory, seq_id) }
    }

    /// Check if memory supports shifting
    pub fn memory_can_shift(&self) -> bool {
        let memory = unsafe { sys::llama_get_memory(self.ctx_ptr) };
        if memory.is_null() {
            return false;
        }
        unsafe { sys::llama_memory_can_shift(memory) }
    }

    // ==================== State Sequence Operations ====================

    /// Get the size of state data for a specific sequence
    pub fn state_seq_size(&self, seq_id: i32) -> usize {
        unsafe { sys::llama_state_seq_get_size(self.ctx_ptr, seq_id) }
    }

    /// Save state for a specific sequence
    pub fn save_state_seq(&self, seq_id: i32) -> Vec<u8> {
        let size = self.state_seq_size(seq_id);
        let mut buffer = vec![0u8; size];
        unsafe {
            let written =
                sys::llama_state_seq_get_data(self.ctx_ptr, buffer.as_mut_ptr(), size, seq_id);
            buffer.truncate(written);
        }
        buffer
    }

    /// Load state for a specific sequence
    pub fn load_state_seq(&mut self, seq_id: i32, data: &[u8]) -> Result<usize, MullamaError> {
        let read = unsafe {
            sys::llama_state_seq_set_data(self.ctx_ptr, data.as_ptr(), data.len(), seq_id)
        };
        if read == 0 && !data.is_empty() {
            Err(MullamaError::ContextError(
                "Failed to load sequence state".to_string(),
            ))
        } else {
            Ok(read)
        }
    }

    /// Save sequence state to file
    pub fn save_state_seq_file(&self, path: &str, seq_id: i32) -> Result<(), MullamaError> {
        let c_path = std::ffi::CString::new(path)
            .map_err(|_| MullamaError::InvalidInput("Invalid path".to_string()))?;

        let result = unsafe {
            sys::llama_state_seq_save_file(
                self.ctx_ptr,
                c_path.as_ptr(),
                seq_id,
                std::ptr::null(),
                0,
            )
        };

        if result > 0 {
            Ok(())
        } else {
            Err(MullamaError::ContextError(
                "Failed to save sequence state file".to_string(),
            ))
        }
    }

    /// Load sequence state from file
    pub fn load_state_seq_file(
        &mut self,
        path: &str,
        dest_seq_id: i32,
    ) -> Result<(), MullamaError> {
        let c_path = std::ffi::CString::new(path)
            .map_err(|_| MullamaError::InvalidInput("Invalid path".to_string()))?;

        let mut n_token_count = 0usize;
        let result = unsafe {
            sys::llama_state_seq_load_file(
                self.ctx_ptr,
                c_path.as_ptr(),
                dest_seq_id,
                std::ptr::null_mut(),
                0,
                &mut n_token_count,
            )
        };

        if result > 0 {
            Ok(())
        } else {
            Err(MullamaError::ContextError(
                "Failed to load sequence state file".to_string(),
            ))
        }
    }

    // ==================== Encoding ====================

    /// Encode a batch (for encoder-decoder models)
    pub fn encode(&mut self, batch: &mut Batch) -> Result<(), MullamaError> {
        if let Some(llama_batch) = batch.take_llama_batch() {
            let result = unsafe { sys::llama_encode(self.ctx_ptr, llama_batch) };

            if result != 0 {
                return Err(MullamaError::ContextError(format!(
                    "Encode failed with code: {}",
                    result
                )));
            }
        }
        Ok(())
    }

    // ==================== Performance ====================

    /// Get performance timings for this context
    pub fn perf_data(&self) -> ContextPerfData {
        let data = unsafe { sys::llama_perf_context(self.ctx_ptr) };
        ContextPerfData {
            t_start_ms: data.t_start_ms,
            t_load_ms: data.t_load_ms,
            t_p_eval_ms: data.t_p_eval_ms,
            t_eval_ms: data.t_eval_ms,
            n_p_eval: data.n_p_eval,
            n_eval: data.n_eval,
        }
    }

    /// Print performance information to stderr
    pub fn perf_print(&self) {
        unsafe {
            sys::llama_perf_context_print(self.ctx_ptr);
        }
    }

    /// Reset performance counters
    pub fn perf_reset(&mut self) {
        unsafe {
            sys::llama_perf_context_reset(self.ctx_ptr);
        }
    }

    // ==================== Advanced Operations ====================

    /// Set an abort callback for long operations
    ///
    /// The callback will be called periodically and can return true to abort.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `user_data` points to valid memory that will remain
    /// valid for the lifetime of this callback, or is null.
    pub unsafe fn set_abort_callback(
        &mut self,
        callback: Option<unsafe extern "C" fn(data: *mut std::os::raw::c_void) -> sys::c_bool>,
        user_data: *mut std::os::raw::c_void,
    ) {
        sys::llama_set_abort_callback(self.ctx_ptr, callback, user_data);
    }
}

/// Performance data for a context
#[derive(Debug, Clone)]
pub struct ContextPerfData {
    pub t_start_ms: f64,
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
}

impl ContextPerfData {
    /// Get tokens per second for prompt evaluation
    pub fn prompt_tokens_per_sec(&self) -> f64 {
        if self.t_p_eval_ms > 0.0 {
            (self.n_p_eval as f64 * 1000.0) / self.t_p_eval_ms
        } else {
            0.0
        }
    }

    /// Get tokens per second for generation
    pub fn generation_tokens_per_sec(&self) -> f64 {
        if self.t_eval_ms > 0.0 {
            (self.n_eval as f64 * 1000.0) / self.t_eval_ms
        } else {
            0.0
        }
    }
}

// Contexts need to be freed when dropped
impl Drop for Context {
    fn drop(&mut self) {
        if !self.ctx_ptr.is_null() {
            unsafe {
                sys::llama_free(self.ctx_ptr);
            }
        }
    }
}

// SAFETY: Context can be sent between threads because:
// 1. The raw pointer is never dereferenced without proper synchronization
// 2. All operations are synchronized via RwLock when used in async contexts
// 3. llama_context is designed to be used from a single thread at a time,
//    which we ensure through the RwLock guard
unsafe impl Send for Context {}

// SAFETY: Context can be shared between threads because:
// 1. All mutable operations are done through &mut self
// 2. When wrapped in RwLock, concurrent access is properly synchronized
unsafe impl Sync for Context {}
