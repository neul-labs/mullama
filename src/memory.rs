//! Memory management for model contexts
//!
//! This module provides comprehensive memory management for:
//! - Key-Value cache management
//! - Sequence-based memory operations
//! - Memory statistics and monitoring
//! - Context state management

use crate::error::MullamaError;
use crate::sys;
use std::collections::HashMap;

/// Memory manager for model contexts
///
/// Wraps llama.cpp's memory management API for KV cache and sequence operations.
/// Can be used standalone or obtained from a Context.
pub struct MemoryManager {
    /// The underlying llama memory handle
    memory_ptr: sys::llama_memory_t,
    /// Whether this manager owns the memory (should free on drop)
    owned: bool,
    /// Memory statistics
    stats: MemoryStats,
}

/// Statistics about memory usage
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total number of clear operations
    pub clears: u64,
    /// Total number of sequence removals
    pub seq_removals: u64,
    /// Total number of sequence copies
    pub seq_copies: u64,
    /// Total number of position shifts
    pub pos_shifts: u64,
    /// Current number of active sequences (estimated)
    pub active_sequences: usize,
}

/// Information about a sequence in memory
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Sequence ID
    pub seq_id: i32,
    /// Minimum position in the sequence
    pub pos_min: i32,
    /// Maximum position in the sequence
    pub pos_max: i32,
    /// Number of tokens (estimated)
    pub token_count: usize,
}

impl MemoryManager {
    /// Create a new empty memory manager
    ///
    /// The manager starts with a null pointer and is not valid until
    /// associated with a context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new memory manager from a raw pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and remains valid
    /// for the lifetime of the MemoryManager.
    pub(crate) fn from_ptr(ptr: sys::llama_memory_t, owned: bool) -> Self {
        Self {
            memory_ptr: ptr,
            owned,
            stats: MemoryStats::default(),
        }
    }

    /// Create a memory manager from a context
    ///
    /// This extracts the memory handle from the context for direct memory operations.
    pub fn from_context(ctx_ptr: *mut sys::llama_context) -> Option<Self> {
        let memory_ptr = unsafe { sys::llama_get_memory(ctx_ptr) };
        if memory_ptr.is_null() {
            None
        } else {
            Some(Self::from_ptr(memory_ptr, false))
        }
    }

    /// Check if the memory manager is valid
    pub fn is_valid(&self) -> bool {
        !self.memory_ptr.is_null()
    }

    /// Clear all memory contents
    ///
    /// # Arguments
    /// * `clear_data` - If true, also clear the underlying data buffers
    pub fn clear(&mut self, clear_data: bool) -> Result<(), MullamaError> {
        if self.memory_ptr.is_null() {
            return Err(MullamaError::MemoryError("Invalid memory handle".to_string()));
        }

        unsafe {
            sys::llama_memory_clear(self.memory_ptr, clear_data);
        }

        self.stats.clears += 1;
        self.stats.active_sequences = 0;
        Ok(())
    }

    /// Remove tokens belonging to a specific sequence
    ///
    /// Removes all tokens in positions [pos_start, pos_end) for the specified sequence.
    /// Use seq_id = -1 to remove from all sequences.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID (-1 for all sequences)
    /// * `pos_start` - Start position (inclusive)
    /// * `pos_end` - End position (exclusive), -1 for all positions
    ///
    /// # Returns
    /// `true` if successful, `false` if the sequence could not be removed
    pub fn remove_sequence_tokens(
        &mut self,
        seq_id: i32,
        pos_start: i32,
        pos_end: i32,
    ) -> Result<bool, MullamaError> {
        if self.memory_ptr.is_null() {
            return Err(MullamaError::MemoryError("Invalid memory handle".to_string()));
        }

        let result = unsafe {
            sys::llama_memory_seq_rm(self.memory_ptr, seq_id, pos_start, pos_end)
        };

        self.stats.seq_removals += 1;
        Ok(result)
    }

    /// Copy tokens from one sequence to another
    ///
    /// Copies all tokens in positions [pos_start, pos_end) from source to destination.
    ///
    /// # Arguments
    /// * `src_seq_id` - Source sequence ID
    /// * `dst_seq_id` - Destination sequence ID
    /// * `pos_start` - Start position (inclusive)
    /// * `pos_end` - End position (exclusive), -1 for all positions
    pub fn copy_sequence_tokens(
        &mut self,
        src_seq_id: i32,
        dst_seq_id: i32,
        pos_start: i32,
        pos_end: i32,
    ) -> Result<(), MullamaError> {
        if self.memory_ptr.is_null() {
            return Err(MullamaError::MemoryError("Invalid memory handle".to_string()));
        }

        unsafe {
            sys::llama_memory_seq_cp(self.memory_ptr, src_seq_id, dst_seq_id, pos_start, pos_end);
        }

        self.stats.seq_copies += 1;
        Ok(())
    }

    /// Keep only the specified sequence, removing all others
    ///
    /// # Arguments
    /// * `seq_id` - The sequence ID to keep
    pub fn keep_sequence(&mut self, seq_id: i32) -> Result<(), MullamaError> {
        if self.memory_ptr.is_null() {
            return Err(MullamaError::MemoryError("Invalid memory handle".to_string()));
        }

        unsafe {
            sys::llama_memory_seq_keep(self.memory_ptr, seq_id);
        }

        self.stats.active_sequences = 1;
        Ok(())
    }

    /// Shift token positions in memory
    ///
    /// Adds `delta` to all token positions in [pos_start, pos_end) for the specified sequence.
    /// Useful for context shifting when the cache is full.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `pos_start` - Start position (inclusive)
    /// * `pos_end` - End position (exclusive), -1 for all positions
    /// * `delta` - Amount to add to positions (can be negative)
    pub fn shift_positions(
        &mut self,
        seq_id: i32,
        pos_start: i32,
        pos_end: i32,
        delta: i32,
    ) -> Result<(), MullamaError> {
        if self.memory_ptr.is_null() {
            return Err(MullamaError::MemoryError("Invalid memory handle".to_string()));
        }

        unsafe {
            sys::llama_memory_seq_add(self.memory_ptr, seq_id, pos_start, pos_end, delta);
        }

        self.stats.pos_shifts += 1;
        Ok(())
    }

    /// Divide token positions by a factor
    ///
    /// Integer division of positions in [pos_start, pos_end) by the divisor.
    /// Useful for position interpolation.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `pos_start` - Start position (inclusive)
    /// * `pos_end` - End position (exclusive), -1 for all positions
    /// * `divisor` - Factor to divide by (must be > 1)
    pub fn divide_positions(
        &mut self,
        seq_id: i32,
        pos_start: i32,
        pos_end: i32,
        divisor: i32,
    ) -> Result<(), MullamaError> {
        if self.memory_ptr.is_null() {
            return Err(MullamaError::MemoryError("Invalid memory handle".to_string()));
        }

        if divisor <= 1 {
            return Err(MullamaError::InvalidInput(
                "Divisor must be greater than 1".to_string()
            ));
        }

        unsafe {
            sys::llama_memory_seq_div(self.memory_ptr, seq_id, pos_start, pos_end, divisor);
        }

        Ok(())
    }

    /// Get the minimum position for a sequence
    ///
    /// Returns -1 if the sequence is empty.
    pub fn get_min_position(&self, seq_id: i32) -> i32 {
        if self.memory_ptr.is_null() {
            return -1;
        }

        unsafe { sys::llama_memory_seq_pos_min(self.memory_ptr, seq_id) }
    }

    /// Get the maximum position for a sequence
    ///
    /// Returns -1 if the sequence is empty.
    pub fn get_max_position(&self, seq_id: i32) -> i32 {
        if self.memory_ptr.is_null() {
            return -1;
        }

        unsafe { sys::llama_memory_seq_pos_max(self.memory_ptr, seq_id) }
    }

    /// Check if the memory supports position shifting
    pub fn can_shift(&self) -> bool {
        if self.memory_ptr.is_null() {
            return false;
        }

        unsafe { sys::llama_memory_can_shift(self.memory_ptr) }
    }

    /// Get information about a sequence
    pub fn get_sequence_info(&self, seq_id: i32) -> Option<SequenceInfo> {
        if self.memory_ptr.is_null() {
            return None;
        }

        let pos_min = self.get_min_position(seq_id);
        let pos_max = self.get_max_position(seq_id);

        if pos_min < 0 || pos_max < 0 {
            return None;
        }

        Some(SequenceInfo {
            seq_id,
            pos_min,
            pos_max,
            token_count: (pos_max - pos_min + 1) as usize,
        })
    }

    /// Get memory statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Reset memory statistics
    pub fn reset_stats(&mut self) {
        self.stats = MemoryStats::default();
    }

    /// Get the raw memory pointer (for advanced use)
    pub fn as_ptr(&self) -> sys::llama_memory_t {
        self.memory_ptr
    }

    /// Perform context shifting to make room for new tokens
    ///
    /// This is a convenience method that:
    /// 1. Removes old tokens from the beginning
    /// 2. Shifts remaining positions to start from 0
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID to shift
    /// * `keep_count` - Number of recent tokens to keep
    pub fn context_shift(&mut self, seq_id: i32, keep_count: i32) -> Result<(), MullamaError> {
        let pos_max = self.get_max_position(seq_id);
        if pos_max < 0 {
            return Ok(()); // Empty sequence, nothing to shift
        }

        let pos_min = self.get_min_position(seq_id);
        let total_tokens = pos_max - pos_min + 1;

        if total_tokens <= keep_count {
            return Ok(()); // Not enough tokens to need shifting
        }

        // Calculate how many tokens to remove
        let remove_count = total_tokens - keep_count;
        let remove_end = pos_min + remove_count;

        // Remove old tokens
        self.remove_sequence_tokens(seq_id, pos_min, remove_end)?;

        // Shift remaining positions to start from 0
        self.shift_positions(seq_id, remove_end, -1, -remove_count)?;

        Ok(())
    }

    /// Fork a sequence by copying it to a new sequence ID
    ///
    /// # Arguments
    /// * `src_seq_id` - Source sequence to copy
    /// * `dst_seq_id` - Destination sequence ID
    ///
    /// # Returns
    /// Information about the new sequence
    pub fn fork_sequence(&mut self, src_seq_id: i32, dst_seq_id: i32) -> Result<SequenceInfo, MullamaError> {
        let pos_min = self.get_min_position(src_seq_id);
        let pos_max = self.get_max_position(src_seq_id);

        if pos_min < 0 || pos_max < 0 {
            return Err(MullamaError::MemoryError(
                format!("Source sequence {} is empty", src_seq_id)
            ));
        }

        // Copy the entire sequence
        self.copy_sequence_tokens(src_seq_id, dst_seq_id, pos_min, pos_max + 1)?;

        Ok(SequenceInfo {
            seq_id: dst_seq_id,
            pos_min,
            pos_max,
            token_count: (pos_max - pos_min + 1) as usize,
        })
    }

    /// Truncate a sequence to a maximum length
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID to truncate
    /// * `max_length` - Maximum number of tokens to keep
    pub fn truncate_sequence(&mut self, seq_id: i32, max_length: i32) -> Result<(), MullamaError> {
        let pos_min = self.get_min_position(seq_id);
        let pos_max = self.get_max_position(seq_id);

        if pos_min < 0 || pos_max < 0 {
            return Ok(()); // Empty sequence
        }

        let current_length = pos_max - pos_min + 1;
        if current_length <= max_length {
            return Ok(()); // Already within limit
        }

        // Remove tokens from the end
        let truncate_start = pos_min + max_length;
        self.remove_sequence_tokens(seq_id, truncate_start, -1)?;

        Ok(())
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self {
            memory_ptr: std::ptr::null_mut(),
            owned: false,
            stats: MemoryStats::default(),
        }
    }
}

/// KV Cache manager for more user-friendly cache operations
pub struct KVCacheManager {
    ctx_ptr: *mut sys::llama_context,
}

impl KVCacheManager {
    /// Create a KV cache manager from a context pointer
    pub fn new(ctx_ptr: *mut sys::llama_context) -> Self {
        Self { ctx_ptr }
    }

    /// Get the memory handle for this context
    fn get_memory(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx_ptr) }
    }

    /// Clear the entire KV cache
    pub fn clear(&mut self) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_clear(mem, false);
        }
    }

    /// Remove tokens from the cache
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID (-1 for all sequences)
    /// * `p0` - Start position (inclusive)
    /// * `p1` - End position (exclusive), -1 for all
    pub fn seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_rm(mem, seq_id, p0, p1)
        }
    }

    /// Copy tokens between sequences
    pub fn seq_cp(&mut self, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
        }
    }

    /// Keep only the specified sequence
    pub fn seq_keep(&mut self, seq_id: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_keep(mem, seq_id);
        }
    }

    /// Add delta to positions
    pub fn seq_add(&mut self, seq_id: i32, p0: i32, p1: i32, delta: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_add(mem, seq_id, p0, p1, delta);
        }
    }

    /// Divide positions by factor
    pub fn seq_div(&mut self, seq_id: i32, p0: i32, p1: i32, d: i32) {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_div(mem, seq_id, p0, p1, d);
        }
    }

    /// Get min position for a sequence
    pub fn seq_pos_min(&self, seq_id: i32) -> i32 {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_pos_min(mem, seq_id)
        }
    }

    /// Get max position for a sequence
    pub fn seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_seq_pos_max(mem, seq_id)
        }
    }

    /// Check if shifting is supported
    pub fn can_shift(&self) -> bool {
        unsafe {
            let mem = self.get_memory();
            sys::llama_memory_can_shift(mem)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats_default() {
        let stats = MemoryStats::default();
        assert_eq!(stats.clears, 0);
        assert_eq!(stats.seq_removals, 0);
        assert_eq!(stats.seq_copies, 0);
        assert_eq!(stats.pos_shifts, 0);
        assert_eq!(stats.active_sequences, 0);
    }

    #[test]
    fn test_sequence_info() {
        let info = SequenceInfo {
            seq_id: 0,
            pos_min: 0,
            pos_max: 99,
            token_count: 100,
        };

        assert_eq!(info.seq_id, 0);
        assert_eq!(info.token_count, 100);
    }

    #[test]
    fn test_memory_manager_default() {
        let manager = MemoryManager::default();
        assert!(!manager.is_valid());
        assert_eq!(manager.get_min_position(0), -1);
        assert_eq!(manager.get_max_position(0), -1);
        assert!(!manager.can_shift());
    }
}