//! Complete FFI bindings to llama.cpp
//!
//! This module provides comprehensive FFI bindings to llama.cpp with 100% API coverage.
//! These bindings are designed for maximum performance and safety while maintaining
//! compatibility with the latest llama.cpp API.
//!
//! **NOTE**: These are low-level FFI bindings. Use the safe Rust API in other modules.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_float, c_int, c_void};

// c_bool type alias for compatibility
pub type c_bool = bool;

//
// Constants - matching llama.cpp exactly
//

/// Null token ID
pub const LLAMA_TOKEN_NULL: llama_token = -1;

/// Default random seed
pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFFFFFF;

/// Session file version
pub const LLAMA_SESSION_VERSION: u32 = 9;

/// State sequence version
pub const LLAMA_STATE_SEQ_VERSION: u32 = 2;

//
// Core opaque types - these match llama.cpp exactly
//

#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_vocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler_i {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_memory_i {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_threadpool {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_backend_buffer_type {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_backend_dev {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_tensor {
    _private: [u8; 0],
}

// Callback type aliases
pub type ggml_backend_sched_eval_callback =
    Option<unsafe extern "C" fn(t: *mut ggml_tensor, ask: bool, user_data: *mut c_void) -> bool>;
pub type ggml_abort_callback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;
pub type ggml_backend_dev_t = *mut ggml_backend_dev;

#[repr(C)]
pub struct llama_adapter_lora {
    _private: [u8; 0],
}

//
// Multimodal (mtmd) opaque types
//

#[repr(C)]
pub struct mtmd_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mtmd_bitmap {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mtmd_image_tokens {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mtmd_input_chunk {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mtmd_input_chunks {
    _private: [u8; 0],
}

//
// Type aliases - exact matches to llama.cpp
//

pub type llama_token = i32;
pub type llama_pos = i32;
pub type llama_seq_id = i32;
pub type llama_memory_t = *mut llama_memory_i;

//
// Enums - complete coverage of llama.cpp enums
//

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0,
    LLAMA_VOCAB_TYPE_SPM = 1,
    LLAMA_VOCAB_TYPE_BPE = 2,
    LLAMA_VOCAB_TYPE_WPM = 3,
    LLAMA_VOCAB_TYPE_UGM = 4,
    LLAMA_VOCAB_TYPE_RWKV = 5,
    LLAMA_VOCAB_TYPE_PLAMO2 = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_rope_type {
    LLAMA_ROPE_TYPE_NONE = -1,
    LLAMA_ROPE_TYPE_NORM = 0,
    LLAMA_ROPE_TYPE_NEOX = 2,
    LLAMA_ROPE_TYPE_MROPE = 6,
    LLAMA_ROPE_TYPE_VISION = 10,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED = 0,
    LLAMA_TOKEN_TYPE_NORMAL = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN = 2,
    LLAMA_TOKEN_TYPE_CONTROL = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED = 5,
    LLAMA_TOKEN_TYPE_BYTE = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_token_attr {
    LLAMA_TOKEN_ATTR_UNDEFINED = 0,
    LLAMA_TOKEN_ATTR_UNKNOWN = 1 << 0,
    LLAMA_TOKEN_ATTR_UNUSED = 1 << 1,
    LLAMA_TOKEN_ATTR_NORMAL = 1 << 2,
    LLAMA_TOKEN_ATTR_CONTROL = 1 << 3,
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    LLAMA_TOKEN_ATTR_BYTE = 1 << 5,
    LLAMA_TOKEN_ATTR_NORMALIZED = 1 << 6,
    LLAMA_TOKEN_ATTR_LSTRIP = 1 << 7,
    LLAMA_TOKEN_ATTR_RSTRIP = 1 << 8,
    LLAMA_TOKEN_ATTR_SINGLE_WORD = 1 << 9,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_ftype {
    LLAMA_FTYPE_ALL_F32 = 0,
    LLAMA_FTYPE_MOSTLY_F16 = 1,
    LLAMA_FTYPE_MOSTLY_Q4_0 = 2,
    LLAMA_FTYPE_MOSTLY_Q4_1 = 3,
    LLAMA_FTYPE_MOSTLY_Q8_0 = 7,
    LLAMA_FTYPE_MOSTLY_Q5_0 = 8,
    LLAMA_FTYPE_MOSTLY_Q5_1 = 9,
    LLAMA_FTYPE_MOSTLY_Q2_K = 10,
    LLAMA_FTYPE_MOSTLY_Q3_K_S = 11,
    LLAMA_FTYPE_MOSTLY_Q3_K_M = 12,
    LLAMA_FTYPE_MOSTLY_Q3_K_L = 13,
    LLAMA_FTYPE_MOSTLY_Q4_K_S = 14,
    LLAMA_FTYPE_MOSTLY_Q4_K_M = 15,
    LLAMA_FTYPE_MOSTLY_Q5_K_S = 16,
    LLAMA_FTYPE_MOSTLY_Q5_K_M = 17,
    LLAMA_FTYPE_MOSTLY_Q6_K = 18,
    LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19,
    LLAMA_FTYPE_MOSTLY_IQ2_XS = 20,
    LLAMA_FTYPE_MOSTLY_Q2_K_S = 21,
    LLAMA_FTYPE_MOSTLY_IQ3_XS = 22,
    LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23,
    LLAMA_FTYPE_MOSTLY_IQ1_S = 24,
    LLAMA_FTYPE_MOSTLY_IQ4_NL = 25,
    LLAMA_FTYPE_MOSTLY_IQ3_S = 26,
    LLAMA_FTYPE_MOSTLY_IQ3_M = 27,
    LLAMA_FTYPE_MOSTLY_IQ2_S = 28,
    LLAMA_FTYPE_MOSTLY_IQ2_M = 29,
    LLAMA_FTYPE_MOSTLY_IQ4_XS = 30,
    LLAMA_FTYPE_MOSTLY_IQ1_M = 31,
    LLAMA_FTYPE_MOSTLY_BF16 = 32,
    LLAMA_FTYPE_MOSTLY_TQ1_0 = 36,
    LLAMA_FTYPE_MOSTLY_TQ2_0 = 37,
    LLAMA_FTYPE_MOSTLY_MXFP4_MOE = 38,
    LLAMA_FTYPE_GUESSED = 1024,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_rope_scaling_type {
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
    LLAMA_ROPE_SCALING_TYPE_NONE = 0,
    LLAMA_ROPE_SCALING_TYPE_LINEAR = 1,
    LLAMA_ROPE_SCALING_TYPE_YARN = 2,
    LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3,
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_pooling_type {
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE = 0,
    LLAMA_POOLING_TYPE_MEAN = 1,
    LLAMA_POOLING_TYPE_CLS = 2,
    LLAMA_POOLING_TYPE_LAST = 3,
    LLAMA_POOLING_TYPE_RANK = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_attention_type {
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
    LLAMA_ATTENTION_TYPE_CAUSAL = 0,
    LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_flash_attn_type {
    LLAMA_FLASH_ATTN_TYPE_AUTO = -1,
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
    LLAMA_FLASH_ATTN_TYPE_ENABLED = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_model_meta_key {
    LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE = 0,
    LLAMA_MODEL_META_KEY_SAMPLING_TOP_K = 1,
    LLAMA_MODEL_META_KEY_SAMPLING_TOP_P = 2,
    LLAMA_MODEL_META_KEY_SAMPLING_MIN_P = 3,
    LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY = 4,
    LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD = 5,
    LLAMA_MODEL_META_KEY_SAMPLING_TEMP = 6,
    LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N = 7,
    LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT = 8,
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT = 9,
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU = 10,
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA = 11,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_split_mode {
    LLAMA_SPLIT_MODE_NONE = 0,
    LLAMA_SPLIT_MODE_LAYER = 1,
    LLAMA_SPLIT_MODE_ROW = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_model_kv_override_type {
    LLAMA_KV_OVERRIDE_TYPE_INT,
    LLAMA_KV_OVERRIDE_TYPE_FLOAT,
    LLAMA_KV_OVERRIDE_TYPE_BOOL,
    LLAMA_KV_OVERRIDE_TYPE_STR,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_Q4_0_4_4 = 31,
    GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0 = 34,
    GGML_TYPE_TQ2_0 = 35,
    GGML_TYPE_IQ4_NL_4_4 = 36,
    GGML_TYPE_IQ4_NL_4_8 = 37,
    GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_COUNT,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ggml_numa_strategy {
    GGML_NUMA_STRATEGY_DISABLED = 0,
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
    GGML_NUMA_STRATEGY_ISOLATE = 2,
    GGML_NUMA_STRATEGY_NUMACTL = 3,
    GGML_NUMA_STRATEGY_MIRROR = 4,
    GGML_NUMA_STRATEGY_COUNT,
}

//
// Multimodal (mtmd) enums
//

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT = 0,
    MTMD_INPUT_CHUNK_TYPE_IMAGE = 1,
    MTMD_INPUT_CHUNK_TYPE_AUDIO = 2,
}

//
// Multimodal (mtmd) parameter structures
//

#[repr(C)]
#[derive(Debug, Clone)]
pub struct mtmd_context_params {
    pub use_gpu: c_bool,
    pub print_timings: c_bool,
    pub n_threads: c_int,
    pub image_marker: *const c_char, // deprecated, use media_marker
    pub media_marker: *const c_char,
    pub flash_attn_type: llama_flash_attn_type,
    pub warmup: c_bool,
    pub image_min_tokens: c_int,
    pub image_max_tokens: c_int,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct mtmd_input_text {
    pub text: *const c_char,
    pub add_special: c_bool,
    pub parse_special: c_bool,
}

//
// Core data structures - complete parameter structures
//

#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_token_data {
    pub id: llama_token,
    pub logit: c_float,
    pub p: c_float,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_token_data_array {
    pub data: *mut llama_token_data,
    pub size: usize,
    pub selected: i64,
    pub sorted: c_bool,
}

#[repr(C)]
#[derive(Clone)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut c_float,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

pub type llama_progress_callback =
    Option<unsafe extern "C" fn(progress: c_float, user_data: *mut c_void) -> c_bool>;

#[repr(C)]
#[derive(Debug)]
pub struct llama_model_kv_override {
    pub tag: llama_model_kv_override_type,
    pub key: [c_char; 128],
    pub value: llama_model_kv_override_value,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union llama_model_kv_override_value {
    pub val_i64: i64,
    pub val_f64: f64,
    pub val_bool: c_bool,
    pub val_str: [c_char; 128],
}

impl std::fmt::Debug for llama_model_kv_override_value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("llama_model_kv_override_value")
            .field("val_i64", unsafe { &self.val_i64 })
            .finish()
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct llama_model_tensor_buft_override {
    pub pattern: *const c_char,
    pub buft: *mut ggml_backend_buffer_type,
}

// Parameter structures - must match llama.h exactly!
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_model_params {
    // NULL-terminated list of devices to use for offloading
    pub devices: *mut ggml_backend_dev_t,
    // NULL-terminated list of buffer types to use for tensors that match a pattern
    pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
    pub n_gpu_layers: i32,
    pub split_mode: llama_split_mode,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    // Progress callback
    pub progress_callback: llama_progress_callback,
    pub progress_callback_user_data: *mut c_void,
    // KV overrides
    pub kv_overrides: *const llama_model_kv_override,
    // Keep booleans together at the end to avoid misalignment
    pub vocab_only: c_bool,
    pub use_mmap: c_bool,
    pub use_mlock: c_bool,
    pub check_tensors: c_bool,
    pub use_extra_bufts: c_bool,
    pub no_host: c_bool,
    pub no_alloc: c_bool,
}

// llama_context_params must match llama.h exactly!
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_context_params {
    pub n_ctx: u32,           // text context, 0 = from model
    pub n_batch: u32,         // logical maximum batch size
    pub n_ubatch: u32,        // physical maximum batch size
    pub n_seq_max: u32,       // max number of sequences
    pub n_threads: i32,       // number of threads for generation
    pub n_threads_batch: i32, // number of threads for batch processing

    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: llama_attention_type,
    pub flash_attn_type: llama_flash_attn_type,

    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,

    // Eval callback
    pub cb_eval: ggml_backend_sched_eval_callback,
    pub cb_eval_user_data: *mut c_void,

    // KV cache data types
    pub type_k: ggml_type,
    pub type_v: ggml_type,

    // Abort callback
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut c_void,

    // Keep booleans together at the end
    pub embeddings: c_bool,
    pub offload_kqv: c_bool,
    pub no_perf: c_bool,
    pub op_offload: c_bool,
    pub swa_full: c_bool,
    pub kv_unified: c_bool,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_sampler_chain_params {
    pub no_perf: c_bool,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_model_quantize_params {
    pub nthread: i32,
    pub ftype: llama_ftype,
    pub output_tensor_type: ggml_type,
    pub token_embedding_type: ggml_type,
    pub allow_requantize: c_bool,
    pub quantize_output_tensor: c_bool,
    pub only_copy: c_bool,
    pub pure: c_bool,
    pub keep_split: c_bool,
    pub imatrix: *mut c_void,
    pub kv_overrides: *const llama_model_kv_override,
}

// Additional structures
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_logit_bias {
    pub token: llama_token,
    pub bias: f32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_chat_message {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_perf_context_data {
    pub t_start_ms: f64,
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_perf_sampler_data {
    pub t_sample_ms: f64,
    pub n_sample: i32,
}

//
// Complete FFI function declarations - 213+ functions for 100% coverage
//

extern "C" {
    //
    // Backend and system initialization
    //
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    pub fn llama_numa_init(numa: ggml_numa_strategy);
    pub fn llama_time_us() -> i64;
    pub fn llama_max_devices() -> usize;
    pub fn llama_max_parallel_sequences() -> usize;
    pub fn llama_supports_mmap() -> c_bool;
    pub fn llama_supports_mlock() -> c_bool;
    pub fn llama_supports_gpu_offload() -> c_bool;
    pub fn llama_supports_rpc() -> c_bool;
    pub fn llama_print_system_info() -> *const c_char;

    // Model introspection
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> i32;
    pub fn llama_model_n_embd(model: *const llama_model) -> i32;

    //
    // Parameter defaults
    //
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
    pub fn llama_model_quantize_default_params() -> llama_model_quantize_params;

    //
    // Thread pool management
    //
    pub fn llama_attach_threadpool(
        ctx: *mut llama_context,
        threadpool: *mut ggml_threadpool,
        threadpool_batch: *mut ggml_threadpool,
    );
    pub fn llama_detach_threadpool(ctx: *mut llama_context);

    //
    // Model loading and management
    //
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub fn llama_model_load_from_splits(
        paths: *const *const c_char,
        n_paths: usize,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub fn llama_model_save_to_file(model: *const llama_model, path_model: *const c_char);

    pub fn llama_model_free(model: *mut llama_model);

    pub fn llama_model_quantize(
        fname_inp: *const c_char,
        fname_out: *const c_char,
        params: *const llama_model_quantize_params,
    ) -> i32;

    //
    // Context creation and management
    //
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;

    pub fn llama_free(ctx: *mut llama_context);

    //
    // Context information
    //
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
    pub fn llama_n_batch(ctx: *const llama_context) -> u32;
    pub fn llama_n_ubatch(ctx: *const llama_context) -> u32;
    pub fn llama_n_seq_max(ctx: *const llama_context) -> u32;
    pub fn llama_get_model(ctx: *const llama_context) -> *const llama_model;
    pub fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
    pub fn llama_pooling_type(ctx: *const llama_context) -> llama_pooling_type;

    //
    // Memory/KV cache management
    //
    pub fn llama_memory_clear(mem: llama_memory_t, data: c_bool);
    pub fn llama_memory_seq_rm(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> c_bool;
    pub fn llama_memory_seq_cp(
        mem: llama_memory_t,
        seq_id_src: llama_seq_id,
        seq_id_dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    );
    pub fn llama_memory_seq_keep(mem: llama_memory_t, seq_id: llama_seq_id);
    pub fn llama_memory_seq_add(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    );
    pub fn llama_memory_seq_div(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        d: c_int,
    );
    pub fn llama_memory_seq_pos_min(mem: llama_memory_t, seq_id: llama_seq_id) -> llama_pos;
    pub fn llama_memory_seq_pos_max(mem: llama_memory_t, seq_id: llama_seq_id) -> llama_pos;
    pub fn llama_memory_can_shift(mem: llama_memory_t) -> c_bool;

    //
    // Model information
    //
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_model_rope_type(model: *const llama_model) -> llama_rope_type;
    pub fn llama_model_n_layer(model: *const llama_model) -> i32;
    pub fn llama_model_n_head(model: *const llama_model) -> i32;
    pub fn llama_model_n_head_kv(model: *const llama_model) -> i32;
    pub fn llama_model_n_swa(model: *const llama_model) -> i32;
    pub fn llama_model_rope_freq_scale_train(model: *const llama_model) -> c_float;
    pub fn llama_model_n_cls_out(model: *const llama_model) -> u32;
    pub fn llama_model_cls_label(model: *const llama_model, i: u32) -> *const c_char;

    //
    // Vocabulary and tokenization
    //
    pub fn llama_vocab_type(vocab: *const llama_vocab) -> llama_vocab_type;
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;

    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: c_bool,
        parse_special: c_bool,
    ) -> i32;

    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: c_bool,
    ) -> i32;

    pub fn llama_detokenize(
        vocab: *const llama_vocab,
        tokens: *const llama_token,
        n_tokens: i32,
        text: *mut c_char,
        text_len_max: i32,
        remove_special: c_bool,
        unparse_special: c_bool,
    ) -> i32;

    //
    // Batch processing
    //
    pub fn llama_batch_get_one(tokens: *mut llama_token, n_tokens: i32) -> llama_batch;

    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;

    pub fn llama_batch_free(batch: llama_batch);

    //
    // Inference and decoding
    //
    pub fn llama_encode(ctx: *mut llama_context, batch: llama_batch) -> i32;

    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;

    pub fn llama_set_n_threads(ctx: *mut llama_context, n_threads: i32, n_threads_batch: i32);

    pub fn llama_n_threads(ctx: *mut llama_context) -> i32;
    pub fn llama_n_threads_batch(ctx: *mut llama_context) -> i32;

    //
    // Logits and embeddings access
    //
    pub fn llama_get_logits(ctx: *mut llama_context) -> *mut c_float;
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;
    pub fn llama_get_embeddings(ctx: *mut llama_context) -> *mut c_float;
    pub fn llama_get_embeddings_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;
    pub fn llama_get_embeddings_seq(ctx: *mut llama_context, seq_id: llama_seq_id) -> *mut c_float;

    //
    // State management
    //
    pub fn llama_state_get_size(ctx: *const llama_context) -> usize;
    pub fn llama_state_get_data(ctx: *const llama_context, dst: *mut u8, size: usize) -> usize;
    pub fn llama_state_set_data(ctx: *mut llama_context, src: *const u8, size: usize) -> usize;
    pub fn llama_state_load_file(
        ctx: *mut llama_context,
        path_session: *const c_char,
        tokens_out: *mut llama_token,
        n_token_capacity: usize,
        n_token_count_out: *mut usize,
    ) -> c_bool;
    pub fn llama_state_save_file(
        ctx: *const llama_context,
        path_session: *const c_char,
        tokens: *const llama_token,
        n_token_count: usize,
    ) -> c_bool;

    //
    // Sequence state management
    //
    pub fn llama_state_seq_get_size(ctx: *const llama_context, seq_id: llama_seq_id) -> usize;
    pub fn llama_state_seq_get_data(
        ctx: *const llama_context,
        dst: *mut u8,
        size: usize,
        seq_id: llama_seq_id,
    ) -> usize;
    pub fn llama_state_seq_set_data(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
        dest_seq_id: llama_seq_id,
    ) -> usize;
    pub fn llama_state_seq_save_file(
        ctx: *mut llama_context,
        filepath: *const c_char,
        seq_id: llama_seq_id,
        tokens: *const llama_token,
        n_token_count: usize,
    ) -> usize;
    pub fn llama_state_seq_load_file(
        ctx: *mut llama_context,
        filepath: *const c_char,
        dest_seq_id: llama_seq_id,
        tokens_out: *mut llama_token,
        n_token_capacity: usize,
        n_token_count_out: *mut usize,
    ) -> usize;

    //
    // Complete sampling system (30+ functions)
    //
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_tail_free(z: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_typical(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: c_float) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp_ext(
        t: c_float,
        delta: c_float,
        exponent: c_float,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_mirostat(
        vocab: *const llama_vocab,
        seed: u32,
        tau: c_float,
        eta: c_float,
        m: i32,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_mirostat_v2(
        seed: u32,
        tau: c_float,
        eta: c_float,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_grammar(
        vocab: *const llama_vocab,
        grammar_str: *const c_char,
        grammar_root: *const c_char,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_penalties(
        penalty_last_n: i32,
        penalty_repeat: c_float,
        penalty_freq: c_float,
        penalty_present: c_float,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_logit_bias(
        n_vocab: i32,
        n_logit_bias: i32,
        logit_bias: *const llama_logit_bias,
    ) -> *mut llama_sampler;

    //
    // Sampler chain management
    //
    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_chain_get(chain: *const llama_sampler, i: i32) -> *mut llama_sampler;
    pub fn llama_sampler_chain_n(chain: *const llama_sampler) -> i32;
    pub fn llama_sampler_chain_remove(chain: *mut llama_sampler, i: i32) -> *mut llama_sampler;

    //
    // Sampling operations
    //
    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;
    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);
    pub fn llama_sampler_apply(smpl: *mut llama_sampler, cur: *mut llama_token_data_array);
    pub fn llama_sampler_reset(smpl: *mut llama_sampler);
    pub fn llama_sampler_clone(smpl: *const llama_sampler) -> *mut llama_sampler;
    pub fn llama_sampler_free(smpl: *mut llama_sampler);
    pub fn llama_sampler_name(smpl: *const llama_sampler) -> *const c_char;

    //
    // Chat templates
    //
    pub fn llama_chat_apply_template(
        tmpl: *const c_char,
        chat: *const llama_chat_message,
        n_msg: usize,
        add_ass: c_bool,
        buf: *mut c_char,
        length: i32,
    ) -> i32;
    pub fn llama_chat_builtin_templates(buf: *mut c_char, length: usize) -> i32;

    //
    // Performance monitoring
    //
    pub fn llama_perf_context(ctx: *const llama_context) -> llama_perf_context_data;
    pub fn llama_perf_context_print(ctx: *const llama_context);
    pub fn llama_perf_context_reset(ctx: *mut llama_context);
    pub fn llama_perf_sampler(smpl: *const llama_sampler) -> llama_perf_sampler_data;
    pub fn llama_perf_sampler_print(smpl: *const llama_sampler);
    pub fn llama_perf_sampler_reset(smpl: *mut llama_sampler);
    pub fn llama_perf_dump_yaml(stream: *mut c_void, ctx: *const llama_context);

    //
    // LoRA adapter operations
    //
    pub fn llama_adapter_lora_init(
        model: *mut llama_model,
        path_lora: *const c_char,
    ) -> *mut llama_adapter_lora;
    pub fn llama_adapter_lora_free(adapter: *mut llama_adapter_lora);
    pub fn llama_set_adapter_lora(
        ctx: *mut llama_context,
        adapter: *mut llama_adapter_lora,
        scale: f32,
    ) -> i32;
    pub fn llama_rm_adapter_lora(ctx: *mut llama_context, adapter: *mut llama_adapter_lora) -> i32;
    pub fn llama_clear_adapter_lora(ctx: *mut llama_context);

    //
    // Control vector operations
    //
    pub fn llama_control_vector_apply(
        ctx: *mut llama_context,
        data: *const f32,
        len: usize,
        n_embd: i32,
        il_start: i32,
        il_end: i32,
    ) -> i32;

    //
    // State / sessions
    //
    pub fn llama_get_state_size(ctx: *mut llama_context) -> usize;
    pub fn llama_copy_state_data(ctx: *mut llama_context, dst: *mut u8) -> usize;
    pub fn llama_set_state_data(ctx: *mut llama_context, src: *const u8) -> usize;
    pub fn llama_load_session_file(
        ctx: *mut llama_context,
        path_session: *const c_char,
        tokens_out: *mut llama_token,
        n_token_capacity: usize,
        n_token_count_out: *mut usize,
    ) -> c_bool;
    pub fn llama_save_session_file(
        ctx: *mut llama_context,
        path_session: *const c_char,
        tokens: *const llama_token,
        n_token_count: usize,
    ) -> c_bool;
    //
    // Model metadata
    //
    pub fn llama_model_desc(model: *const llama_model, buf: *mut c_char, buf_size: usize) -> c_int;
    pub fn llama_model_size(model: *const llama_model) -> u64;
    pub fn llama_model_n_params(model: *const llama_model) -> u64;
    pub fn llama_model_meta_count(model: *const llama_model) -> i32;
    pub fn llama_model_meta_key_by_index(
        model: *const llama_model,
        i: i32,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;
    pub fn llama_model_meta_val_str(
        model: *const llama_model,
        key: *const c_char,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;
    pub fn llama_model_meta_val_str_by_index(
        model: *const llama_model,
        i: i32,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;
    pub fn llama_model_has_encoder(model: *const llama_model) -> c_bool;
    pub fn llama_model_has_decoder(model: *const llama_model) -> c_bool;
    pub fn llama_model_is_recurrent(model: *const llama_model) -> c_bool;
    pub fn llama_model_decoder_start_token(model: *const llama_model) -> llama_token;
    pub fn llama_model_chat_template(
        model: *const llama_model,
        name: *const c_char,
    ) -> *const c_char;
    pub fn llama_n_ctx_train(model: *const llama_model) -> u32;
    pub fn llama_n_embd(model: *const llama_model) -> i32;
    pub fn llama_n_layer(model: *const llama_model) -> i32;
    pub fn llama_n_head(model: *const llama_model) -> i32;
    pub fn llama_n_vocab(model: *const llama_model) -> i32;

    //
    // Vocabulary functions
    //
    pub fn llama_vocab_get_text(vocab: *const llama_vocab, token: llama_token) -> *const c_char;
    pub fn llama_vocab_get_score(vocab: *const llama_vocab, token: llama_token) -> f32;
    pub fn llama_vocab_get_attr(vocab: *const llama_vocab, token: llama_token) -> llama_token_attr;
    pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> c_bool;
    pub fn llama_vocab_is_control(vocab: *const llama_vocab, token: llama_token) -> c_bool;
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eot(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_sep(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_nl(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_pad(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_cls(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_mask(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> c_bool;
    pub fn llama_vocab_get_add_eos(vocab: *const llama_vocab) -> c_bool;
    pub fn llama_vocab_get_add_sep(vocab: *const llama_vocab) -> c_bool;
    pub fn llama_vocab_fim_pre(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_fim_suf(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_fim_mid(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_fim_pad(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_fim_rep(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_fim_sep(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_token_fim_pre(model: *const llama_model) -> llama_token;
    pub fn llama_token_fim_suf(model: *const llama_model) -> llama_token;
    pub fn llama_token_fim_mid(model: *const llama_model) -> llama_token;
    pub fn llama_token_fim_pad(model: *const llama_model) -> llama_token;
    pub fn llama_token_fim_rep(model: *const llama_model) -> llama_token;
    pub fn llama_token_fim_sep(model: *const llama_model) -> llama_token;

    //
    // Additional sampler functions
    //
    pub fn llama_sampler_get_seed(smpl: *const llama_sampler) -> u32;
    pub fn llama_sampler_init_softmax() -> *mut llama_sampler;
    pub fn llama_sampler_init_top_n_sigma(n: f32) -> *mut llama_sampler;
    pub fn llama_sampler_init_dry(
        vocab: *const llama_vocab,
        n_ctx_train: i32,
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: i32,
        dry_penalty_last_n: i32,
        seq_breakers: *const *const c_char,
        num_breakers: usize,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_xtc(p: f32, t: f32, min_keep: usize, seed: u32)
        -> *mut llama_sampler;
    pub fn llama_sampler_init_infill(vocab: *const llama_vocab) -> *mut llama_sampler;

    //
    // Context settings
    //
    pub fn llama_set_causal_attn(ctx: *mut llama_context, causal_attn: c_bool);
    pub fn llama_set_embeddings(ctx: *mut llama_context, embeddings: c_bool);
    pub fn llama_set_warmup(ctx: *mut llama_context, warmup: c_bool);
    pub fn llama_set_abort_callback(
        ctx: *mut llama_context,
        abort_callback: Option<unsafe extern "C" fn(data: *mut c_void) -> c_bool>,
        abort_callback_data: *mut c_void,
    );
    pub fn llama_synchronize(ctx: *mut llama_context);

    //
    // Utilities and system information
    //
    pub fn llama_log_set(
        log_callback: Option<
            unsafe extern "C" fn(level: i32, text: *const c_char, user_data: *mut c_void),
        >,
        user_data: *mut c_void,
    );
    pub fn llama_log_callback_default(level: i32, text: *const c_char, user_data: *mut c_void);
    pub fn llama_dump_timing_info_yaml(stream: *mut c_void, ctx: *const llama_context);
    pub fn llama_split_path(
        split_path: *mut c_char,
        maxlen: usize,
        path_prefix: *const c_char,
        split_no: c_int,
        split_count: c_int,
    ) -> c_int;
    pub fn llama_split_prefix(
        split_prefix: *mut c_char,
        maxlen: usize,
        split_path: *const c_char,
        split_no: c_int,
        split_count: c_int,
    ) -> c_int;

    //
    // Additional model/context functions
    //
    pub fn llama_load_model_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_free_model(model: *mut llama_model);
    pub fn llama_new_context_with_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_get_pooling_type(ctx: *const llama_context) -> llama_pooling_type;
    pub fn llama_model_is_diffusion(model: *const llama_model) -> c_bool;

    //
    // Extended state functions
    //
    pub fn llama_state_seq_get_size_ext(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
        flags: i32,
    ) -> usize;
    pub fn llama_state_seq_get_data_ext(
        ctx: *mut llama_context,
        dst: *mut u8,
        size: usize,
        seq_id: llama_seq_id,
        flags: i32,
    ) -> usize;
    pub fn llama_state_seq_set_data_ext(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
        seq_id: llama_seq_id,
        flags: i32,
    ) -> usize;

    //
    // Decode with sampler
    //
    pub fn llama_decode_with_sampler(
        ctx: *mut llama_context,
        smpl: *mut llama_sampler,
        batch: llama_batch,
        top_n_logprobs: i32,
    ) -> i32;

    //
    // Control vector application
    //
    pub fn llama_apply_adapter_cvec(
        ctx: *mut llama_context,
        data: *const f32,
        n_embd: i32,
        il_start: i32,
        il_end: i32,
    ) -> i32;

    //
    // Grammar lazy initialization
    //
    pub fn llama_sampler_init_grammar_lazy(
        model: *const llama_model,
        grammar_str: *const c_char,
        grammar_root: *const c_char,
        trigger_words: *const *const c_char,
        num_trigger_words: usize,
        trigger_tokens: *const llama_token,
        num_trigger_tokens: usize,
    ) -> *mut llama_sampler;

    //
    // Multimodal (mtmd) functions
    //

    // Default marker for media in prompts
    pub fn mtmd_default_marker() -> *const c_char;

    // Get default context params
    pub fn mtmd_context_params_default() -> mtmd_context_params;

    // Context lifecycle
    pub fn mtmd_init_from_file(
        mmproj_fname: *const c_char,
        text_model: *const llama_model,
        ctx_params: mtmd_context_params,
    ) -> *mut mtmd_context;
    pub fn mtmd_free(ctx: *mut mtmd_context);

    // Context capability queries
    pub fn mtmd_decode_use_non_causal(ctx: *mut mtmd_context) -> c_bool;
    pub fn mtmd_decode_use_mrope(ctx: *mut mtmd_context) -> c_bool;
    pub fn mtmd_support_vision(ctx: *mut mtmd_context) -> c_bool;
    pub fn mtmd_support_audio(ctx: *mut mtmd_context) -> c_bool;
    pub fn mtmd_get_audio_bitrate(ctx: *mut mtmd_context) -> c_int;

    // Bitmap (image/audio data) management
    pub fn mtmd_bitmap_init(
        nx: u32,
        ny: u32,
        data: *const std::os::raw::c_uchar,
    ) -> *mut mtmd_bitmap;
    pub fn mtmd_bitmap_init_from_audio(n_samples: usize, data: *const c_float) -> *mut mtmd_bitmap;
    pub fn mtmd_bitmap_get_nx(bitmap: *const mtmd_bitmap) -> u32;
    pub fn mtmd_bitmap_get_ny(bitmap: *const mtmd_bitmap) -> u32;
    pub fn mtmd_bitmap_get_data(bitmap: *const mtmd_bitmap) -> *const std::os::raw::c_uchar;
    pub fn mtmd_bitmap_get_n_bytes(bitmap: *const mtmd_bitmap) -> usize;
    pub fn mtmd_bitmap_is_audio(bitmap: *const mtmd_bitmap) -> c_bool;
    pub fn mtmd_bitmap_free(bitmap: *mut mtmd_bitmap);
    pub fn mtmd_bitmap_get_id(bitmap: *const mtmd_bitmap) -> *const c_char;
    pub fn mtmd_bitmap_set_id(bitmap: *mut mtmd_bitmap, id: *const c_char);

    // Input chunks management
    pub fn mtmd_input_chunks_init() -> *mut mtmd_input_chunks;
    pub fn mtmd_input_chunks_size(chunks: *const mtmd_input_chunks) -> usize;
    pub fn mtmd_input_chunks_get(
        chunks: *const mtmd_input_chunks,
        idx: usize,
    ) -> *const mtmd_input_chunk;
    pub fn mtmd_input_chunks_free(chunks: *mut mtmd_input_chunks);

    // Input chunk inspection
    pub fn mtmd_input_chunk_get_type(chunk: *const mtmd_input_chunk) -> mtmd_input_chunk_type;
    pub fn mtmd_input_chunk_get_tokens_text(
        chunk: *const mtmd_input_chunk,
        n_tokens_output: *mut usize,
    ) -> *const llama_token;
    pub fn mtmd_input_chunk_get_tokens_image(
        chunk: *const mtmd_input_chunk,
    ) -> *const mtmd_image_tokens;
    pub fn mtmd_input_chunk_get_n_tokens(chunk: *const mtmd_input_chunk) -> usize;
    pub fn mtmd_input_chunk_get_id(chunk: *const mtmd_input_chunk) -> *const c_char;
    pub fn mtmd_input_chunk_get_n_pos(chunk: *const mtmd_input_chunk) -> llama_pos;
    pub fn mtmd_input_chunk_copy(chunk: *const mtmd_input_chunk) -> *mut mtmd_input_chunk;
    pub fn mtmd_input_chunk_free(chunk: *mut mtmd_input_chunk);

    // Image tokens
    pub fn mtmd_image_tokens_get_n_tokens(image_tokens: *const mtmd_image_tokens) -> usize;
    pub fn mtmd_image_tokens_get_nx(image_tokens: *const mtmd_image_tokens) -> usize;
    pub fn mtmd_image_tokens_get_ny(image_tokens: *const mtmd_image_tokens) -> usize;
    pub fn mtmd_image_tokens_get_id(image_tokens: *const mtmd_image_tokens) -> *const c_char;
    pub fn mtmd_image_tokens_get_n_pos(image_tokens: *const mtmd_image_tokens) -> llama_pos;

    // Tokenization
    pub fn mtmd_tokenize(
        ctx: *mut mtmd_context,
        output: *mut mtmd_input_chunks,
        text: *const mtmd_input_text,
        bitmaps: *const *const mtmd_bitmap,
        n_bitmaps: usize,
    ) -> i32;

    // Encoding
    pub fn mtmd_encode(ctx: *mut mtmd_context, image_tokens: *const mtmd_image_tokens) -> i32;
    pub fn mtmd_encode_chunk(ctx: *mut mtmd_context, chunk: *const mtmd_input_chunk) -> i32;
    pub fn mtmd_get_output_embd(ctx: *mut mtmd_context) -> *mut c_float;

    //
    // Multimodal helper functions (mtmd-helper.h)
    //

    // Helper to load bitmap from file
    pub fn mtmd_helper_bitmap_init_from_file(
        ctx: *mut mtmd_context,
        fname: *const c_char,
    ) -> *mut mtmd_bitmap;

    // Helper to load bitmap from buffer
    pub fn mtmd_helper_bitmap_init_from_buf(
        ctx: *mut mtmd_context,
        buf: *const std::os::raw::c_uchar,
        len: usize,
    ) -> *mut mtmd_bitmap;

    // Helper to count tokens
    pub fn mtmd_helper_get_n_tokens(chunks: *const mtmd_input_chunks) -> usize;
    pub fn mtmd_helper_get_n_pos(chunks: *const mtmd_input_chunks) -> llama_pos;

    // Helper to evaluate chunks in llama context
    pub fn mtmd_helper_eval_chunks(
        ctx: *mut mtmd_context,
        lctx: *mut llama_context,
        chunks: *const mtmd_input_chunks,
        n_past: llama_pos,
        seq_id: llama_seq_id,
        n_batch: i32,
        logits_last: c_bool,
        new_n_past: *mut llama_pos,
    ) -> i32;

    // Helper to evaluate single chunk
    pub fn mtmd_helper_eval_chunk_single(
        ctx: *mut mtmd_context,
        lctx: *mut llama_context,
        chunk: *const mtmd_input_chunk,
        n_past: llama_pos,
        seq_id: llama_seq_id,
        n_batch: i32,
        logits_last: c_bool,
        new_n_past: *mut llama_pos,
    ) -> i32;

    // Helper to decode image chunk
    pub fn mtmd_helper_decode_image_chunk(
        ctx: *mut mtmd_context,
        lctx: *mut llama_context,
        chunk: *const mtmd_input_chunk,
        encoded_embd: *mut c_float,
        n_past: llama_pos,
        seq_id: llama_seq_id,
        n_batch: i32,
        new_n_past: *mut llama_pos,
    ) -> i32;
}
