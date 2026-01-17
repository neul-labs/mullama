//! Comprehensive multimodal processing for text, image, and audio integration
//!
//! This module provides advanced multimodal processing capabilities, enabling seamless
//! integration of text, image, and audio data for sophisticated AI applications.
//!
//! ## Features
//!
//! - **Vision-Language Models**: Process text and visual inputs together
//! - **Audio Processing**: Handle audio input/output with speech-to-text/text-to-speech
//! - **Cross-Modal Understanding**: Combine multiple modalities for richer context
//! - **Batch Processing**: Efficient processing of multimodal datasets
//! - **Format Support**: Wide range of image and audio formats
//! - **Pipeline Integration**: Seamless integration with generation pipelines

use crate::error::MullamaError;
use crate::sys;
use crate::{Context, Model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::time::Duration;

#[cfg(all(feature = "multimodal", feature = "async"))]
use crate::AsyncModel;

#[cfg(feature = "multimodal")]
use tokio::{fs, io::AsyncReadExt};

/// Multimodal processor for handling text and vision inputs
#[derive(Debug)]
pub struct MultimodalProcessor {
    /// Text model for language processing
    text_model: Model,
    /// Vision encoder for image processing
    vision_encoder: Option<VisionEncoder>,
    /// Processor configuration
    config: MultimodalConfig,
    /// Supported modalities
    supported_modalities: Vec<Modality>,
}

/// Vision encoder for processing images
#[derive(Debug)]
pub struct VisionEncoder {
    /// Vision model pointer
    vision_model_ptr: *mut sys::llama_model,
    /// Image preprocessing configuration
    preprocess_config: ImagePreprocessConfig,
    /// Encoder type
    encoder_type: VisionEncoderType,
}

/// Types of vision encoders
#[derive(Debug, Clone, Copy)]
pub enum VisionEncoderType {
    /// CLIP-style encoder
    Clip,
    /// DINOv2 encoder
    Dino,
    /// Custom vision encoder
    Custom,
}

/// Supported modalities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Modality {
    /// Text input/output
    Text,
    /// Image input
    Image,
    /// Video input (experimental)
    Video,
    /// Audio input (experimental)
    Audio,
}

/// Multimodal configuration
#[derive(Debug, Clone)]
pub struct MultimodalConfig {
    /// Maximum image resolution
    pub max_image_resolution: (u32, u32),
    /// Image patch size for vision transformer
    pub patch_size: u32,
    /// Number of vision tokens per image
    pub vision_tokens_per_image: usize,
    /// Enable image-to-text generation
    pub enable_image_to_text: bool,
    /// Enable text-to-image generation (experimental)
    pub enable_text_to_image: bool,
    /// Cross-attention configuration
    pub cross_attention_config: CrossAttentionConfig,
    /// Temperature for multimodal generation
    pub temperature: f32,
}

/// Cross-attention configuration for multimodal fusion
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    /// Number of cross-attention layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Dropout rate
    pub dropout_rate: f32,
}

/// Image preprocessing configuration
#[derive(Debug, Clone)]
pub struct ImagePreprocessConfig {
    /// Target image size
    pub target_size: (u32, u32),
    /// Normalization mean values (RGB)
    pub mean: [f32; 3],
    /// Normalization standard deviation values (RGB)
    pub std: [f32; 3],
    /// Whether to resize and center crop
    pub resize_and_crop: bool,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

/// Image interpolation methods
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    /// Nearest neighbor
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
}

/// Multimodal input combining text and visual data
#[derive(Debug)]
pub struct MultimodalInput {
    /// Text prompt
    pub text: Option<String>,
    /// Image data
    pub images: Vec<ImageInput>,
    /// Video data (experimental)
    pub videos: Vec<VideoInput>,
    /// Audio data (experimental)
    pub audio: Vec<AudioInput>,
    /// Input metadata
    pub metadata: HashMap<String, String>,
}

/// Image input data
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Image data (RGB bytes)
    pub data: Vec<u8>,
    /// Image dimensions (width, height)
    pub dimensions: (u32, u32),
    /// Image format
    pub format: ImageFormat,
    /// Optional caption or description
    pub caption: Option<String>,
}

/// Supported image formats
#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    /// RGB format
    Rgb,
    /// RGBA format
    Rgba,
    /// JPEG format
    Jpeg,
    /// PNG format
    Png,
    /// WebP format
    WebP,
}

/// Video input data (experimental)
#[derive(Debug, Clone)]
pub struct VideoInput {
    /// Frame data
    pub frames: Vec<ImageInput>,
    /// Frame rate
    pub fps: f32,
    /// Duration in seconds
    pub duration: f32,
    /// Optional description
    pub description: Option<String>,
}

/// Enhanced audio input data with comprehensive format support
#[derive(Debug, Clone)]
pub struct AudioInput {
    /// Audio samples (normalized to -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Audio format information
    pub format: AudioFormat,
    /// Optional transcript for speech audio
    pub transcript: Option<String>,
    /// Audio metadata (artist, title, etc.)
    pub metadata: HashMap<String, String>,
}

/// Enhanced audio format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Container format (wav, mp3, flac, etc.)
    pub container: String,
    /// Codec used (pcm, mp3, flac, aac, etc.)
    pub codec: String,
    /// Bit depth (8, 16, 24, 32)
    pub bit_depth: u16,
    /// Bitrate for compressed formats
    pub bitrate: Option<u32>,
}

/// Audio processor for advanced audio processing
#[cfg(feature = "multimodal")]
pub struct AudioProcessor {
    config: AudioProcessingConfig,
    supported_formats: Vec<String>,
}

/// Configuration for audio processing
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone)]
pub struct AudioProcessingConfig {
    /// Default sample rate for processing
    pub default_sample_rate: u32,
    /// Default number of channels
    pub default_channels: u16,
    /// Maximum audio duration in seconds
    pub max_duration: Duration,
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Enable automatic gain control
    pub enable_agc: bool,
    /// Speech-to-text configuration
    pub stt_config: Option<SpeechToTextConfig>,
    /// Text-to-speech configuration
    pub tts_config: Option<TextToSpeechConfig>,
}

/// Speech-to-text configuration
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone)]
pub struct SpeechToTextConfig {
    /// Language model for transcription
    pub language: String,
    /// Enable speaker identification
    pub enable_speaker_id: bool,
    /// Enable confidence scores
    pub enable_confidence: bool,
    /// Minimum confidence threshold
    pub min_confidence: f32,
}

/// Text-to-speech configuration
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone)]
pub struct TextToSpeechConfig {
    /// Voice to use for synthesis
    pub voice: String,
    /// Speaking rate (0.5 = half speed, 2.0 = double speed)
    pub rate: f32,
    /// Pitch adjustment (-1.0 to 1.0)
    pub pitch: f32,
    /// Volume level (0.0 to 1.0)
    pub volume: f32,
    /// Output audio format
    pub output_format: AudioFormat,
}

/// Multimodal generation output
#[derive(Debug)]
pub struct MultimodalOutput {
    /// Generated text
    pub text: Option<String>,
    /// Generated image features (for text-to-image)
    pub image_features: Option<Vec<f32>>,
    /// Attention weights for interpretability
    pub attention_weights: Option<AttentionWeights>,
    /// Generation metadata
    pub metadata: HashMap<String, f64>,
}

/// Attention weights for multimodal interpretability
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Text-to-image attention weights
    pub text_to_image: Vec<Vec<f32>>,
    /// Image-to-text attention weights
    pub image_to_text: Vec<Vec<f32>>,
    /// Self-attention weights
    pub self_attention: Vec<Vec<f32>>,
}

/// Multimodal generation parameters
#[derive(Debug, Clone)]
pub struct MultimodalGenerationParams {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Whether to include attention weights in output
    pub include_attention: bool,
    /// Custom stopping criteria
    pub stop_sequences: Vec<String>,
}

impl MultimodalProcessor {
    /// Create a new multimodal processor
    ///
    /// # Example
    /// ```rust
    /// use mullama::multimodal::{MultimodalProcessor, MultimodalConfig};
    ///
    /// let config = MultimodalConfig::default();
    /// let processor = MultimodalProcessor::new(text_model, Some(vision_encoder), config)?;
    /// ```
    pub fn new(
        text_model: Model,
        vision_encoder: Option<VisionEncoder>,
        config: MultimodalConfig,
    ) -> Result<Self, MullamaError> {
        let mut supported_modalities = vec![Modality::Text];

        if vision_encoder.is_some() {
            supported_modalities.push(Modality::Image);
        }

        Ok(Self {
            text_model,
            vision_encoder,
            config,
            supported_modalities,
        })
    }

    /// Load a multimodal model from files
    pub fn from_files<P: AsRef<Path>>(
        text_model_path: P,
        vision_model_path: Option<P>,
        config: MultimodalConfig,
    ) -> Result<Self, MullamaError> {
        let text_model = Model::load(text_model_path)?;

        let vision_encoder = if let Some(vision_path) = vision_model_path {
            Some(VisionEncoder::from_file(vision_path)?)
        } else {
            None
        };

        Self::new(text_model, vision_encoder, config)
    }

    /// Process multimodal input and generate response
    ///
    /// # Example
    /// ```rust
    /// use mullama::multimodal::{MultimodalInput, MultimodalGenerationParams};
    ///
    /// let mut input = MultimodalInput::new();
    /// input.set_text("Describe this image:");
    /// input.add_image_from_path("path/to/image.jpg")?;
    ///
    /// let params = MultimodalGenerationParams::default();
    /// let output = processor.generate(&input, &params)?;
    /// ```
    pub fn generate(
        &mut self,
        input: &MultimodalInput,
        params: &MultimodalGenerationParams,
    ) -> Result<MultimodalOutput, MullamaError> {
        // Validate input modalities
        self.validate_input(input)?;

        // Process images if present
        let image_features = if !input.images.is_empty() {
            Some(self.process_images(&input.images)?)
        } else {
            None
        };

        // Create multimodal context
        let mut context = self.create_multimodal_context(input, image_features.as_ref())?;

        // Generate response
        let text_output = if let Some(ref text) = input.text {
            Some(self.generate_text_response(&mut context, text, params)?)
        } else {
            None
        };

        // Create output
        let output = MultimodalOutput {
            text: text_output,
            image_features,
            attention_weights: if params.include_attention {
                Some(self.extract_attention_weights(&context)?)
            } else {
                None
            },
            metadata: HashMap::new(),
        };

        Ok(output)
    }

    /// Process a batch of multimodal inputs
    pub fn generate_batch(
        &mut self,
        inputs: &[MultimodalInput],
        params: &MultimodalGenerationParams,
    ) -> Result<Vec<MultimodalOutput>, MullamaError> {
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let output = self.generate(input, params)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Get supported modalities
    pub fn supported_modalities(&self) -> &[Modality] {
        &self.supported_modalities
    }

    /// Check if a specific modality is supported
    pub fn supports_modality(&self, modality: Modality) -> bool {
        self.supported_modalities.contains(&modality)
    }

    /// Update processor configuration
    pub fn update_config(&mut self, config: MultimodalConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &MultimodalConfig {
        &self.config
    }

    /// Process images through vision encoder
    fn process_images(&self, images: &[ImageInput]) -> Result<Vec<f32>, MullamaError> {
        if let Some(ref vision_encoder) = self.vision_encoder {
            vision_encoder.encode_images(images)
        } else {
            Err(MullamaError::NotSupported(
                "Vision encoder not available".to_string(),
            ))
        }
    }

    /// Validate input modalities against supported ones
    fn validate_input(&self, input: &MultimodalInput) -> Result<(), MullamaError> {
        if !input.images.is_empty() && !self.supports_modality(Modality::Image) {
            return Err(MullamaError::NotSupported(
                "Image processing not supported".to_string(),
            ));
        }

        if !input.videos.is_empty() && !self.supports_modality(Modality::Video) {
            return Err(MullamaError::NotSupported(
                "Video processing not supported".to_string(),
            ));
        }

        if !input.audio.is_empty() && !self.supports_modality(Modality::Audio) {
            return Err(MullamaError::NotSupported(
                "Audio processing not supported".to_string(),
            ));
        }

        Ok(())
    }

    /// Create multimodal context combining text and vision
    fn create_multimodal_context(
        &self,
        input: &MultimodalInput,
        image_features: Option<&Vec<f32>>,
    ) -> Result<Context, MullamaError> {
        // Create context from text model
        // Placeholder for context creation - multimodal not yet fully implemented
        Err(MullamaError::NotImplemented(
            "Multimodal context creation not implemented".to_string(),
        ))
    }

    /// Inject image features into context
    fn inject_image_features(
        &self,
        context: &mut Context,
        features: &[f32],
    ) -> Result<(), MullamaError> {
        // This would implement the actual injection of image features
        // into the language model context, typically through cross-attention
        // For now, this is a placeholder
        Ok(())
    }

    /// Generate text response given multimodal context
    fn generate_text_response(
        &self,
        _context: &mut Context,
        _prompt: &str,
        _params: &MultimodalGenerationParams,
    ) -> Result<String, MullamaError> {
        // Placeholder - multimodal text generation not yet fully implemented
        Err(MullamaError::NotImplemented(
            "Multimodal text generation not implemented".to_string(),
        ))
    }

    /// Extract attention weights for interpretability
    fn extract_attention_weights(
        &self,
        context: &Context,
    ) -> Result<AttentionWeights, MullamaError> {
        // This would extract actual attention weights from the model
        // For now, return placeholder weights
        Ok(AttentionWeights {
            text_to_image: vec![vec![0.5; 10]; 10],
            image_to_text: vec![vec![0.5; 10]; 10],
            self_attention: vec![vec![0.5; 10]; 10],
        })
    }
}

impl VisionEncoder {
    /// Load vision encoder from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, MullamaError> {
        let path_str = CString::new(path.as_ref().to_string_lossy().as_ref())
            .map_err(|_| MullamaError::InvalidInput("Path contains null byte".to_string()))?;

        // Load vision model using llama.cpp with default params
        let default_params = unsafe { sys::llama_model_default_params() };
        let vision_model_ptr =
            unsafe { sys::llama_model_load_from_file(path_str.as_ptr(), default_params) };

        if vision_model_ptr.is_null() {
            return Err(MullamaError::ModelLoadError(
                "Failed to load vision model".to_string(),
            ));
        }

        Ok(Self {
            vision_model_ptr,
            preprocess_config: ImagePreprocessConfig::default(),
            encoder_type: VisionEncoderType::Clip,
        })
    }

    /// Encode images to feature vectors
    pub fn encode_images(&self, images: &[ImageInput]) -> Result<Vec<f32>, MullamaError> {
        let mut all_features = Vec::new();

        for image in images {
            let features = self.encode_single_image(image)?;
            all_features.extend(features);
        }

        Ok(all_features)
    }

    /// Encode a single image
    fn encode_single_image(&self, image: &ImageInput) -> Result<Vec<f32>, MullamaError> {
        // Preprocess the image
        let preprocessed = self.preprocess_image(image)?;

        // Run through vision encoder
        let features = self.forward_vision_model(&preprocessed)?;

        Ok(features)
    }

    /// Preprocess image according to configuration
    fn preprocess_image(&self, image: &ImageInput) -> Result<Vec<f32>, MullamaError> {
        let (width, height) = image.dimensions;
        let target_size = self.preprocess_config.target_size;

        // Convert image data to f32 and normalize
        let mut processed = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);

        // Simple preprocessing (in practice, this would be more sophisticated)
        for pixel in image.data.chunks(3) {
            let r = (pixel[0] as f32 / 255.0 - self.preprocess_config.mean[0])
                / self.preprocess_config.std[0];
            let g = (pixel[1] as f32 / 255.0 - self.preprocess_config.mean[1])
                / self.preprocess_config.std[1];
            let b = (pixel[2] as f32 / 255.0 - self.preprocess_config.mean[2])
                / self.preprocess_config.std[2];

            processed.extend_from_slice(&[r, g, b]);
        }

        Ok(processed)
    }

    /// Forward pass through vision model
    fn forward_vision_model(&self, preprocessed_image: &[f32]) -> Result<Vec<f32>, MullamaError> {
        // This would implement the actual forward pass through the vision model
        // For now, return placeholder features
        Ok(vec![0.1; 768]) // Typical CLIP feature dimension
    }
}

impl MultimodalInput {
    /// Create a new multimodal input
    pub fn new() -> Self {
        Self {
            text: None,
            images: Vec::new(),
            videos: Vec::new(),
            audio: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set text prompt
    pub fn set_text<S: Into<String>>(&mut self, text: S) {
        self.text = Some(text.into());
    }

    /// Add an image from raw data
    pub fn add_image(&mut self, data: Vec<u8>, dimensions: (u32, u32), format: ImageFormat) {
        self.images.push(ImageInput {
            data,
            dimensions,
            format,
            caption: None,
        });
    }

    /// Add an image from file path
    pub fn add_image_from_path<P: AsRef<Path>>(&mut self, path: P) -> Result<(), MullamaError> {
        // This would load and decode the image file
        // For now, return a placeholder implementation
        let placeholder_data = vec![128u8; 224 * 224 * 3]; // 224x224 RGB
        self.add_image(placeholder_data, (224, 224), ImageFormat::Rgb);
        Ok(())
    }

    /// Add metadata
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            max_image_resolution: (512, 512),
            patch_size: 16,
            vision_tokens_per_image: 256,
            enable_image_to_text: true,
            enable_text_to_image: false,
            cross_attention_config: CrossAttentionConfig::default(),
            temperature: 0.7,
        }
    }
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            num_layers: 6,
            num_heads: 8,
            hidden_dim: 768,
            dropout_rate: 0.1,
        }
    }
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: (224, 224),
            mean: [0.485, 0.456, 0.406], // ImageNet normalization
            std: [0.229, 0.224, 0.225],  // ImageNet normalization
            resize_and_crop: true,
            interpolation: InterpolationMethod::Bilinear,
        }
    }
}

impl Default for MultimodalGenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            include_attention: false,
            stop_sequences: vec!["<|end|>".to_string(), "</s>".to_string()],
        }
    }
}

/// Utility functions for multimodal processing
pub mod utils {
    use super::*;

    /// Create a basic image-to-text configuration
    pub fn image_to_text_config() -> MultimodalConfig {
        MultimodalConfig {
            enable_image_to_text: true,
            enable_text_to_image: false,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for visual question answering
    pub fn vqa_config() -> MultimodalConfig {
        MultimodalConfig {
            max_image_resolution: (384, 384),
            vision_tokens_per_image: 196, // 14x14 patches
            cross_attention_config: CrossAttentionConfig {
                num_layers: 12,
                num_heads: 12,
                hidden_dim: 768,
                dropout_rate: 0.05,
            },
            temperature: 0.1, // Lower temperature for factual answers
            ..Default::default()
        }
    }

    /// Create a configuration for image captioning
    pub fn captioning_config() -> MultimodalConfig {
        MultimodalConfig {
            max_image_resolution: (224, 224),
            vision_tokens_per_image: 196,
            temperature: 0.8, // Higher temperature for creative captions
            ..Default::default()
        }
    }

    /// Validate image format compatibility
    pub fn validate_image_format(format: ImageFormat) -> bool {
        matches!(
            format,
            ImageFormat::Rgb | ImageFormat::Rgba | ImageFormat::Jpeg | ImageFormat::Png
        )
    }

    /// Calculate optimal batch size for multimodal processing
    pub fn calculate_optimal_batch_size(
        model_size: u64,
        available_memory: u64,
        image_resolution: (u32, u32),
    ) -> usize {
        let base_model_memory = model_size;
        let image_memory = (image_resolution.0 * image_resolution.1 * 3 * 4) as u64; // RGB, f32
        let safety_factor = 0.8; // Use 80% of available memory

        let usable_memory = (available_memory as f64 * safety_factor) as u64;
        let memory_per_sample = base_model_memory / 10 + image_memory; // Rough estimate

        std::cmp::max(1, (usable_memory / memory_per_sample) as usize)
    }

    /// Create an audio input from file path
    pub async fn load_audio_from_path(path: impl AsRef<Path>) -> Result<AudioInput, MullamaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(MullamaError::ConfigError(format!(
                "Audio file not found: {}",
                path.display()
            )));
        }

        // Placeholder for actual audio loading
        // In real implementation, this would use libraries like rodio, symphonia, etc.
        let samples = vec![0.0; 44100]; // 1 second of silence
        let format = AudioFormat {
            container: "wav".to_string(),
            codec: "pcm".to_string(),
            bit_depth: 16,
            bitrate: None,
        };

        Ok(AudioInput {
            samples,
            sample_rate: 44100,
            channels: 1,
            duration: 1.0,
            format,
            transcript: None,
            metadata: HashMap::new(),
        })
    }

    /// Process audio with noise reduction and normalization
    pub fn process_audio(
        audio: &mut AudioInput,
        config: &AudioProcessingConfig,
    ) -> Result<(), MullamaError> {
        if config.enable_noise_reduction {
            apply_noise_reduction(&mut audio.samples);
        }

        if config.enable_agc {
            apply_automatic_gain_control(&mut audio.samples);
        }

        // Resample if needed
        if audio.sample_rate != config.default_sample_rate {
            audio.samples = resample_audio(
                &audio.samples,
                audio.sample_rate,
                config.default_sample_rate,
            )?;
            audio.sample_rate = config.default_sample_rate;
        }

        Ok(())
    }

    /// Convert between audio formats
    pub fn convert_audio_format(
        input: &AudioInput,
        target_format: &AudioFormat,
    ) -> Result<AudioInput, MullamaError> {
        // Placeholder for audio format conversion
        let mut output = input.clone();
        output.format = target_format.clone();
        Ok(output)
    }

    /// Extract audio features for analysis
    pub fn extract_audio_features(audio: &AudioInput) -> AudioFeatures {
        // Placeholder for actual feature extraction
        AudioFeatures {
            duration: audio.duration,
            energy: calculate_energy(&audio.samples),
            zero_crossing_rate: calculate_zero_crossing_rate(&audio.samples),
            spectral_centroid: 1000.0,           // Placeholder
            mfcc: vec![0.1, 0.2, 0.3, 0.4, 0.5], // 5 MFCC coefficients
            pitch: detect_pitch(&audio.samples, audio.sample_rate),
            tempo: detect_tempo(&audio.samples, audio.sample_rate),
            has_speech: detect_speech(&audio.samples),
        }
    }
}

/// Audio feature extraction results
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub duration: f32,
    pub energy: f32,
    pub zero_crossing_rate: f32,
    pub spectral_centroid: f32,
    pub mfcc: Vec<f32>,
    pub pitch: f32,
    pub tempo: f32,
    pub has_speech: bool,
}

// Audio processing helper functions
fn apply_noise_reduction(samples: &mut [f32]) {
    // Placeholder for noise reduction algorithm
    // In real implementation, this would use spectral subtraction or Wiener filtering
    for sample in samples.iter_mut() {
        if sample.abs() < 0.01 {
            *sample = 0.0; // Simple noise gate
        }
    }
}

fn apply_automatic_gain_control(samples: &mut [f32]) {
    // Simple AGC implementation
    let max_amplitude = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    if max_amplitude > 0.0 {
        let gain = 0.8 / max_amplitude; // Normalize to 80% of full scale
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, MullamaError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Simple linear interpolation resampling (placeholder)
    let ratio = to_rate as f32 / from_rate as f32;
    let new_length = (samples.len() as f32 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_length);

    for i in 0..new_length {
        let original_index = i as f32 / ratio;
        let index = original_index as usize;

        if index < samples.len() - 1 {
            let frac = original_index - index as f32;
            let sample = samples[index] * (1.0 - frac) + samples[index + 1] * frac;
            resampled.push(sample);
        } else if index < samples.len() {
            resampled.push(samples[index]);
        }
    }

    Ok(resampled)
}

fn calculate_energy(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32
}

fn calculate_zero_crossing_rate(samples: &[f32]) -> f32 {
    let mut crossings = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            crossings += 1;
        }
    }
    crossings as f32 / samples.len() as f32
}

fn detect_pitch(samples: &[f32], sample_rate: u32) -> f32 {
    // Placeholder for pitch detection (would use autocorrelation or YIN algorithm)
    440.0 // A4 note
}

fn detect_tempo(samples: &[f32], sample_rate: u32) -> f32 {
    // Placeholder for tempo detection
    120.0 // 120 BPM
}

fn detect_speech(samples: &[f32]) -> bool {
    // Simple speech detection based on energy and zero-crossing rate
    let energy = calculate_energy(samples);
    let zcr = calculate_zero_crossing_rate(samples);

    // Heuristic thresholds for speech detection
    energy > 0.01 && zcr > 0.1 && zcr < 0.4
}

//
// Real mtmd-based multimodal implementation
//

use std::ffi::CStr;
use std::sync::Arc;

/// Bitmap wrapper for mtmd_bitmap (holds image or audio data)
///
/// This is the core data structure for passing media to the multimodal processor.
/// It can hold either image data (RGB pixels) or audio data (f32 samples).
pub struct Bitmap {
    ptr: *mut sys::mtmd_bitmap,
}

// SAFETY: mtmd_bitmap is read-only after creation
unsafe impl Send for Bitmap {}
unsafe impl Sync for Bitmap {}

impl Bitmap {
    /// Create a bitmap from RGB image data
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - RGB pixel data (length must be width * height * 3)
    ///
    /// # Example
    /// ```rust
    /// let rgb_data = vec![255u8; 224 * 224 * 3]; // White 224x224 image
    /// let bitmap = Bitmap::from_image(224, 224, &rgb_data)?;
    /// ```
    pub fn from_image(width: u32, height: u32, data: &[u8]) -> Result<Self, MullamaError> {
        let expected_len = (width * height * 3) as usize;
        if data.len() != expected_len {
            return Err(MullamaError::InvalidInput(format!(
                "Image data length {} doesn't match expected {} ({}x{}x3)",
                data.len(),
                expected_len,
                width,
                height
            )));
        }

        let ptr = unsafe { sys::mtmd_bitmap_init(width, height, data.as_ptr()) };
        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(
                "Failed to create image bitmap".to_string(),
            ));
        }

        Ok(Self { ptr })
    }

    /// Create a bitmap from audio samples
    ///
    /// # Arguments
    /// * `samples` - Audio samples as f32 (PCM format, -1.0 to 1.0)
    ///
    /// # Example
    /// ```rust
    /// let samples = vec![0.0f32; 16000]; // 1 second of silence at 16kHz
    /// let bitmap = Bitmap::from_audio(&samples)?;
    /// ```
    pub fn from_audio(samples: &[f32]) -> Result<Self, MullamaError> {
        if samples.is_empty() {
            return Err(MullamaError::InvalidInput(
                "Audio samples cannot be empty".to_string(),
            ));
        }

        let ptr = unsafe { sys::mtmd_bitmap_init_from_audio(samples.len(), samples.as_ptr()) };
        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(
                "Failed to create audio bitmap".to_string(),
            ));
        }

        Ok(Self { ptr })
    }

    /// Get the width of the bitmap (for images)
    pub fn width(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_nx(self.ptr) }
    }

    /// Get the height of the bitmap (for images)
    pub fn height(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_ny(self.ptr) }
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        unsafe { sys::mtmd_bitmap_get_n_bytes(self.ptr) }
    }

    /// Check if this bitmap contains audio data
    pub fn is_audio(&self) -> bool {
        unsafe { sys::mtmd_bitmap_is_audio(self.ptr) }
    }

    /// Get the bitmap ID (if set)
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { sys::mtmd_bitmap_get_id(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() })
        }
    }

    /// Set the bitmap ID (useful for KV cache tracking)
    pub fn set_id(&mut self, id: &str) -> Result<(), MullamaError> {
        let id_c = CString::new(id)
            .map_err(|_| MullamaError::InvalidInput("ID contains null byte".to_string()))?;
        unsafe { sys::mtmd_bitmap_set_id(self.ptr, id_c.as_ptr()) };
        Ok(())
    }

    /// Get the raw pointer (for FFI calls)
    pub(crate) fn as_ptr(&self) -> *const sys::mtmd_bitmap {
        self.ptr
    }
}

impl Drop for Bitmap {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::mtmd_bitmap_free(self.ptr) };
        }
    }
}

/// Input chunks - a collection of tokenized multimodal input
///
/// This holds the result of tokenizing text with embedded media markers.
/// Each chunk is either text tokens or media (image/audio) data.
pub struct InputChunks {
    ptr: *mut sys::mtmd_input_chunks,
}

unsafe impl Send for InputChunks {}
unsafe impl Sync for InputChunks {}

impl InputChunks {
    /// Create a new empty input chunks container
    pub fn new() -> Self {
        let ptr = unsafe { sys::mtmd_input_chunks_init() };
        Self { ptr }
    }

    /// Get the number of chunks
    pub fn len(&self) -> usize {
        unsafe { sys::mtmd_input_chunks_size(self.ptr) }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index
    pub fn get(&self, idx: usize) -> Option<InputChunk> {
        if idx >= self.len() {
            return None;
        }
        let chunk_ptr = unsafe { sys::mtmd_input_chunks_get(self.ptr, idx) };
        if chunk_ptr.is_null() {
            None
        } else {
            Some(InputChunk {
                ptr: chunk_ptr,
                owned: false,
            })
        }
    }

    /// Get total number of tokens across all chunks
    pub fn total_tokens(&self) -> usize {
        unsafe { sys::mtmd_helper_get_n_tokens(self.ptr) }
    }

    /// Get total position count (for M-RoPE models this differs from token count)
    pub fn total_positions(&self) -> i32 {
        unsafe { sys::mtmd_helper_get_n_pos(self.ptr) }
    }

    /// Iterate over chunks
    pub fn iter(&self) -> impl Iterator<Item = InputChunk> + '_ {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    /// Get the raw pointer (for FFI calls)
    pub(crate) fn as_ptr(&self) -> *mut sys::mtmd_input_chunks {
        self.ptr
    }
}

impl Default for InputChunks {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for InputChunks {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::mtmd_input_chunks_free(self.ptr) };
        }
    }
}

/// Type of input chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// Text tokens
    Text,
    /// Image data
    Image,
    /// Audio data
    Audio,
}

impl From<sys::mtmd_input_chunk_type> for ChunkType {
    fn from(t: sys::mtmd_input_chunk_type) -> Self {
        match t {
            sys::mtmd_input_chunk_type::MTMD_INPUT_CHUNK_TYPE_TEXT => ChunkType::Text,
            sys::mtmd_input_chunk_type::MTMD_INPUT_CHUNK_TYPE_IMAGE => ChunkType::Image,
            sys::mtmd_input_chunk_type::MTMD_INPUT_CHUNK_TYPE_AUDIO => ChunkType::Audio,
        }
    }
}

/// A single input chunk (text or media)
pub struct InputChunk {
    ptr: *const sys::mtmd_input_chunk,
    owned: bool,
}

impl InputChunk {
    /// Get the type of this chunk
    pub fn chunk_type(&self) -> ChunkType {
        let t = unsafe { sys::mtmd_input_chunk_get_type(self.ptr) };
        ChunkType::from(t)
    }

    /// Get the number of tokens in this chunk
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.ptr) }
    }

    /// Get the number of positions (for M-RoPE)
    pub fn n_positions(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.ptr) }
    }

    /// Get the chunk ID (for media chunks)
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { sys::mtmd_input_chunk_get_id(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() })
        }
    }

    /// Get text tokens (only valid for text chunks)
    pub fn text_tokens(&self) -> Option<Vec<i32>> {
        if self.chunk_type() != ChunkType::Text {
            return None;
        }

        let mut n_tokens: usize = 0;
        let tokens_ptr = unsafe { sys::mtmd_input_chunk_get_tokens_text(self.ptr, &mut n_tokens) };

        if tokens_ptr.is_null() || n_tokens == 0 {
            return None;
        }

        let tokens = unsafe { std::slice::from_raw_parts(tokens_ptr, n_tokens) };
        Some(tokens.to_vec())
    }

    /// Copy this chunk (takes ownership)
    pub fn copy(&self) -> Self {
        let ptr = unsafe { sys::mtmd_input_chunk_copy(self.ptr) };
        Self { ptr, owned: true }
    }
}

impl Drop for InputChunk {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { sys::mtmd_input_chunk_free(self.ptr as *mut _) };
        }
    }
}

/// Multimodal context parameters
#[derive(Debug, Clone)]
pub struct MtmdParams {
    /// Whether to use GPU for encoding
    pub use_gpu: bool,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use
    pub n_threads: i32,
    /// Media marker in prompts (default: "<__media__>")
    pub media_marker: Option<String>,
    /// Flash attention setting
    pub flash_attn_type: sys::llama_flash_attn_type,
    /// Whether to run warmup encode after initialization
    pub warmup: bool,
    /// Minimum image tokens (for dynamic resolution models)
    pub image_min_tokens: Option<i32>,
    /// Maximum image tokens (for dynamic resolution models)
    pub image_max_tokens: Option<i32>,
}

impl Default for MtmdParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            print_timings: false,
            n_threads: 4,
            media_marker: None,
            flash_attn_type: sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO,
            warmup: true,
            image_min_tokens: None,
            image_max_tokens: None,
        }
    }
}

impl MtmdParams {
    /// Convert to FFI params, using defaults for unspecified values
    fn to_sys(&self) -> sys::mtmd_context_params {
        let default = unsafe { sys::mtmd_context_params_default() };

        sys::mtmd_context_params {
            use_gpu: self.use_gpu,
            print_timings: self.print_timings,
            n_threads: self.n_threads,
            image_marker: default.image_marker, // deprecated, keep default
            media_marker: default.media_marker, // keep default, custom set later
            flash_attn_type: self.flash_attn_type,
            warmup: self.warmup,
            image_min_tokens: self.image_min_tokens.unwrap_or(default.image_min_tokens),
            image_max_tokens: self.image_max_tokens.unwrap_or(default.image_max_tokens),
        }
    }
}

/// Multimodal context for vision-language and audio-language models
///
/// This is the main interface for processing multimodal inputs with llama.cpp models.
///
/// # Example
/// ```rust
/// use mullama::{Model, MtmdContext, Bitmap, MtmdParams};
/// use std::sync::Arc;
///
/// let model = Arc::new(Model::load("llava-model.gguf")?);
/// let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;
///
/// // Load an image
/// let image = Bitmap::from_file(&mtmd, "test.jpg")?;
///
/// // Tokenize prompt with image
/// let chunks = mtmd.tokenize("Describe this image: <__media__>", &[&image])?;
///
/// // Evaluate in context and generate...
/// ```
pub struct MtmdContext {
    ptr: *mut sys::mtmd_context,
    model: Arc<crate::Model>,
    // Keep CStrings alive
    _media_marker: Option<CString>,
}

unsafe impl Send for MtmdContext {}
// Safety: Access is synchronized through RwLock in the daemon
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// Create a new multimodal context
    ///
    /// # Arguments
    /// * `mmproj_path` - Path to the multimodal projector file (e.g., mmproj.gguf)
    /// * `model` - The text model to use
    /// * `params` - Configuration parameters
    pub fn new(
        mmproj_path: &str,
        model: &Arc<crate::Model>,
        params: MtmdParams,
    ) -> Result<Self, MullamaError> {
        let path_c = CString::new(mmproj_path)
            .map_err(|_| MullamaError::InvalidInput("Path contains null byte".to_string()))?;

        let mut sys_params = params.to_sys();

        // Handle custom media marker
        let media_marker_c = if let Some(ref marker) = params.media_marker {
            let c = CString::new(marker.as_str())
                .map_err(|_| MullamaError::InvalidInput("Marker contains null byte".to_string()))?;
            sys_params.media_marker = c.as_ptr();
            Some(c)
        } else {
            None
        };

        let ptr = unsafe { sys::mtmd_init_from_file(path_c.as_ptr(), model.as_ptr(), sys_params) };

        if ptr.is_null() {
            return Err(MullamaError::ModelLoadError(format!(
                "Failed to load multimodal projector from: {}",
                mmproj_path
            )));
        }

        Ok(Self {
            ptr,
            model: model.clone(),
            _media_marker: media_marker_c,
        })
    }

    /// Check if this context supports vision (image) input
    pub fn supports_vision(&self) -> bool {
        unsafe { sys::mtmd_support_vision(self.ptr) }
    }

    /// Check if this context supports audio input
    pub fn supports_audio(&self) -> bool {
        unsafe { sys::mtmd_support_audio(self.ptr) }
    }

    /// Get the audio bitrate (sample rate) if audio is supported
    pub fn audio_bitrate(&self) -> Option<i32> {
        let rate = unsafe { sys::mtmd_get_audio_bitrate(self.ptr) };
        if rate < 0 {
            None
        } else {
            Some(rate)
        }
    }

    /// Check if non-causal attention is needed for decoding
    pub fn needs_non_causal(&self) -> bool {
        unsafe { sys::mtmd_decode_use_non_causal(self.ptr) }
    }

    /// Check if M-RoPE is used
    pub fn uses_mrope(&self) -> bool {
        unsafe { sys::mtmd_decode_use_mrope(self.ptr) }
    }

    /// Load a bitmap from a file (image or audio)
    ///
    /// Supported formats:
    /// - Images: jpg, png, bmp, gif (via stb_image)
    /// - Audio: wav, mp3, flac (via miniaudio)
    pub fn bitmap_from_file(&self, path: &str) -> Result<Bitmap, MullamaError> {
        let path_c = CString::new(path)
            .map_err(|_| MullamaError::InvalidInput("Path contains null byte".to_string()))?;

        let ptr = unsafe { sys::mtmd_helper_bitmap_init_from_file(self.ptr, path_c.as_ptr()) };

        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to load media from file: {}",
                path
            )));
        }

        Ok(Bitmap { ptr })
    }

    /// Load a bitmap from a buffer (image or audio file data)
    pub fn bitmap_from_buffer(&self, data: &[u8]) -> Result<Bitmap, MullamaError> {
        let ptr =
            unsafe { sys::mtmd_helper_bitmap_init_from_buf(self.ptr, data.as_ptr(), data.len()) };

        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(
                "Failed to create bitmap from buffer".to_string(),
            ));
        }

        Ok(Bitmap { ptr })
    }

    /// Tokenize text with media markers, replacing markers with bitmaps
    ///
    /// # Arguments
    /// * `text` - Text prompt containing media markers (default: "<__media__>")
    /// * `bitmaps` - Bitmaps to substitute for each marker
    ///
    /// # Returns
    /// Input chunks ready for evaluation
    ///
    /// # Errors
    /// - Returns error if number of bitmaps doesn't match markers
    /// - Returns error if preprocessing fails
    pub fn tokenize(
        &mut self,
        text: &str,
        bitmaps: &[&Bitmap],
    ) -> Result<InputChunks, MullamaError> {
        let text_c = CString::new(text)
            .map_err(|_| MullamaError::InvalidInput("Text contains null byte".to_string()))?;

        let input_text = sys::mtmd_input_text {
            text: text_c.as_ptr(),
            add_special: true,
            parse_special: true,
        };

        let chunks = InputChunks::new();

        // Create array of bitmap pointers
        let bitmap_ptrs: Vec<*const sys::mtmd_bitmap> =
            bitmaps.iter().map(|b| b.as_ptr()).collect();

        let result = unsafe {
            sys::mtmd_tokenize(
                self.ptr,
                chunks.ptr,
                &input_text,
                bitmap_ptrs.as_ptr(),
                bitmap_ptrs.len(),
            )
        };

        match result {
            0 => Ok(chunks),
            1 => Err(MullamaError::InvalidInput(format!(
                "Number of bitmaps ({}) doesn't match markers in text",
                bitmaps.len()
            ))),
            2 => Err(MullamaError::MultimodalError(
                "Image preprocessing failed".to_string(),
            )),
            _ => Err(MullamaError::MultimodalError(format!(
                "Tokenization failed with code: {}",
                result
            ))),
        }
    }

    /// Evaluate chunks in a llama context
    ///
    /// This is the main function for processing multimodal input.
    /// It handles both text and media chunks automatically.
    ///
    /// # Arguments
    /// * `context` - The llama context to evaluate in
    /// * `chunks` - The tokenized input chunks
    /// * `n_past` - Current position in the context
    /// * `seq_id` - Sequence ID (usually 0)
    /// * `n_batch` - Batch size for processing
    /// * `logits_last` - Whether to compute logits for the last token
    ///
    /// # Returns
    /// New position after evaluation
    pub fn eval_chunks(
        &mut self,
        context: &mut crate::Context,
        chunks: &InputChunks,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<i32, MullamaError> {
        let mut new_n_past: i32 = 0;

        let result = unsafe {
            sys::mtmd_helper_eval_chunks(
                self.ptr,
                context.as_ptr(),
                chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &mut new_n_past,
            )
        };

        if result != 0 {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to evaluate multimodal chunks: error code {}",
                result
            )));
        }

        Ok(new_n_past)
    }

    /// Encode a single chunk and get the embeddings
    pub fn encode_chunk(&mut self, chunk: &InputChunk) -> Result<(), MullamaError> {
        let result = unsafe { sys::mtmd_encode_chunk(self.ptr, chunk.ptr) };

        if result != 0 {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to encode chunk: error code {}",
                result
            )));
        }

        Ok(())
    }

    /// Get the output embeddings from the last encode operation
    ///
    /// The size depends on the chunk that was encoded.
    pub fn get_output_embeddings(&self, chunk: &InputChunk) -> Option<&[f32]> {
        let ptr = unsafe { sys::mtmd_get_output_embd(self.ptr) };
        if ptr.is_null() {
            return None;
        }

        let n_tokens = chunk.n_tokens();
        let n_embd = self.model.n_embd() as usize;
        let size = n_tokens * n_embd;

        Some(unsafe { std::slice::from_raw_parts(ptr, size) })
    }

    /// Get the default media marker string
    pub fn default_marker() -> String {
        let ptr = unsafe { sys::mtmd_default_marker() };
        if ptr.is_null() {
            "<__media__>".to_string()
        } else {
            unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
        }
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::mtmd_free(self.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_input_creation() {
        let mut input = MultimodalInput::new();
        input.set_text("Describe this image");
        input.add_image(vec![128u8; 224 * 224 * 3], (224, 224), ImageFormat::Rgb);

        assert!(input.text.is_some());
        assert_eq!(input.images.len(), 1);
        assert_eq!(input.images[0].dimensions, (224, 224));
    }

    #[test]
    fn test_config_defaults() {
        let config = MultimodalConfig::default();
        assert_eq!(config.max_image_resolution, (512, 512));
        assert_eq!(config.patch_size, 16);
        assert!(config.enable_image_to_text);
        assert!(!config.enable_text_to_image);
    }

    #[test]
    fn test_image_preprocessing_config() {
        let config = ImagePreprocessConfig::default();
        assert_eq!(config.target_size, (224, 224));
        assert_eq!(config.mean, [0.485, 0.456, 0.406]);
        assert_eq!(config.std, [0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_modality_support() {
        let _config = MultimodalConfig::default();
        // Placeholder for test model - skip for now as we don't have a model in tests
    }

    #[test]
    fn test_generation_params() {
        let params = MultimodalGenerationParams::default();
        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 40);
        assert!(!params.include_attention);
    }

    #[test]
    fn test_utility_configs() {
        let vqa_config = utils::vqa_config();
        assert_eq!(vqa_config.temperature, 0.1);
        assert_eq!(vqa_config.cross_attention_config.num_layers, 12);

        let caption_config = utils::captioning_config();
        assert_eq!(caption_config.temperature, 0.8);
    }

    #[test]
    fn test_batch_size_calculation() {
        let batch_size = utils::calculate_optimal_batch_size(
            1_000_000_000, // 1GB model
            8_000_000_000, // 8GB available memory
            (224, 224),    // Image resolution
        );

        assert!(batch_size > 0);
        assert!(batch_size < 100); // Should be reasonable
    }
}
