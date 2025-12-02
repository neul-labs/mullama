# 🚀 Getting Started with Mullama

This guide will help you get up and running with Mullama's advanced integration features quickly.

## 📋 Table of Contents

- [Installation](#-installation)
- [Basic Setup](#-basic-setup)
- [Feature Overview](#-feature-overview)
- [Your First Application](#-your-first-application)
- [Common Patterns](#-common-patterns)
- [Next Steps](#-next-steps)

## 🛠️ Installation

### Basic Installation

Add Mullama to your `Cargo.toml`:

```toml
[dependencies]
mullama = "0.1.0"

# For complete feature experience
mullama = { version = "0.1.0", features = ["full"] }
```

### Platform-Specific Setup

> **📋 For detailed setup instructions**: See [Platform Setup Guide](./PLATFORM_SETUP.md)

#### Quick Setup by Platform

**Windows (PowerShell as Administrator)**
```powershell
# Install package manager and dependencies
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install cmake git rustup.install llvm ninja -y
refreshenv

# Audio dependencies
choco install vcredist-all -y

# For CUDA support (optional)
# Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
```

**Linux (Ubuntu/Debian)**
```bash
# Essential build tools
sudo apt update
sudo apt install -y build-essential cmake pkg-config git curl

# Audio dependencies (required for streaming-audio feature)
sudo apt install -y libasound2-dev libpulse-dev pulseaudio-utils

# Additional audio formats
sudo apt install -y libflac-dev libvorbis-dev libopus-dev libmp3lame-dev

# Media processing (for format-conversion feature)
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev

# Image processing (for multimodal feature)
sudo apt install -y libpng-dev libjpeg-dev libtiff-dev libwebp-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**Linux (CentOS/RHEL/Fedora)**
```bash
# For CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y epel-release cmake3 pkg-config git alsa-lib-devel pulseaudio-libs-devel

# For Fedora
sudo dnf groupinstall -y "Development Tools" "Development Libraries"
sudo dnf install -y cmake pkg-config git alsa-lib-devel pulseaudio-libs-devel \
    flac-devel libvorbis-devel opus-devel lame-devel ffmpeg-devel

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**macOS**
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake pkg-config git llvm ninja

# Audio dependencies
brew install portaudio libsamplerate libsndfile flac libvorbis opus lame jack

# Media processing
brew install ffmpeg imagemagick

# Image processing
brew install libpng jpeg libtiff webp

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### GPU Acceleration Setup

Enable GPU acceleration for improved performance:

```bash
# NVIDIA CUDA (Windows/Linux)
# 1. Install CUDA Toolkit 12.0+
# 2. Set environment variable:
export LLAMA_CUDA=1  # Linux
$env:LLAMA_CUDA=1    # Windows PowerShell

# AMD ROCm (Linux only)
# 1. Install ROCm drivers
# 2. Set environment variable:
export LLAMA_HIPBLAS=1

# Apple Metal (macOS only)
# Automatically available on Apple Silicon
export LLAMA_METAL=1

# Intel OpenCL (Cross-platform)
export LLAMA_CLBLAST=1
```

### Verify Installation

Test your setup with a simple build:

```bash
# Clone repository
git clone --recurse-submodules https://github.com/username/mullama.git
cd mullama

# Test basic build
cargo check

# Test with features
cargo check --features "async,streaming"

# Full build with all features
cargo build --release --features full
```

## 🎯 Basic Setup

### Simple Text Generation

```rust
use mullama::prelude::*;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    // Load a model
    let model = ModelBuilder::new()
        .path("path/to/model.gguf")
        .context_size(2048)
        .build().await?;

    // Generate text
    let response = model.generate("Hello, AI!", 50).await?;
    println!("Response: {}", response);

    Ok(())
}
```

### Configuration-Based Setup

```rust
use mullama::{MullamaConfig, ModelConfig, ContextConfig};

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    let config = MullamaConfig {
        model: ModelConfig {
            path: "model.gguf".to_string(),
            gpu_layers: 20,
            context_size: 4096,
            ..Default::default()
        },
        context: ContextConfig {
            n_ctx: 4096,
            n_batch: 1024,
            n_threads: 8,
            ..Default::default()
        },
        ..Default::default()
    };

    let model = config.load_model().await?;
    let response = model.generate("Explain AI", 100).await?;
    println!("{}", response);

    Ok(())
}
```

## 🌟 Feature Overview

### Choose Your Features

```toml
# Web applications
features = ["web", "websockets", "async", "streaming"]

# Multimodal AI
features = ["multimodal", "streaming-audio", "format-conversion"]

# High-performance
features = ["parallel", "tokio-runtime", "async"]

# Everything
features = ["full"]
```

### Feature Capabilities

| Feature | Use Case | Example |
|---------|----------|---------|
| `async` | Non-blocking operations | Real-time applications |
| `streaming` | Token-by-token generation | Interactive chat |
| `web` | REST APIs | AI services |
| `websockets` | Real-time communication | Live chat |
| `multimodal` | Text + Image + Audio | Content analysis |
| `streaming-audio` | Live audio processing | Voice assistants |
| `format-conversion` | Media format support | File processing |
| `parallel` | Batch processing | High throughput |
| `tokio-runtime` | Advanced async control | Production services |

## 🎮 Your First Application

Let's build a simple voice-enabled AI assistant:

### Step 1: Basic Setup

```rust
use mullama::prelude::*;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("🤖 Starting Voice AI Assistant");

    // Load model
    let model = ModelBuilder::new()
        .path("model.gguf")
        .context_size(4096)
        .build().await?;

    println!("✅ Model loaded successfully");
    Ok(())
}
```

### Step 2: Add Audio Processing

```rust
use mullama::{StreamingAudioProcessor, AudioStreamConfig};

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    // ... model setup ...

    // Setup audio processing
    let audio_config = AudioStreamConfig::new()
        .sample_rate(16000)
        .channels(1)
        .enable_voice_detection(true)
        .enable_noise_reduction(true);

    let mut audio_processor = StreamingAudioProcessor::new(audio_config)?;
    println!("🎤 Audio processor ready");

    Ok(())
}
```

### Step 3: Add Multimodal Processing

```rust
use mullama::MultimodalProcessor;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    // ... previous setup ...

    // Setup multimodal processing
    let multimodal = MultimodalProcessor::new()
        .enable_audio_processing()
        .build();

    println!("🎭 Multimodal processor ready");
    Ok(())
}
```

### Step 4: Create the Main Loop

```rust
#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    // ... setup code ...

    println!("🚀 Starting voice assistant (speak something!)");

    // Start audio capture
    let mut audio_stream = audio_processor.start_capture().await?;

    while let Some(chunk) = audio_stream.next().await {
        let processed = audio_processor.process_chunk(&chunk).await?;

        if processed.voice_detected {
            println!("🎙️ Voice detected, processing...");

            let audio_input = processed.to_audio_input();
            let result = multimodal.process_audio(&audio_input).await?;

            if let Some(transcript) = result.transcript {
                println!("📝 You said: {}", transcript);

                // Generate AI response
                let ai_response = model.generate(&format!("User said: {}", transcript), 100).await?;
                println!("🤖 AI: {}", ai_response);
            }
        }
    }

    Ok(())
}
```

### Complete Example

```rust
use mullama::prelude::*;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("🤖 Voice AI Assistant Starting...");

    // 1. Load model
    let model = ModelBuilder::new()
        .path("model.gguf")
        .context_size(4096)
        .build().await?;
    println!("✅ Model loaded");

    // 2. Setup audio processing
    let audio_config = AudioStreamConfig::new()
        .sample_rate(16000)
        .channels(1)
        .enable_voice_detection(true)
        .enable_noise_reduction(true)
        .vad_threshold(0.3);

    let mut audio_processor = StreamingAudioProcessor::new(audio_config)?;
    println!("🎤 Audio processor ready");

    // 3. Setup multimodal processing
    let multimodal = MultimodalProcessor::new()
        .enable_audio_processing()
        .build();
    println!("🎭 Multimodal processor ready");

    // 4. Start processing
    println!("🚀 Assistant ready! Speak something...");

    let mut audio_stream = audio_processor.start_capture().await?;

    while let Some(chunk) = audio_stream.next().await {
        let processed = audio_processor.process_chunk(&chunk).await?;

        if processed.voice_detected {
            println!("🎙️ Processing voice...");

            let audio_input = processed.to_audio_input();
            let result = multimodal.process_audio(&audio_input).await?;

            if let Some(transcript) = result.transcript {
                if !transcript.trim().is_empty() {
                    println!("📝 You: {}", transcript);

                    let prompt = format!("User said: \"{}\". Please respond helpfully.", transcript);
                    let response = model.generate(&prompt, 150).await?;
                    println!("🤖 AI: {}", response);
                    println!("---");
                }
            }
        }
    }

    Ok(())
}
```

## 🔄 Common Patterns

### Pattern 1: Async Text Generation

```rust
use mullama::prelude::*;

async fn generate_multiple_responses(model: &AsyncModel, prompts: &[&str]) -> Result<Vec<String>, MullamaError> {
    let mut tasks = Vec::new();

    for prompt in prompts {
        let model_clone = model.clone();
        let prompt = prompt.to_string();

        let task = tokio::spawn(async move {
            model_clone.generate(&prompt, 100).await
        });

        tasks.push(task);
    }

    let mut results = Vec::new();
    for task in tasks {
        let result = task.await.map_err(|e| MullamaError::AsyncError(e.to_string()))??;
        results.push(result);
    }

    Ok(results)
}
```

### Pattern 2: Streaming Responses

```rust
use mullama::{TokenStream, StreamConfig};

async fn stream_response(model: &AsyncModel, prompt: &str) -> Result<(), MullamaError> {
    let config = StreamConfig::new()
        .max_tokens(200)
        .temperature(0.7);

    let mut stream = model.generate_stream(prompt, config).await?;

    print!("AI: ");
    while let Some(token) = stream.next().await {
        let token_data = token?;
        print!("{}", token_data.text);
        tokio::io::stdout().flush().await?;
    }
    println!();

    Ok(())
}
```

### Pattern 3: Error Handling

```rust
use mullama::{MullamaError, ModelBuilder};

async fn robust_model_loading(path: &str) -> Result<AsyncModel, MullamaError> {
    match ModelBuilder::new().path(path).build().await {
        Ok(model) => {
            println!("✅ Model loaded successfully");
            Ok(model)
        }
        Err(MullamaError::ModelLoadError(msg)) => {
            eprintln!("❌ Failed to load model: {}", msg);

            // Try fallback model
            println!("🔄 Trying fallback model...");
            ModelBuilder::new()
                .path("fallback_model.gguf")
                .build().await
        }
        Err(e) => {
            eprintln!("❌ Unexpected error: {}", e);
            Err(e)
        }
    }
}
```

### Pattern 4: Configuration Management

```rust
use mullama::{MullamaConfig, load_config_from_file};
use serde_json;

async fn load_from_config() -> Result<AsyncModel, MullamaError> {
    // Load from JSON file
    let config: MullamaConfig = load_config_from_file("config.json").await?;

    // Or create programmatically
    let config = MullamaConfig::new()
        .model(ModelConfig::new()
            .path("model.gguf")
            .gpu_layers(40)
            .context_size(8192))
        .performance(PerformanceConfig::new()
            .enable_monitoring(true)
            .memory_optimization(3))
        .build()?;

    config.load_model().await
}
```

## 🎯 Next Steps

### Beginner
1. **Try the examples**: Run `cargo run --example simple --features async`
2. **Read the API docs**: Check out [docs.rs/mullama](https://docs.rs/mullama)
3. **Join the community**: Visit our [Discord](https://discord.gg/mullama)

### Intermediate
1. **Build a web service**: Try the web framework integration
2. **Add multimodal features**: Process images and audio
3. **Optimize performance**: Use parallel processing features

### Advanced
1. **Create custom integrations**: Build your own feature modules
2. **Contribute**: Help improve Mullama for everyone
3. **Production deployment**: Use advanced runtime management

### Learning Resources

- 📚 **[API Documentation](https://docs.rs/mullama)** - Complete API reference
- 🎯 **[Examples Directory](../examples/)** - Practical code examples
- 📖 **[Integration Guide](./FEATURES.md)** - Detailed feature documentation
- 💬 **[Community Discord](https://discord.gg/mullama)** - Get help and share projects

### Troubleshooting

#### Common Issues

**Model loading fails**
```rust
// Ensure path is correct and file exists
let path = std::path::Path::new("model.gguf");
if !path.exists() {
    eprintln!("Model file not found: {}", path.display());
}
```

**Audio features not working**
```bash
# Install audio dependencies (Linux)
sudo apt install libasound2-dev

# Enable streaming-audio feature
cargo build --features streaming-audio
```

**Performance issues**
```rust
// Optimize for your hardware
let config = MullamaConfig::new()
    .context(ContextConfig::new()
        .n_threads(num_cpus::get())  // Use all CPU cores
        .n_batch(1024))              // Larger batch size
    .performance(PerformanceConfig::new()
        .memory_optimization(3))     // Max memory optimization
    .build()?;
```

Ready to build amazing AI applications with Mullama! 🚀