//! # Mullama - Unified CLI
//!
//! A multi-model LLM server with IPC and OpenAI-compatible HTTP API.
//!
//! ## Commands
//!
//! ```bash
//! mullama serve       # Start the daemon server
//! mullama chat        # Interactive TUI client
//! mullama run "..."   # One-shot text generation
//! mullama models      # List loaded models
//! mullama load        # Load a model
//! mullama unload      # Unload a model
//! mullama status      # Show daemon status
//! mullama cache       # Manage model cache
//! mullama pull        # Download a model from HuggingFace
//! ```
//!
//! ## HuggingFace Model Support
//!
//! ```bash
//! # Download and serve HuggingFace models
//! mullama serve --model hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf
//!
//! # Auto-detect best quantization
//! mullama serve --model hf:TheBloke/Llama-2-7B-GGUF
//!
//! # With custom alias
//! mullama serve --model llama:hf:TheBloke/Llama-2-7B-GGUF
//!
//! # Pre-download model
//! mullama pull hf:TheBloke/Llama-2-7B-GGUF
//! ```

// Use mimalloc for better allocation performance
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use mullama::daemon::{
    create_openai_router, resolve_model_path, Daemon, DaemonBuilder, DaemonClient, GgufFileInfo,
    HfDownloader, HfModelSpec, HfSearchResult, TuiApp, DEFAULT_HTTP_PORT, DEFAULT_SOCKET,
};

#[derive(Parser)]
#[command(name = "mullama")]
#[command(author, version, about = "Multi-model LLM server and client")]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the daemon server
    #[command(alias = "start")]
    Serve {
        /// Models to load (format: alias:path or just path)
        /// Can be specified multiple times
        #[arg(short, long, value_name = "SPEC")]
        model: Vec<String>,

        /// Path to multimodal projector for vision models (applies to first model)
        #[arg(long)]
        mmproj: Option<PathBuf>,

        /// IPC socket address
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// HTTP port for OpenAI-compatible API (0 to disable)
        #[arg(short = 'p', long, default_value_t = DEFAULT_HTTP_PORT)]
        http_port: u16,

        /// HTTP bind address
        #[arg(long, default_value = "0.0.0.0")]
        http_addr: String,

        /// Default GPU layers to offload
        #[arg(short, long, default_value = "0")]
        gpu_layers: i32,

        /// Default context size
        #[arg(short, long, default_value = "4096")]
        context_size: u32,

        /// Threads per model
        #[arg(short, long)]
        threads: Option<i32>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Interactive TUI chat client
    #[command(alias = "tui")]
    Chat {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Connection timeout in seconds
        #[arg(short, long, default_value = "10")]
        timeout: u64,
    },

    /// One-shot text generation
    Run {
        /// The prompt to send
        prompt: String,

        /// Model to use (default: daemon's default model)
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: u32,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Image file for vision models
        #[arg(short, long)]
        image: Option<PathBuf>,

        /// HTTP port for vision requests (uses HTTP API instead of IPC)
        #[arg(long, default_value = "8080")]
        http_port: u16,

        /// Show generation stats
        #[arg(long)]
        stats: bool,
    },

    /// List loaded models
    Models {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Load a model into the daemon
    Load {
        /// Model specification (format: alias:path or just path)
        spec: String,

        /// Number of GPU layers to offload
        #[arg(short, long, default_value = "0")]
        gpu_layers: i32,

        /// Context size
        #[arg(short, long, default_value = "4096")]
        context_size: u32,

        /// Path to multimodal projector for vision models (mmproj.gguf)
        #[arg(long)]
        mmproj: Option<PathBuf>,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Unload a model from the daemon
    Unload {
        /// Model alias to unload
        alias: String,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Set the default model
    Default {
        /// Model alias to set as default
        alias: String,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Show daemon status
    Status {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Ping the daemon
    Ping {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Shutdown the daemon
    Stop {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Force shutdown even with active requests
        #[arg(short, long)]
        force: bool,
    },

    /// Tokenize text using a model
    Tokenize {
        /// Text to tokenize
        text: String,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Generate embeddings for text
    Embed {
        /// Text(s) to embed
        text: Vec<String>,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Download a model from HuggingFace
    #[command(alias = "download")]
    Pull {
        /// Model specification (e.g., hf:TheBloke/Llama-2-7B-GGUF:model.Q4_K_M.gguf)
        spec: String,

        /// Quiet mode (no progress bar)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Manage the model cache
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Search for models on HuggingFace
    #[command(alias = "find")]
    Search {
        /// Search query (e.g., "llama 7b", "mistral gguf", "phi")
        query: String,

        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,

        /// Show all models (not just GGUF)
        #[arg(long)]
        all: bool,

        /// Show available GGUF files for each result
        #[arg(short, long)]
        files: bool,
    },

    /// Show details about a HuggingFace repository
    Info {
        /// Repository ID (e.g., TheBloke/Llama-2-7B-GGUF)
        repo: String,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// List cached models
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show cache directory path
    Path,

    /// Show cache size
    Size,

    /// Remove a cached model
    Remove {
        /// Repository ID (e.g., TheBloke/Llama-2-7B-GGUF)
        repo_id: String,

        /// Filename to remove (if not specified, removes all files from repo)
        #[arg(short, long)]
        filename: Option<String>,
    },

    /// Clear all cached models
    Clear {
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            model,
            mmproj,
            socket,
            http_port,
            http_addr,
            gpu_layers,
            context_size,
            threads,
            verbose,
        } => {
            run_server(
                model,
                mmproj,
                socket,
                http_port,
                http_addr,
                gpu_layers,
                context_size,
                threads,
                verbose,
            )
            .await?;
        }

        Commands::Chat { socket, timeout } => {
            run_chat(&socket, timeout)?;
        }

        Commands::Run {
            prompt,
            model,
            max_tokens,
            temperature,
            socket,
            image,
            http_port,
            stats,
        } => {
            if image.is_some() {
                run_vision_prompt(
                    http_port,
                    &prompt,
                    model.as_deref(),
                    max_tokens,
                    temperature,
                    image.as_ref().unwrap(),
                    stats,
                )
                .await?;
            } else {
                run_prompt(
                    &socket,
                    &prompt,
                    model.as_deref(),
                    max_tokens,
                    temperature,
                    stats,
                )?;
            }
        }

        Commands::Models { socket, verbose } => {
            list_models(&socket, verbose)?;
        }

        Commands::Load {
            spec,
            gpu_layers,
            context_size,
            mmproj,
            socket,
        } => {
            load_model(&socket, &spec, gpu_layers, context_size, mmproj)?;
        }

        Commands::Unload { alias, socket } => {
            unload_model(&socket, &alias)?;
        }

        Commands::Default { alias, socket } => {
            set_default(&socket, &alias)?;
        }

        Commands::Status { socket, json } => {
            show_status(&socket, json)?;
        }

        Commands::Ping { socket } => {
            ping_daemon(&socket)?;
        }

        Commands::Stop { socket, force: _ } => {
            stop_daemon(&socket)?;
        }

        Commands::Tokenize {
            text,
            model,
            socket,
        } => {
            tokenize_text(&socket, &text, model.as_deref())?;
        }

        Commands::Embed {
            text,
            model,
            socket,
            json,
        } => {
            embed_text(&socket, &text, model.as_deref(), json)?;
        }

        Commands::Pull { spec, quiet } => {
            pull_model(&spec, !quiet).await?;
        }

        Commands::Cache { action } => {
            handle_cache_action(action).await?;
        }

        Commands::Search {
            query,
            limit,
            all,
            files,
        } => {
            search_models(&query, limit, !all, files).await?;
        }

        Commands::Info { repo } => {
            show_repo_info(&repo).await?;
        }
    }

    Ok(())
}

// ==================== Server ====================

async fn run_server(
    models: Vec<String>,
    mmproj: Option<PathBuf>,
    socket: String,
    http_port: u16,
    http_addr: String,
    gpu_layers: i32,
    context_size: u32,
    threads: Option<i32>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize backend
    mullama::backend_init();

    println!("Starting Mullama Daemon...");
    println!("  IPC Socket: {}", socket);
    if http_port > 0 {
        println!("  HTTP API:   http://{}:{}", http_addr, http_port);
    }
    println!("  GPU Layers: {}", gpu_layers);
    println!("  Context:    {}", context_size);
    if let Some(ref mmp) = mmproj {
        println!("  MMProj:     {}", mmp.display());
    }
    println!();

    // Resolve model paths (download HF models if needed)
    let mut resolved_models: Vec<(String, PathBuf)> = Vec::new();
    for spec in &models {
        if HfModelSpec::is_hf_spec(spec) {
            println!("Resolving HuggingFace model: {}", spec);
            match resolve_model_path(spec, true).await {
                Ok((alias, path)) => {
                    println!("  -> {} at {}", alias, path.display());
                    resolved_models.push((alias, path));
                }
                Err(e) => {
                    eprintln!("Failed to resolve {}: {}", spec, e);
                    continue;
                }
            }
        } else {
            // Local path - parse alias:path format
            let (alias, path) = if let Some(pos) = spec.find(':') {
                let alias = &spec[..pos];
                let path_str = &spec[pos + 1..];
                // Check for Windows drive letter
                if alias.len() == 1 && path_str.starts_with('\\') {
                    let p = PathBuf::from(spec);
                    let a = p
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "model".to_string());
                    (a, p)
                } else {
                    (alias.to_string(), PathBuf::from(path_str))
                }
            } else {
                let p = PathBuf::from(spec);
                let a = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                (a, p)
            };
            resolved_models.push((alias, path));
        }
    }
    println!();

    // Build daemon configuration
    let mut builder = DaemonBuilder::new()
        .ipc_socket(&socket)
        .default_gpu_layers(gpu_layers)
        .default_context_size(context_size);

    if http_port > 0 {
        builder = builder.http_port(http_port).http_addr(&http_addr);
    } else {
        builder = builder.disable_http();
    }

    if let Some(t) = threads {
        builder = builder.threads_per_model(t);
    }

    // Add resolved models
    for (alias, path) in &resolved_models {
        builder = builder.model(format!("{}:{}", alias, path.display()));
    }

    let (daemon, mut initial_models) = builder.build();
    let daemon = std::sync::Arc::new(daemon);

    // Apply mmproj to the first model if specified
    if let Some(ref mmp) = mmproj {
        if let Some(first) = initial_models.first_mut() {
            first.mmproj_path = Some(mmp.display().to_string());
        }
    }

    // Load initial models
    for config in initial_models {
        print!("Loading model '{}'... ", config.alias);
        io::stdout().flush()?;

        match daemon.models.load(config.clone()).await {
            Ok(info) => {
                println!("OK");
                if verbose {
                    println!("    Path: {}", info.path);
                    println!("    Parameters: {}M", info.parameters / 1_000_000);
                    println!("    Context: {}", info.context_size);
                }
            }
            Err(e) => {
                println!("FAILED");
                eprintln!("    Error: {}", e);
            }
        }
    }

    if resolved_models.is_empty() {
        println!("No models specified. Use --model to load models.");
        println!("You can also load models via the API or TUI.");
        println!();
        println!("Examples:");
        println!("  mullama serve --model ./model.gguf");
        println!("  mullama serve --model hf:TheBloke/Llama-2-7B-GGUF");
        println!(
            "  mullama serve --model llama:hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf"
        );
    }

    println!();
    println!("Daemon ready. Press Ctrl+C to stop.");
    println!();

    // Start IPC server
    let ipc_daemon = daemon.clone();
    let ipc_socket = socket.clone();
    let ipc_handle = tokio::spawn(async move {
        if let Err(e) = run_ipc_server(ipc_daemon, &ipc_socket).await {
            eprintln!("IPC server error: {}", e);
        }
    });

    // Start HTTP server if enabled
    let http_handle = if http_port > 0 {
        let http_daemon = daemon.clone();
        let addr = format!("{}:{}", http_addr, http_port);
        Some(tokio::spawn(async move {
            let router = create_openai_router(http_daemon);
            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
            if let Err(e) = axum::serve(listener, router).await {
                eprintln!("HTTP server error: {}", e);
            }
        }))
    } else {
        None
    };

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    println!("\nShutting down...");

    // Signal shutdown
    daemon
        .shutdown
        .store(true, std::sync::atomic::Ordering::SeqCst);

    // Give servers time to cleanup
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Cleanup
    mullama::backend_free();

    Ok(())
}

async fn run_ipc_server(
    daemon: std::sync::Arc<Daemon>,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use mullama::daemon::{Request, Response};
    use nng::{Protocol, Socket};

    let socket = Socket::new(Protocol::Rep0)?;
    socket.listen(addr)?;

    loop {
        if daemon.is_shutdown() {
            break;
        }

        // Non-blocking receive with timeout
        match socket.recv() {
            Ok(msg) => {
                let request = match Request::from_bytes(&msg) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Invalid request: {}", e);
                        continue;
                    }
                };

                let response = daemon.handle_request(request).await;

                let resp_bytes = match response.to_bytes() {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Serialization error: {}", e);
                        continue;
                    }
                };

                if let Err(e) = socket.send(nng::Message::from(resp_bytes.as_slice())) {
                    eprintln!("Send error: {:?}", e);
                }
            }
            Err(nng::Error::TimedOut) => {
                continue;
            }
            Err(e) => {
                if !daemon.is_shutdown() {
                    eprintln!("Receive error: {}", e);
                }
                break;
            }
        }
    }

    Ok(())
}

// ==================== Client Commands ====================

fn connect(socket: &str) -> Result<DaemonClient, Box<dyn std::error::Error>> {
    DaemonClient::connect_with_timeout(socket, Duration::from_secs(5))
        .map_err(|e| format!("Failed to connect to daemon: {}\nIs the daemon running?", e).into())
}

fn run_chat(socket: &str, timeout: u64) -> Result<(), Box<dyn std::error::Error>> {
    let client = DaemonClient::connect_with_timeout(socket, Duration::from_secs(timeout))?;

    // Verify connection
    match client.ping() {
        Ok((uptime, version)) => {
            println!(
                "Connected to Mullama daemon v{} (uptime: {}s)",
                version, uptime
            );
        }
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            eprintln!("Make sure the daemon is running: mullama serve --model <path>");
            return Err(e.into());
        }
    }

    // Start TUI
    let mut app = TuiApp::new(client);
    app.run()?;

    Ok(())
}

fn run_prompt(
    socket: &str,
    prompt: &str,
    model: Option<&str>,
    max_tokens: u32,
    temperature: f32,
    stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let result = client.chat(prompt, model, max_tokens, temperature)?;

    println!("{}", result.text);

    if stats {
        eprintln!();
        eprintln!(
            "--- {} tokens in {}ms ({:.1} tok/s) using {} ---",
            result.completion_tokens,
            result.duration_ms,
            result.tokens_per_second(),
            result.model
        );
    }

    Ok(())
}

/// Run vision prompt using HTTP API
async fn run_vision_prompt(
    http_port: u16,
    prompt: &str,
    model: Option<&str>,
    max_tokens: u32,
    temperature: f32,
    image_path: &PathBuf,
    stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use base64::Engine;

    let start = std::time::Instant::now();

    // Read and encode image
    let image_data = std::fs::read(image_path).map_err(|e| {
        format!(
            "Failed to read image file '{}': {}",
            image_path.display(),
            e
        )
    })?;

    // Detect image type from extension
    let mime_type = match image_path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        Some("webp") => "image/webp",
        _ => "image/jpeg", // Default to JPEG
    };

    let base64_image = base64::engine::general_purpose::STANDARD.encode(&image_data);
    let image_url = format!("data:{};base64,{}", mime_type, base64_image);

    // Build OpenAI vision API request
    let request_body = serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false
    });

    // Send request to HTTP API
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", http_port);

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Failed to connect to daemon at {}: {}", url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("Vision request failed ({}): {}", status, error_text).into());
    }

    // Parse response
    let resp_json: serde_json::Value = response.json().await?;

    let text = resp_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("(no response)");

    println!("{}", text);

    if stats {
        let duration = start.elapsed();
        let completion_tokens = resp_json["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or(0);
        let model_used = resp_json["model"].as_str().unwrap_or("unknown");

        eprintln!();
        eprintln!(
            "--- {} tokens in {}ms ({:.1} tok/s) using {} ---",
            completion_tokens,
            duration.as_millis(),
            completion_tokens as f64 / duration.as_secs_f64(),
            model_used
        );
    }

    Ok(())
}

fn list_models(socket: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let models = client.list_models()?;

    if models.is_empty() {
        println!("No models loaded.");
        println!("Use 'mullama load <path>' to load a model.");
        return Ok(());
    }

    println!("Loaded models:\n");
    for model in models {
        let default_marker = if model.is_default { " (default)" } else { "" };
        println!("  {}{}", model.alias, default_marker);

        if verbose {
            println!("    Path:       {}", model.info.path);
            println!("    Parameters: {}M", model.info.parameters / 1_000_000);
            println!("    Context:    {}", model.info.context_size);
            println!("    GPU layers: {}", model.info.gpu_layers);
            if model.active_requests > 0 {
                println!("    Active:     {} requests", model.active_requests);
            }
            println!();
        }
    }

    Ok(())
}

fn load_model(
    socket: &str,
    spec: &str,
    gpu_layers: i32,
    context_size: u32,
    mmproj: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if mmproj.is_some() {
        eprintln!("Warning: --mmproj is not yet supported via IPC protocol.");
        eprintln!("         Vision models can be loaded directly via the server:");
        eprintln!("         mullama serve --model model.gguf (with mmproj support coming soon)");
        eprintln!();
    }

    let client = connect(socket)?;

    // Parse spec
    let (alias, path) = if let Some(pos) = spec.find(':') {
        (spec[..pos].to_string(), spec[pos + 1..].to_string())
    } else {
        let p = std::path::Path::new(spec);
        let alias = p
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        (alias, spec.to_string())
    };

    print!("Loading model '{}'... ", alias);
    io::stdout().flush()?;

    match client.load_model_with_options(&alias, &path, gpu_layers, context_size) {
        Ok((alias, info)) => {
            println!("OK");
            println!("  Parameters: {}M", info.parameters / 1_000_000);
            println!("  Context:    {}", info.context_size);
        }
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

fn unload_model(socket: &str, alias: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    print!("Unloading model '{}'... ", alias);
    io::stdout().flush()?;

    match client.unload_model(alias) {
        Ok(()) => println!("OK"),
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

fn set_default(socket: &str, alias: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    match client.set_default_model(alias) {
        Ok(()) => println!("Default model set to '{}'", alias),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}

fn show_status(socket: &str, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let status = client.status()?;

    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("Mullama Daemon Status");
        println!("=====================");
        println!("Version:         {}", status.version);
        println!("Uptime:          {}s", status.uptime_secs);
        println!("Models loaded:   {}", status.models_loaded);
        if let Some(ref default) = status.default_model {
            println!("Default model:   {}", default);
        }
        if let Some(ref http) = status.http_endpoint {
            println!("HTTP endpoint:   {}", http);
        }
        println!("IPC endpoint:    {}", status.ipc_endpoint);
        println!();
        println!("Statistics:");
        println!("  Total requests:   {}", status.stats.requests_total);
        println!("  Tokens generated: {}", status.stats.tokens_generated);
        println!("  Active requests:  {}", status.stats.active_requests);
        println!("  GPU available:    {}", status.stats.gpu_available);
    }

    Ok(())
}

fn ping_daemon(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let client = connect(socket)?;
    let (uptime, version) = client.ping()?;
    let latency = start.elapsed();

    println!("Pong from mullama v{}", version);
    println!("  Daemon uptime: {}s", uptime);
    println!("  Round-trip:    {:?}", latency);

    Ok(())
}

fn stop_daemon(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    print!("Shutting down daemon... ");
    io::stdout().flush()?;

    match client.shutdown() {
        Ok(()) => println!("OK"),
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

fn tokenize_text(
    socket: &str,
    text: &str,
    model: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let tokens = client.tokenize(text, model)?;

    println!("Tokens ({}): {:?}", tokens.len(), tokens);

    Ok(())
}

fn embed_text(
    socket: &str,
    texts: &[String],
    model: Option<&str>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    if texts.is_empty() {
        return Err("No text provided".into());
    }

    if texts.len() == 1 {
        // Single embedding
        let result = client.embed(&texts[0], model)?;

        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "model": result.model,
                    "dimension": result.dimension(),
                    "prompt_tokens": result.prompt_tokens,
                    "embedding": result.embedding,
                }))?
            );
        } else {
            println!("Model: {}", result.model);
            println!("Dimension: {}", result.dimension());
            println!("Tokens: {}", result.prompt_tokens);
            println!(
                "Embedding (first 10): {:?}...",
                &result.embedding[..result.embedding.len().min(10)]
            );
        }
    } else {
        // Batch embedding
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let result = client.embed_batch(&text_refs, model)?;

        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "model": result.model,
                    "count": result.count(),
                    "dimension": result.dimension(),
                    "prompt_tokens": result.prompt_tokens,
                    "embeddings": result.embeddings,
                }))?
            );
        } else {
            println!("Model: {}", result.model);
            println!("Count: {}", result.count());
            println!("Dimension: {}", result.dimension());
            println!("Tokens: {}", result.prompt_tokens);
            for (i, emb) in result.embeddings.iter().enumerate() {
                println!("  [{}]: {:?}...", i, &emb[..emb.len().min(5)]);
            }
        }
    }

    Ok(())
}

// ==================== HuggingFace / Cache Commands ====================

async fn pull_model(spec: &str, show_progress: bool) -> Result<(), Box<dyn std::error::Error>> {
    let hf_spec = HfModelSpec::parse(spec).ok_or_else(|| {
        format!(
            "Invalid HuggingFace spec: {}\n\
             Expected format: hf:owner/repo:filename.gguf or hf:owner/repo",
            spec
        )
    })?;

    let downloader = HfDownloader::new()?;

    println!("Downloading from HuggingFace...");
    println!("  Repository: {}", hf_spec.repo_id);

    if let Some(ref filename) = hf_spec.filename {
        println!("  File: {}", filename);
    } else {
        println!("  File: (auto-detecting best GGUF)");
    }
    println!();

    let path = downloader.download_spec(&hf_spec, show_progress).await?;

    println!();
    println!("Model downloaded successfully!");
    println!("  Path: {}", path.display());
    println!();
    println!("To use this model:");
    println!(
        "  mullama serve --model {}:{}",
        hf_spec.get_alias(),
        path.display()
    );

    Ok(())
}

async fn handle_cache_action(action: CacheAction) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;

    match action {
        CacheAction::List { verbose } => {
            let models = downloader.list_cached();

            if models.is_empty() {
                println!("No cached models.");
                println!();
                println!("Download models with:");
                println!("  mullama pull hf:TheBloke/Llama-2-7B-GGUF");
                return Ok(());
            }

            println!("Cached models:\n");
            for model in models {
                println!("  {} / {}", model.repo_id, model.filename);
                if verbose {
                    println!("    Path: {}", model.local_path.display());
                    println!(
                        "    Size: {:.2} GB",
                        model.size_bytes as f64 / 1_073_741_824.0
                    );
                    println!("    Downloaded: {}", model.downloaded_at);
                    println!();
                }
            }

            if !verbose {
                println!();
                println!("Use --verbose for more details.");
            }
        }

        CacheAction::Path => {
            println!("{}", downloader.cache_dir().display());
            println!();
            println!("Override with MULLAMA_CACHE_DIR environment variable.");
        }

        CacheAction::Size => {
            let size = downloader.cache_size();
            let models = downloader.list_cached();

            println!("Cache size: {:.2} GB", size as f64 / 1_073_741_824.0);
            println!("Models cached: {}", models.len());
            println!("Cache directory: {}", downloader.cache_dir().display());
        }

        CacheAction::Remove { repo_id, filename } => {
            if let Some(filename) = filename {
                print!("Removing {} / {}... ", repo_id, filename);
                io::stdout().flush()?;
                downloader.remove_cached(&repo_id, &filename)?;
                println!("OK");
            } else {
                // Remove all files from repo
                let models = downloader.list_cached();
                let to_remove: Vec<_> = models.iter().filter(|m| m.repo_id == repo_id).collect();

                if to_remove.is_empty() {
                    println!("No cached files found for {}", repo_id);
                    return Ok(());
                }

                for model in to_remove {
                    print!("Removing {}... ", model.filename);
                    io::stdout().flush()?;
                    downloader.remove_cached(&model.repo_id, &model.filename)?;
                    println!("OK");
                }
            }
        }

        CacheAction::Clear { force } => {
            if !force {
                let models = downloader.list_cached();
                let size = downloader.cache_size();

                println!(
                    "This will remove {} models ({:.2} GB).",
                    models.len(),
                    size as f64 / 1_073_741_824.0
                );
                print!("Are you sure? [y/N] ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;

                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Cancelled.");
                    return Ok(());
                }
            }

            print!("Clearing cache... ");
            io::stdout().flush()?;
            downloader.clear_cache()?;
            println!("OK");
        }
    }

    Ok(())
}

// ==================== Search Commands ====================

async fn search_models(
    query: &str,
    limit: usize,
    gguf_only: bool,
    show_files: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;

    println!("Searching HuggingFace for '{}'...\n", query);

    let results = downloader.search(query, gguf_only, limit).await?;

    if results.is_empty() {
        println!("No models found.");
        if gguf_only {
            println!("Try --all to search all models (not just GGUF).");
        }
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        // Header line
        print!("{}. ", i + 1);
        print!("{}", result.id);
        if result.is_gguf() {
            print!(" [GGUF]");
        }
        println!();

        // Metadata line
        print!("   ");
        print!("Downloads: {}", result.downloads_formatted());
        if let Some(likes) = result.likes {
            print!(" | Likes: {}", likes);
        }
        if let Some(ref pipeline) = result.pipeline_tag {
            print!(" | {}", pipeline);
        }
        println!();

        // Usage hint
        println!("   Use: mullama serve --model hf:{}", result.id);

        // Show files if requested
        if show_files && result.is_gguf() {
            match downloader.list_gguf_files(&result.id).await {
                Ok(files) => {
                    println!("   Files:");
                    for file in files.iter().take(5) {
                        print!("     - {}", file.filename);
                        print!(" ({})", file.size_formatted());
                        if let Some(ref q) = file.quantization {
                            print!(" [{}]", q);
                        }
                        println!();
                    }
                    if files.len() > 5 {
                        println!("     ... and {} more files", files.len() - 5);
                    }
                }
                Err(_) => {
                    println!("   (Could not fetch file list)");
                }
            }
        }

        println!();
    }

    println!("Found {} models.", results.len());
    if !show_files && gguf_only {
        println!("Use --files to show available GGUF files.");
    }

    Ok(())
}

async fn show_repo_info(repo_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;

    println!("Fetching info for {}...\n", repo_id);

    // Get GGUF files
    let files = downloader.list_gguf_files(repo_id).await?;

    println!("Repository: {}", repo_id);
    println!("URL: https://huggingface.co/{}", repo_id);
    println!();
    println!("Available GGUF files ({}):", files.len());
    println!();

    // Group by quantization type
    let mut by_quant: std::collections::HashMap<String, Vec<&GgufFileInfo>> =
        std::collections::HashMap::new();
    for file in &files {
        let key = file
            .quantization
            .clone()
            .unwrap_or_else(|| "Other".to_string());
        by_quant.entry(key).or_default().push(file);
    }

    // Sort quantization types by preference
    let quant_order = [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0", "Q4_1", "Q8_0", "Q6_K", "Q3_K_M", "Q3_K_S",
        "Q3_K_L", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "IQ2_M",
        "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S", "F16", "F32", "Other",
    ];

    for quant in quant_order {
        if let Some(files) = by_quant.get(quant) {
            for file in files {
                println!(
                    "  {:12} {:>10}  {}",
                    file.quantization.as_deref().unwrap_or("-"),
                    file.size_formatted(),
                    file.filename
                );
            }
        }
    }

    // Show any remaining that weren't in our order
    for (quant, files) in &by_quant {
        if !quant_order.contains(&quant.as_str()) {
            for file in files {
                println!(
                    "  {:12} {:>10}  {}",
                    file.quantization.as_deref().unwrap_or("-"),
                    file.size_formatted(),
                    file.filename
                );
            }
        }
    }

    println!();
    println!("Quick start:");
    println!("  mullama pull hf:{}", repo_id);
    println!("  mullama serve --model hf:{}", repo_id);

    // Check if any are cached
    let cached = downloader.list_cached();
    let cached_from_repo: Vec<_> = cached.iter().filter(|c| c.repo_id == repo_id).collect();
    if !cached_from_repo.is_empty() {
        println!();
        println!("Cached locally:");
        for c in cached_from_repo {
            println!(
                "  {} ({:.2} GB)",
                c.filename,
                c.size_bytes as f64 / 1_073_741_824.0
            );
        }
    }

    Ok(())
}
