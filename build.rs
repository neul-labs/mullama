use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The llama.cpp version to use when downloading
const LLAMA_CPP_VERSION: &str = "b7542";
const LLAMA_CPP_REPO: &str = "https://github.com/ggml-org/llama.cpp";

fn main() {
    // Tell cargo to invalidate the built crate whenever wrapper files change
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_METAL");
    println!("cargo:rerun-if-env-changed=LLAMA_HIPBLAS");
    println!("cargo:rerun-if-env-changed=LLAMA_CLBLAST");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_PATH");

    // Set up platform-specific configurations
    setup_platform_specific();

    // Print dependency errors if needed
    print_dependency_errors();

    // Determine the path to llama.cpp
    let llama_cpp_path = get_llama_cpp_path();

    // Check if the llama.cpp directory has the required files
    if !llama_cpp_path.join("include").join("llama.h").exists() {
        panic!(
            "llama.h not found in llama.cpp include directory at {:?}. \
             The llama.cpp source may be incomplete or corrupted.",
            llama_cpp_path
        );
    }

    // Build the C++ library using CMake
    let dst = build_llama_cpp(&llama_cpp_path);

    // Generate bindings
    generate_bindings(&llama_cpp_path, &dst);
}

/// Get the path to llama.cpp, downloading it if necessary
fn get_llama_cpp_path() -> PathBuf {
    // First, check for user-specified path via environment variable
    if let Ok(path) = env::var("LLAMA_CPP_PATH") {
        let path = PathBuf::from(path);
        if path.exists() && path.join("include").join("llama.h").exists() {
            println!(
                "cargo:warning=Using llama.cpp from LLAMA_CPP_PATH: {:?}",
                path
            );
            return path;
        }
        panic!(
            "LLAMA_CPP_PATH is set to {:?} but it doesn't contain valid llama.cpp sources",
            path
        );
    }

    // Check for submodule in the manifest directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let submodule_path = manifest_dir.join("llama.cpp");

    if submodule_path.exists() && submodule_path.join("include").join("llama.h").exists() {
        return submodule_path;
    }

    // If submodule exists but is empty (common after git clone without --recurse-submodules)
    if submodule_path.exists() {
        println!("cargo:warning=llama.cpp submodule exists but appears empty, attempting to initialize...");
        let status = Command::new("git")
            .args(["submodule", "update", "--init", "--recursive"])
            .current_dir(&manifest_dir)
            .status();

        if status.is_ok() && submodule_path.join("include").join("llama.h").exists() {
            return submodule_path;
        }
    }

    // Download llama.cpp to OUT_DIR for crates.io builds
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let download_path = out_dir.join("llama.cpp");

    if download_path.exists() && download_path.join("include").join("llama.h").exists() {
        return download_path;
    }

    println!(
        "cargo:warning=Downloading llama.cpp {} from GitHub...",
        LLAMA_CPP_VERSION
    );
    download_llama_cpp(&download_path);

    if !download_path.join("include").join("llama.h").exists() {
        panic!(
            "Failed to download llama.cpp. Please either:\n\
             1. Clone with submodules: git clone --recurse-submodules {}\n\
             2. Initialize submodule: git submodule update --init --recursive\n\
             3. Set LLAMA_CPP_PATH environment variable to a llama.cpp checkout",
            env::var("CARGO_PKG_REPOSITORY").unwrap_or_default()
        );
    }

    download_path
}

/// Download llama.cpp from GitHub
fn download_llama_cpp(target_path: &PathBuf) {
    // Clean up any partial download
    if target_path.exists() {
        fs::remove_dir_all(target_path).ok();
    }

    let archive_url = format!(
        "{}/archive/refs/tags/{}.tar.gz",
        LLAMA_CPP_REPO, LLAMA_CPP_VERSION
    );

    let archive_path = target_path.parent().unwrap().join("llama-cpp.tar.gz");

    // Try curl first, then wget
    let download_result = Command::new("curl")
        .args(["-L", "-o"])
        .arg(&archive_path)
        .arg(&archive_url)
        .status()
        .or_else(|_| {
            Command::new("wget")
                .args(["-O"])
                .arg(&archive_path)
                .arg(&archive_url)
                .status()
        });

    if download_result.is_err() || !archive_path.exists() {
        panic!(
            "Failed to download llama.cpp. Please install curl or wget, or manually download from {}",
            archive_url
        );
    }

    // Extract the archive
    fs::create_dir_all(target_path).expect("Failed to create target directory");

    let extract_result = Command::new("tar")
        .args(["xzf"])
        .arg(&archive_path)
        .args(["--strip-components=1", "-C"])
        .arg(target_path)
        .status();

    // Clean up archive
    fs::remove_file(&archive_path).ok();

    if extract_result.is_err() {
        panic!("Failed to extract llama.cpp archive. Please ensure tar is installed.");
    }

    println!(
        "cargo:warning=Successfully downloaded llama.cpp {}",
        LLAMA_CPP_VERSION
    );
}

fn setup_platform_specific() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    match target_os.as_str() {
        "windows" => setup_windows(),
        "macos" => setup_macos(&target_arch),
        "linux" => setup_linux(),
        _ => println!("cargo:warning=Unsupported target OS: {}", target_os),
    }
}

fn setup_windows() {
    println!("cargo:rustc-cfg=target_platform=\"windows\"");

    // Link Windows-specific libraries
    println!("cargo:rustc-link-lib=ole32");
    println!("cargo:rustc-link-lib=oleaut32");
    println!("cargo:rustc-link-lib=winmm");
    println!("cargo:rustc-link-lib=dsound");
    println!("cargo:rustc-link-lib=dxguid");
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=kernel32");

    // Check for Visual Studio
    if let Ok(vs_path) = env::var("VCINSTALLDIR") {
        println!("cargo:rustc-link-search=native={}/lib/x64", vs_path);
    }

    // Windows-specific compiler flags
    if env::var("PROFILE").unwrap() == "release" {
        println!("cargo:rustc-env=CFLAGS=/O2 /GL /DNDEBUG");
        println!("cargo:rustc-env=CXXFLAGS=/O2 /GL /DNDEBUG");
    }
}

fn setup_macos(target_arch: &str) {
    println!("cargo:rustc-cfg=target_platform=\"macos\"");

    // Link macOS frameworks
    println!("cargo:rustc-link-lib=framework=CoreAudio");
    println!("cargo:rustc-link-lib=framework=AudioToolbox");
    println!("cargo:rustc-link-lib=framework=AudioUnit");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=CoreServices");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Apple Silicon specific optimizations
    if target_arch == "aarch64" {
        println!("cargo:rustc-cfg=target_arch_apple_silicon");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");

        // Enable Metal by default on Apple Silicon
        if env::var("LLAMA_METAL").is_err() {
            env::set_var("LLAMA_METAL", "1");
        }
    }

    // macOS-specific compiler flags
    if env::var("PROFILE").unwrap() == "release" {
        if target_arch == "aarch64" {
            println!("cargo:rustc-env=CFLAGS=-O3 -mcpu=apple-m1");
            println!("cargo:rustc-env=CXXFLAGS=-O3 -mcpu=apple-m1");
        } else {
            println!("cargo:rustc-env=CFLAGS=-O3 -march=native");
            println!("cargo:rustc-env=CXXFLAGS=-O3 -march=native");
        }
    }
}

fn setup_linux() {
    println!("cargo:rustc-cfg=target_platform=\"linux\"");

    // Check for audio libraries using pkg-config
    check_audio_libraries();

    // Linux-specific compiler flags
    if env::var("PROFILE").unwrap() == "release" {
        println!("cargo:rustc-env=CFLAGS=-O3 -march=native -mtune=native -DNDEBUG");
        println!("cargo:rustc-env=CXXFLAGS=-O3 -march=native -mtune=native -DNDEBUG");
    }

    // Check for NUMA support
    if pkg_config::probe_library("numa").is_ok() {
        println!("cargo:rustc-cfg=feature=\"numa\"");
        println!("cargo:rustc-link-lib=numa");
    }

    // Standard Linux libraries
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
}

fn check_audio_libraries() {
    // Check for ALSA
    if pkg_config::probe_library("alsa").is_ok() {
        println!("cargo:rustc-cfg=feature=\"alsa\"");
        println!("cargo:rustc-link-lib=asound");
    } else {
        println!("cargo:warning=ALSA development libraries not found. Install libasound2-dev");
    }

    // Check for PulseAudio
    if pkg_config::probe_library("libpulse").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pulseaudio\"");
        println!("cargo:rustc-link-lib=pulse");
    } else {
        println!("cargo:warning=PulseAudio development libraries not found. Install libpulse-dev");
    }

    // Check for JACK
    if pkg_config::probe_library("jack").is_ok() {
        println!("cargo:rustc-cfg=feature=\"jack\"");
        println!("cargo:rustc-link-lib=jack");
    }

    // Check for additional audio libraries
    for lib in &["flac", "vorbis", "vorbisenc", "opus"] {
        if pkg_config::probe_library(lib).is_ok() {
            println!("cargo:rustc-cfg=feature=\"{}\"", lib);
        }
    }
}

fn build_llama_cpp(llama_cpp_path: &PathBuf) -> PathBuf {
    let mut cmake_config = cmake::Config::new(llama_cpp_path);

    // Set build type
    if env::var("PROFILE").unwrap() == "release" {
        cmake_config.define("CMAKE_BUILD_TYPE", "Release");
    } else {
        cmake_config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    // Platform-specific CMake configurations
    if cfg!(target_os = "windows") {
        cmake_config.define("CMAKE_GENERATOR_PLATFORM", "x64");
        cmake_config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
    }

    // GPU acceleration configurations (using new GGML_* naming)
    if env::var("LLAMA_CUDA").is_ok() {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        cmake_config.define("GGML_CUDA", "ON");
        cmake_config.define("CMAKE_CUDA_ARCHITECTURES", "61;70;75;80;86;89");
        configure_cuda_linking();
    } else {
        cmake_config.define("GGML_CUDA", "OFF");
    }

    if env::var("LLAMA_METAL").is_ok() {
        println!("cargo:rustc-cfg=feature=\"metal\"");
        cmake_config.define("GGML_METAL", "ON");
    } else {
        cmake_config.define("GGML_METAL", "OFF");
    }

    if env::var("LLAMA_HIPBLAS").is_ok() {
        println!("cargo:rustc-cfg=feature=\"rocm\"");
        cmake_config.define("GGML_HIP", "ON");
        configure_rocm_linking();
    } else {
        cmake_config.define("GGML_HIP", "OFF");
    }

    if env::var("LLAMA_CLBLAST").is_ok() {
        println!("cargo:rustc-cfg=feature=\"opencl\"");
        cmake_config.define("GGML_OPENCL", "ON");
        configure_opencl_linking();
    } else {
        cmake_config.define("GGML_OPENCL", "OFF");
    }

    // General optimizations (using new GGML_* naming)
    cmake_config.define("GGML_NATIVE", "ON");
    cmake_config.define("GGML_LTO", "ON");
    cmake_config.define("GGML_AVX", "ON");
    cmake_config.define("GGML_AVX2", "ON");
    cmake_config.define("GGML_FMA", "ON");
    cmake_config.define("GGML_F16C", "ON");
    cmake_config.define("GGML_OPENMP", "ON");

    // Build configuration
    cmake_config.define("LLAMA_BUILD_TESTS", "OFF");
    cmake_config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    cmake_config.define("LLAMA_CURL", "OFF");
    cmake_config.define("BUILD_SHARED_LIBS", "OFF");
    cmake_config.define("GGML_STATIC", "ON");

    // Build mtmd (multimodal) library for vision/audio support
    cmake_config.define("LLAMA_BUILD_TOOLS", "ON");

    let dst = cmake_config.build();

    // Link the built library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    // Platform-specific library linking
    // The ggml libraries have circular dependencies, so we use +whole-archive
    // to include all symbols (Rust 1.61+ feature)
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml_static");
        // Link mtmd for multimodal support
        println!("cargo:rustc-link-lib=static=mtmd");
    } else if cfg!(target_os = "macos") {
        // On macOS, use +whole-archive,-bundle for proper symbol inclusion
        println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=ggml-base");
        println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=ggml-cpu");
        println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=ggml");
        println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=llama");
        // Link mtmd for multimodal support
        println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=mtmd");
    } else {
        // On Linux, use linker group to handle circular dependencies between libraries
        // The order matters: llama depends on ggml, ggml depends on ggml-base
        // Using --start-group/--end-group ensures all symbols are resolved
        println!("cargo:rustc-link-arg=-Wl,--start-group");
        println!("cargo:rustc-link-lib=static:+whole-archive=llama");
        println!("cargo:rustc-link-lib=static:+whole-archive=mtmd");
        println!("cargo:rustc-link-lib=static:+whole-archive=ggml");
        println!("cargo:rustc-link-lib=static:+whole-archive=ggml-cpu");
        println!("cargo:rustc-link-lib=static:+whole-archive=ggml-base");
        println!("cargo:rustc-link-arg=-Wl,--end-group");
    }

    // Link standard libraries
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp"); // OpenMP
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=msvcrt");
    }

    dst
}

fn configure_cuda_linking() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_ROOT"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

    let cuda_lib_path = if cfg!(target_os = "windows") {
        format!("{}\\lib\\x64", cuda_path)
    } else {
        format!("{}/lib64", cuda_path)
    };

    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=curand");

    // Check CUDA version
    if let Ok(output) = Command::new("nvcc").args(["--version"]).output() {
        let version_str = String::from_utf8_lossy(&output.stdout);
        if version_str.contains("release 12") {
            println!("cargo:rustc-cfg=cuda_version=\"12\"");
        } else if version_str.contains("release 11") {
            println!("cargo:rustc-cfg=cuda_version=\"11\"");
        }
    }
}

fn configure_rocm_linking() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:rustc-link-search=native={}/lib", rocm_path);
    println!("cargo:rustc-link-lib=hipblas");
    println!("cargo:rustc-link-lib=rocblas");
    println!("cargo:rustc-link-lib=amdhip64");
}

fn configure_opencl_linking() {
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=OpenCL");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
    } else if pkg_config::probe_library("OpenCL").is_ok() {
        println!("cargo:rustc-link-lib=OpenCL");
    } else {
        println!("cargo:warning=OpenCL not found. Install opencl-headers and ocl-icd-opencl-dev");
    }

    // CLBlast for improved OpenCL performance
    if pkg_config::probe_library("clblast").is_ok() {
        println!("cargo:rustc-link-lib=clblast");
    }
}

// Print helpful error messages for missing dependencies
fn print_dependency_errors() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    match target_os.as_str() {
        "windows" => {
            if !command_exists("cl") && !command_exists("gcc") {
                println!("cargo:warning=No C++ compiler found. Install Visual Studio Build Tools or MinGW.");
            }
            if !command_exists("cmake") {
                println!("cargo:warning=CMake not found. Install CMake and add it to PATH.");
            }
        }
        "macos" => {
            if !command_exists("clang") {
                println!(
                    "cargo:warning=Xcode command line tools not found. Run: xcode-select --install"
                );
            }
            if !command_exists("cmake") {
                println!("cargo:warning=CMake not found. Install with: brew install cmake");
            }
        }
        "linux" => {
            if !command_exists("gcc") && !command_exists("clang") {
                println!("cargo:warning=No C++ compiler found. Install build-essential or clang.");
            }
            if !command_exists("cmake") {
                println!("cargo:warning=CMake not found. Install with your package manager.");
            }
            if !command_exists("pkg-config") {
                println!("cargo:warning=pkg-config not found. Install with your package manager.");
            }
        }
        _ => {}
    }
}

// Helper function to check if a command exists
fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn generate_bindings(llama_cpp_path: &Path, _build_path: &Path) {
    let include_path = llama_cpp_path.join("include");
    let ggml_include_path = llama_cpp_path.join("ggml").join("include");
    let mtmd_include_path = llama_cpp_path.join("tools").join("mtmd");

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_path.display()))
        .clang_arg(format!("-I{}", ggml_include_path.display()))
        .clang_arg(format!("-I{}/ggml/src", llama_cpp_path.display()))
        .clang_arg(format!("-I{}", mtmd_include_path.display()))
        // Use C11 standard to ensure stdbool.h and other standard headers are available
        .clang_arg("-std=c11")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Blocklist problematic types
        .blocklist_type("max_align_t")
        .blocklist_type("__off_t")
        .blocklist_type("__off64_t")
        .blocklist_type("_IO_lock_t")
        // Allow specific functions
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_function("clip_.*")
        // Allow specific types
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("mtmd_.*")
        .allowlist_type("clip_.*");

    // On Linux, try to find system include paths for clang
    #[cfg(target_os = "linux")]
    {
        // Try to get GCC's include paths
        if let Ok(output) = Command::new("gcc")
            .args(["-E", "-Wp,-v", "-xc", "/dev/null"])
            .output()
        {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                let line = line.trim();
                if line.starts_with('/')
                    && !line.contains(' ')
                    && std::path::Path::new(line).exists()
                {
                    builder = builder.clang_arg(format!("-isystem{}", line));
                }
            }
        }
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
