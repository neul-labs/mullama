//! HuggingFace Model Downloader
//!
//! Downloads and caches GGUF models from HuggingFace Hub.
//!
//! ## Model Specification Formats
//!
//! ```text
//! # Full HF spec with filename
//! hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf
//!
//! # Auto-detect best GGUF file
//! hf:TheBloke/Llama-2-7B-GGUF
//!
//! # With alias
//! llama:hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf
//!
//! # Short format (searches for repo)
//! hf:llama-2-7b-q4
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::MullamaError;

/// HuggingFace API base URL
const HF_API_URL: &str = "https://huggingface.co/api";
const HF_CDN_URL: &str = "https://huggingface.co";

/// Cache directory name
const CACHE_DIR: &str = "mullama";
const MODELS_SUBDIR: &str = "models";

/// Model file info from HF API
#[derive(Debug, Clone, Deserialize)]
pub struct HfFileInfo {
    #[serde(rename = "rfilename")]
    pub filename: String,
    pub size: Option<u64>,
}

/// Repository info from HF API
#[derive(Debug, Clone, Deserialize)]
pub struct HfRepoInfo {
    pub id: String,
    pub siblings: Option<Vec<HfFileInfo>>,
}

/// Cached model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel {
    pub repo_id: String,
    pub filename: String,
    pub local_path: PathBuf,
    pub size_bytes: u64,
    pub downloaded_at: String,
    pub etag: Option<String>,
}

/// Model cache index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheIndex {
    pub models: Vec<CachedModel>,
}

/// HuggingFace model specification
#[derive(Debug, Clone)]
pub struct HfModelSpec {
    pub alias: Option<String>,
    pub repo_id: String,
    pub filename: Option<String>,
    pub revision: String,
}

impl HfModelSpec {
    /// Parse a model specification string
    ///
    /// Formats:
    /// - `hf:owner/repo:filename.gguf`
    /// - `hf:owner/repo` (auto-detect GGUF)
    /// - `alias:hf:owner/repo:filename.gguf`
    pub fn parse(spec: &str) -> Option<Self> {
        let (alias, rest) = if spec.contains(":hf:") {
            let parts: Vec<&str> = spec.splitn(2, ":hf:").collect();
            (Some(parts[0].to_string()), parts.get(1).copied()?)
        } else if spec.starts_with("hf:") {
            (None, spec.strip_prefix("hf:")?)
        } else {
            return None;
        };

        // Parse repo:filename or just repo
        let (repo_id, filename) = if let Some(pos) = rest.rfind(':') {
            // Check if this is owner/repo:filename or owner:repo (git ref)
            let before = &rest[..pos];
            let after = &rest[pos + 1..];

            if after.contains('.') && (after.ends_with(".gguf") || after.ends_with(".bin")) {
                (before.to_string(), Some(after.to_string()))
            } else {
                (rest.to_string(), None)
            }
        } else {
            (rest.to_string(), None)
        };

        Some(Self {
            alias,
            repo_id,
            filename,
            revision: "main".to_string(),
        })
    }

    /// Check if a string looks like an HF spec
    pub fn is_hf_spec(spec: &str) -> bool {
        spec.starts_with("hf:") || spec.contains(":hf:")
    }

    /// Get the alias to use (explicit or derived from repo)
    pub fn get_alias(&self) -> String {
        self.alias.clone().unwrap_or_else(|| {
            // Use repo name as alias
            self.repo_id
                .split('/')
                .last()
                .unwrap_or("model")
                .to_lowercase()
                .replace("-gguf", "")
                .replace("_gguf", "")
        })
    }
}

/// HuggingFace model downloader with caching
pub struct HfDownloader {
    client: Client,
    cache_dir: PathBuf,
    hf_token: Option<String>,
}

impl HfDownloader {
    /// Create a new downloader
    pub fn new() -> Result<Self, MullamaError> {
        let cache_dir = Self::default_cache_dir()?;
        fs::create_dir_all(&cache_dir).map_err(|e| {
            MullamaError::OperationFailed(format!("Failed to create cache dir: {}", e))
        })?;

        let hf_token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();

        Ok(Self {
            client: Client::new(),
            cache_dir,
            hf_token,
        })
    }

    /// Create with custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, MullamaError> {
        fs::create_dir_all(&cache_dir).map_err(|e| {
            MullamaError::OperationFailed(format!("Failed to create cache dir: {}", e))
        })?;

        let hf_token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();

        Ok(Self {
            client: Client::new(),
            cache_dir,
            hf_token,
        })
    }

    /// Get default cache directory
    pub fn default_cache_dir() -> Result<PathBuf, MullamaError> {
        let base = dirs::cache_dir()
            .or_else(dirs::data_local_dir)
            .ok_or_else(|| MullamaError::OperationFailed("No cache directory available".into()))?;

        Ok(base.join(CACHE_DIR).join(MODELS_SUBDIR))
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Load the cache index
    fn load_index(&self) -> CacheIndex {
        let index_path = self.cache_dir.join("index.json");
        if index_path.exists() {
            if let Ok(content) = fs::read_to_string(&index_path) {
                if let Ok(index) = serde_json::from_str(&content) {
                    return index;
                }
            }
        }
        CacheIndex::default()
    }

    /// Save the cache index
    fn save_index(&self, index: &CacheIndex) -> Result<(), MullamaError> {
        let index_path = self.cache_dir.join("index.json");
        let content = serde_json::to_string_pretty(index)
            .map_err(|e| MullamaError::OperationFailed(format!("Failed to serialize index: {}", e)))?;
        fs::write(&index_path, content)
            .map_err(|e| MullamaError::OperationFailed(format!("Failed to write index: {}", e)))?;
        Ok(())
    }

    /// Check if a model is cached
    pub fn get_cached(&self, repo_id: &str, filename: &str) -> Option<CachedModel> {
        let index = self.load_index();
        index
            .models
            .into_iter()
            .find(|m| m.repo_id == repo_id && m.filename == filename && m.local_path.exists())
    }

    /// List all cached models
    pub fn list_cached(&self) -> Vec<CachedModel> {
        let index = self.load_index();
        index
            .models
            .into_iter()
            .filter(|m| m.local_path.exists())
            .collect()
    }

    /// Get repository info from HF API
    pub async fn get_repo_info(&self, repo_id: &str) -> Result<HfRepoInfo, MullamaError> {
        let url = format!("{}/models/{}", HF_API_URL, repo_id);

        let mut req = self.client.get(&url);
        if let Some(ref token) = self.hf_token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req.send().await.map_err(|e| {
            MullamaError::OperationFailed(format!("Failed to fetch repo info: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(MullamaError::OperationFailed(format!(
                "HF API error: {} - {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            )));
        }

        resp.json().await.map_err(|e| {
            MullamaError::OperationFailed(format!("Failed to parse repo info: {}", e))
        })
    }

    /// Find the best GGUF file in a repository
    pub async fn find_best_gguf(&self, repo_id: &str) -> Result<String, MullamaError> {
        let info = self.get_repo_info(repo_id).await?;

        let siblings = info.siblings.ok_or_else(|| {
            MullamaError::OperationFailed("No files found in repository".into())
        })?;

        // Find GGUF files and sort by preference
        let mut gguf_files: Vec<_> = siblings
            .into_iter()
            .filter(|f| f.filename.ends_with(".gguf"))
            .collect();

        if gguf_files.is_empty() {
            return Err(MullamaError::OperationFailed(
                "No GGUF files found in repository".into(),
            ));
        }

        // Sort by quantization preference (Q4_K_M is a good default)
        let preference_order = [
            "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0", "Q4_1",
            "Q8_0", "Q6_K", "Q3_K_M", "Q3_K_S", "Q2_K",
        ];

        gguf_files.sort_by(|a, b| {
            let a_score = preference_order
                .iter()
                .position(|q| a.filename.contains(q))
                .unwrap_or(100);
            let b_score = preference_order
                .iter()
                .position(|q| b.filename.contains(q))
                .unwrap_or(100);
            a_score.cmp(&b_score)
        });

        Ok(gguf_files[0].filename.clone())
    }

    /// Download a model file with progress
    pub async fn download(
        &self,
        repo_id: &str,
        filename: &str,
        show_progress: bool,
    ) -> Result<PathBuf, MullamaError> {
        // Check cache first
        if let Some(cached) = self.get_cached(repo_id, filename) {
            if show_progress {
                println!("Using cached model: {}", cached.local_path.display());
            }
            return Ok(cached.local_path);
        }

        // Create repo directory
        let repo_dir = self.cache_dir.join(repo_id.replace('/', "--"));
        fs::create_dir_all(&repo_dir).map_err(|e| {
            MullamaError::OperationFailed(format!("Failed to create repo dir: {}", e))
        })?;

        let local_path = repo_dir.join(filename);

        // Build download URL
        let url = format!(
            "{}/{}/resolve/main/{}",
            HF_CDN_URL, repo_id, filename
        );

        if show_progress {
            println!("Downloading {} from {}", filename, repo_id);
        }

        // Start download
        let mut req = self.client.get(&url);
        if let Some(ref token) = self.hf_token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req.send().await.map_err(|e| {
            MullamaError::OperationFailed(format!("Download failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(MullamaError::OperationFailed(format!(
                "Download failed: {} - {}",
                resp.status(),
                if resp.status().as_u16() == 401 {
                    "Unauthorized. Set HF_TOKEN for gated models."
                } else {
                    "Check repo and filename"
                }
            )));
        }

        let total_size = resp.content_length().unwrap_or(0);
        let etag = resp
            .headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(String::from);

        // Setup progress bar
        let progress = if show_progress && total_size > 0 {
            let pb = ProgressBar::new(total_size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        // Download to temp file first
        let temp_path = local_path.with_extension("part");
        let mut file = File::create(&temp_path).map_err(|e| {
            MullamaError::OperationFailed(format!("Failed to create file: {}", e))
        })?;

        let mut downloaded: u64 = 0;
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                MullamaError::OperationFailed(format!("Download error: {}", e))
            })?;

            file.write_all(&chunk).map_err(|e| {
                MullamaError::OperationFailed(format!("Write error: {}", e))
            })?;

            downloaded += chunk.len() as u64;
            if let Some(ref pb) = progress {
                pb.set_position(downloaded);
            }
        }

        if let Some(pb) = progress {
            pb.finish_with_message("Download complete");
        }

        // Rename temp file to final
        fs::rename(&temp_path, &local_path).map_err(|e| {
            MullamaError::OperationFailed(format!("Failed to finalize download: {}", e))
        })?;

        // Update cache index
        let mut index = self.load_index();
        index.models.retain(|m| !(m.repo_id == repo_id && m.filename == filename));
        index.models.push(CachedModel {
            repo_id: repo_id.to_string(),
            filename: filename.to_string(),
            local_path: local_path.clone(),
            size_bytes: downloaded,
            downloaded_at: chrono::Utc::now().to_rfc3339(),
            etag,
        });
        self.save_index(&index)?;

        if show_progress {
            println!("Saved to: {}", local_path.display());
        }

        Ok(local_path)
    }

    /// Download from spec, auto-detecting filename if needed
    pub async fn download_spec(
        &self,
        spec: &HfModelSpec,
        show_progress: bool,
    ) -> Result<PathBuf, MullamaError> {
        let filename = match &spec.filename {
            Some(f) => f.clone(),
            None => {
                if show_progress {
                    println!("Finding best GGUF file in {}...", spec.repo_id);
                }
                self.find_best_gguf(&spec.repo_id).await?
            }
        };

        self.download(&spec.repo_id, &filename, show_progress).await
    }

    /// Remove a cached model
    pub fn remove_cached(&self, repo_id: &str, filename: &str) -> Result<(), MullamaError> {
        let mut index = self.load_index();

        if let Some(pos) = index.models.iter().position(|m| m.repo_id == repo_id && m.filename == filename) {
            let model = index.models.remove(pos);
            if model.local_path.exists() {
                fs::remove_file(&model.local_path).map_err(|e| {
                    MullamaError::OperationFailed(format!("Failed to remove file: {}", e))
                })?;
            }
            self.save_index(&index)?;
        }

        Ok(())
    }

    /// Clear all cached models
    pub fn clear_cache(&self) -> Result<(), MullamaError> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir).map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to clear cache: {}", e))
            })?;
            fs::create_dir_all(&self.cache_dir).map_err(|e| {
                MullamaError::OperationFailed(format!("Failed to recreate cache dir: {}", e))
            })?;
        }
        Ok(())
    }

    /// Get cache size in bytes
    pub fn cache_size(&self) -> u64 {
        self.list_cached().iter().map(|m| m.size_bytes).sum()
    }
}

impl Default for HfDownloader {
    fn default() -> Self {
        Self::new().expect("Failed to create HfDownloader")
    }
}

/// Resolve a model spec to a local path, downloading if needed
pub async fn resolve_model_path(
    spec: &str,
    show_progress: bool,
) -> Result<(String, PathBuf), MullamaError> {
    if let Some(hf_spec) = HfModelSpec::parse(spec) {
        let downloader = HfDownloader::new()?;
        let path = downloader.download_spec(&hf_spec, show_progress).await?;
        let alias = hf_spec.get_alias();
        Ok((alias, path))
    } else {
        // Local file path
        let path = PathBuf::from(spec);

        // Check for alias:path format
        let (alias, path) = if let Some(pos) = spec.find(':') {
            let alias = &spec[..pos];
            let path_str = &spec[pos + 1..];
            // Make sure this isn't a Windows drive letter (C:\...)
            if alias.len() == 1 && path_str.starts_with('\\') {
                // Windows path
                let p = PathBuf::from(spec);
                let alias = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                (alias, p)
            } else {
                (alias.to_string(), PathBuf::from(path_str))
            }
        } else {
            let alias = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string());
            (alias, path)
        };

        if !path.exists() {
            return Err(MullamaError::OperationFailed(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        Ok((alias, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hf_spec() {
        // Basic HF spec
        let spec = HfModelSpec::parse("hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf");
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert_eq!(spec.repo_id, "TheBloke/Llama-2-7B-GGUF");
        assert_eq!(spec.filename, Some("llama-2-7b.Q4_K_M.gguf".to_string()));
        assert!(spec.alias.is_none());

        // With alias
        let spec = HfModelSpec::parse("llama:hf:TheBloke/Llama-2-7B-GGUF:model.gguf");
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert_eq!(spec.alias, Some("llama".to_string()));
        assert_eq!(spec.repo_id, "TheBloke/Llama-2-7B-GGUF");

        // Auto-detect filename
        let spec = HfModelSpec::parse("hf:TheBloke/Llama-2-7B-GGUF");
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert!(spec.filename.is_none());

        // Not an HF spec
        assert!(HfModelSpec::parse("./model.gguf").is_none());
        assert!(HfModelSpec::parse("model:./path.gguf").is_none());
    }

    #[test]
    fn test_is_hf_spec() {
        assert!(HfModelSpec::is_hf_spec("hf:owner/repo"));
        assert!(HfModelSpec::is_hf_spec("alias:hf:owner/repo"));
        assert!(!HfModelSpec::is_hf_spec("./local/path.gguf"));
        assert!(!HfModelSpec::is_hf_spec("alias:./local/path.gguf"));
    }

    #[test]
    fn test_get_alias() {
        let spec = HfModelSpec {
            alias: Some("custom".to_string()),
            repo_id: "TheBloke/Llama-2-7B-GGUF".to_string(),
            filename: None,
            revision: "main".to_string(),
        };
        assert_eq!(spec.get_alias(), "custom");

        let spec = HfModelSpec {
            alias: None,
            repo_id: "TheBloke/Llama-2-7B-GGUF".to_string(),
            filename: None,
            revision: "main".to_string(),
        };
        assert_eq!(spec.get_alias(), "llama-2-7b");
    }
}
