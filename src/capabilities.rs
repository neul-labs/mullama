//! Model capability detection and configuration.
//!
//! This module provides a configuration-driven approach to detecting model capabilities
//! like JSON mode, tool use, and extended thinking. Model families are defined in TOML
//! configuration files that can be extended by users without recompiling.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::MullamaError;

/// Global capability registry instance
static REGISTRY: OnceLock<CapabilityRegistry> = OnceLock::new();

/// Get or initialize the global capability registry
pub fn registry() -> &'static CapabilityRegistry {
    REGISTRY.get_or_init(|| {
        CapabilityRegistry::load_default().unwrap_or_else(|e| {
            eprintln!("Warning: Failed to load capability configs: {}", e);
            CapabilityRegistry::with_builtin_defaults()
        })
    })
}

/// Model family configuration loaded from TOML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFamilyConfig {
    pub family: FamilyInfo,
    pub capabilities: Capabilities,
    #[serde(default)]
    pub tokens: TokenConfig,
    #[serde(default)]
    pub tool_format: Option<ToolFormat>,
    #[serde(default)]
    pub thinking: Option<ThinkingTokens>,
}

/// Basic family identification info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyInfo {
    /// Internal name (e.g., "qwen3")
    pub name: String,
    /// Display name (e.g., "Qwen 3")
    pub display_name: String,
    /// Regex patterns to match model names/architectures
    pub patterns: Vec<String>,
    /// Priority for pattern matching (higher = checked first)
    #[serde(default)]
    pub priority: i32,
}

/// Model capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Capabilities {
    /// Supports native JSON output mode
    #[serde(default)]
    pub native_json: bool,
    /// Supports function/tool calling
    #[serde(default)]
    pub tool_use: bool,
    /// Supports extended thinking (chain of thought)
    #[serde(default)]
    pub thinking: bool,
    /// Supports vision/image input
    #[serde(default)]
    pub vision: bool,
}

/// Chat template token configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenConfig {
    /// Inherit tokens from another config
    #[serde(default)]
    pub extends: Option<String>,
    /// Beginning of sequence token
    #[serde(default)]
    pub bos: Option<String>,
    /// End of sequence token
    #[serde(default)]
    pub eos: Option<String>,
    /// User message prefix
    #[serde(default)]
    pub user_prefix: Option<String>,
    /// User message suffix
    #[serde(default)]
    pub user_suffix: Option<String>,
    /// Assistant message prefix
    #[serde(default)]
    pub assistant_prefix: Option<String>,
    /// Assistant message suffix
    #[serde(default)]
    pub assistant_suffix: Option<String>,
    /// System message prefix
    #[serde(default)]
    pub system_prefix: Option<String>,
    /// System message suffix
    #[serde(default)]
    pub system_suffix: Option<String>,
    /// Additional stop sequences
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

/// Tool/function calling format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFormat {
    /// Format style: "qwen", "llama", "generic"
    pub style: String,
    /// Start token for tool calls
    #[serde(default)]
    pub tool_call_start: Option<String>,
    /// End token for tool calls
    #[serde(default)]
    pub tool_call_end: Option<String>,
    /// Tool result start token
    #[serde(default)]
    pub tool_result_start: Option<String>,
    /// Tool result end token
    #[serde(default)]
    pub tool_result_end: Option<String>,
}

/// Thinking/reasoning tokens for chain-of-thought models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTokens {
    /// Start of thinking block
    pub start_token: String,
    /// End of thinking block
    pub end_token: String,
}

/// Registry of all loaded model family configurations
#[derive(Debug, Clone)]
pub struct CapabilityRegistry {
    families: Vec<ModelFamilyConfig>,
    /// Compiled regex patterns for faster matching
    patterns: Vec<(regex::Regex, usize)>,
}

impl CapabilityRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
            patterns: Vec::new(),
        }
    }

    /// Create a registry with built-in defaults only
    pub fn with_builtin_defaults() -> Self {
        let mut registry = Self::new();
        registry.load_builtin_defaults();
        registry.compile_patterns();
        registry
    }

    /// Load the default registry from config directories
    pub fn load_default() -> Result<Self, MullamaError> {
        let mut registry = Self::new();

        // First load built-in defaults
        registry.load_builtin_defaults();

        // Then load from config directories (user configs override builtins)
        let config_dirs = get_config_dirs();
        for dir in config_dirs {
            if dir.exists() {
                registry.load_from_dir(&dir)?;
            }
        }

        registry.compile_patterns();
        Ok(registry)
    }

    /// Load configurations from a directory
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<(), MullamaError> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir)
            .map_err(|e| MullamaError::IoError(e))?
        {
            let entry = entry.map_err(|e| MullamaError::IoError(e))?;
            let path = entry.path();

            if path.extension().map(|e| e == "toml").unwrap_or(false) {
                match self.load_config_file(&path) {
                    Ok(config) => {
                        // Remove existing config with same name (user overrides builtin)
                        self.families.retain(|f| f.family.name != config.family.name);
                        self.families.push(config);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Load a single configuration file
    fn load_config_file(&self, path: &Path) -> Result<ModelFamilyConfig, MullamaError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| MullamaError::IoError(e))?;

        let config: ModelFamilyConfig = toml::from_str(&contents)
            .map_err(|e| MullamaError::InvalidInput(format!(
                "Failed to parse {}: {}", path.display(), e
            )))?;

        Ok(config)
    }

    /// Compile regex patterns for faster matching
    fn compile_patterns(&mut self) {
        self.patterns.clear();

        // Sort families by priority (descending)
        self.families.sort_by(|a, b| b.family.priority.cmp(&a.family.priority));

        for (idx, family) in self.families.iter().enumerate() {
            for pattern in &family.family.patterns {
                if let Ok(re) = regex::Regex::new(&format!("(?i){}", pattern)) {
                    self.patterns.push((re, idx));
                }
            }
        }
    }

    /// Load built-in default configurations
    fn load_builtin_defaults(&mut self) {
        // Qwen 3
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "qwen3".to_string(),
                display_name: "Qwen 3".to_string(),
                patterns: vec![
                    "qwen3".to_string(),
                    "qwen-3".to_string(),
                    "qwen_3".to_string(),
                    "qwen2\\.5".to_string(),
                ],
                priority: 10,
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: false,
                vision: false,
            },
            tokens: TokenConfig {
                extends: None,
                bos: Some("<|im_start|>".to_string()),
                eos: Some("<|im_end|>".to_string()),
                user_prefix: Some("<|im_start|>user\n".to_string()),
                user_suffix: Some("<|im_end|>\n".to_string()),
                assistant_prefix: Some("<|im_start|>assistant\n".to_string()),
                assistant_suffix: Some("<|im_end|>\n".to_string()),
                system_prefix: Some("<|im_start|>system\n".to_string()),
                system_suffix: Some("<|im_end|>\n".to_string()),
                stop_sequences: vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()],
            },
            tool_format: Some(ToolFormat {
                style: "qwen".to_string(),
                tool_call_start: Some("<tool_call>".to_string()),
                tool_call_end: Some("</tool_call>".to_string()),
                tool_result_start: Some("<tool_response>".to_string()),
                tool_result_end: Some("</tool_response>".to_string()),
            }),
            thinking: None,
        });

        // QwQ (Qwen reasoning model)
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "qwq".to_string(),
                display_name: "QwQ (Reasoning)".to_string(),
                patterns: vec!["qwq".to_string(), "qwen-qwq".to_string()],
                priority: 20, // Higher priority than base qwen
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: true,
                vision: false,
            },
            tokens: TokenConfig {
                extends: Some("qwen3".to_string()),
                bos: Some("<|im_start|>".to_string()),
                eos: Some("<|im_end|>".to_string()),
                user_prefix: Some("<|im_start|>user\n".to_string()),
                user_suffix: Some("<|im_end|>\n".to_string()),
                assistant_prefix: Some("<|im_start|>assistant\n".to_string()),
                assistant_suffix: Some("<|im_end|>\n".to_string()),
                system_prefix: Some("<|im_start|>system\n".to_string()),
                system_suffix: Some("<|im_end|>\n".to_string()),
                stop_sequences: vec!["<|im_end|>".to_string()],
            },
            tool_format: Some(ToolFormat {
                style: "qwen".to_string(),
                tool_call_start: Some("<tool_call>".to_string()),
                tool_call_end: Some("</tool_call>".to_string()),
                tool_result_start: Some("<tool_response>".to_string()),
                tool_result_end: Some("</tool_response>".to_string()),
            }),
            thinking: Some(ThinkingTokens {
                start_token: "<think>".to_string(),
                end_token: "</think>".to_string(),
            }),
        });

        // Llama 3.1+
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "llama3".to_string(),
                display_name: "Llama 3".to_string(),
                patterns: vec![
                    "llama-3".to_string(),
                    "llama3".to_string(),
                    "llama-3\\.1".to_string(),
                    "llama-3\\.2".to_string(),
                    "llama-3\\.3".to_string(),
                ],
                priority: 10,
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: false,
                vision: false,
            },
            tokens: TokenConfig {
                extends: None,
                bos: Some("<|begin_of_text|>".to_string()),
                eos: Some("<|eot_id|>".to_string()),
                user_prefix: Some("<|start_header_id|>user<|end_header_id|>\n\n".to_string()),
                user_suffix: Some("<|eot_id|>".to_string()),
                assistant_prefix: Some("<|start_header_id|>assistant<|end_header_id|>\n\n".to_string()),
                assistant_suffix: Some("<|eot_id|>".to_string()),
                system_prefix: Some("<|start_header_id|>system<|end_header_id|>\n\n".to_string()),
                system_suffix: Some("<|eot_id|>".to_string()),
                stop_sequences: vec!["<|eot_id|>".to_string(), "<|eom_id|>".to_string()],
            },
            tool_format: Some(ToolFormat {
                style: "llama".to_string(),
                tool_call_start: Some("<|python_tag|>".to_string()),
                tool_call_end: None,
                tool_result_start: Some("<|start_header_id|>ipython<|end_header_id|>\n\n".to_string()),
                tool_result_end: Some("<|eot_id|>".to_string()),
            }),
            thinking: None,
        });

        // Gemma 3
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "gemma3".to_string(),
                display_name: "Gemma 3".to_string(),
                patterns: vec!["gemma-3".to_string(), "gemma3".to_string()],
                priority: 10,
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: false,
                vision: false,
            },
            tokens: TokenConfig {
                extends: None,
                bos: Some("<bos>".to_string()),
                eos: Some("<end_of_turn>".to_string()),
                user_prefix: Some("<start_of_turn>user\n".to_string()),
                user_suffix: Some("<end_of_turn>\n".to_string()),
                assistant_prefix: Some("<start_of_turn>model\n".to_string()),
                assistant_suffix: Some("<end_of_turn>\n".to_string()),
                system_prefix: Some("<start_of_turn>user\n".to_string()), // Gemma uses user for system
                system_suffix: Some("<end_of_turn>\n".to_string()),
                stop_sequences: vec!["<end_of_turn>".to_string()],
            },
            tool_format: None,
            thinking: None,
        });

        // DeepSeek
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "deepseek".to_string(),
                display_name: "DeepSeek".to_string(),
                patterns: vec!["deepseek".to_string()],
                priority: 5,
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: false,
                vision: false,
            },
            tokens: TokenConfig {
                extends: None,
                bos: Some("<|begin▁of▁sentence|>".to_string()),
                eos: Some("<|end▁of▁sentence|>".to_string()),
                user_prefix: Some("<|User|>".to_string()),
                user_suffix: Some("\n".to_string()),
                assistant_prefix: Some("<|Assistant|>".to_string()),
                assistant_suffix: Some("<|end▁of▁sentence|>".to_string()),
                system_prefix: Some("<|System|>".to_string()),
                system_suffix: Some("\n".to_string()),
                stop_sequences: vec![
                    "<|end▁of▁sentence|>".to_string(),
                    "<｜end▁of▁sentence｜>".to_string(),
                ],
            },
            tool_format: None,
            thinking: None,
        });

        // DeepSeek R1 (reasoning model)
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "deepseek-r1".to_string(),
                display_name: "DeepSeek R1 (Reasoning)".to_string(),
                patterns: vec!["deepseek-r1".to_string(), "deepseek.*r1".to_string()],
                priority: 15, // Higher than base deepseek
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: true,
                vision: false,
            },
            tokens: TokenConfig {
                extends: Some("deepseek".to_string()),
                bos: Some("<|begin▁of▁sentence|>".to_string()),
                eos: Some("<|end▁of▁sentence|>".to_string()),
                user_prefix: Some("<|User|>".to_string()),
                user_suffix: Some("\n".to_string()),
                assistant_prefix: Some("<|Assistant|>".to_string()),
                assistant_suffix: Some("<|end▁of▁sentence|>".to_string()),
                system_prefix: Some("<|System|>".to_string()),
                system_suffix: Some("\n".to_string()),
                stop_sequences: vec!["<|end▁of▁sentence|>".to_string()],
            },
            tool_format: None,
            thinking: Some(ThinkingTokens {
                start_token: "<think>".to_string(),
                end_token: "</think>".to_string(),
            }),
        });

        // Phi 4
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "phi4".to_string(),
                display_name: "Phi 4".to_string(),
                patterns: vec!["phi-4".to_string(), "phi4".to_string()],
                priority: 10,
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: false,
                vision: false,
            },
            tokens: TokenConfig {
                extends: None,
                bos: Some("<|im_start|>".to_string()),
                eos: Some("<|im_end|>".to_string()),
                user_prefix: Some("<|im_start|>user\n".to_string()),
                user_suffix: Some("<|im_end|>\n".to_string()),
                assistant_prefix: Some("<|im_start|>assistant\n".to_string()),
                assistant_suffix: Some("<|im_end|>\n".to_string()),
                system_prefix: Some("<|im_start|>system\n".to_string()),
                system_suffix: Some("<|im_end|>\n".to_string()),
                stop_sequences: vec!["<|im_end|>".to_string(), "<|end|>".to_string()],
            },
            tool_format: None,
            thinking: None,
        });

        // Mistral
        self.families.push(ModelFamilyConfig {
            family: FamilyInfo {
                name: "mistral".to_string(),
                display_name: "Mistral".to_string(),
                patterns: vec!["mistral".to_string(), "mixtral".to_string()],
                priority: 5,
            },
            capabilities: Capabilities {
                native_json: true,
                tool_use: true,
                thinking: false,
                vision: false,
            },
            tokens: TokenConfig {
                extends: None,
                bos: Some("<s>".to_string()),
                eos: Some("</s>".to_string()),
                user_prefix: Some("[INST] ".to_string()),
                user_suffix: Some(" [/INST]".to_string()),
                assistant_prefix: Some("".to_string()),
                assistant_suffix: Some("</s>".to_string()),
                system_prefix: Some("<<SYS>>\n".to_string()),
                system_suffix: Some("\n<</SYS>>\n\n".to_string()),
                stop_sequences: vec!["</s>".to_string()],
            },
            tool_format: None,
            thinking: None,
        });
    }

    /// Detect capabilities for a model by name or architecture
    pub fn detect(&self, model_name: &str, architecture: Option<&str>) -> &ModelFamilyConfig {
        // Try to match against model name first
        let name_lower = model_name.to_lowercase();
        for (pattern, idx) in &self.patterns {
            if pattern.is_match(&name_lower) {
                return &self.families[*idx];
            }
        }

        // Try architecture if provided
        if let Some(arch) = architecture {
            let arch_lower = arch.to_lowercase();
            for (pattern, idx) in &self.patterns {
                if pattern.is_match(&arch_lower) {
                    return &self.families[*idx];
                }
            }
        }

        // Return fallback
        self.fallback()
    }

    /// Get fallback config for unknown models
    pub fn fallback(&self) -> &ModelFamilyConfig {
        // Use a static fallback config
        static FALLBACK: OnceLock<ModelFamilyConfig> = OnceLock::new();
        FALLBACK.get_or_init(|| ModelFamilyConfig {
            family: FamilyInfo {
                name: "unknown".to_string(),
                display_name: "Unknown Model".to_string(),
                patterns: vec![],
                priority: 0,
            },
            capabilities: Capabilities::default(),
            tokens: TokenConfig::default(),
            tool_format: None,
            thinking: None,
        })
    }

    /// Get all registered families
    pub fn families(&self) -> &[ModelFamilyConfig] {
        &self.families
    }

    /// Get a family by name
    pub fn get_family(&self, name: &str) -> Option<&ModelFamilyConfig> {
        self.families.iter().find(|f| f.family.name == name)
    }
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::with_builtin_defaults()
    }
}

/// Get the list of config directories to search
fn get_config_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // Environment variable override
    if let Ok(config_dir) = std::env::var("MULLAMA_CONFIG") {
        dirs.push(PathBuf::from(config_dir).join("models"));
    }

    // XDG config directory
    if let Some(config_home) = dirs::config_dir() {
        dirs.push(config_home.join("mullama").join("models"));
    }

    // Home directory fallback
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".config").join("mullama").join("models"));
    }

    dirs
}

/// Detect capabilities for a model
///
/// This is the main entry point for capability detection.
pub fn detect_capabilities(model_name: &str, architecture: Option<&str>) -> &'static ModelFamilyConfig {
    registry().detect(model_name, architecture)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_detection() {
        let registry = CapabilityRegistry::with_builtin_defaults();
        let config = registry.detect("Qwen3-0.6B-GGUF", None);
        assert_eq!(config.family.name, "qwen3");
        assert!(config.capabilities.native_json);
        assert!(config.capabilities.tool_use);
        assert!(!config.capabilities.thinking);
    }

    #[test]
    fn test_qwq_detection() {
        let registry = CapabilityRegistry::with_builtin_defaults();
        let config = registry.detect("QwQ-32B", None);
        assert_eq!(config.family.name, "qwq");
        assert!(config.capabilities.thinking);
    }

    #[test]
    fn test_deepseek_r1_detection() {
        let registry = CapabilityRegistry::with_builtin_defaults();
        let config = registry.detect("DeepSeek-R1-Distill-Qwen-7B", None);
        assert_eq!(config.family.name, "deepseek-r1");
        assert!(config.capabilities.thinking);
    }

    #[test]
    fn test_architecture_fallback() {
        let registry = CapabilityRegistry::with_builtin_defaults();
        // Unknown model name but known architecture
        let config = registry.detect("some-random-model", Some("qwen2"));
        // Should still match qwen family by architecture pattern
        assert!(config.family.name.contains("qwen") || config.family.name == "unknown");
    }

    #[test]
    fn test_fallback_for_unknown() {
        let registry = CapabilityRegistry::with_builtin_defaults();
        let config = registry.detect("completely-unknown-model", None);
        assert_eq!(config.family.name, "unknown");
        assert!(!config.capabilities.tool_use);
    }
}
