//! Default Modelfiles for quick-start experience
//!
//! This module provides bundled Modelfiles that users can see in the UI
//! and use with a single click. Each includes proper templates, stop tokens,
//! system prompts, and parameters.

use include_dir::{include_dir, Dir};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::modelfile::{Modelfile, ModelfileParser};

/// Embedded default modelfiles directory
static DEFAULT_MODELFILES: Dir = include_dir!("$CARGO_MANIFEST_DIR/configs/modelfiles");

/// Metadata for a default model (extracted from modelfile + hardcoded info)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultModelInfo {
    /// Model name (derived from filename)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Model size hint (e.g., "1B", "7B", "9B")
    pub size_hint: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// The FROM directive value
    pub from: String,
    /// Whether this model supports thinking/reasoning
    pub has_thinking: bool,
    /// Whether this model supports vision/images
    pub has_vision: bool,
    /// Whether this model supports tool calling
    pub has_tools: bool,
}

/// A default model with its parsed Modelfile
#[derive(Debug, Clone)]
pub struct DefaultModel {
    /// Metadata for display
    pub info: DefaultModelInfo,
    /// The parsed Modelfile
    pub modelfile: Modelfile,
    /// Raw content of the modelfile
    pub content: String,
}

/// Static metadata for default models (description, size, tags)
fn get_model_metadata() -> HashMap<&'static str, (&'static str, &'static str, &'static [&'static str])> {
    let mut meta = HashMap::new();

    // (description, size_hint, tags)
    meta.insert("llama3.2-1b", (
        "Meta Llama 3.2 1B - Fast and lightweight",
        "1B",
        &["chat", "instruct", "fast", "lightweight"][..],
    ));
    meta.insert("llama3.2-3b", (
        "Meta Llama 3.2 3B - Balanced size and capability",
        "3B",
        &["chat", "instruct", "tools", "balanced"][..],
    ));
    meta.insert("qwen2.5-7b", (
        "Qwen 2.5 7B - Multilingual with tool support",
        "7B",
        &["chat", "multilingual", "tools", "coding"][..],
    ));
    meta.insert("deepseek-r1-7b", (
        "DeepSeek R1 7B - Advanced reasoning model",
        "7B",
        &["reasoning", "thinking", "chain-of-thought"][..],
    ));
    meta.insert("mistral-7b", (
        "Mistral 7B v0.3 - Strong general purpose",
        "7B",
        &["chat", "general", "tools", "coding"][..],
    ));
    meta.insert("phi3-mini", (
        "Phi-3 Mini - Microsoft's compact powerhouse",
        "3.8B",
        &["chat", "compact", "efficient"][..],
    ));
    meta.insert("gemma2-9b", (
        "Gemma 2 9B - Google's capable model",
        "9B",
        &["chat", "reasoning", "tools"][..],
    ));
    meta.insert("llava-7b", (
        "LLaVA 1.5 7B - Vision and language model",
        "7B",
        &["vision", "multimodal", "image-understanding"][..],
    ));

    meta
}

/// List all available default models
pub fn list_defaults() -> Vec<DefaultModel> {
    let parser = ModelfileParser::new();
    let metadata = get_model_metadata();
    let mut defaults = Vec::new();

    for file in DEFAULT_MODELFILES.files() {
        let path = file.path();

        // Get the model name from filename (without extension)
        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Only process .modelfile files
        let ext = path.extension().and_then(|s| s.to_str());
        if ext != Some("modelfile") {
            continue;
        }

        // Parse the content
        let content = match std::str::from_utf8(file.contents()) {
            Ok(c) => c.to_string(),
            Err(_) => continue,
        };

        let modelfile = match parser.parse_str(&content) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to parse default modelfile {}: {}", name, e);
                continue;
            }
        };

        // Get metadata for this model
        let (description, size_hint, tags) = metadata
            .get(name.as_str())
            .copied()
            .unwrap_or(("Unknown model", "?B", &[][..]));

        let info = DefaultModelInfo {
            name: name.clone(),
            description: description.to_string(),
            size_hint: size_hint.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            from: modelfile.from.clone(),
            has_thinking: modelfile.capabilities.thinking,
            has_vision: modelfile.capabilities.vision,
            has_tools: modelfile.capabilities.tools,
        };

        defaults.push(DefaultModel {
            info,
            modelfile,
            content,
        });
    }

    // Sort by size (smallest first) for better UX
    defaults.sort_by(|a, b| {
        let size_order = |s: &str| -> u32 {
            match s {
                "1B" => 1,
                "3B" => 3,
                "3.8B" => 4,
                "7B" => 7,
                "9B" => 9,
                _ => 100,
            }
        };
        size_order(&a.info.size_hint).cmp(&size_order(&b.info.size_hint))
    });

    defaults
}

/// Get a specific default model by name
pub fn get_default(name: &str) -> Option<DefaultModel> {
    list_defaults().into_iter().find(|d| d.info.name == name)
}

/// Get just the info for all defaults (lighter weight for API responses)
pub fn list_default_infos() -> Vec<DefaultModelInfo> {
    list_defaults().into_iter().map(|d| d.info).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_defaults() {
        let defaults = list_defaults();
        assert!(!defaults.is_empty(), "Should have at least one default model");

        for model in &defaults {
            assert!(!model.info.name.is_empty());
            assert!(!model.info.description.is_empty());
            assert!(!model.info.from.is_empty());
            assert!(!model.modelfile.from.is_empty());
        }
    }

    #[test]
    fn test_get_default() {
        let llama = get_default("llama3.2-3b");
        assert!(llama.is_some());

        let model = llama.unwrap();
        assert_eq!(model.info.size_hint, "3B");
        assert!(model.info.from.contains("Llama-3.2-3B"));
    }

    #[test]
    fn test_thinking_model() {
        let deepseek = get_default("deepseek-r1-7b");
        assert!(deepseek.is_some());

        let model = deepseek.unwrap();
        assert!(model.info.has_thinking);
        assert!(model.modelfile.thinking.is_some());
    }

    #[test]
    fn test_vision_model() {
        let llava = get_default("llava-7b");
        assert!(llava.is_some());

        let model = llava.unwrap();
        assert!(model.info.has_vision);
    }
}
