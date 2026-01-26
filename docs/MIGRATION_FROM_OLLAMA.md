# Migration from Ollama

Mullama is a drop-in replacement for Ollama. This guide covers everything you need to switch.

## Quick Checklist

- [ ] Install Mullama: `curl -fsSL https://mullama.dev/install.sh | sh`
- [ ] Your CLI commands work unchanged
- [ ] Your Modelfiles work unchanged
- [ ] Update API endpoint from `localhost:11434` to `localhost:8080`
- [ ] (Optional) Explore native library bindings

## CLI Commands

**Same commands, same syntax:**

| Command | Ollama | Mullama |
|---------|--------|---------|
| Run model | `ollama run llama3.2` | `mullama run llama3.2` |
| Pull model | `ollama pull llama3.2` | `mullama pull llama3.2` |
| List models | `ollama list` | `mullama list` |
| Remove model | `ollama rm model` | `mullama rm model` |
| Show info | `ollama show model` | `mullama show model` |
| Start server | `ollama serve` | `mullama serve` |
| Running models | `ollama ps` | `mullama ps` |

## Modelfile Compatibility

Your existing Modelfiles work without changes:

```dockerfile
FROM llama3.2:1b

PARAMETER temperature 0.7
PARAMETER num_ctx 8192

SYSTEM """
You are a helpful assistant.
"""
```

```bash
mullama create my-assistant -f ./Modelfile
mullama run my-assistant "Hello!"
```

**Mullama extensions** (optional):

```dockerfile
# These are Mullama-specific, ignored by Ollama
GPU_LAYERS 32
FLASH_ATTENTION true
ADAPTER ./my-lora.safetensors
```

## API Endpoint Changes

**Default ports:**

| | Ollama | Mullama |
|--|--------|---------|
| HTTP API | `localhost:11434` | `localhost:8080` |

**OpenAI-compatible endpoints:**

```bash
# Ollama
curl http://localhost:11434/v1/chat/completions ...

# Mullama
curl http://localhost:8080/v1/chat/completions ...
```

**Anthropic-compatible endpoints (Mullama only):**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "max_tokens": 1024, "messages": [...]}'
```

## Model Storage

Models are stored in different locations:

| Platform | Ollama | Mullama |
|----------|--------|---------|
| Linux | `~/.ollama/models` | `~/.cache/mullama/models` |
| macOS | `~/.ollama/models` | `~/Library/Caches/mullama/models` |
| Windows | `%USERPROFILE%\.ollama\models` | `%LOCALAPPDATA%\mullama\models` |

**Note:** Mullama downloads models from HuggingFace directly. Models are not shared between Ollama and Mullama.

## What's New in Mullama

Features not available in Ollama:

| Feature | Description |
|---------|-------------|
| Native bindings | Python, Node.js, Go, PHP, C/C++ |
| Anthropic API | `/v1/messages` endpoint |
| Web UI | Built-in at `/ui/` |
| TUI chat | `mullama chat` |
| Embed in app | No daemon required for library use |
| ColBERT | Late interaction embeddings |
| Streaming audio | Real-time voice input |

## Environment Variables

| Variable | Ollama | Mullama |
|----------|--------|---------|
| API host | `OLLAMA_HOST` | N/A (use `--http-addr`) |
| Model storage | `OLLAMA_MODELS` | `MULLAMA_CACHE_DIR` |
| HuggingFace token | N/A | `HF_TOKEN` |

## Need Help?

- [Full documentation](https://docs.neullabs.com/mullama/)
- [Detailed comparison](../documentation/docs/comparison/vs-ollama.md)
- [GitHub issues](https://github.com/neul-labs/mullama/issues)
