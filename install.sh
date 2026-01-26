#!/bin/sh
# Mullama installer
# Usage: curl -fsSL https://mullama.dev/install.sh | sh
#
# Drop-in Ollama replacement. All-in-one LLM toolkit.
# https://github.com/neul-labs/mullama

set -e

# Delegate to the full installer script
exec curl -fsSL https://raw.githubusercontent.com/neul-labs/mullama/main/scripts/install.sh | sh
