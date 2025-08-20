#!/usr/bin/env bash
set -euo pipefail

echo "=== Alexa_FAST setup script ==="

# --- 1. Install Ollama if missing ---
if ! command -v ollama >/dev/null 2>&1; then
  echo "[*] Ollama not found, installing..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "[*] Starting Ollama service..."
  sudo systemctl enable ollama
  sudo systemctl start ollama
else
  echo "[*] Ollama already installed."
fi

# --- 2. Wait for Ollama API to be ready ---
echo "[*] Waiting for Ollama service to be ready..."
for i in {1..10}; do
  if curl -s http://localhost:11434/api/version >/dev/null; then
    echo "[*] Ollama is running."
    break
  fi
  sleep 2
done

# --- 3. Pull the required model ---
echo "[*] Pulling model (edit script to switch models)..."

# Uncomment the model you want to use:

# Qwen3 8B quantized Q4_K_M
ollama pull qwen3:8b-q4_K_M

# Llama 3.1 8B Instruct quantized Q4_K_M
# ollama pull llama3.1:8b-instruct-q4_K_M

echo "=== Setup complete ==="
echo "To test: run 'ollama run qwen3:8b-q4_K_M' or the model you pulled."
