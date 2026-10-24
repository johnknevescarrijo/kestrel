#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/models"
TARGET_FILE="${TARGET_DIR}/qwen2.5-0.5b-q4.gguf"
DOWNLOAD_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

mkdir -p "${TARGET_DIR}"

if [[ -f "${TARGET_FILE}" ]]; then
    echo "Model already exists at: ${TARGET_FILE}"
    echo "Size: $(du -h "${TARGET_FILE}" | cut -f1)"
    exit 0
fi

echo "Fetching Qwen2.5-0.5B-Instruct Q4_K_M GGUF from Hugging Face..."
TMP_FILE="${TARGET_FILE}.tmp.$$"
trap 'rm -f "${TMP_FILE}"' EXIT

curl -fL --progress-bar "${DOWNLOAD_URL}" -o "${TMP_FILE}"
mv "${TMP_FILE}" "${TARGET_FILE}"

echo "Model successfully downloaded to: ${TARGET_FILE}"
echo "Size: $(du -h "${TARGET_FILE}" | cut -f1)"
