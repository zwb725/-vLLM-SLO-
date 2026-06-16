#!/usr/bin/env bash
set -euo pipefail

echo "date: $(date -Iseconds)"
echo "kernel: $(uname -a)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi: not found"
fi

if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  echo "python3: not found"
fi

if command -v vllm >/dev/null 2>&1; then
  vllm --version
else
  echo "vllm: not found"
fi
