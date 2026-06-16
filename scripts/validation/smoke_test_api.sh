#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl: not found" >&2
  exit 1
fi

curl --fail --silent --show-error "${BASE_URL}/health"
