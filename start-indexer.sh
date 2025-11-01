#!/usr/bin/env bash
set -euo pipefail

# Directory resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

# Parse CLI overrides (CLI > ENV_FILE values)
CLI_ENV_FILE=""
CLI_CODEBASE_PATH=""
CLI_COLLECTION_NAME=""
CLI_GPT2GIGA_URL=""
CLI_VECTOR_DB_TYPE=""
CLI_VECTOR_DB_URL=""
CLI_VECTOR_DB_API_KEY=""
CLI_BATCH_SIZE=""
CLI_CHUNK_SIZE=""
CLI_INCLUDE_PATTERNS=""
CLI_EXCLUDE_PATTERNS=""
CLI_INCREMENTAL=""
CLI_FORCE=""
CLI_LOG_LEVEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      CLI_ENV_FILE="${2:-}"; shift 2;;
    --codebase-path)
      CLI_CODEBASE_PATH="${2:-}"; shift 2;;
    --collection-name)
      CLI_COLLECTION_NAME="${2:-}"; shift 2;;
    --gpt2giga-url)
      CLI_GPT2GIGA_URL="${2:-}"; shift 2;;
    --vector-db-type)
      CLI_VECTOR_DB_TYPE="${2:-}"; shift 2;;
    --vector-db-url)
      CLI_VECTOR_DB_URL="${2:-}"; shift 2;;
    --vector-db-api-key)
      CLI_VECTOR_DB_API_KEY="${2:-}"; shift 2;;
    --batch-size)
      CLI_BATCH_SIZE="${2:-}"; shift 2;;
    --chunk-size)
      CLI_CHUNK_SIZE="${2:-}"; shift 2;;
    --include-patterns)
      CLI_INCLUDE_PATTERNS="${2:-}"; shift 2;;
    --exclude-patterns)
      CLI_EXCLUDE_PATTERNS="${2:-}"; shift 2;;
    --incremental)
      CLI_INCREMENTAL="true"; shift;;
    --force)
      CLI_FORCE="true"; shift;;
    --log-level)
      CLI_LOG_LEVEL="${2:-}"; shift 2;;
    --help)
      echo "Usage: $0 [OPTIONS] [CODEBASE_PATH]"
      echo ""
      echo "Index a codebase folder for RAG using gpt2giga."
      echo ""
      echo "Arguments:"
      echo "  CODEBASE_PATH              Path to codebase directory to index (required)"
      echo ""
      echo "Options:"
      echo "  --env-file FILE            Path to .env file (default: .env in project root)"
      echo "  --collection-name NAME    Vector DB collection name (default: codebase)"
      echo "  --gpt2giga-url URL         gpt2giga API URL (default: http://localhost:8090)"
      echo "  --vector-db-type TYPE      Vector DB type: simple, qdrant (default: simple)"
      echo "  --vector-db-url URL        Vector DB URL (required for qdrant)"
      echo "  --vector-db-api-key KEY    Vector DB API key (optional)"
      echo "  --batch-size SIZE          Batch size for embeddings (default: 100)"
      echo "  --chunk-size SIZE          Target tokens per chunk (default: 2000)"
      echo "  --include-patterns PATS   Comma-separated file extensions (default: .py,.js,.ts,...)"
      echo "  --exclude-patterns PATS    Comma-separated patterns to exclude"
      echo "  --incremental               Only index changed files (default)"
      echo "  --force                     Force re-index all files"
      echo "  --log-level LEVEL          Log level: CRITICAL, ERROR, WARNING, INFO, DEBUG (default: INFO)"
      echo "  --help                      Show this help message"
      echo ""
      echo "Environment variables:"
      echo "  See .env.defaults for all available options"
      echo ""
      echo "Examples:"
      echo "  $0 ./spec"
      echo "  $0 /path/to/spec --collection-name spec-tests"
      echo "  $0 ./spec --vector-db-type qdrant --vector-db-url http://localhost:6333"
      exit 0;;
    *)
      # First unknown arg is codebase path, rest are passed through
      if [[ -z "$CLI_CODEBASE_PATH" ]]; then
        CLI_CODEBASE_PATH="$1"
      fi
      shift;;
  esac
done

# .env handling (optional): point ENV_FILE to a custom path if needed
ENV_FILE="${CLI_ENV_FILE:-${ENV_FILE:-$ROOT_DIR/.env}}"
if [ -f "$ENV_FILE" ]; then
  # Export valid KEY=VALUE lines from .env (skip comments and invalid lines)
  while IFS= read -r line || [ -n "$line" ]; do
    # Trim whitespace
    line_trimmed=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    # Skip empty lines and comments
    if [ -z "$line_trimmed" ] || [ "${line_trimmed#\#}" != "$line_trimmed" ]; then
      continue
    fi
    # Check if line contains '=' (is a KEY=VALUE assignment)
    if [[ "$line_trimmed" == *"="* ]]; then
      # Extract key part (before first =)
      key="${line_trimmed%%=*}"
      key_trimmed=$(echo "$key" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
      # Extract value part (after first =)
      value="${line_trimmed#*=}"
      value_trimmed=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
      # Check if key is a valid identifier (starts with letter/underscore, contains only alphanumeric/underscore)
      if echo "$key_trimmed" | grep -qE '^[a-zA-Z_][a-zA-Z0-9_]*$'; then
        # Export the variable (handles quoted and unquoted values)
        export "${key_trimmed}=${value_trimmed}" 2>/dev/null || true
      fi
    fi
  done < "$ENV_FILE"
fi

# Validate codebase path (required)
CODEBASE_PATH="${CLI_CODEBASE_PATH:-${CODEBASE_PATH:-}}"
if [ -z "$CODEBASE_PATH" ]; then
  echo "Error: CODEBASE_PATH is required"
  echo "Usage: $0 [OPTIONS] CODEBASE_PATH"
  echo "Run '$0 --help' for more information"
  exit 1
fi

# Validate codebase path exists
if [ ! -d "$CODEBASE_PATH" ]; then
  echo "Error: Codebase path does not exist: $CODEBASE_PATH"
  exit 1
fi

# Resolve absolute path
CODEBASE_PATH="$(cd "$CODEBASE_PATH" && pwd)"

# Build command arguments
CMD_ARGS=()

# Required: codebase path
CMD_ARGS+=("$CODEBASE_PATH")

# Optional arguments (CLI > ENV)
if [[ -n "$CLI_COLLECTION_NAME" ]]; then
  CMD_ARGS+=(--collection-name "$CLI_COLLECTION_NAME")
elif [[ -n "${GPT2GIGA_INDEXER_COLLECTION_NAME:-}" ]]; then
  CMD_ARGS+=(--collection-name "$GPT2GIGA_INDEXER_COLLECTION_NAME")
fi

if [[ -n "$CLI_GPT2GIGA_URL" ]]; then
  CMD_ARGS+=(--gpt2giga-url "$CLI_GPT2GIGA_URL")
elif [[ -n "${GPT2GIGA_INDEXER_GPT2GIGA_URL:-}" ]]; then
  CMD_ARGS+=(--gpt2giga-url "$GPT2GIGA_INDEXER_GPT2GIGA_URL")
fi

if [[ -n "$CLI_VECTOR_DB_TYPE" ]]; then
  CMD_ARGS+=(--vector-db-type "$CLI_VECTOR_DB_TYPE")
elif [[ -n "${GPT2GIGA_INDEXER_VECTOR_DB_TYPE:-}" ]]; then
  CMD_ARGS+=(--vector-db-type "$GPT2GIGA_INDEXER_VECTOR_DB_TYPE")
fi

if [[ -n "$CLI_VECTOR_DB_URL" ]]; then
  CMD_ARGS+=(--vector-db-url "$CLI_VECTOR_DB_URL")
elif [[ -n "${GPT2GIGA_INDEXER_VECTOR_DB_URL:-}" ]]; then
  CMD_ARGS+=(--vector-db-url "$GPT2GIGA_INDEXER_VECTOR_DB_URL")
fi

if [[ -n "$CLI_VECTOR_DB_API_KEY" ]]; then
  CMD_ARGS+=(--vector-db-api-key "$CLI_VECTOR_DB_API_KEY")
elif [[ -n "${GPT2GIGA_INDEXER_VECTOR_DB_API_KEY:-}" ]]; then
  CMD_ARGS+=(--vector-db-api-key "$GPT2GIGA_INDEXER_VECTOR_DB_API_KEY")
fi

if [[ -n "$CLI_BATCH_SIZE" ]]; then
  CMD_ARGS+=(--batch-size "$CLI_BATCH_SIZE")
elif [[ -n "${GPT2GIGA_INDEXER_BATCH_SIZE:-}" ]]; then
  CMD_ARGS+=(--batch-size "$GPT2GIGA_INDEXER_BATCH_SIZE")
fi

if [[ -n "$CLI_CHUNK_SIZE" ]]; then
  CMD_ARGS+=(--chunk-size "$CLI_CHUNK_SIZE")
elif [[ -n "${GPT2GIGA_INDEXER_CHUNK_SIZE:-}" ]]; then
  CMD_ARGS+=(--chunk-size "$GPT2GIGA_INDEXER_CHUNK_SIZE")
fi

if [[ -n "$CLI_INCLUDE_PATTERNS" ]]; then
  CMD_ARGS+=(--include-patterns "$CLI_INCLUDE_PATTERNS")
elif [[ -n "${GPT2GIGA_INDEXER_INCLUDE_PATTERNS:-}" ]]; then
  CMD_ARGS+=(--include-patterns "$GPT2GIGA_INDEXER_INCLUDE_PATTERNS")
fi

if [[ -n "$CLI_EXCLUDE_PATTERNS" ]]; then
  CMD_ARGS+=(--exclude-patterns "$CLI_EXCLUDE_PATTERNS")
elif [[ -n "${GPT2GIGA_INDEXER_EXCLUDE_PATTERNS:-}" ]]; then
  CMD_ARGS+=(--exclude-patterns "$GPT2GIGA_INDEXER_EXCLUDE_PATTERNS")
fi

if [[ -n "$CLI_FORCE" ]]; then
  CMD_ARGS+=(--force)
elif [[ -n "$CLI_INCREMENTAL" ]] && [[ "$CLI_INCREMENTAL" == "true" ]]; then
  CMD_ARGS+=(--incremental)
fi

if [[ -n "$CLI_LOG_LEVEL" ]]; then
  CMD_ARGS+=(--log-level "$CLI_LOG_LEVEL")
elif [[ -n "${GPT2GIGA_LOG_LEVEL:-}" ]]; then
  CMD_ARGS+=(--log-level "$GPT2GIGA_LOG_LEVEL")
fi

# Determine Python executable
PYTHON=""
if command -v poetry &> /dev/null && [ -f "$ROOT_DIR/pyproject.toml" ]; then
  # Try poetry first
  PYTHON="poetry run python"
elif [ -f "$ROOT_DIR/.venv/bin/python" ]; then
  # Try local venv
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif [ -f "$ROOT_DIR/venv/bin/python" ]; then
  # Try venv (legacy)
  PYTHON="$ROOT_DIR/venv/bin/python"
elif command -v python3 &> /dev/null; then
  # Fallback to system python3
  PYTHON="python3"
else
  echo "Error: Python not found. Please install Python 3.9+ or activate a virtual environment."
  exit 1
fi

# Run the indexer
cd "$ROOT_DIR"
exec $PYTHON -m gpt2giga index-codebase "${CMD_ARGS[@]}"

