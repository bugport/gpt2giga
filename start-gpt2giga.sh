#!/usr/bin/env bash
set -euo pipefail

# Directory resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

# Parse CLI overrides (CLI > ENV_FILE values)
CLI_ENV_FILE=""
CLI_HOST=""
CLI_PORT=""
CLI_USE_HTTPS=""
CLI_HTTPS_KEY_FILE=""
CLI_HTTPS_CERT_FILE=""
CLI_PASS_MODEL=""
CLI_PASS_TOKEN=""
CLI_LOG_LEVEL=""
CLI_BASE_URL=""
CLI_VERIFY_SSL_CERTS=""
CLI_MTLS=""
CLI_CERT_FILE=""
CLI_KEY_FILE=""
CLI_KEY_FILE_PASSWORD=""
CLI_CA_BUNDLE_FILE=""
CLI_NO_VERIFY_SSL_CERTS=""
CLI_AUTH_INSECURE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      CLI_ENV_FILE="${2:-}"; shift 2;;
    --host)
      CLI_HOST="${2:-}"; shift 2;;
    --port)
      CLI_PORT="${2:-}"; shift 2;;
    --use-https)
      CLI_USE_HTTPS="true"; shift;;
    --https-key-file)
      CLI_HTTPS_KEY_FILE="${2:-}"; shift 2;;
    --https-cert-file)
      CLI_HTTPS_CERT_FILE="${2:-}"; shift 2;;
    --pass-model)
      CLI_PASS_MODEL="true"; shift;;
    --pass-token)
      CLI_PASS_TOKEN="true"; shift;;
    --log-level)
      CLI_LOG_LEVEL="${2:-}"; shift 2;;
    --base-url)
      CLI_BASE_URL="${2:-}"; shift 2;;
    --verify-ssl-certs)
      CLI_VERIFY_SSL_CERTS="true"; shift;;
    --no-verify-ssl-certs)
      CLI_NO_VERIFY_SSL_CERTS="true"; shift;;
    --auth-insecure)
      CLI_AUTH_INSECURE="true"; shift;;
    --mtls)
      CLI_MTLS="true"; shift;;
    --cert-file)
      CLI_CERT_FILE="${2:-}"; shift 2;;
    --key-file)
      CLI_KEY_FILE="${2:-}"; shift 2;;
    --key-file-password)
      CLI_KEY_FILE_PASSWORD="${2:-}"; shift 2;;
    --ca-bundle-file)
      CLI_CA_BUNDLE_FILE="${2:-}"; shift 2;;
    --)
      shift; break;;
    *)
      # Unknown arg, stop parsing to allow passing through in future
      break;;
  esac
done

# .env handling (optional): point ENV_FILE to a custom path if needed
ENV_FILE="${CLI_ENV_FILE:-${ENV_FILE:-$ROOT_DIR/.env}}"
if [ -f "$ENV_FILE" ]; then
  # Export valid KEY=VALUE lines from .env (skip comments and invalid lines)
  while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines, comments, and lines without '='
    line_trimmed=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    if [ -n "$line_trimmed" ] && [ "${line_trimmed#\#}" = "$line_trimmed" ] && [ "${line_trimmed#*=}" != "$line_trimmed" ]; then
      # Only export if it looks like a valid KEY=VALUE assignment
      key="${line_trimmed%%=*}"
      # Check if key is a valid identifier (starts with letter/underscore, contains only alphanumeric/underscore)
      if echo "$key" | grep -qE '^[a-zA-Z_][a-zA-Z0-9_]*$'; then
        export "$line_trimmed" 2>/dev/null || true
      fi
    fi
  done < "$ENV_FILE"
fi

# If forcing no verification, override env and clear global cert envs
if [[ -n "$CLI_NO_VERIFY_SSL_CERTS" ]]; then
  export GIGACHAT_VERIFY_SSL_CERTS=False
  unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE || true
fi

# If auth insecure requested, inform server via env var (picked by ProxySettings)
if [[ -n "$CLI_AUTH_INSECURE" ]]; then
  export GPT2GIGA_AUTH_INSECURE=True
fi

# Resolve proxy host/port with sensible defaults (CLI overrides)
HOST="${CLI_HOST:-${GPT2GIGA_HOST:-${GPT2GIGA_PROXY_HOST:-0.0.0.0}}}"
PORT="${CLI_PORT:-${GPT2GIGA_PORT:-${GPT2GIGA_PROXY_PORT:-8090}}}"

# Build CLI args for gpt2giga
ARGS=(
  --proxy-host "$HOST"
  --proxy-port "$PORT"
)

# Include .env path explicitly so the CLI loader picks it up
if [ -f "$ENV_FILE" ]; then
  ARGS+=(--env-path "$ENV_FILE")
fi

# Log level (e.g., INFO, DEBUG)
if [ -n "${CLI_LOG_LEVEL:-}" ]; then
  ARGS+=(--proxy-log-level "$CLI_LOG_LEVEL")
elif [ -n "${GPT2GIGA_LOG_LEVEL:-}" ]; then
  ARGS+=(--proxy-log-level "$GPT2GIGA_LOG_LEVEL")
fi

# Optional HTTPS for the proxy itself
USE_HTTPS_EFFECTIVE="${CLI_USE_HTTPS:-${GPT2GIGA_USE_HTTPS:-${GPT2GIGA_PROXY_USE_HTTPS:-}}}"
if [[ "$USE_HTTPS_EFFECTIVE" == "True" || "$USE_HTTPS_EFFECTIVE" == "true" ]]; then
  ARGS+=(--proxy-use-https)
  if [ -n "${CLI_HTTPS_KEY_FILE:-${GPT2GIGA_HTTPS_KEY_FILE:-${GPT2GIGA_PROXY_HTTPS_KEY_FILE:-}}}" ]; then
    ARGS+=(--proxy-https-key-file "${CLI_HTTPS_KEY_FILE:-${GPT2GIGA_HTTPS_KEY_FILE:-${GPT2GIGA_PROXY_HTTPS_KEY_FILE}}}")
  fi
  if [ -n "${CLI_HTTPS_CERT_FILE:-${GPT2GIGA_HTTPS_CERT_FILE:-${GPT2GIGA_PROXY_HTTPS_CERT_FILE:-}}}" ]; then
    ARGS+=(--proxy-https-cert-file "${CLI_HTTPS_CERT_FILE:-${GPT2GIGA_HTTPS_CERT_FILE:-${GPT2GIGA_PROXY_HTTPS_CERT_FILE}}}")
  fi
fi

# Optional pass-through toggles
if [[ "${CLI_PASS_MODEL:-${GPT2GIGA_PASS_MODEL:-}}" == "True" || "${CLI_PASS_MODEL:-${GPT2GIGA_PASS_MODEL:-}}" == "true" ]]; then
  ARGS+=(--proxy-pass-model)
fi
if [[ "${CLI_PASS_TOKEN:-${GPT2GIGA_PASS_TOKEN:-}}" == "True" || "${CLI_PASS_TOKEN:-${GPT2GIGA_PASS_TOKEN:-}}" == "true" ]]; then
  ARGS+=(--proxy-pass-token)
fi

# Backend base URL (no-auth or mTLS target)
if [ -n "${CLI_BASE_URL:-${GIGACHAT_BASE_URL:-}}" ]; then
  ARGS+=(--gigachat-base-url "${CLI_BASE_URL:-${GIGACHAT_BASE_URL}}")
fi

# Backend TLS verification toggle
if [[ -z "$CLI_NO_VERIFY_SSL_CERTS" && ( "${CLI_VERIFY_SSL_CERTS:-${GIGACHAT_VERIFY_SSL_CERTS:-}}" == "True" || "${CLI_VERIFY_SSL_CERTS:-${GIGACHAT_VERIFY_SSL_CERTS:-}}" == "true" ) ]]; then
  ARGS+=(--gigachat-verify-ssl-certs)
fi

# Optional mTLS to backend
if [[ "${CLI_MTLS:-${GIGACHAT_MTLS_AUTH:-}}" == "True" || "${CLI_MTLS:-${GIGACHAT_MTLS_AUTH:-}}" == "true" ]]; then
  ARGS+=(--gigachat-mtls-auth)
  if [ -n "${CLI_CERT_FILE:-${GIGACHAT_CERT_FILE:-}}" ]; then
    ARGS+=(--gigachat-cert-file "${CLI_CERT_FILE:-${GIGACHAT_CERT_FILE}}")
  fi
  if [ -n "${CLI_KEY_FILE:-${GIGACHAT_KEY_FILE:-}}" ]; then
    ARGS+=(--gigachat-key-file "${CLI_KEY_FILE:-${GIGACHAT_KEY_FILE}}")
  fi
  if [ -n "${CLI_KEY_FILE_PASSWORD:-${GIGACHAT_KEY_FILE_PASSWORD:-}}" ]; then
    ARGS+=(--gigachat-key-file-password "${CLI_KEY_FILE_PASSWORD:-${GIGACHAT_KEY_FILE_PASSWORD}}")
  fi
  if [ -n "${CLI_CA_BUNDLE_FILE:-${GIGACHAT_CA_BUNDLE_FILE:-}}" ]; then
    ARGS+=(--gigachat-ca-bundle-file "${CLI_CA_BUNDLE_FILE:-${GIGACHAT_CA_BUNDLE_FILE}}")
  fi
fi

echo "Starting gpt2giga on $HOST:$PORT"

# Prefer Poetry if available; fallback to python -m
if command -v poetry >/dev/null 2>&1; then
  exec poetry run gpt2giga "${ARGS[@]}"
else
  exec python -m gpt2giga.api_server "${ARGS[@]}"
fi


