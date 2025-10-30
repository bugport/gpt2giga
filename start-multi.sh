#!/usr/bin/env bash
set -euo pipefail

# Start multiple gpt2giga instances, each with its own .env file.
# Edit CONFIGS to list your instance env files (relative to this script dir).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}\")" && pwd)"

# List your config files here. Each must set a unique GPT2GIGA_PORT.
CONFIGS=(
  ".env.instance1"
  ".env.instance2"
  # Add more as needed, e.g. ".env.http.example" ".env.oauth.example"
)

start_instance() {
  local env_path="$1"
  if [[ ! -f "$env_path" ]]; then
    echo "Warning: $env_path does not exist, skipping." >&2
    return 0
  fi

  # Extract port for logging purposes (fallback to 8090)
  local port
  port=$(grep -E '^(GPT2GIGA_PORT|GPT2GIGA_PROXY_PORT)=' "$env_path" | tail -n1 | cut -d= -f2 | tr -d '"')
  port=${port:-8090}

  echo "Starting gpt2giga with $(basename "$env_path") on port $port ..."
  ENV_FILE="$env_path" "$SCRIPT_DIR/start-gpt2giga.sh" >"$SCRIPT_DIR/gpt2giga-$port.log" 2>&1 &
}

for cfg in "${CONFIGS[@]}"; do
  start_instance "$SCRIPT_DIR/$cfg"
done

echo "Launched all instances. Logs: $SCRIPT_DIR/gpt2giga-<PORT>.log"
echo "List processes:   ps aux | grep gpt2giga"
echo "Stop all:         pkill -f start-gpt2giga.sh"


