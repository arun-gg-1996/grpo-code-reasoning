#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for a fresh GPU server with persistent disk.
# - Creates/uses project directory on persistent storage
# - Clones repo (or updates if already present)
# - Creates venv and installs requirements
# - Prompts for W&B / HF / Gemini keys (optional)
#
# Usage examples:
#   bash scripts/setup_persistent_server.sh
#   REPO_URL=https://github.com/<you>/<repo>.git bash scripts/setup_persistent_server.sh
#   PERSIST_ROOT=/mnt/persist PROJECT_DIR=/mnt/persist/grpo-code-reasoning bash scripts/setup_persistent_server.sh

PERSIST_ROOT="${PERSIST_ROOT:-/mnt/persist}"
PROJECT_DIR="${PROJECT_DIR:-$PERSIST_ROOT/grpo-code-reasoning}"
REPO_URL="${REPO_URL:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"

echo "=== GRPO Server Setup ==="
echo "PERSIST_ROOT: $PERSIST_ROOT"
echo "PROJECT_DIR : $PROJECT_DIR"
echo

if [[ ! -d "$PERSIST_ROOT" ]]; then
  echo "ERROR: $PERSIST_ROOT does not exist."
  echo "Mount your persistent disk first, then rerun."
  exit 1
fi

mkdir -p "$PROJECT_DIR"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  if [[ -z "$REPO_URL" ]]; then
    echo "Project repo not found at $PROJECT_DIR."
    read -r -p "Enter Git repo URL to clone: " REPO_URL
  fi
  echo "Cloning repo..."
  git clone "$REPO_URL" "$PROJECT_DIR"
else
  echo "Repo already present. Pulling latest..."
  git -C "$PROJECT_DIR" pull --ff-only || true
fi

cd "$PROJECT_DIR"

echo "Creating virtualenv at $VENV_DIR..."
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing Python deps..."
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

mkdir -p checkpoints results/eval logs

ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<'EOF'
# Fill values below (or set via script prompts)
wandb_api_key=
hf_token=
gemini_api_key=
EOF
  echo "Created $ENV_FILE"
fi

echo
read -r -p "Set W&B API key now? [y/N]: " SET_WANDB
if [[ "${SET_WANDB,,}" == "y" ]]; then
  read -r -s -p "Enter W&B API key: " WANDB_KEY
  echo
  if [[ -n "${WANDB_KEY:-}" ]]; then
    wandb login "$WANDB_KEY"
    sed -i.bak "s|^wandb_api_key=.*|wandb_api_key=${WANDB_KEY}|g" "$ENV_FILE" || true
  fi
fi

read -r -p "Set Hugging Face token now? [y/N]: " SET_HF
if [[ "${SET_HF,,}" == "y" ]]; then
  read -r -s -p "Enter HF token: " HF_KEY
  echo
  if [[ -n "${HF_KEY:-}" ]]; then
    hf auth login --token "$HF_KEY"
    sed -i.bak "s|^hf_token=.*|hf_token=${HF_KEY}|g" "$ENV_FILE" || true
  fi
fi

read -r -p "Set Gemini API key in .env now? [y/N]: " SET_GEM
if [[ "${SET_GEM,,}" == "y" ]]; then
  read -r -s -p "Enter Gemini API key: " GEM_KEY
  echo
  if [[ -n "${GEM_KEY:-}" ]]; then
    sed -i.bak "s|^gemini_api_key=.*|gemini_api_key=${GEM_KEY}|g" "$ENV_FILE" || true
  fi
fi

echo
echo "=== Setup Complete ==="
echo "Project: $PROJECT_DIR"
echo "Venv   : $VENV_DIR"
echo
echo "Next commands:"
cat <<'EOF'
source venv/bin/activate
python smoke_test.py
python scripts/stress_execution.py --rounds-clean 10 --rounds-mixed 5 --batch-size 32 --workers 16 --timeout 5
python eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --save-debug-details
# then training:
python train.py
EOF

