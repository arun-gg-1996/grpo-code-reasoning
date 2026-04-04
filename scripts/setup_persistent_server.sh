#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for a fresh Ubuntu GPU server.
# - Installs required system packages
# - Creates/uses project directory
# - Clones repo (or updates if already present)
# - Creates venv with Python 3.10+ and installs requirements
# - Prompts for W&B / HF / Gemini keys (optional)
#
# Usage examples:
#   bash scripts/setup_persistent_server.sh
#   REPO_URL=https://github.com/<you>/<repo>.git bash scripts/setup_persistent_server.sh
#   PERSIST_ROOT=/mnt/persist PROJECT_DIR=/mnt/persist/grpo-code-reasoning bash scripts/setup_persistent_server.sh

PERSIST_ROOT="${PERSIST_ROOT:-/mnt/persist}"
PROJECT_DIR="${PROJECT_DIR:-$PERSIST_ROOT/grpo-code-reasoning}"
REPO_URL="${REPO_URL:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

APT_UPDATED=0
apt_update_once() {
  if [[ "$APT_UPDATED" -eq 1 ]]; then
    return 0
  fi
  if need_cmd sudo; then
    sudo apt-get update
  else
    apt-get update
  fi
  APT_UPDATED=1
}

apt_install_pkg() {
  local pkg="$1"
  apt_update_once
  if need_cmd sudo; then
    sudo apt-get install -y "$pkg"
  else
    apt-get install -y "$pkg"
  fi
}

ensure_system_deps() {
  local missing=()
  need_cmd git || missing+=("git")
  need_cmd python3 || missing+=("python3")
  need_cmd python3-venv || true  # this binary may not exist even when module exists
  if ! python3 -c "import venv" >/dev/null 2>&1; then
    missing+=("python3-venv")
  fi
  need_cmd pip3 || missing+=("python3-pip")

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Installing system packages: ${missing[*]}"
    apt_update_once
    if need_cmd sudo; then
      sudo apt-get install -y "${missing[@]}"
    else
      apt-get install -y "${missing[@]}"
    fi
  fi
}

select_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  for p in python3.12 python3.11 python3.10 python3; do
    if need_cmd "$p"; then
      echo "$p"
      return
    fi
  done
  echo "python3"
}

check_python_version() {
  local py="$1"
  "$py" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    raise SystemExit("ERROR: Python 3.10+ is required.")
print(f"Using Python {major}.{minor}")
PY
}

python_xy_version() {
  local py="$1"
  "$py" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

ensure_python_venv_support() {
  local py="$1"
  local pyver
  pyver="$(python_xy_version "$py")"
  local candidates=("python${pyver}-venv" "python3-venv")

  echo "Installing missing venv support for Python ${pyver}..."
  local installed=0
  for pkg in "${candidates[@]}"; do
    if apt_install_pkg "$pkg"; then
      echo "Installed: $pkg"
      installed=1
      break
    fi
    echo "Package unavailable or failed: $pkg"
  done

  if [[ "$installed" -ne 1 ]]; then
    echo "ERROR: Unable to install Python venv support automatically."
    echo "Tried: ${candidates[*]}"
    echo "Please install manually, then rerun:"
    echo "  sudo apt-get install -y python${pyver}-venv"
    echo "or"
    echo "  sudo apt-get install -y python3-venv"
    exit 1
  fi
}

create_venv_with_repair() {
  local py="$1"
  local venv_dir="$2"
  local err_log
  err_log="$(mktemp)"

  if "$py" -m venv "$venv_dir" 2>"$err_log"; then
    rm -f "$err_log"
    return 0
  fi

  if grep -qiE "ensurepip is not available|No module named ensurepip" "$err_log"; then
    cat "$err_log"
    ensure_python_venv_support "$py"
    rm -rf "$venv_dir"
    "$py" -m venv "$venv_dir"
    rm -f "$err_log"
    return 0
  fi

  echo "ERROR: Failed to create virtualenv."
  cat "$err_log"
  rm -f "$err_log"
  exit 1
}

echo "=== GRPO Server Setup ==="
echo "PERSIST_ROOT: $PERSIST_ROOT"
echo "PROJECT_DIR : $PROJECT_DIR"
echo

ensure_system_deps
PYTHON_BIN="$(select_python)"
check_python_version "$PYTHON_BIN"

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
create_venv_with_repair "$PYTHON_BIN" "$VENV_DIR"
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
echo "Python : $PYTHON_BIN"
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
