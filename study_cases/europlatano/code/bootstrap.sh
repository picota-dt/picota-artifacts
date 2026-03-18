#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${CODE_DIR}/.venv"

INSTALL_ML_DEPS=1
CPU_TORCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-ml-deps)
      INSTALL_ML_DEPS=0
      shift
      ;;
    --cpu-torch)
      CPU_TORCH=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--skip-ml-deps] [--cpu-torch]" >&2
      exit 1
      ;;
  esac
done

echo "[bootstrap] code root: ${CODE_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[bootstrap] Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo "[bootstrap] Upgrading pip/setuptools/wheel"
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_ML_DEPS}" -eq 1 ]]; then
  echo "[bootstrap] Installing runtime dependencies (numpy/scipy/shap)"
  "${VENV_PIP}" install numpy scipy shap

  if ! "${VENV_PY}" -c "import torch" >/dev/null 2>&1; then
    echo "[bootstrap] Installing PyTorch"
    if [[ "${CPU_TORCH}" -eq 1 ]]; then
      "${VENV_PIP}" install torch --index-url https://download.pytorch.org/whl/cpu
    else
      "${VENV_PIP}" install torch
    fi
  else
    echo "[bootstrap] PyTorch already installed"
  fi
fi

echo "[bootstrap] Verifying imports"
(
  cd "${CODE_DIR}"
  "${VENV_PY}" - <<'PY'
import importlib

for mod in [
    "EuroplatanoRuleCatalog",
    "KanTrainer",
    "MetamorphicAlternativeKanTrainer",
    "TabNetModel",
    "TabNetTrainer",
    "metamorphic_evaluation",
]:
    importlib.import_module(mod)
print("Local case imports OK")
PY
)

echo "[bootstrap] Done"
echo "[bootstrap] Activate with: source ${VENV_DIR}/bin/activate"
