#!/usr/bin/env bash
set -euo pipefail
STAGE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export STAGE_REPO_ROOT
export STAGE_RUN_ROOT="$STAGE_REPO_ROOT/stage/runs"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$STAGE_REPO_ROOT/stage/runtime/mk-mini${PYTHONPATH:+:$PYTHONPATH}"
STAGE_PYTHON_BIN="${STAGE_PYTHON:-/home/linghu/miniconda3/envs/mk_mujoco/bin/python}"
if [[ ! -x "$STAGE_PYTHON_BIN" ]]; then STAGE_PYTHON_BIN="$(command -v python3)"; fi
cd "$STAGE_REPO_ROOT"
if [[ $# -eq 0 ]]; then
  exec "$STAGE_PYTHON_BIN" stage/tools/check_local_runtime.py
fi
exec "$STAGE_PYTHON_BIN" "$@"
