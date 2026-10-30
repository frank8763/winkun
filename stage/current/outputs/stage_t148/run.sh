#!/usr/bin/env bash
set -euo pipefail
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$job_dir/process.lock"
flock -n 9 || { echo 'T148_ALREADY_RUNNING'; exit 1; }
cd /home/linghu/catkin_ws/src/mk-mini
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONPATH=/home/linghu/catkin_ws/src/mk-mini
unset T148_RESTORE_ONLY
python_bin=/home/linghu/miniconda3/envs/mk_mujoco/bin/python
"$python_bin" -u "$job_dir/verify_resume.py"
unset T148_RESTORE_ONLY
"$python_bin" -u "$job_dir/train.py"
"$python_bin" -u "$job_dir/evaluate.py"
