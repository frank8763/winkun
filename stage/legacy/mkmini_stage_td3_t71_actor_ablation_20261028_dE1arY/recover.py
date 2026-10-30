from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

H = Path.home()
JOB = Path(os.environ["T71_JOB"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fail(message):
    print("T71_RECOVERY_BLOCKED=", message, flush=True)
    raise RuntimeError(message)


print("T71_STAGE_ONLY=TRUE")
print("T71_READ_ONLY_FROZEN=TRUE")

# 1. 打印真实异常位置，而不是只看最后一行。
for name in ("T71A", "T71B"):
    log = JOB / name / "train.log"

    print("\n==========", name, "原始异常 ==========")

    if log.is_file():
        lines = log.read_text(errors="replace").splitlines()
        for line in lines[-65:]:
            print(line)
    else:
        fail(name + "_TRAIN_LOG_MISSING")

# 2. 确认原有分支没有被部分训练。
for name in ("T71A", "T71B"):
    folder = JOB / name

    status = json.loads(
        (folder / "status.json").read_text()
    )

    meta = json.loads(
        (folder / "prepare.json").read_text()
    )

    if (
        status.get("state") != "RUNNING"
        or int(status.get("frames", -1)) != 134491
    ):
        fail(name + "_STATE_NOT_INITIAL")

    checkpoint = (
        folder / "checkpoints"
        / status["checkpoint"]
    )

    if (
        not checkpoint.is_file()
        or sha(checkpoint)
            != status["checkpoint_sha256"]
    ):
        fail(name + "_CHECKPOINT_CHANGED")

    for filename, expected in meta["files"].items():
        path = folder / filename

        if not path.is_file() or sha(path) != expected:
            fail(name + "_SOURCE_CHANGED:" + filename)

    print(name + "_INITIAL_STATE=PASS")

# 3. 检查原Stage源码中对g5b的引用。
sources = [
    JOB / "T71A/train.py",
    JOB / "T71A/td3_core_goal_contrast.py",
    H / (
        "mkmini_stage_td3_t67_3m_nightrelay_"
        "20261027_kEPAW0/T67C022/train.py"
    ),
    H / (
        "mkmini_stage_td3_independent_"
        "20261027_105202_577031_v1/td3_smoke.py"
    ),
]

print("\n========== g5b源码引用 ==========")

for path in sources:
    if not path.is_file():
        continue

    lines = path.read_text(errors="replace").splitlines()

    for index, line in enumerate(lines):
        if (
            "g5b" in line.lower()
            or "sys.path.insert" in line
            or "sys.modules[" in line
        ):
            print(
                path.name,
                "LINE", index + 1,
                line.strip()
            )

# 4. 查询当前环境能否找到g5b。
spec = importlib.util.find_spec("g5b")
paths = []

if spec is not None:
    print("G5B_ALREADY_IMPORTABLE=", spec.origin)

else:
    print("\n========== Stage本地模块搜索 ==========")

    roots = [
        path for path in H.iterdir()
        if path.is_dir()
        and path.name.startswith("mkmini_stage_")
    ]

    roots.append(H / "catkin_ws/src/mk-mini")

    for root in roots:
        if not root.exists():
            continue

        for current, dirs, files in os.walk(root):
            current_path = Path(current)

            depth = len(
                current_path.relative_to(root).parts
            )

            if depth >= 7:
                dirs[:] = []
                continue

            dirs[:] = [
                d for d in dirs
                if d not in (
                    "__pycache__", ".git",
                    "checkpoints", "logs"
                )
            ]

            if "g5b.py" in files:
                paths.append(current_path)

            if (
                current_path.name == "g5b"
                and "__init__.py" in files
            ):
                paths.append(current_path.parent)

    paths = sorted(set(paths))

    for path in paths:
        print("G5B_SOURCE_CANDIDATE=", path)

    if len(paths) != 1:
        fail(
            "G5B_SOURCE_NOT_UNAMBIGUOUS:"
            + str(len(paths))
        )

    os.environ["PYTHONPATH"] = (
        str(paths[0])
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )

# 5. 在独立Python进程中检验真实导入。
test = subprocess.run(
    [
        sys.executable,
        "-c",
        "import g5b; print('G5B_IMPORT_PASS', "
        "getattr(g5b, '__file__', None))"
    ],
    env=os.environ.copy(),
    capture_output=True,
    text=True,
    timeout=30,
    check=False
)

print("G5B_IMPORT_STDOUT=", test.stdout.strip())
print("G5B_IMPORT_STDERR=", test.stderr.strip())

if test.returncode != 0:
    fail("G5B_REAL_IMPORT_FAILED")

# 6. 修复已确认的异常检测缺陷。
runner = JOB / "runner.py"
source = runner.read_text()

old = '''    check(
        result.returncode == 0,
        "SUBPROCESS_FAILED:" + str(script)
    )'''

new = '''    output_text = Path(logfile).read_text(
        errors="replace"
    )

    check(
        result.returncode == 0
        and "TD3_T6_BLOCKED=" not in output_text
        and "T71E_BLOCKED=" not in output_text,
        "SUBPROCESS_FAILED_OR_INTERNALLY_BLOCKED:"
        + str(script)
    )'''

if source.count(old) != 1:
    fail("RUNNER_GUARD_PATCH_ANCHOR_INVALID")

import ast
patched = source.replace(old, new, 1)
ast.parse(patched)

print("RUNNER_ORIGINAL_SHA256=", sha(runner))

backup = JOB / "runner.initial.py"

if backup.exists():
    fail("RUNNER_BACKUP_ALREADY_EXISTS")

shutil.copy2(runner, backup)
runner.write_text(patched)

print("RUNNER_PATCHED_SHA256=", sha(runner))

# 7. 保留首次失败日志。
for name in ("T71A", "T71B"):
    original = JOB / name / "train.log"
    archive = JOB / name / "initial_import_failure.log"

    if not archive.exists():
        shutil.copy2(original, archive)

# 8. 把经过验证的模块路径交给启动命令。
env_file = JOB / "verified_pythonpath.txt"

env_file.write_text(
    os.environ["PYTHONPATH"] + "\n"
)

print("T71_G5B_IMPORT_VERIFIED=PASS")
print("T71_INITIAL_CHECKPOINTS=PASS")
print("T71_ERROR_GUARD_PATCHED=PASS")
print("T71_RECOVERY_PREFLIGHT=PASS")
print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
