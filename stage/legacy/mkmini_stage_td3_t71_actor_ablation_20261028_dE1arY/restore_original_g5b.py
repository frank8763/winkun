from pathlib import Path, PurePosixPath
import hashlib
import zipfile
import tempfile
import shutil
import subprocess
import importlib.util
import json
import os
import sys
import traceback

H = Path.home()
JOB = Path(os.environ["T71_JOB"])

NAME = "stage_daad_g5b_selector_recovery_v1"
DEST = Path("/tmp") / NAME

BACKUPS = [
    H / "下载" / (NAME + ".zip"),
    H / "Downloads" / (NAME + ".zip"),
]

EXPECTED = (
    "d218726bf0b08c19d0fbfee0e09821f6"
    "2b6c0f95fb3575dfa0d45855880bb5f6"
)

G20 = H / (
    "mkmini_stage_g20f6c19_turn_learning_"
    "20261022_221621_577031_v1/"
    "stage_c13_B30/"
    "train_g20f6c13_global_navigation.py"
)

DAAD = (
    H / "stage_daad_g4h_multibranch_night_v1/runner"
)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    return digest(Path(path).read_bytes())

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

def report(label, value):
    print(f"{label}={value}", flush=True)

def main():
    print("=" * 75)
    print("T71 ORIGINAL G5B RECOVERY — STAGE ONLY")
    print("=" * 75)

    report("TRAINING", False)
    report("FROZEN_MODEL_WRITE", False)

    backup = next(
        (p for p in BACKUPS if p.is_file()),
        None
    )

    check(
        backup is not None,
        "ORIGINAL_ZIP_NOT_FOUND"
    )

    report("ORIGINAL_BACKUP", backup)

    with zipfile.ZipFile(backup) as archive:
        bad = archive.testzip()

        check(
            bad is None,
            "ORIGINAL_ZIP_CRC_FAILED:" + str(bad)
        )

        prefix = NAME + "/"
        entries = []

        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)

            # 只恢复这个Stage模块目录。
            if not name.startswith(prefix):
                continue

            check(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in name,
                "UNSAFE_ARCHIVE_PATH:" + name
            )

            mode = (info.external_attr >> 16) & 0o170000

            check(
                mode != 0o120000,
                "ARCHIVE_SYMLINK_NOT_ALLOWED"
            )

            check(
                info.file_size <= 100 * 1024 * 1024,
                "ARCHIVE_FILE_TOO_LARGE"
            )

            entries.append(info)

        original_name = prefix + "g5b.py"

        check(
            sum(x.filename == original_name for x in entries) == 1,
            "ORIGINAL_G5B_NOT_IN_ZIP"
        )

        original_bytes = archive.read(original_name)
        original_sha = digest(original_bytes)

        report("BACKUP_G5B_SHA256", original_sha)

        check(
            original_sha == EXPECTED,
            "ORIGINAL_G5B_HASH_MISMATCH"
        )

        report("ORIGINAL_BACKUP_HASH_GATE", "PASS")

        if DEST.exists():
            check(
                (DEST / "g5b.py").is_file(),
                "DEST_EXISTS_WITHOUT_G5B"
            )

            check(
                sha(DEST / "g5b.py") == EXPECTED,
                "DEST_G5B_HASH_CONFLICT"
            )

            report("RESTORE", "ALREADY_VALID")

        else:
            temporary = Path(
                tempfile.mkdtemp(
                    prefix="t71_native_restore_",
                    dir="/tmp"
                )
            )

            try:
                for info in entries:
                    target = temporary / info.filename

                    if info.is_dir():
                        target.mkdir(
                            parents=True,
                            exist_ok=True
                        )
                        continue

                    target.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    target.write_bytes(
                        archive.read(info)
                    )

                staged = temporary / NAME

                check(
                    staged.is_dir()
                    and sha(staged / "g5b.py") == EXPECTED,
                    "STAGED_G5B_INVALID"
                )

                check(
                    not DEST.exists(),
                    "RESTORE_DEST_APPEARED"
                )

                staged.rename(DEST)

                report("RESTORE", "ORIGINAL_ARCHIVE")

            finally:
                shutil.rmtree(
                    temporary,
                    ignore_errors=True
                )

    check(
        (DEST / "g5b.py").is_file()
        and sha(DEST / "g5b.py") == EXPECTED,
        "RESTORED_G5B_INVALID"
    )

    report("RESTORED_G5B", DEST / "g5b.py")
    report("RESTORED_G5B_HASH_GATE", "PASS")

    # 保持原Stage加载路径及依赖。
    env = os.environ.copy()

    env["PYTHONPATH"] = os.pathsep.join([
        str(H / "catkin_ws/src/mk-mini"),
        str(DEST),
        str(DAAD),
        env.get("PYTHONPATH", "")
    ])

    probe = r'''
from pathlib import Path
import importlib.util
import hashlib
import sys

p = Path(sys.argv[1])
expected = sys.argv[2]

spec = importlib.util.spec_from_file_location(
    "t71_original_stage_native", str(p)
)

assert spec is not None and spec.loader is not None

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

native = mod.load_native()

assert len(native) == 5

g5b = native[3][0]

actual = hashlib.sha256(
    Path(g5b.__file__).read_bytes()
).hexdigest()

print("NATIVE_G5B_SHA256=", actual, flush=True)

assert actual == expected

print("T71_ORIGINAL_NATIVE_GATE=PASS", flush=True)
'''

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            probe,
            str(G20),
            EXPECTED
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False
    )

    report("NATIVE_LOAD_RETURN_CODE", result.returncode)

    print(result.stdout[-12000:], flush=True)

    if result.stderr:
        print(result.stderr[-12000:], flush=True)

    check(
        result.returncode == 0
        and "T71_ORIGINAL_NATIVE_GATE=PASS"
            in result.stdout,
        "ORIGINAL_NATIVE_LOAD_FAILED"
    )

    # 确认T71两组初始检查点未变化。
    for label in ("T71A", "T71B"):
        folder = JOB / label

        status = json.loads(
            (folder / "status.json").read_text()
        )

        meta = json.loads(
            (folder / "prepare.json").read_text()
        )

        ck = (
            folder / "checkpoints"
            / status["checkpoint"]
        )

        check(
            status["state"] == "RUNNING"
            and int(status["frames"]) == 134491
            and ck.is_file()
            and sha(ck) == status["checkpoint_sha256"],
            label + "_INITIAL_CHECKPOINT_INVALID"
        )

        for filename, expected in meta["files"].items():
            path = folder / filename

            check(
                path.is_file()
                and sha(path) == expected,
                label + "_SOURCE_INTEGRITY_FAILED:"
                + filename
            )

        report(
            label + "_FROZEN_START_GATE",
            "PASS"
        )

    runner = JOB / "runner.py"

    check(
        "SUBPROCESS_FAILED_OR_INTERNAL_ERROR:"
        in runner.read_text(),
        "RUNNER_ERROR_GUARD_NOT_REPAIRED"
    )

    (JOB / "verified_native_pythonpath.txt").write_text(
        env["PYTHONPATH"] + "\n"
    )

    report("T71_G5B_RECOVERY", "PASS")
    report("T71_NATIVE_PREFLIGHT", "PASS")
    report("TRAINING_LAUNCHED", False)
    report("INTERACTIVE_TERMINAL_REMAINS_OPEN", True)

try:
    main()

except Exception as error:
    print(
        "T71_RECOVERY_BLOCKED=",
        repr(error),
        flush=True
    )
    traceback.print_exc()
    print(
        "TRAINING_LAUNCHED=FALSE",
        flush=True
    )
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
        flush=True
    )
