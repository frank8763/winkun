from pathlib import Path
import os
import sys
import json
import hashlib
import subprocess
import importlib.machinery

H = Path.home()
JOB = Path(os.environ["T71_JOB"])

TMP = Path("/tmp/stage_daad_g5b_selector_recovery_v1")
DAAD = H / "stage_daad_g4h_multibranch_night_v1/runner"

G20 = H / (
    "mkmini_stage_g20f6c19_turn_learning_"
    "20261022_221621_577031_v1/"
    "stage_c13_B30/"
    "train_g20f6c13_global_navigation.py"
)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def info(s):
    print(s, flush=True)

def heading(s):
    print("\n" + "=" * 75)
    print(s)
    print("=" * 75)

heading("T71 NATIVE STAGE DEPENDENCY VERIFICATION")

info("STAGE_ONLY=TRUE")
info("TRAINING=FALSE")
info("MODEL_MODIFIED=FALSE")

heading("1. EXACT ORIGINAL DEPENDENCY LOCATIONS")

for p in (TMP, DAAD):
    info(f"PATH={p}")
    info(f"EXISTS={p.exists()}")
    info(f"IS_SYMLINK={p.is_symlink()}")

    if p.exists() and p.is_dir():
        for f in sorted(p.iterdir())[:35]:
            info(f"CHILD={f.name}")

heading("2. REAL G5B FILE SEARCH")

roots = [TMP, DAAD]

roots.extend(
    p for p in H.glob("stage_daad*")
    if p.is_dir()
)

roots.extend(
    p for p in H.glob("mkmini_stage_g5*")
    if p.is_dir()
)

roots.extend(
    p for p in Path("/tmp").glob("stage_daad*")
    if p.is_dir()
)

roots = sorted(set(roots))

suffixes = tuple(
    importlib.machinery.EXTENSION_SUFFIXES
)

found = {}

for root in roots:
    if not root.is_dir():
        continue

    for current, dirs, files in os.walk(root):
        here = Path(current)

        depth = len(here.relative_to(root).parts)

        dirs[:] = [
            name for name in dirs
            if name not in (
                ".git", "__pycache__",
                "checkpoints", "rootfs",
                "site-packages", "node_modules"
            )
        ]

        if depth >= 10:
            dirs[:] = []

        for name in files:
            valid = (
                name == "g5b.py"
                or (
                    name.startswith("g5b.")
                    and name.endswith(suffixes)
                )
                or (
                    here.name == "g5b"
                    and name == "__init__.py"
                )
            )

            if not valid:
                continue

            p = here / name
            if not p.is_file():
                continue

            parent = (
                here.parent
                if name == "__init__.py"
                else here
            )

            found[str(p.resolve())] = {
                "path": str(p),
                "parent": str(parent),
                "sha256": sha(p),
            }

for item in found.values():
    info("REAL_MODULE=" + json.dumps(item))

info(f"REAL_MODULE_COUNT={len(found)}")

heading("3. ORIGINAL STAGE LOAD CHAIN")

if len(found) != 1:
    info("T71_NATIVE_GATE=BLOCKED")
    info("REASON=REAL_G5B_NOT_UNIQUELY_VERIFIED")
    info("TRAINING_LAUNCHED=FALSE")
    info("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")

else:
    item = next(iter(found.values()))

    original_py = os.environ.get("PYTHONPATH", "")

    candidate_py = (
        item["parent"]
        + os.pathsep
        + original_py
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = candidate_py

    probe = r'''
import sys
import hashlib
import importlib.util
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]

spec = importlib.util.spec_from_file_location(
    "t71_verified_native_g20", str(path)
)

assert spec is not None and spec.loader is not None

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

native = mod.load_native()
assert len(native) == 5

assert "g5b" in sys.modules

g5b = sys.modules["g5b"]
actual = Path(g5b.__file__)

h = hashlib.sha256(actual.read_bytes()).hexdigest()

print("NATIVE_G5B_FILE=", actual, flush=True)
print("NATIVE_G5B_SHA256=", h, flush=True)

assert h == expected, "NATIVE_G5B_SHA_MISMATCH"

print("T71_NATIVE_LOAD_GATE=PASS", flush=True)
'''

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                probe,
                str(G20),
                item["sha256"],
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )

        info("NATIVE_PROBE_RC=" + str(result.returncode))

        for line in result.stdout.splitlines()[-45:]:
            info(line)

        native_pass = (
            result.returncode == 0
            and "T71_NATIVE_LOAD_GATE=PASS"
                in result.stdout
        )

    except Exception as e:
        info("NATIVE_PROBE_ERROR=" + repr(e))
        native_pass = False

    heading("4. T71 CHECKPOINT AND SOURCE INTEGRITY")

    states_ok = True

    for label in ("T71A", "T71B"):
        folder = JOB / label

        try:
            status = json.loads(
                (folder / "status.json").read_text()
            )

            meta = json.loads(
                (folder / "prepare.json").read_text()
            )

            ck = (
                folder
                / "checkpoints"
                / status["checkpoint"]
            )

            good = (
                status["state"] == "RUNNING"
                and int(status["frames"]) == 134491
                and ck.is_file()
                and sha(ck) == status["checkpoint_sha256"]
                and all(
                    (folder / filename).is_file()
                    and sha(folder / filename) == digest
                    for filename, digest
                    in meta["files"].items()
                )
            )

        except Exception as e:
            info(f"{label}_CHECK_ERROR={e!r}")
            good = False

        states_ok = states_ok and good
        info(f"{label}_RESUME_INPUT_PASS={good}")

    guard = JOB / "runner.py"

    guard_ok = (
        guard.is_file()
        and "SUBPROCESS_FAILED_OR_INTERNAL_ERROR:"
            in guard.read_text()
    )

    info(f"T71_ERROR_GUARD_PASS={guard_ok}")

    heading("5. FINAL VERDICT")

    if native_pass and states_ok and guard_ok:
        (JOB / "verified_native_pythonpath.txt").write_text(
            candidate_py + "\n"
        )
        info("T71_RECOVERY_PREFLIGHT=PASS")
        info("NEXT=RESUME_EXISTING_T71_AB_EXPERIMENT")
    else:
        info("T71_RECOVERY_PREFLIGHT=BLOCKED")
        info("NEXT=REPAIR_EXACT_FAILED_CHECK")

    info("TRAINING_LAUNCHED=FALSE")
    info("MODEL_MODIFIED=FALSE")
    info("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
