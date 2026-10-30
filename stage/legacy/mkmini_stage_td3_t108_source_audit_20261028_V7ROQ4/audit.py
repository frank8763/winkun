from pathlib import Path
import hashlib
import json
import traceback

H = Path.home()

patterns = [
    "mkmini_stage_td3_t87*",
    "mkmini_stage_td3_t90s*",
    "mkmini_stage_td3_t102_policy_interpolation_20261028_pklqP9",
]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

try:
    print("=" * 76)
    print("T108 — STAGE TD3 TRAINING SOURCE AUDIT")
    print("=" * 76)
    print("TRAINING=FALSE")
    print("SIMULATION=FALSE")
    print("MODEL_WRITE=FALSE")
    print("MODEL_PROMOTION=FALSE")

    roots = []
    for pattern in patterns:
        roots.extend(sorted(H.glob(pattern)))

    roots = list(dict.fromkeys(roots))

    if not roots:
        print("SOURCE_ROOTS_NOT_FOUND")
    else:
        for root in roots:
            print("\nROOT=", root)

    files = []
    for root in roots:
        if not root.is_dir():
            continue

        for path in root.rglob("train.py"):
            if any(
                part in {
                    "checkpoints", "__pycache__",
                    "archive", "backup"
                }
                for part in path.parts
            ):
                continue

            try:
                if path.stat().st_size > 3_000_000:
                    continue

                source = path.read_text(errors="replace")

                if (
                    "_goal_state" in source
                    and "replay.append" in source
                ):
                    files.append((path, source))

            except Exception as exc:
                print("SOURCE_READ_WARNING=", path, repr(exc))

    if not files:
        print("TRAINING_SOURCES_NOT_FOUND")
    else:
        for path, source in files:
            lines = source.splitlines()

            print("\n" + "=" * 76)
            print("SOURCE=", path)
            print("SHA256=", sha(path))
            print("LINES=", len(lines))

            markers = [
                "env.step(action)",
                "step_checked",
                "_goal_state",
                "_angle_before",
                "replay.append(",
                "agent.update(",
            ]

            for marker in markers:
                found = [
                    i + 1 for i, line in enumerate(lines)
                    if marker in line
                ]
                if found:
                    print(
                        "MARKER=", marker,
                        "LINES=", found[:20]
                    )

            replay_positions = [
                i for i, line in enumerate(lines)
                if "replay.append(" in line
            ]

            selected = None
            for pos in replay_positions:
                start = max(0, pos - 130)
                section = "\n".join(lines[start:pos + 8])

                if (
                    "_goal_state" in section
                    and (
                        "env.step" in section
                        or "step_checked" in section
                    )
                ):
                    selected = (start, min(len(lines), pos + 8))
                    break

            if selected is None:
                print("REWARD_LOOP_ANCHOR_NOT_FOUND")
                continue

            start, end = selected

            print("\n--- REWARD / STEP / REPLAY CONTEXT ---")
            for i in range(start, end):
                print(f"{i + 1:05d}: {lines[i]}")

    b35 = (
        H / "mkmini_stage_td3_t102_policy_interpolation_"
        "20261028_pklqP9/T102_B35"
    )

    print("\n" + "=" * 76)
    print("T102-B35 CANDIDATE IDENTITY")
    print("=" * 76)

    for name in ("identity.json", "manifest.json"):
        path = b35 / name
        if path.is_file():
            print("FILE=", path)
            print(path.read_text(errors="replace")[:12000])

    candidate = b35 / "candidate.pt"
    if candidate.is_file():
        print("CANDIDATE_PATH=", candidate)
        print("CANDIDATE_SHA256=", sha(candidate))

    print("\nT108_SOURCE_AUDIT=COMPLETE")

except Exception as exc:
    print("T108_SOURCE_AUDIT_BLOCKED=", repr(exc))
    traceback.print_exc()

finally:
    print("TRAINING_LAUNCHED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
