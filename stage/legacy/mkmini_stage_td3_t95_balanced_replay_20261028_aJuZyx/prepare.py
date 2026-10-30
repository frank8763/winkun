from pathlib import Path
import os
import ast
import json
import hashlib
import traceback

import numpy as np
import torch

H = Path.home()
JOB = Path(os.environ["T95_JOB"])

SOURCES = {
    "RIGHT": (
        H / (
            "mkmini_stage_td3_t87_direction_learning_"
            "20261028_Mnr36j/checkpoints/episode_0023.pt"
        ),
        "e29e3ccc177c7c8487b3ea9c99b72fc678"
        "b5255891dcea34a6f5a42ab6619ba9"
    ),
    "LEFT": (
        H / (
            "mkmini_stage_td3_t87_direction_learning_"
            "20261028_Mnr36j/checkpoints/episode_0025.pt"
        ),
        "0378e4af4296683d5d612a2d71607149"
        "e8fb3b0ff5235e35643766059ffaf94e"
    )
}

TRAIN_SOURCE = H / (
    "mkmini_stage_td3_t87_direction_learning_"
    "20261028_Mnr36j/train.py"
)

SAMPLES_PER_SIDE = 7000
RNG = np.random.default_rng(2026102895)

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()

def collect(label, checkpoint):
    selected = []
    raw = checkpoint["replay"]

    for row in raw:
        if not isinstance(row, (tuple, list)):
            continue

        if len(row) < 5:
            continue

        try:
            state = np.asarray(
                row[0], dtype=np.float32
            ).reshape(-1)

            action = np.asarray(
                row[1], dtype=np.float32
            ).reshape(-1)

            next_state = np.asarray(
                row[3], dtype=np.float32
            ).reshape(-1)

            if (
                state.shape != (112,)
                or action.shape != (2,)
                or next_state.shape != (112,)
            ):
                continue

            if not (
                np.isfinite(state).all()
                and np.isfinite(action).all()
                and np.isfinite(next_state).all()
            ):
                continue

            angle = np.degrees(
                np.arctan2(
                    float(state[49]),
                    float(state[50])
                )
            )

            distance = float(state[48]) * 10.0

            if not (0.5 <= distance <= 3.5):
                continue

            if label == "RIGHT":
                accepted = 12.0 <= angle < 75.0
            else:
                accepted = -75.0 < angle <= -8.0

            if accepted:
                selected.append(row)

        except (ValueError, TypeError, IndexError):
            continue

    return selected

try:
    print("=" * 78)
    print("T95 — STAGE TD3 PERSISTENT BALANCED REPLAY ARCHIVE")
    print("=" * 78)

    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("SIMULATION=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("HISTORICAL_MODEL_WRITE=FALSE")

    pools = {}
    provenance = {}

    for label, (path, expected_sha) in SOURCES.items():

        check(
            path.is_file(),
            "CHECKPOINT_MISSING:" + label
        )

        check(
            sha(path) == expected_sha,
            "CHECKPOINT_IDENTITY_FAILED:" + label
        )

        saved = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        pool = collect(label, saved)

        print(
            "SOURCE=", label,
            "AVAILABLE_TRANSITIONS=", len(pool),
            "FRAMES=", saved.get("frames")
        )

        check(
            len(pool) >= SAMPLES_PER_SIDE,
            "SIDE_COVERAGE_INSUFFICIENT:" + label
        )

        ids = RNG.choice(
            len(pool),
            SAMPLES_PER_SIDE,
            replace=False
        )

        pools[label] = [
            pool[int(i)] for i in ids
        ]

        provenance[label] = {
            "source": str(path),
            "source_sha256": expected_sha,
            "source_frames": int(saved["frames"]),
            "available_transitions": len(pool),
            "archived_transitions": len(pools[label])
        }

        del saved

    check(
        len(pools["LEFT"]) == SAMPLES_PER_SIDE
        and len(pools["RIGHT"]) == SAMPLES_PER_SIDE,
        "BALANCE_INVALID"
    )

    archive_path = JOB / "balanced_archive.pt"

    check(
        not archive_path.exists(),
        "ARCHIVE_ALREADY_EXISTS"
    )

    torch.save(
        {
            "LEFT": pools["LEFT"],
            "RIGHT": pools["RIGHT"],
            "provenance": provenance
        },
        archive_path
    )

    archive_sha = sha(archive_path)

    # 重新加载，确保归档可供后续训练程序使用。
    verified = torch.load(
        archive_path,
        map_location="cpu",
        weights_only=False
    )

    check(
        len(verified["LEFT"]) == SAMPLES_PER_SIDE
        and len(verified["RIGHT"]) == SAMPLES_PER_SIDE,
        "ARCHIVE_RELOAD_FAILED"
    )

    manifest = {
        "experiment": "STAGE_TD3_T95",
        "status": "ARCHIVE_PREPARED_NOT_TRAINED",
        "left_transitions": SAMPLES_PER_SIDE,
        "right_transitions": SAMPLES_PER_SIDE,
        "total_transitions": 2 * SAMPLES_PER_SIDE,
        "archive_sha256": archive_sha,
        "provenance": provenance,
        "automatic_promotion": False
    }

    (JOB / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("\nBALANCED_ARCHIVE=PASS")
    print("LEFT_COUNT=", len(verified["LEFT"]))
    print("RIGHT_COUNT=", len(verified["RIGHT"]))
    print("ARCHIVE_SHA256=", archive_sha)

    # 检查训练器现有采样位置。
    # 只显示源码，不修改训练器。
    print("\n" + "=" * 78)
    print("TD3 TRAINING SAMPLE INTERFACE")
    print("=" * 78)

    check(
        TRAIN_SOURCE.is_file(),
        "TRAIN_SOURCE_MISSING"
    )

    source = TRAIN_SOURCE.read_text()
    ast.parse(source)

    lines = source.splitlines()

    print(
        "TRAIN_SOURCE_SHA256=",
        sha(TRAIN_SOURCE)
    )

    positions = [
        i for i, line in enumerate(lines)
        if (
            "for row in samples" in line
            or "samples =" in line
            or "agent.update(" in line
        )
    ]

    for i in positions[-8:]:
        print("\nCODE_AT_LINE=", i + 1)

        for j in range(
            max(0, i - 5),
            min(len(lines), i + 7)
        ):
            print(
                f"{j + 1:05d}: {lines[j]}"
            )

    print("\nT95_ARCHIVE_PREFLIGHT=PASS")
    print("T95_TRAINING_STARTED=FALSE")

except Exception as error:
    print(
        "T95_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("MODEL_PROMOTED=FALSE")
    print("HISTORICAL_MODEL_MODIFIED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
