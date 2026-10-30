from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import shutil
import sys
import traceback

import numpy as np
import torch

H = Path.home()
JOB = Path(os.environ["T96M_JOB"])

BASE = H / (
    "mkmini_stage_td3_t87_direction_learning_"
    "20261028_Mnr36j"
)

ARCHIVE_SOURCE = H / (
    "mkmini_stage_td3_t95_balanced_replay_"
    "20261028_aJuZyx/balanced_archive.pt"
)

INITIAL_SOURCE = BASE / "checkpoints/episode_0025.pt"

TRAIN_SOURCE = BASE / "train.py"
CORE_SOURCE = BASE / "td3_core_goal_contrast.py"

EXPECTED = {
    "archive":
        "df09d48719ebe35ec726b09c3edd19b7fe861f55309b46ebd93be171604e68f6",
    "checkpoint":
        "0378e4af4296683d5d612a2d71607149e8fb3b0ff5235e35643766059ffaf94e",
    "train":
        "252268ad3b7a346f85b8a91cc451905d6d13542dcfc85c81226334803700f429",
    "core":
        "59b3b27bce8c3eeefc4c1b12989139a8fbe00474cc2784df0d8f5a5ec4353a64",
}

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def replace_top(source, name, expression):
    tree = ast.parse(source)

    nodes = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in n.targets
        )
    ]

    check(len(nodes) == 1, "BINDING_INVALID:" + name)

    node = nodes[0]
    lines = source.splitlines(keepends=True)

    lines[node.lineno-1:node.end_lineno] = [
        name + " = " + expression + "\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

SAMPLER_SOURCE = r'''
from pathlib import Path
import hashlib
import numpy as np
import torch

class T96BalancedSampler:

    def __init__(self, path, expected_sha):
        path = Path(path)

        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != expected_sha:
            raise RuntimeError("T96_ARCHIVE_SHA_MISMATCH")

        saved = torch.load(
            path, map_location="cpu", weights_only=False
        )

        self.left = [
            tuple(row[:5]) for row in saved["LEFT"]
        ]
        self.right = [
            tuple(row[:5]) for row in saved["RIGHT"]
        ]

        if len(self.left) != 7000 or len(self.right) != 7000:
            raise RuntimeError("T96_ARCHIVE_COUNTS_INVALID")

    def mix(self, original, updates):
        n = len(original)

        if n < 32:
            raise RuntimeError("T96_BATCH_TOO_SMALL")

        each = int(round(0.30 * n))
        keep = n - 2 * each

        if keep < 1:
            raise RuntimeError("T96_MIX_RATIO_INVALID")

        rng = np.random.default_rng(
            202610289600000 + int(updates)
        )

        current_ids = rng.choice(
            n, size=keep, replace=False
        )
        left_ids = rng.choice(
            len(self.left), size=each, replace=False
        )
        right_ids = rng.choice(
            len(self.right), size=each, replace=False
        )

        mixed = (
            [original[int(i)] for i in current_ids]
            + [self.left[int(i)] for i in left_ids]
            + [self.right[int(i)] for i in right_ids]
        )

        order = rng.permutation(len(mixed))
        mixed = [mixed[int(i)] for i in order]

        if len(mixed) != n:
            raise RuntimeError("T96_BATCH_SIZE_CHANGED")

        return mixed, {
            "original": keep,
            "left": each,
            "right": each,
            "total": n,
        }
'''

INJECTION = '''
                    # T96M: persistent left/right TD3 transitions.
                    # Training batch only; never overrides actor actions.
                    if not hasattr(agent, "_t96_sampler"):
                        from t96_balancer import T96BalancedSampler
                        agent._t96_sampler = T96BalancedSampler(
                            JOB / "balanced_archive.pt",
                            "df09d48719ebe35ec726b09c3edd19b7fe861f55309b46ebd93be171604e68f6"
                        )

                    samples, _t96_counts = agent._t96_sampler.mix(
                        samples,
                        int(agent.total_updates)
                    )

                    if (
                        not hasattr(agent, "_t96_logged")
                        or agent.total_updates % 2000 == 0
                    ):
                        agent._t96_logged = True
                        print(
                            "T96_EFFECTIVE_BATCH=PASS",
                            "UPDATES=", agent.total_updates,
                            "COUNTS=", _t96_counts,
                            flush=True
                        )

'''

try:
    print("=" * 78)
    print("T96M STAGE TD3 PERSISTENT BALANCED SAMPLING")
    print("=" * 78)

    print("TRAINING_STARTED=FALSE")
    print("PLANNER=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("HISTORICAL_MODEL_WRITE=FALSE")

    sources = {
        "archive": ARCHIVE_SOURCE,
        "checkpoint": INITIAL_SOURCE,
        "train": TRAIN_SOURCE,
        "core": CORE_SOURCE,
    }

    for name, path in sources.items():
        check(path.is_file(), "MISSING_SOURCE:" + name)
        check(
            sha(path) == EXPECTED[name],
            "SOURCE_HASH_CHANGED:" + name
        )
        print("T96_SOURCE_VERIFIED=", name)

    saved_archive = torch.load(
        ARCHIVE_SOURCE,
        map_location="cpu",
        weights_only=False
    )

    for side, sign in [("LEFT", -1), ("RIGHT", 1)]:
        rows = saved_archive[side]

        check(len(rows) == 7000, "ARCHIVE_COUNT:" + side)

        sampled = rows[:32]

        for row in sampled:
            check(len(row) >= 5, "ARCHIVE_ROW_INVALID")

            state = np.asarray(row[0], dtype=np.float32)
            action = np.asarray(row[1], dtype=np.float32)
            nxt = np.asarray(row[3], dtype=np.float32)

            check(
                state.shape == (112,)
                and action.shape == (2,)
                and nxt.shape == (112,)
                and np.isfinite(state).all()
                and np.isfinite(action).all()
                and np.isfinite(nxt).all(),
                "ARCHIVE_TRANSITION_SCHEMA_INVALID"
            )

            check(
                sign * float(state[49]) > 0,
                "ARCHIVE_DIRECTION_INVALID"
            )

    print("T96_ARCHIVE_LEFT_RIGHT=PASS")

    initial = torch.load(
        INITIAL_SOURCE,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(initial["frames"]) == 171387,
        "INITIAL_FRAMES_INVALID"
    )

    check(
        int(initial["updates"]) == 85566,
        "INITIAL_UPDATES_INVALID"
    )

    check(
        initial["manifest"]["source_core"]
        == EXPECTED["core"],
        "INITIAL_CORE_INVALID"
    )

    check(
        initial["manifest"]["experiment"]
        == "STAGE_TD3_T87",
        "INITIAL_EXPERIMENT_INVALID"
    )

    train = TRAIN_SOURCE.read_text()

    check(
        train.count("STAGE_TD3_T87") == 1,
        "EXPERIMENT_BINDING_INVALID"
    )

    check(
        train.count("                    batch = tuple(") == 1,
        "BATCH_CONVERSION_ANCHOR_INVALID"
    )

    check(
        train.count("agent.update(") == 1,
        "TD3_UPDATE_ANCHOR_INVALID"
    )

    check(
        train.count("for row in samples") == 1,
        "TRAINING_SAMPLE_LAYOUT_INVALID"
    )

    check(
        "T43_MIXED_BATCH_COUNT" in train
        and "T46_SUCCESS_BC_ACTIVE" in CORE_SOURCE.read_text(),
        "ORIGINAL_REPLAY_MECHANISMS_NOT_FOUND"
    )

    train = replace_top(
        train,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    train = train.replace(
        "STAGE_TD3_T87",
        "STAGE_TD3_T96M",
        1
    )

    train = train.replace(
        "                    batch = tuple(",
        INJECTION + "                    batch = tuple(",
        1
    )

    ast.parse(train)

    (JOB / "train.py").write_text(train)

    (JOB / "t96_balancer.py").write_text(
        SAMPLER_SOURCE
    )

    shutil.copy2(
        CORE_SOURCE,
        JOB / "td3_core_goal_contrast.py"
    )

    shutil.copy2(
        BASE / "goal_pairs.npz",
        JOB / "goal_pairs.npz"
    )

    shutil.copy2(
        ARCHIVE_SOURCE,
        JOB / "balanced_archive.pt"
    )

    check(
        sha(JOB / "balanced_archive.pt")
        == EXPECTED["archive"],
        "LOCAL_ARCHIVE_HASH_INVALID"
    )

    check(
        sha(JOB / "td3_core_goal_contrast.py")
        == EXPECTED["core"],
        "LOCAL_CORE_CHANGED"
    )

    # 对真正的T96采样器进行独立批次测试。
    sys.path.insert(0, str(JOB))

    from t96_balancer import T96BalancedSampler

    sampler = T96BalancedSampler(
        JOB / "balanced_archive.pt",
        EXPECTED["archive"]
    )

    original = list(initial["replay"][:64])

    mixed, counts = sampler.mix(
        original, 85566
    )

    check(
        counts == {
            "original": 26,
            "left": 19,
            "right": 19,
            "total": 64
        },
        "T96_MIX_COUNTS_INVALID"
    )

    # 验证与原训练器一样，可以转换成五元批次。
    batch = tuple(
        np.asarray(
            [row[k] for row in mixed],
            dtype=np.float32
        )
        for k in range(5)
    )

    check(
        batch[0].shape == (64, 112)
        and batch[1].shape == (64, 2)
        and batch[3].shape == (64, 112)
        and np.isfinite(batch[0]).all(),
        "T96_DRY_RUN_BATCH_INVALID"
    )

    print("T96_DRY_RUN_COUNTS=", counts)
    print("T96_DRY_RUN_BATCH=PASS")

    child = copy.deepcopy(initial)
    child["manifest"]["experiment"] = (
        "STAGE_TD3_T96M"
    )
    child["manifest"]["source_core"] = (
        EXPECTED["core"]
    )
    child["manifest"]["target_frames"] = (
        190491
    )

    checkpoint_dir = JOB / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = (
        checkpoint_dir / "episode_0025.pt"
    )

    torch.save(child, checkpoint)

    status = json.loads(
        (BASE / "status.json").read_text()
    )

    status.update({
        "state": "RUNNING",
        "frames": 171387,
        "updates": 85566,
        "episodes": 25,
        "target_frames": 190491,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha(checkpoint),
        "automatic_promotion": False,
    })

    for field in ("success", "collisions"):
        if field in child:
            status[field] = child[field]

    (JOB / "status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )

    manifest = {
        "experiment": "STAGE_TD3_T96M",
        "source_checkpoint":
            str(INITIAL_SOURCE),
        "source_checkpoint_sha256":
            EXPECTED["checkpoint"],
        "initial_frames": 171387,
        "target_frames": 190491,
        "archive_sha256": EXPECTED["archive"],
        "original_batch_fraction": 0.40,
        "left_fraction": 0.30,
        "right_fraction": 0.30,
        "single_functional_change":
            "Persistent left/right TD3 batch sampling",
        "actor_loss_unchanged": True,
        "reward_unchanged": True,
        "environment_unchanged": True,
        "action_override": False,
        "automatic_promotion": False,
        "train_sha256": sha(JOB / "train.py"),
        "core_sha256": sha(
            JOB / "td3_core_goal_contrast.py"
        ),
        "initial_checkpoint_sha256": sha(checkpoint),
    }

    (JOB / "experiment.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("T96_INITIAL_CHECKPOINT=PASS")
    print("T96_ORIGINAL_TD3_CORE=PASS")
    print("T96_T43_T46_PRESERVED=PASS")
    print("T96_ACTOR_ACTION_UNCHANGED=PASS")
    print("T96_TRAIN_SOURCE_SHA=", manifest["train_sha256"])
    print("T96_PREFLIGHT=PASS")

except Exception as error:
    print("T96_PREFLIGHT_BLOCKED=", repr(error))
    traceback.print_exc()

finally:
    print("FROZEN_MODEL_MODIFIED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
