from pathlib import Path
from datetime import datetime
import ast
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()
ORIGINAL = Path(os.environ["T62_JOB"])
JOB = Path(os.environ["T65_JOB"])

END = datetime.fromisoformat(
    "2026-10-28T08:00:00"
).astimezone().timestamp()

STAGE_LOCK = H / ".mkmini_stage_td3_independent.lock"

PILOT = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

# 固定顺序：先困难右转，再困难左转，最后旧能力。
STANDARD_TASKS = (
    (-17.3, 3.0),
    (13.6, 3.0),
    (18.8, 3.0),
    (-12.5, 3.0),
)

STANDARD_SEEDS = (
    2029653601,
    2029653602,
    2029654802,
    2029654801,
)

# 另一分支增加此前已验证的新方向。
DIFFICULT_TASKS = (
    (-33.7, 3.0),
    (37.9, 3.0),
    (18.8, 3.0),
    (-12.5, 3.0),
)

DIFFICULT_SEEDS = (
    2029654803,
    2029654804,
    2029654802,
    2029654801,
)

# 全部配置预先固定，不依据测试结果修改。
BRANCHES = [
    {
        "name": "T65A",
        "tasks": STANDARD_TASKS,
        "seeds": STANDARD_SEEDS,
        "front": ("0.12 * risk(front, 0.48)",
                  "0.22 * risk(front, 0.55)"),
        "rear": ("0.05 * risk(rear, 0.35)",
                 "0.08 * risk(rear, 0.43)"),
        "chassis": ("0.08 * risk(chassis, 0.45)",
                    "0.12 * risk(chassis, 0.53)"),
        "speed": ("0.6 + 1.6 *",
                  "0.8 + 2.0 *"),
    },
    {
        "name": "T65B",
        "tasks": DIFFICULT_TASKS,
        "seeds": DIFFICULT_SEEDS,
        "front": ("0.12 * risk(front, 0.48)",
                  "0.19 * risk(front, 0.55)"),
        "rear": ("0.05 * risk(rear, 0.35)",
                 "0.09 * risk(rear, 0.46)"),
        "chassis": ("0.08 * risk(chassis, 0.45)",
                    "0.15 * risk(chassis, 0.56)"),
        "speed": ("0.6 + 1.6 *",
                  "0.8 + 2.4 *"),
    },
]


def announce(*items):
    print(*items, flush=True)


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()


def read_text(path):
    path = Path(path)
    return (
        path.read_text(errors="replace")
        if path.is_file()
        else ""
    )


def save_json(path, value):
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )
    os.replace(temporary, path)


def remaining():
    return max(0, int(END - time.time()))


def load_verified_t62_functions():
    """
    仅加载T62已有的函数定义和常量。
    不执行T62脚本末尾的--night入口。
    """
    source = ORIGINAL / "night.py"

    check(source.is_file(), "T62_SOURCE_MISSING")

    tree = ast.parse(source.read_text())

    safe_types = (
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.FunctionDef,
    )

    selected = [
        node for node in tree.body
        if isinstance(node, safe_types)
    ]

    selected_tree = ast.Module(
        body=selected,
        type_ignores=[]
    )

    ast.fix_missing_locations(selected_tree)

    namespace = {
        "__name__": "_t62_readonly_functions",
    }

    exec(
        compile(
            selected_tree,
            str(source),
            "exec"
        ),
        namespace
    )

    required = (
        "create_branch",
        "completed_model",
        "build_evaluator",
        "check_eval",
        "check_sources",
        "read_model",
    )

    for name in required:
        check(
            callable(namespace.get(name)),
            "T62_FUNCTION_MISSING:" + name
        )

    check(
        Path(namespace["JOB"]) == ORIGINAL,
        "T62_JOB_MISMATCH"
    )

    check(
        sha(namespace["PARENT"])
        == namespace["PARENT_SHA"],
        "T53_PARENT_CHANGED"
    )

    namespace["check_sources"](
        namespace["T53"]
    )

    namespace["check_sources"](
        namespace["T59"]
    )

    namespace["check_sources"](
        ORIGINAL / "stage_t62_train"
    )

    return namespace


def patch_reward(folder, cfg):
    train = folder / "train.py"

    source = train.read_text()

    replacements = {
        "max_range = 0.60":
            "max_range = 0.65",
        cfg["front"][0]: cfg["front"][1],
        cfg["rear"][0]: cfg["rear"][1],
        cfg["chassis"][0]: cfg["chassis"][1],
        cfg["speed"][0]: cfg["speed"][1],
    }

    for old, new in replacements.items():
        check(
            source.count(old) == 1,
            "T65_PATCH_ANCHOR_INVALID:" + old
        )

        source = source.replace(old, new, 1)

    ast.parse(source)

    train.write_text(source)

    manifest_path = folder / "prepare.json"
    manifest = json.loads(
        manifest_path.read_text()
    )

    manifest["files"]["train.py"] = sha(train)
    manifest["relay_configuration"] = cfg["name"]

    save_json(manifest_path, manifest)

    announce(
        "T65_REWARD_PATCH=PASS",
        cfg["name"]
    )


def prepare_branch(lib, cfg):
    folder = JOB / cfg["name"]
    target = 22491 + 56000

    if not (folder / "prepare.json").exists():

        lib["create_branch"](
            folder,
            cfg["name"],
            lib["PARENT"],
            22491,
            ORIGINAL / "stage_t62_train",
            target,
            14000,
            cfg["tasks"],
            cfg["seeds"],
            False,
            7200
        )

        patch_reward(folder, cfg)

    lib["check_sources"](folder)

    status = json.loads(
        (folder / "status.json").read_text()
    )

    checkpoint = (
        folder / "checkpoints"
        / status["checkpoint"]
    )

    check(
        checkpoint.is_file()
        and sha(checkpoint)
            == status["checkpoint_sha256"],
        "T65_CHECKPOINT_INTEGRITY_FAILED"
    )

    return folder, target


def run_checked(script, logfile, time_limit):
    seconds = min(
        time_limit,
        remaining() - 90
    )

    check(
        seconds >= 180,
        "T65_NOT_ENOUGH_TIME"
    )

    announce(
        "T65_START=",
        script,
        "TIME_BUDGET_S=",
        seconds
    )

    with Path(logfile).open("a") as stream:
        result = subprocess.run(
            [
                "timeout",
                "--foreground",
                "--signal=TERM",
                f"{seconds}s",
                sys.executable,
                "-u",
                str(script),
            ],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False
        )

    output = read_text(logfile)

    announce(
        "T65_SCRIPT_RC=",
        result.returncode,
        "SCRIPT=",
        script
    )

    check(
        result.returncode == 0,
        "T65_SUBPROCESS_FAILED:"
        + str(script)
    )

    check(
        "TD3_T6_BLOCKED=" not in output
        and "T63_BLOCKED=" not in output,
        "T65_TRAIN_OR_EVAL_BLOCKED"
    )

    return output


def build_and_evaluate(
    lib, cfg, checkpoint, digest, frames,
    heldout=False
):
    name = cfg["name"]

    if heldout:
        # 每个分支使用不同的预先固定测试集合。
        index = next(
            i for i, c in enumerate(BRANCHES)
            if c["name"] == name
        )

        rng = np.random.default_rng(
            202610276500 + index
        )

        angles = rng.uniform(
            8.0, 43.0, size=40
        )

        tasks = [
            (
                2029730001 + index * 100 + i,
                round(
                    float(
                        -angles[i] if i % 2 == 0
                        else angles[i]
                    ),
                    3
                ),
                3.0,
            )
            for i in range(40)
        ]

        folder = JOB / f"{name}_heldout40"
        wall = 24000
        budget = 14400

    else:
        tasks = PILOT
        folder = JOB / f"{name}_pilot4"
        wall = 6000
        budget = 5400

    evaluator = lib["build_evaluator"](
        folder,
        name,
        tasks,
        checkpoint,
        digest,
        frames,
        wall
    )

    run_checked(
        evaluator,
        folder / "live.log",
        budget
    )

    lib["check_eval"](
        folder,
        name,
        tasks,
        heldout=heldout
    )

    return True


def launch_long_distance(
    lib, cfg, checkpoint, digest, frames
):
    check(
        remaining() >= 300,
        "LONG_DISTANCE_TIME_WINDOW_CLOSED"
    )

    folder = JOB / "T65_LONG_5M"

    tasks = (
        (-15.0, 5.0),
        (15.0, 5.0),
        (-30.0, 5.0),
        (30.0, 5.0),
    )

    seeds = (
        2029688101,
        2029688102,
        2029688103,
        2029688104,
    )

    lib["create_branch"](
        folder,
        "T64",
        checkpoint,
        frames,
        ORIGINAL / "stage_t62_train",
        frames + 60000,
        20000,
        tasks,
        seeds,
        False,
        28800
    )

    # 将获胜的安全奖励配置同步到长距离训练。
    patch_reward(folder, cfg)

    lib["check_sources"](folder)

    announce(
        "T65_LONG_DISTANCE_5M_STARTED=TRUE"
    )

    save_json(
        JOB / "handoff.json",
        {
            "state": "LONG_DISTANCE_STARTED",
            "parent": cfg["name"],
            "parent_checkpoint_sha256": digest,
            "distance_m": 5.0,
            "automatic_promotion": False,
        }
    )

    run_checked(
        folder / "train.py",
        folder / "night_train.log",
        28800
    )

    announce(
        "T65_LONG_DISTANCE_TRAINING_FINISHED=TRUE"
    )


def execute_fallback(lib):
    announce(
        "T65_RELAY_TAKING_OVER=TRUE"
    )

    results = []

    for cfg in BRANCHES:
        if remaining() < 300:
            announce(
                "T65_DEADLINE_APPROACHING=TRUE"
            )
            break

        name = cfg["name"]

        record = {
            "branch": name,
            "state": "STARTED",
        }

        try:
            announce(
                "T65_BRANCH_START=",
                name
            )

            folder, target = prepare_branch(
                lib, cfg
            )

            status = json.loads(
                (folder / "status.json").read_text()
            )

            if int(status["frames"]) < target:
                run_checked(
                    folder / "train.py",
                    folder / "night_train.log",
                    14400
                )

            checkpoint, digest = (
                lib["completed_model"](
                    folder, target
                )
            )

            announce(
                "T65_TRAINING_COMPLETED=",
                name
            )

            build_and_evaluate(
                lib, cfg,
                checkpoint, digest, target,
                heldout=False
            )

            announce(
                "T65_PILOT_4_GATE=PASS",
                name
            )

            record["state"] = "PILOT_PASS"

            build_and_evaluate(
                lib, cfg,
                checkpoint, digest, target,
                heldout=True
            )

            announce(
                "T65_RANDOM_40_GATE=PASS",
                name
            )

            record["state"] = "40_SCENE_PASS"
            results.append(record)

            save_json(
                JOB / "relay_results.json",
                results
            )

            launch_long_distance(
                lib, cfg,
                checkpoint, digest, target
            )

            return

        except Exception as error:
            record["state"] = "FAILED_OR_INCOMPLETE"
            record["error"] = repr(error)

            announce(
                "T65_BRANCH_BLOCKED=",
                name,
                repr(error)
            )

            traceback.print_exc()

            results.append(record)

            save_json(
                JOB / "relay_results.json",
                results
            )

    announce(
        "T65_FALLBACKS_FINISHED_WITHOUT_LONG_GATE=TRUE"
    )
    announce(
        "LONG_DISTANCE=LOCKED"
    )


def watch():
    announce("T65_STAGE_ONLY=TRUE")
    announce("T65_T62_UNMODIFIED=TRUE")
    announce("T65_PHASE_ISOLATED=TRUE")
    announce("T65_NO_AUTO_PROMOTION=TRUE")

    announce(
        "T65_MACHINE_LOCAL_TIME=",
        datetime.now().astimezone().isoformat()
    )

    announce(
        "T65_DEADLINE_LOCAL=",
        datetime.fromtimestamp(
            END
        ).astimezone().isoformat()
    )

    lib = load_verified_t62_functions()

    done = False
    last_heartbeat = 0

    while remaining() > 0:
        now = time.time()

        if now - last_heartbeat >= 300:
            announce(
                "T65_HEARTBEAT",
                "LOCAL=",
                datetime.now().astimezone().isoformat(),
                "REMAINING_S=",
                remaining()
            )
            last_heartbeat = now

        original_log = read_text(
            ORIGINAL / "overnight.log"
        )

        if "LONG_DISTANCE_5M_STARTED=TRUE" in original_log:
            if not done:
                announce(
                    "T65_ORIGINAL_LONG_DISTANCE_DETECTED=TRUE"
                )
                announce(
                    "T65_NO_DUPLICATE_STAGE_START=TRUE"
                )
                done = True

            time.sleep(min(30, remaining()))
            continue

        if done:
            time.sleep(min(30, remaining()))
            continue

        # T62运行时持有全局Stage锁。
        # 接力程序只在锁释放后接管。
        fd = os.open(
            STAGE_LOCK,
            os.O_CREAT | os.O_RDWR,
            0o600
        )

        acquired = False

        try:
            try:
                fcntl.flock(
                    fd,
                    fcntl.LOCK_EX | fcntl.LOCK_NB
                )
                acquired = True

            except BlockingIOError:
                pass

            if acquired:
                # 二次核验，防止交接瞬间重复启动。
                original_log = read_text(
                    ORIGINAL / "overnight.log"
                )

                if "LONG_DISTANCE_5M_STARTED=TRUE" in original_log:
                    announce(
                        "T65_T62_ALREADY_ENTERED_LONG_DISTANCE=TRUE"
                    )

                else:
                    announce(
                        "T65_T62_LOCK_RELEASED=TRUE"
                    )

                    execute_fallback(lib)

                done = True

        finally:
            if acquired:
                fcntl.flock(
                    fd,
                    fcntl.LOCK_UN
                )

            os.close(fd)

        time.sleep(min(30, remaining()))

    announce("T65_MONITORING_END_AT_08=TRUE")
    announce("T65_NO_NEW_WORK_AFTER_DEADLINE=TRUE")
    announce("MODEL_PROMOTED=FALSE")


def main():
    mode = sys.argv[1]

    if mode == "--check":
        check(
            remaining() > 0,
            "T65_DEADLINE_ALREADY_PASSED"
        )

        load_verified_t62_functions()

        announce("T65_PREFLIGHT=PASS")
        announce("T65_T62_RUNNING_JOB_NOT_TOUCHED=TRUE")

    elif mode == "--watch":
        watch()

    else:
        raise RuntimeError(
            "T65_INVALID_MODE"
        )


try:
    main()

except Exception as error:
    announce(
        "T65_BLOCKED=",
        repr(error)
    )

    traceback.print_exc()
    raise

finally:
    announce(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
