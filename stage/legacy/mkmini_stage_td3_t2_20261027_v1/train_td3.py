from pathlib import Path
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()

JOB = H / "mkmini_stage_td3_t2_20261027_v1"

T1_DIR = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1"
)

T1_SCRIPT = T1_DIR / "td3_smoke.py"
T1_CK = T1_DIR / "td3_t1_research_only.pt"

T1_SHA = (
    "436dc372657f4543cf511f51ee493f4d7938"
    "af132d9f3bc207f92de6dc2df8fa"
)

CORE = H / (
    "mkmini_stage_td3_adapter_20261027_v1"
    "/td3_core.py"
)

CORE_SHA = (
    "e78f7f981b3fdc9d576bbc08b0222d42dc1a"
    "599865796180a4b45b1adcf361b8"
)

STAGE = H / (
    "mkmini_stage_c147_fixedgoal_sac_20261027_v1"
    "/GOAL_COMMIT_24000/train/train.py"
)

STAGE_SHA = (
    "4e9855bd90f523fb5c7c12ac4ea5c7ca5a72"
    "42cd08a63808ef3b3c89120c1556"
)

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_"
    "20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

FROZEN_SHA = (
    "cb650414dab55dbc0a913cc439fe947a7979"
    "fd0755f5aff3912fda68c5ab35d8"
)

TARGET = 6000
EP_CAP = 1500
WALL = 4500

# 全部是最终目标，不存在中间路点。
TASKS = (
    (2029461000, -20, 0.4),
    (2029461001,  20, 0.4),
    (2029461000, -30, 0.8),
    (2029461001,  30, 0.8),
    (2029461002, -25, 1.0),
    (2029461003,  25, 1.0),
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def emit(key, value):
    print(
        key,
        json.dumps(
            value,
            ensure_ascii=False,
            default=str
        ),
        flush=True
    )


def write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str
        ) + "\n"
    )

    os.replace(tmp, path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        name, path
    )

    check(
        spec is not None
        and spec.loader is not None,
        "IMPORT_SPEC_ERROR:" + str(path)
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def preflight():
    expected = (
        (T1_CK, T1_SHA),
        (CORE, CORE_SHA),
        (STAGE, STAGE_SHA),
        (FROZEN, FROZEN_SHA),
    )

    for path, digest in expected:
        check(
            path.is_file() and sha(path) == digest,
            "SOURCE_INTEGRITY_FAIL:" + str(path)
        )

    check(
        T1_SCRIPT.is_file(),
        "T1_HELPER_MISSING"
    )

    import ast
    ast.parse(T1_SCRIPT.read_text())

    emit("TD3_T2_PREFLIGHT=PASS", {
        "additional_physics_budget": TARGET,
        "episode_cap": EP_CAP,
        "wall_seconds": WALL,
        "stage_sha256": STAGE_SHA,
        "warmstart_sha256": T1_SHA,
        "sac_resume": False,
        "automatic_promotion": False
    })


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261027)

    rng = np.random.default_rng(20261027)

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    # 仅导入T1中已验证的环境辅助函数。
    # 不运行原有SAC训练入口。
    helper = load(
        T1_SCRIPT,
        "stage_td3_t1_helper_for_t2"
    )

    core = load(
        CORE,
        "stage_td3_core_for_t2"
    )

    legacy = helper.load_module(
        STAGE,
        "stage_td3_stage_bootstrap_for_t2"
    )

    for path, digest in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == digest,
            "NATIVE_SOURCE_MISMATCH:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_td3_native_runner_for_t2"
    )

    p2, v, native_mod, prep, hashes = (
        runner.load_native()
    )

    check(
        isinstance(hashes, dict)
        and len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]

    check(
        callable(N.step_checked),
        "NATIVE_STEP_MISSING"
    )

    make_base = legacy.make_short_task(
        runner.make_task
    )

    agent = core.TD3(
        state_dim=112,
        action_dim=2,
        device="cpu"
    )

    latest = JOB / "latest.pt"

    if latest.is_file():
        status_path = JOB / "status.json"

        check(
            status_path.is_file(),
            "RESUME_STATUS_MISSING"
        )

        previous_status = json.loads(
            status_path.read_text()
        )

        check(
            sha(latest)
            == previous_status.get(
                "checkpoint_sha256"
            ),
            "RESUME_CHECKPOINT_HASH_INVALID"
        )

        checkpoint_file = latest
    else:
        checkpoint_file = T1_CK

    state = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False
    )

    check(
        state.get("algorithm", "TD3") == "TD3",
        "TD3_CHECKPOINT_ALGORITHM_INVALID"
    )

    if checkpoint_file == T1_CK:
        check(
            state.get("stage_source_sha256")
            == STAGE_SHA
            and state.get("td3_core_sha256")
            == CORE_SHA,
            "T1_CHECKPOINT_SOURCE_MISMATCH"
        )
    else:
        check(
            state.get("stage_sha256") == STAGE_SHA
            and state.get("core_sha256") == CORE_SHA
            and state.get("source_t1_sha256")
            == T1_SHA,
            "T2_CHECKPOINT_SOURCE_MISMATCH"
        )

    for key, network in (
        ("actor", agent.actor),
        ("actor_target", agent.actor_target),
        ("critic", agent.critic),
        ("critic_target", agent.critic_target),
    ):
        network.load_state_dict(state[key])

    agent.actor_optimizer.load_state_dict(
        state["actor_optimizer"]
    )

    agent.critic_optimizer.load_state_dict(
        state["critic_optimizer"]
    )

    agent.total_updates = int(state["updates"])

    replay = deque(
        state["replay"],
        maxlen=60000
    )

    rng.bit_generator.state = (
        state["numpy_rng_state"]
    )

    torch.set_rng_state(
        state["torch_rng_state"]
    )

    additional = int(
        state.get("additional_frames", 0)
    )

    episode_index = int(
        state.get("episode_index", 0)
    )

    decisions = int(state["decisions"])

    episodes = list(
        state.get("episodes", [])
    )

    base_t1 = int(
        state.get(
            "t1_physics_frames",
            state["physics_frames"]
        )
    )

    check(
        0 <= additional <= TARGET,
        "FRAME_COUNT_INVALID"
    )

    check(
        episode_index == len(episodes),
        "EPISODE_INDEX_INVALID"
    )

    check(
        base_t1 == 1602,
        "T1_FRAME_BASE_UNEXPECTED"
    )

    emit("TD3_T2_BOOTSTRAP=PASS", {
        "checkpoint": str(checkpoint_file),
        "t1_frames": base_t1,
        "additional_frames": additional,
        "replay": len(replay),
        "updates": agent.total_updates,
        "episode_index": episode_index
    })

    deadline = time.monotonic() + WALL

    while (
        additional < TARGET
        and time.monotonic() < deadline
    ):

        base_seed, angle, distance = (
            TASKS[episode_index % len(TASKS)]
        )

        seed = base_seed + 10000 * (
            episode_index // len(TASKS)
        )

        env = None
        renderer = None

        try:
            env, renderer, obs, actual_seed = (
                helper.make_goal_task(
                    make_base,
                    v,
                    native_mod,
                    N,
                    seed,
                    angle,
                    distance
                )
            )

            check(
                obs.shape == (112,),
                "INITIAL_OBSERVATION_INVALID"
            )

            actual = float(
                env._goal_state()[1]
            )

            check(
                abs(actual - distance) < 0.03,
                "RUNTIME_GOAL_MISMATCH"
            )

            emit("TD3_T2_GOAL_GATE=", {
                "episode": episode_index + 1,
                "seed": actual_seed,
                "angle": angle,
                "requested_m": distance,
                "actual_m": actual
            })

            initial = actual

            used = 0
            rewards = []
            contacts = 0
            reached_goal = False
            end_reason = "CAP"

            while (
                used < EP_CAP
                and additional < TARGET
            ):

                if time.monotonic() >= deadline:
                    end_reason = "WALL_BUDGET"
                    break

                # 仅TD3网络输出与随机探索噪声。
                raw = np.clip(
                    agent.act(obs)
                    + rng.normal(0, 0.18, 2),
                    -1.0,
                    1.0
                ).astype(np.float32)

                # Stage真实动作：[油门, 转向]
                action = np.array([
                    (float(raw[0]) + 1.0) / 2.0,
                    float(raw[1])
                ], dtype=np.float32)

                before = float(env.data.time)

                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = N.step_checked(env, action)

                nframe = int(round(
                    (
                        float(env.data.time)
                        - before
                    ) / float(
                        env.model.opt.timestep
                    )
                ))

                check(
                    nframe == 1,
                    "UNEXPECTED_PHYSICS_DELTA:"
                    + str(nframe)
                )

                next_obs = np.asarray(
                    next_obs,
                    dtype=np.float32
                ).reshape(-1)

                check(
                    next_obs.size == 112
                    and np.isfinite(next_obs).all()
                    and math.isfinite(float(reward)),
                    "INVALID_TRANSITION"
                )

                collision = bool(
                    info.get("collision", False)
                )

                reached = bool(
                    info.get("reach_goal", False)
                )

                # 真正终止与时间截断分开记录。
                terminal = bool(
                    terminated
                    or collision
                    or reached
                )

                replay.append((
                    obs.copy(),
                    raw.copy(),
                    float(reward),
                    next_obs.copy(),
                    float(terminal)
                ))

                obs = next_obs

                used += nframe
                additional += nframe
                decisions += 1

                rewards.append(float(reward))

                contacts += int(collision)

                reached_goal = (
                    reached_goal or reached
                )

                if (
                    len(replay) >= 128
                    and decisions % 2 == 0
                ):

                    ids = rng.choice(
                        len(replay),
                        size=64,
                        replace=False
                    )

                    samples = [
                        replay[int(i)]
                        for i in ids
                    ]

                    batch = tuple(
                        np.asarray(
                            [row[k] for row in samples],
                            dtype=np.float32
                        )
                        for k in range(5)
                    )

                    result = agent.update(batch)

                    check(
                        math.isfinite(
                            result["critic_loss"]
                        ),
                        "NONFINITE_CRITIC_LOSS"
                    )

                    if result["actor_loss"] is not None:
                        check(
                            math.isfinite(
                                result["actor_loss"]
                            ),
                            "NONFINITE_ACTOR_LOSS"
                        )

                if used % 250 == 0:
                    emit("TD3_T2_PROGRESS=", {
                        "additional_frames": additional,
                        "episode_frames": used,
                        "updates": agent.total_updates,
                        "remaining_m": round(
                            float(
                                env._goal_state()[1]
                            ),
                            4
                        ),
                        "mean_reward": round(
                            float(np.mean(rewards)),
                            5
                        ),
                        "contact": collision,
                        "goal": reached_goal
                    })

                if (
                    reached
                    or collision
                    or terminated
                    or truncated
                ):

                    end_reason = (
                        "GOAL" if reached
                        else "CONTACT" if collision
                        else "TRUNCATED"
                    )

                    break

            final_distance = float(
                env._goal_state()[1]
            )

            row = {
                "episode": episode_index + 1,
                "seed": actual_seed,
                "angle_deg": angle,
                "start_m": initial,
                "remaining_m": final_distance,
                "progress_m": (
                    initial - final_distance
                ),
                "physics_frames": used,
                "end": end_reason,
                "reward_sum": float(
                    np.sum(rewards)
                ),
                "reward_min": (
                    float(np.min(rewards))
                    if rewards else None
                ),
                "reward_max": (
                    float(np.max(rewards))
                    if rewards else None
                ),
                "contacts": contacts,
                "success": bool(reached_goal)
            }

            check(
                used > 0,
                "NO_TRANSITIONS_IN_EPISODE"
            )

            episodes.append(row)
            episode_index += 1

            emit("TD3_T2_EPISODE=", row)

        finally:
            if renderer is not None:
                renderer.close()

            if env is not None:
                env.close()

        # 每回合保存一次。保留最新研究检查点，
        # 不覆盖T1，也不覆盖冻结SAC。
        checkpoint = {
            "algorithm": "TD3",
            "actor": agent.actor.state_dict(),
            "actor_target":
                agent.actor_target.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target":
                agent.critic_target.state_dict(),
            "actor_optimizer":
                agent.actor_optimizer.state_dict(),
            "critic_optimizer":
                agent.critic_optimizer.state_dict(),
            "updates": agent.total_updates,
            "physics_frames":
                base_t1 + additional,
            "t1_physics_frames": base_t1,
            "additional_frames": additional,
            "episode_index": episode_index,
            "decisions": decisions,
            "episodes": episodes,
            "replay": list(replay),
            "numpy_rng_state":
                rng.bit_generator.state,
            "torch_rng_state":
                torch.get_rng_state(),
            "stage_sha256": sha(STAGE),
            "core_sha256": sha(CORE),
            "source_t1_sha256": T1_SHA,
            "automatic_promotion": False
        }

        temp = JOB / "latest.pt.partial"

        torch.save(checkpoint, temp)

        os.replace(
            temp,
            latest
        )

        summary = {
            "state": (
                "TRAINED_BUDGET"
                if additional >= TARGET
                else "RUNNING"
            ),
            "additional_frames": additional,
            "target_additional_frames": TARGET,
            "total_physics_frames":
                base_t1 + additional,
            "episodes": len(episodes),
            "updates": agent.total_updates,
            "success_episodes": sum(
                bool(e["success"])
                for e in episodes
            ),
            "contact_episodes": sum(
                bool(e["contacts"])
                for e in episodes
            ),
            "checkpoint_sha256": sha(latest),
            "automatic_promotion": False,
            "scientific_eval_pass": False
        }

        write_json(
            JOB / "status.json",
            summary
        )

        emit("TD3_T2_CHECKPOINT=", summary)

    if not (JOB / "status.json").is_file():
        raise RuntimeError(
            "TRAINING_DID_NOT_EXECUTE"
        )

    if additional < TARGET:
        summary = json.loads(
            (JOB / "status.json").read_text()
        )

        summary["state"] = (
            "PAUSED_WALL_BUDGET"
        )

        write_json(
            JOB / "status.json",
            summary
        )

    check(
        sha(FROZEN) == FROZEN_SHA,
        "FROZEN_PARENT_CHANGED"
    )

    emit(
        "TD3_T2_END=",
        json.loads(
            (JOB / "status.json").read_text()
        )
    )


if __name__ == "__main__":
    try:
        JOB.mkdir(
            parents=True,
            exist_ok=True
        )

        preflight()
        main()

    except Exception as error:
        emit("TD3_T2_BLOCKED=", {
            "error": repr(error)
        })

        traceback.print_exc()

        # 保留上一个已完成回合的检查点。
        raise

    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
