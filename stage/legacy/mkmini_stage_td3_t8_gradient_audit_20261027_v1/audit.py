from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import sys
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t8_gradient_audit_20261027_v1"

CORE = H / (
    "mkmini_stage_td3_adapter_20261027_v1"
    "/td3_core.py"
)

T6 = H / "mkmini_stage_td3_t6_balanced_20261027_v1"

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1"
    "/train/work_resume.pt"
)

HASHES = {
    CORE: (
        "e78f7f981b3fdc9d576bbc08b0222d42dc1a599865796180a4b45b1adcf361b8"
    ),
    FROZEN: (
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8"
    ),
    T6 / "checkpoints/episode_0002.pt": (
        "ee66ed967bf8bc2477d2d6c49f008d84b405b4650b8745c95d2e97e79d590150"
    ),
    T6 / "checkpoints/episode_0003.pt": (
        "bcc65bacaac38a5f4fb7a817eedddd200d66ab2c46529e8575ae34cfc502c8c0"
    ),
    T6 / "checkpoints/episode_0005.pt": (
        "048e82702a923a4f04b10fc4e4e2b9f0973a096cefc7c23c02068a85c5c2aa6f"
    )
}


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for part in iter(
            lambda: stream.read(1048576),
            b""
        ):
            h.update(part)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def emit(name, value):
    print(
        name,
        json.dumps(
            value,
            ensure_ascii=False,
            default=str
        ),
        flush=True
    )


def stats(values):
    a = np.asarray(values, dtype=np.float64)

    return {
        "mean": round(float(a.mean()), 7),
        "min": round(float(a.min()), 7),
        "max": round(float(a.max()), 7),
        "positive_fraction": round(
            float(np.mean(a > 0)), 4
        )
    }


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "stage_td3_t8_verified_core",
        str(path)
    )

    check(
        spec is not None and spec.loader is not None,
        "CORE_IMPORT_ERROR"
    )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    return mod


def sample_states(replay, side, count=64):
    samples = []

    for transition in replay:
        if len(transition) < 6:
            continue

        if transition[5] != side:
            continue

        state = np.asarray(
            transition[0],
            dtype=np.float32
        ).reshape(-1)

        if state.size != 112:
            raise RuntimeError(
                "REPLAY_STATE_DIMENSION_MISMATCH"
            )

        if not np.isfinite(state).all():
            raise RuntimeError(
                "NONFINITE_REPLAY_STATE"
            )

        samples.append(state)

    check(
        len(samples) >= count,
        "INSUFFICIENT_" + side + "_REPLAY"
    )

    indices = np.linspace(
        0,
        len(samples) - 1,
        count,
        dtype=int
    )

    return torch.from_numpy(
        np.stack(
            [samples[int(i)] for i in indices]
        )
    )


def critic_gradient(critic, states, actions):
    # 仅对动作求导，不更新Critic参数。
    actions = actions.detach().clone()
    actions.requires_grad_(True)

    q1, q2 = critic(states, actions)

    gradient = torch.autograd.grad(
        q1.sum(),
        actions
    )[0]

    return {
        "q1": stats(
            q1.detach().numpy().reshape(-1)
        ),
        "q2": stats(
            q2.detach().numpy().reshape(-1)
        ),
        "dq1_dthrottle": stats(
            gradient[:, 0].detach().numpy()
        ),
        "dq1_dsteer": stats(
            gradient[:, 1].detach().numpy()
        ),
    }


def q_grid(critic, states):
    rows = []

    throttle_values = (
        0.0, 0.5, 0.95
    )

    steer_values = (
        -0.9, -0.5, 0.0, 0.5, 0.9
    )

    with torch.no_grad():

        for throttle in throttle_values:
            scores = []

            for steer in steer_values:

                action = torch.empty(
                    (len(states), 2),
                    dtype=torch.float32
                )

                action[:, 0] = throttle
                action[:, 1] = steer

                q1, q2 = critic(
                    states,
                    action
                )

                scores.append({
                    "steer": steer,
                    "q1": float(q1.mean()),
                    "q2": float(q2.mean()),
                    "min_q": float(
                        torch.minimum(
                            q1,
                            q2
                        ).mean()
                    )
                })

            best_q1 = max(
                scores,
                key=lambda row: row["q1"]
            )

            best_min = max(
                scores,
                key=lambda row: row["min_q"]
            )

            rows.append({
                "raw_throttle": throttle,
                "best_q1_steer": best_q1["steer"],
                "best_min_q_steer": best_min["steer"],
                "all_scores": scores
            })

    return rows


def main():
    torch.set_num_threads(2)

    print("========== TD3-T8 GRADIENT AUDIT ==========")
    print("STAGE_ONLY=TRUE")
    print("READ_ONLY=TRUE")
    print("SIMULATION=FALSE")
    print("TRAINING=FALSE")
    print("MODEL_WRITE=FALSE")
    print("AUTOMATIC_PROMOTION=FALSE")

    for path, expected in HASHES.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_HASH_MISMATCH:" + str(path)
        )

        print(
            "SOURCE_VERIFIED=",
            path,
            flush=True
        )

    core = load_module(CORE)

    print()
    print("========== ACTOR UPDATE SOURCE ==========")

    update_source = inspect.getsource(
        core.TD3.update
    )

    for number, line in enumerate(
        update_source.splitlines(),
        1
    ):
        if any(
            term in line
            for term in (
                "actor_loss",
                "critic_loss",
                "backward",
                "optimizer",
                "policy_delay",
                "total_updates",
                "clamp",
                "noise"
            )
        ):
            print(
                f"TD3_UPDATE_SOURCE {number}: {line}"
            )

    results = []

    for episode in (2, 3, 5):

        path = (
            T6
            / "checkpoints"
            / f"episode_{episode:04d}.pt"
        )

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            state.get("algorithm") == "TD3",
            "CHECKPOINT_ALGORITHM_INVALID"
        )

        actor = core.Actor(112, 2)
        actor.load_state_dict(
            state["actor"]
        )
        actor.eval()

        critic = core.Critic(112, 2)
        critic.load_state_dict(
            state["critic"]
        )
        critic.eval()

        for side in ("LEFT", "RIGHT"):

            states = sample_states(
                state["replay"],
                side,
                count=64
            )

            with torch.no_grad():
                actual_actions = actor(states)

            actual_gradient = critic_gradient(
                critic,
                states,
                actual_actions
            )

            # 比较相同Critic在中性动作附近的梯度。
            neutral_actions = torch.zeros(
                (len(states), 2),
                dtype=torch.float32
            )

            neutral_actions[:, 0] = 0.2

            neutral_gradient = critic_gradient(
                critic,
                states,
                neutral_actions
            )

            # 检查高油门、负向大转向处的梯度。
            saturated_actions = torch.zeros(
                (len(states), 2),
                dtype=torch.float32
            )

            saturated_actions[:, 0] = 0.95
            saturated_actions[:, 1] = -0.9

            saturated_gradient = critic_gradient(
                critic,
                states,
                saturated_actions
            )

            raw = actual_actions.detach().numpy()

            # Actor末端Tanh的局部导数。
            tanh_derivative = (
                1.0 - raw * raw
            )

            rows = q_grid(
                critic,
                states
            )

            result = {
                "episode": episode,
                "frames": int(state["frames"]),
                "side": side,
                "actor_raw_throttle": stats(
                    raw[:, 0]
                ),
                "actor_raw_steer": stats(
                    raw[:, 1]
                ),
                "actor_tanh_derivative_throttle":
                    stats(tanh_derivative[:, 0]),
                "actor_tanh_derivative_steer":
                    stats(tanh_derivative[:, 1]),
                "gradient_at_actor": actual_gradient,
                "gradient_at_neutral": neutral_gradient,
                "gradient_at_high_throttle":
                    saturated_gradient,
                "critic_q_grid": rows
            }

            results.append(result)

            emit("TD3_T8_ACTOR=", {
                "episode": episode,
                "side": side,
                "frames": int(state["frames"]),
                "raw_throttle": result[
                    "actor_raw_throttle"
                ],
                "raw_steer": result[
                    "actor_raw_steer"
                ],
                "tanh_steer_derivative": result[
                    "actor_tanh_derivative_steer"
                ]
            })

            emit("TD3_T8_GRADIENT=", {
                "episode": episode,
                "side": side,
                "at_actor": actual_gradient[
                    "dq1_dsteer"
                ],
                "at_neutral": neutral_gradient[
                    "dq1_dsteer"
                ],
                "at_high_throttle":
                    saturated_gradient[
                        "dq1_dsteer"
                    ]
            })

            for row in rows:
                emit("TD3_T8_Q_GRID=", {
                    "episode": episode,
                    "side": side,
                    "raw_throttle": row[
                        "raw_throttle"
                    ],
                    "best_q1_steer": row[
                        "best_q1_steer"
                    ],
                    "best_min_q_steer": row[
                        "best_min_q_steer"
                    ]
                })

    check(
        sha(FROZEN) == HASHES[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    report = {
        "state": "AUDIT_COMPLETED",
        "algorithm": "TD3",
        "results": results,
        "training_started": False,
        "frozen_model_verified": True,
        "automatic_promotion": False
    }

    output = JOB / "gradient_audit.json"
    temporary = JOB / "gradient_audit.json.partial"

    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    temporary.replace(output)

    print()
    print("TD3_T8_GRADIENT_AUDIT=PASS")
    print("RESULTS_FILE=", output)
    print("FROZEN_MODEL_MODIFIED=FALSE")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:

        print(
            "TD3_T8_GRADIENT_AUDIT=BLOCKED",
            repr(error),
            flush=True
        )

        traceback.print_exc()

    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
