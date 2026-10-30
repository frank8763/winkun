from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t5_collapse_audit_20261027_v1"

CORE = H / (
    "mkmini_stage_td3_adapter_20261027_v1/td3_core.py"
)

T1 = H / (
    "mkmini_stage_td3_independent_20261027_105202_577031_v1/"
    "td3_t1_research_only.pt"
)

T2 = H / (
    "mkmini_stage_td3_t2_20261027_v1/latest.pt"
)

T3 = H / (
    "mkmini_stage_td3_t3_20261027_v1/latest.pt"
)

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

EXPECTED = {
    CORE: "e78f7f981b3fdc9d576bbc08b0222d42dc1a599865796180a4b45b1adcf361b8",
    T1: "436dc372657f4543cf511f51ee493f4d7938af132d9f3bc207f92de6dc2df8fa",
    T2: "baae3c4dd8c20fa91ebe427c47b3e07c8d72cbb6197394860b127e8146f3aa54",
    T3: "43e228b52715bd20cbbfdea339b30502f36fbaf985340c90a778e374f62ee041",
    FROZEN: "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "td3_collapse_audit_core",
        path
    )
    check(spec and spec.loader, "CORE_IMPORT_FAILED")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def stats(array):
    x = np.asarray(array, dtype=np.float64)
    return {
        "mean": round(float(x.mean()), 6),
        "std": round(float(x.std()), 6),
        "min": round(float(x.min()), 6),
        "max": round(float(x.max()), 6),
    }


def main():
    torch.set_num_threads(2)

    print("STAGE_ONLY=TRUE")
    print("READ_ONLY_CHECKPOINTS=TRUE")
    print("SIMULATION=FALSE")
    print("TRAINING=FALSE")
    print("AUTOMATIC_PROMOTION=FALSE")

    for path, expected in EXPECTED.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_SHA256_MISMATCH:" + str(path)
        )
        print("SOURCE_VERIFIED=", path.name, expected)

    core = load_module(CORE)

    # 这些检查点由本地受控训练程序生成。
    states = {
        name: torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
        for name, path in (
            ("T1", T1),
            ("T2", T2),
            ("T3", T3)
        )
    }

    latest = states["T3"]

    replay = latest["replay"]
    episodes = latest["episodes"]

    initial_frames = int(
        latest.get("t1_physics_frames", 1602)
    )

    trained_frames = int(
        latest["additional_frames"]
    )

    episode_frames = sum(
        int(e["physics_frames"])
        for e in episodes
    )

    check(
        trained_frames == 24000,
        "T3_TRAINING_BUDGET_INVALID"
    )

    check(
        episode_frames == trained_frames,
        "EPISODE_FRAME_SUM_INVALID"
    )

    check(
        len(replay) >= initial_frames + trained_frames,
        "REPLAY_NOT_FULL_CHRONOLOGICAL"
    )

    print()
    print("========== REPLAY INTEGRITY ==========")
    print("T1_FRAMES=", initial_frames)
    print("T3_TRAIN_FRAMES=", trained_frames)
    print("EPISODES=", len(episodes))
    print("REPLAY_SIZE=", len(replay))
    print("REPLAY_CHRONOLOGY=PASS")

    group = {
        "LEFT": [],
        "RIGHT": []
    }

    replay_actions = {
        "LEFT": [],
        "RIGHT": []
    }

    cursor = len(replay) - episode_frames

    for ep in episodes:
        frames = int(ep["physics_frames"])
        angle = float(ep["angle_deg"])

        side = "LEFT" if angle < 0 else "RIGHT"

        check(
            frames > 0,
            "EPISODE_EMPTY"
        )

        # 每个训练回合抽取起点及不同时间位置。
        positions = sorted(set([
            0,
            frames // 8,
            frames // 4,
            frames // 2,
            3 * frames // 4,
            frames - 1
        ]))

        for pos in positions:
            transition = replay[cursor + pos]

            obs = np.asarray(
                transition[0],
                dtype=np.float32
            ).reshape(-1)

            act = np.asarray(
                transition[1],
                dtype=np.float32
            ).reshape(-1)

            check(
                obs.shape == (112,)
                and act.shape == (2,)
                and np.isfinite(obs).all()
                and np.isfinite(act).all(),
                "REPLAY_TRANSITION_INVALID"
            )

            group[side].append(obs)
            replay_actions[side].append(act)

        cursor += frames

    check(
        cursor == len(replay),
        "REPLAY_EPISODE_ALIGNMENT_FAILED"
    )

    check(
        len(group["LEFT"]) > 0
        and len(group["RIGHT"]) > 0,
        "LEFT_RIGHT_SAMPLE_MISSING"
    )

    print()
    print("========== ACTUAL REPLAY ACTION COVERAGE ==========")

    coverage = {}

    for side in ("LEFT", "RIGHT"):
        actions = np.asarray(
            replay_actions[side],
            dtype=np.float32
        )

        steering = actions[:, 1]

        record = {
            "sample_count": len(actions),
            "raw_steer": stats(steering),
            "positive_gt_0p25_fraction":
                float(np.mean(steering > 0.25)),
            "negative_lt_minus0p25_fraction":
                float(np.mean(steering < -0.25))
        }

        coverage[side] = record

        print(
            side,
            json.dumps(record, ensure_ascii=False)
        )

    print()
    print("========== ACTOR OUTPUT ON IDENTICAL OBSERVATIONS ==========")

    comparisons = {}

    for name in ("T1", "T2", "T3"):
        actor = core.Actor(
            state_dim=112,
            action_dim=2
        )

        actor.load_state_dict(
            states[name]["actor"]
        )

        actor.eval()

        comparison = {}

        for side in ("LEFT", "RIGHT"):
            sample = np.asarray(
                group[side],
                dtype=np.float32
            )

            tensor = torch.from_numpy(sample)

            with torch.no_grad():
                raw = actor(tensor).numpy()

                # 最后一层Tanh之前的logits。
                logits = actor.net[:-1](
                    tensor
                ).numpy()

            steer = raw[:, 1]
            throttle = (raw[:, 0] + 1.0) * 0.5

            record = {
                "sample_count": len(sample),
                "steer": stats(steer),
                "throttle": stats(throttle),
                "steer_negative_0p98_fraction":
                    float(np.mean(steer < -0.98)),
                "steer_positive_0p98_fraction":
                    float(np.mean(steer > 0.98)),
                "throttle_gt_0p98_fraction":
                    float(np.mean(throttle > 0.98)),
                "steer_logit": stats(logits[:, 1]),
                "throttle_logit": stats(logits[:, 0])
            }

            comparison[side] = record

            print(
                name,
                side,
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )

        comparisons[name] = comparison

    # 评估双Critic在右侧状态下对相反转向
    # 的预测，作为偏置诊断；不将其当成真实性能。
    critic = core.Critic(
        state_dim=112,
        action_dim=2
    )

    critic.load_state_dict(
        latest["critic"]
    )

    critic.eval()

    right = torch.from_numpy(
        np.asarray(
            group["RIGHT"],
            dtype=np.float32
        )
    )

    n = len(right)

    action_left = torch.tensor(
        [[0.8, -1.0]],
        dtype=torch.float32
    ).repeat(n, 1)

    action_right = torch.tensor(
        [[0.8, 1.0]],
        dtype=torch.float32
    ).repeat(n, 1)

    with torch.no_grad():
        q1_left, q2_left = critic(
            right, action_left
        )
        q1_right, q2_right = critic(
            right, action_right
        )

    critic_probe = {
        "right_goal_q_left_steer": [
            round(float(q1_left.mean()), 6),
            round(float(q2_left.mean()), 6)
        ],
        "right_goal_q_right_steer": [
            round(float(q1_right.mean()), 6),
            round(float(q2_right.mean()), 6)
        ],
        "note": "critic prediction only, not ground-truth return"
    }

    print()
    print("========== RIGHT-GOAL CRITIC PREFERENCE ==========")
    print(json.dumps(critic_probe, ensure_ascii=False))

    def saturated(name):
        left = comparisons[name]["LEFT"]
        right = comparisons[name]["RIGHT"]

        return all(
            row["steer_negative_0p98_fraction"] >= 0.95
            and row["throttle_gt_0p98_fraction"] >= 0.95
            for row in (left, right)
        )

    saturation = {
        name: saturated(name)
        for name in ("T1", "T2", "T3")
    }

    if not saturation["T2"]:
        recommendation = (
            "USE_T2_FOR_CONTROLLED_CORRECTION"
        )
    elif not saturation["T1"]:
        recommendation = (
            "USE_T1_FOR_CONTROLLED_CORRECTION"
        )
    else:
        recommendation = (
            "REINITIALIZE_TD3_WITH_BALANCED_EXPLORATION"
        )

    report = {
        "state": "AUDIT_COMPLETE",
        "checkpoints": {
            "T1": sha(T1),
            "T2": sha(T2),
            "T3": sha(T3)
        },
        "replay_coverage": coverage,
        "actor_comparison": comparisons,
        "critic_probe": critic_probe,
        "policy_saturated": saturation,
        "recommended_next_experiment": recommendation,
        "training_started": False,
        "automatic_promotion": False
    }

    output = JOB / "collapse_audit.json"

    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    print()
    print("=" * 76)
    print("TD3-T5 DECISION")
    print("=" * 76)

    print("T1_SATURATED=", saturation["T1"])
    print("T2_SATURATED=", saturation["T2"])
    print("T3_SATURATED=", saturation["T3"])
    print("NEXT_EXPERIMENT=", recommendation)
    print("TD3_T5_COLLAPSE_AUDIT=PASS")
    print("FROZEN_MODELS_UNCHANGED=TRUE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            "TD3_T5_COLLAPSE_AUDIT=BLOCKED",
            repr(error),
            flush=True
        )
        traceback.print_exc()
    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
