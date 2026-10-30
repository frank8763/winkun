from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from training.e2e_phase10_lidar_v7_curriculum.train_stage2a1_safety_reward import (
    set_learning_rate,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a11_overnight_autonomous import (
    OvernightObstacleEnv,
    evaluate_task,
    save_json,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a13_terminal_braking_fix import (
    CAUTION_COLLISION_SEEDS,
    LIGHT_COLLISION_SEEDS,
    Stage2A13BrakingEnv,
    VEHICLE_CONFIG,
    evaluate_exact_seeds,
)


SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage2a15_speed_head_interpolation/best_model.zip"
)
OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage2a17_terminal_chatter_fix"
)

FOCUS_TASK = "light"
FOCUS_SEED = 143097
FOCUS_NEIGHBOR_SEEDS = list(range(143092, 143103))


class Stage2A17ChatterEnv(Stage2A13BrakingEnv):
    """Fix the terminal two-step speed/steering limit cycle."""

    def __init__(self, seed: int = 81700) -> None:
        super().__init__(seed=seed)
        self._focus_rng = np.random.default_rng(seed + 177)
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._previous_delta = np.zeros(2, dtype=np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        draw = float(self._focus_rng.random())

        if draw < 0.20:
            task = FOCUS_TASK
            selected_seed = int(
                self._focus_rng.choice(FOCUS_NEIGHBOR_SEEDS)
            )
            sampling_mode = "focus_143097"
        elif draw < 0.70:
            task = str(
                self._focus_rng.choice(
                    ["a3", "bridge"],
                    p=[0.56, 0.44],
                )
            )
            selected_seed = int(
                self._focus_rng.integers(200000, 2_000_000_000)
            )
            sampling_mode = "retention"
        else:
            task = str(
                self._focus_rng.choice(
                    ["caution", "light"],
                    p=[0.58, 0.42],
                )
            )
            selected_seed = int(
                self._focus_rng.integers(200000, 2_000_000_000)
            )
            sampling_mode = "normal_obstacle"

        observation, info = OvernightObstacleEnv.reset(
            self,
            seed=selected_seed,
            options={"task_mode": task},
        )

        self._sampling_mode = sampling_mode
        self._actual_training_seed = selected_seed
        self._last_front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )
        self._previous_action.fill(0.0)
        self._previous_delta.fill(0.0)

        info["sampling_mode"] = sampling_mode
        info["training_seed"] = selected_seed
        return observation, info

    def step(self, action: np.ndarray):
        action_array = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)
        if action_array.size != 2:
            raise RuntimeError(
                f"expected two SAC actions, got {action_array.shape}"
            )

        observation, reward, terminated, truncated, info = super().step(
            action_array
        )

        goal_dist = float(info.get("goal_dist", 999.0))
        front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )

        delta = action_array - self._previous_action
        acceleration = delta - self._previous_delta

        terminal_risk = float(
            np.clip(
                min(
                    (0.45 - goal_dist) / 0.25,
                    (0.30 - front_clearance) / 0.12,
                ),
                0.0,
                1.0,
            )
        )

        chatter_penalty = 0.0
        if terminal_risk > 0.0:
            speed_delta = float(delta[0])
            steer_delta = float(delta[1])
            speed_acceleration = float(acceleration[0])
            steer_acceleration = float(acceleration[1])

            chatter_penalty -= terminal_risk * (
                0.90 * speed_delta**2
                + 1.35 * steer_delta**2
                + 0.45 * speed_acceleration**2
                + 0.70 * steer_acceleration**2
            )

            # Extra penalty for the exact two-step alternating pattern.
            speed_flip = (
                speed_delta * float(self._previous_delta[0]) < 0.0
                and abs(speed_delta) > 0.08
                and abs(float(self._previous_delta[0])) > 0.08
            )
            steer_flip = (
                steer_delta * float(self._previous_delta[1]) < 0.0
                and abs(steer_delta) > 0.20
                and abs(float(self._previous_delta[1])) > 0.20
            )
            if speed_flip:
                chatter_penalty -= 0.20 * terminal_risk
            if steer_flip:
                chatter_penalty -= 0.35 * terminal_risk

            if (
                abs(speed_delta) <= 0.035
                and abs(steer_delta) <= 0.08
            ):
                chatter_penalty += 0.035 * terminal_risk

        reward = float(reward + chatter_penalty)

        self._previous_action = action_array.copy()
        self._previous_delta = delta.copy()

        info["stage2a17_chatter_penalty"] = float(chatter_penalty)
        info["stage2a17_speed_delta"] = float(delta[0])
        info["stage2a17_steer_delta"] = float(delta[1])
        info["sampling_mode"] = self._sampling_mode
        info["training_seed"] = self._actual_training_seed
        return observation, reward, terminated, truncated, info


def configure_final_action_layer(model: SAC) -> dict[str, Any]:
    actor = model.actor

    for parameter in actor.parameters():
        parameter.requires_grad = False

    actor.mu.weight.requires_grad = True
    actor.mu.bias.requires_grad = True

    if int(actor.mu.out_features) != 2:
        raise RuntimeError(
            f"expected two actions, got {actor.mu.out_features}"
        )

    # Important: set the SB3 LR schedule, not only optimizer groups.
    set_learning_rate(model, 4.0e-6)

    trainable = [
        name
        for name, parameter in actor.named_parameters()
        if parameter.requires_grad
    ]

    return {
        "actor_trainable_parameters": trainable,
        "actor_outputs_updated": ["speed", "steering"],
        "feature_layers_frozen": True,
        "scheduled_learning_rate": 4.0e-6,
    }


def evaluate_focus_trace(
    model: SAC,
    seed: int = FOCUS_SEED,
) -> dict[str, Any]:
    env = OvernightObstacleEnv(
        vehicle_config=VEHICLE_CONFIG,
        fixed_task=FOCUS_TASK,
        seed=seed + 5000,
    )

    rows: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=seed)
        previous_action: np.ndarray | None = None
        previous_delta: np.ndarray | None = None
        final_info: dict[str, Any] = {}

        for step in range(env.max_steps):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )
            action = np.asarray(action, dtype=np.float32).reshape(-1)

            observation, _, terminated, truncated, info = env.step(
                action
            )
            final_info = dict(info)

            delta = (
                np.zeros(2, dtype=np.float32)
                if previous_action is None
                else action - previous_action
            )
            acceleration = (
                np.zeros(2, dtype=np.float32)
                if previous_delta is None
                else delta - previous_delta
            )

            rows.append(
                {
                    "step": step + 1,
                    "speed_action": float(action[0]),
                    "steer_action": float(action[1]),
                    "speed_delta": float(delta[0]),
                    "steer_delta": float(delta[1]),
                    "speed_acceleration": float(acceleration[0]),
                    "steer_acceleration": float(acceleration[1]),
                    "goal_dist": float(info.get("goal_dist", 999.0)),
                    "front_clearance": float(
                        info.get("front_clearance", 999.0)
                    ),
                    "collision": bool(info.get("collision", False)),
                    "reach_goal": bool(info.get("reach_goal", False)),
                }
            )

            previous_action = action.copy()
            previous_delta = delta.copy()

            if terminated or truncated:
                break
    finally:
        env.close()

    danger_rows = [
        row
        for row in rows
        if row["goal_dist"] <= 0.45
        and row["front_clearance"] <= 0.30
    ]

    speed_flips = 0
    steer_flips = 0
    for index in range(1, len(danger_rows)):
        previous = danger_rows[index - 1]
        current = danger_rows[index]

        if (
            previous["speed_delta"] * current["speed_delta"] < 0.0
            and abs(previous["speed_delta"]) > 0.08
            and abs(current["speed_delta"]) > 0.08
        ):
            speed_flips += 1

        if (
            previous["steer_delta"] * current["steer_delta"] < 0.0
            and abs(previous["steer_delta"]) > 0.20
            and abs(current["steer_delta"]) > 0.20
        ):
            steer_flips += 1

    denominator = max(len(danger_rows) - 1, 1)

    summary = {
        "seed": seed,
        "steps": len(rows),
        "collision": bool(final_info.get("collision", False)),
        "reach_goal": bool(final_info.get("reach_goal", False)),
        "final_goal_dist": float(
            final_info.get("goal_dist", 999.0)
        ),
        "danger_steps": len(danger_rows),
        "danger_mean_abs_speed_delta": (
            float(
                np.mean(
                    [
                        abs(float(row["speed_delta"]))
                        for row in danger_rows
                    ]
                )
            )
            if danger_rows
            else 0.0
        ),
        "danger_mean_abs_steer_delta": (
            float(
                np.mean(
                    [
                        abs(float(row["steer_delta"]))
                        for row in danger_rows
                    ]
                )
            )
            if danger_rows
            else 0.0
        ),
        "speed_flip_ratio": speed_flips / denominator,
        "steer_flip_ratio": steer_flips / denominator,
        "rows": rows,
    }
    return summary


def evaluate_candidate(
    model: SAC,
    label: str,
    seed_offset: int,
) -> dict[str, Any]:
    a3 = evaluate_task(
        model,
        VEHICLE_CONFIG,
        task="a3",
        episodes=50,
        seed=10730 + seed_offset,
    )
    bridge = evaluate_task(
        model,
        VEHICLE_CONFIG,
        task="bridge",
        episodes=50,
        seed=51000 + seed_offset,
    )
    caution = evaluate_task(
        model,
        VEHICLE_CONFIG,
        task="caution",
        episodes=80,
        seed=62000 + seed_offset,
    )
    light = evaluate_task(
        model,
        VEHICLE_CONFIG,
        task="light",
        episodes=80,
        seed=63000 + seed_offset,
    )
    hard_caution = evaluate_exact_seeds(
        model,
        "caution",
        CAUTION_COLLISION_SEEDS,
    )
    hard_light = evaluate_exact_seeds(
        model,
        "light",
        LIGHT_COLLISION_SEEDS,
    )
    focus = evaluate_focus_trace(model)

    hard_collision_count = (
        int(hard_caution["collision_count"])
        + int(hard_light["collision_count"])
    )
    max_collision = max(
        float(a3["collision_rate"]),
        float(bridge["collision_rate"]),
        float(caution["collision_rate"]),
        float(light["collision_rate"]),
    )
    weighted_success = (
        0.25 * float(a3["success_rate"])
        + 0.25 * float(bridge["success_rate"])
        + 0.30 * float(caution["success_rate"])
        + 0.20 * float(light["success_rate"])
    )
    chatter_score = (
        float(focus["danger_mean_abs_speed_delta"])
        + float(focus["danger_mean_abs_steer_delta"])
        + float(focus["speed_flip_ratio"])
        + float(focus["steer_flip_ratio"])
    )

    acceptable = bool(
        a3["success_rate"] >= 0.78
        and bridge["success_rate"] >= 0.78
        and caution["success_rate"] >= 0.70
        and light["success_rate"] >= 0.60
    )
    gate = bool(
        hard_collision_count == 0
        and not bool(focus["collision"])
        and a3["success_rate"] >= 0.80
        and a3["collision_rate"] == 0.0
        and bridge["success_rate"] >= 0.80
        and bridge["collision_rate"] == 0.0
        and caution["success_rate"] >= 0.70
        and caution["collision_rate"] <= 0.025
        and light["success_rate"] >= 0.60
        and light["collision_rate"] <= 0.05
        and focus["speed_flip_ratio"] <= 0.15
        and focus["steer_flip_ratio"] <= 0.15
    )

    result = {
        "label": label,
        "a3": a3,
        "bridge": bridge,
        "caution": caution,
        "light": light,
        "hard_caution": hard_caution,
        "hard_light": hard_light,
        "focus": focus,
        "hard_collision_count": hard_collision_count,
        "max_collision": max_collision,
        "weighted_success": weighted_success,
        "chatter_score": chatter_score,
        "acceptable": acceptable,
        "gate": gate,
        "selection_key": [
            0 if acceptable else 1,
            hard_collision_count,
            1 if focus["collision"] else 0,
            chatter_score,
            max_collision,
            -weighted_success,
        ],
    }

    print("\n========== STAGE2A17", label, "==========")
    print("A3_SUCCESS_RATE =", a3["success_rate"])
    print("A3_COLLISION_RATE =", a3["collision_rate"])
    print("BRIDGE_SUCCESS_RATE =", bridge["success_rate"])
    print("BRIDGE_COLLISION_RATE =", bridge["collision_rate"])
    print("CAUTION_SUCCESS_RATE =", caution["success_rate"])
    print("CAUTION_COLLISION_RATE =", caution["collision_rate"])
    print("LIGHT_SUCCESS_RATE =", light["success_rate"])
    print("LIGHT_COLLISION_RATE =", light["collision_rate"])
    print("HARD_COLLISION_COUNT =", hard_collision_count)
    print("FOCUS_COLLISION =", focus["collision"])
    print(
        "FOCUS_SPEED_FLIP_RATIO =",
        focus["speed_flip_ratio"],
    )
    print(
        "FOCUS_STEER_FLIP_RATIO =",
        focus["steer_flip_ratio"],
    )
    print(
        "FOCUS_MEAN_ABS_SPEED_DELTA =",
        focus["danger_mean_abs_speed_delta"],
    )
    print(
        "FOCUS_MEAN_ABS_STEER_DELTA =",
        focus["danger_mean_abs_steer_delta"],
    )
    print("WEIGHTED_SUCCESS =", weighted_success)
    print("STAGE2A17_GATE =", "PASS" if gate else "FAIL")
    return result


def better(
    candidate: dict[str, Any],
    best: dict[str, Any],
) -> bool:
    return tuple(candidate["selection_key"]) < tuple(
        best["selection_key"]
    )


def save_focus_csv(
    evaluation: dict[str, Any],
    path: Path,
) -> None:
    rows = evaluation["focus"]["rows"]
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("SOURCE_MODEL =", SOURCE_MODEL)
    print("DEVICE =", device)
    print("ROOT_CAUSE = TERMINAL_TWO_STEP_ACTION_CHATTER")
    print("FOCUS_TASK =", FOCUS_TASK)
    print("FOCUS_SEED =", FOCUS_SEED)
    print("FEATURE_LAYERS_FROZEN = True")
    print("SPEED_AND_STEERING_OUTPUTS_TRAINABLE = True")
    print("SB3_LR_SCHEDULE_FIXED = True")

    source = SAC.load(str(SOURCE_MODEL), device=device)
    baseline = evaluate_candidate(
        source,
        label="SOURCE_STAGE2A15",
        seed_offset=0,
    )
    save_json(OUTPUT_ROOT / "baseline_evaluation.json", baseline)
    save_focus_csv(
        baseline,
        OUTPUT_ROOT / "baseline_focus_trace.csv",
    )

    source.save(str(OUTPUT_ROOT / "best_model"))
    save_json(OUTPUT_ROOT / "best_evaluation.json", baseline)
    save_json(
        OUTPUT_ROOT / "best_metadata.json",
        {"source": "SOURCE_STAGE2A15", "chunk": 0},
    )

    base_env = Stage2A17ChatterEnv()
    monitor_env = Monitor(
        base_env,
        filename=str(OUTPUT_ROOT / "monitor.csv"),
        info_keywords=(
            "is_success",
            "collision",
            "goal_dist",
            "front_clearance",
            "task_mode",
            "sampling_mode",
            "training_seed",
            "stage2a17_chatter_penalty",
            "stage2a17_speed_delta",
            "stage2a17_steer_delta",
        ),
    )

    model = SAC.load(
        str(SOURCE_MODEL),
        env=monitor_env,
        device=device,
    )
    configure_report = configure_final_action_layer(model)
    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.asarray([0.003, 0.006], dtype=np.float32),
    )
    save_json(
        OUTPUT_ROOT / "configure_report.json",
        configure_report,
    )

    print("CONFIGURE_REPORT =", configure_report)
    print("ACTION_NOISE_SIGMA = [0.003, 0.006]")

    best = baseline
    no_improvement = 0
    records = [baseline]

    for chunk in range(1, 7):
        print("\nTRAIN_CHUNK =", chunk, "/ 6")
        print("CHUNK_TIMESTEPS = 4000")

        model.learn(
            total_timesteps=4_000,
            reset_num_timesteps=False,
            log_interval=4,
            progress_bar=False,
        )

        record = evaluate_candidate(
            model,
            label=f"CHUNK{chunk}_4K",
            seed_offset=chunk * 1000,
        )
        records.append(record)
        save_json(
            OUTPUT_ROOT / f"chunk_{chunk}_evaluation.json",
            record,
        )
        save_focus_csv(
            record,
            OUTPUT_ROOT / f"chunk_{chunk}_focus_trace.csv",
        )
        model.save(str(OUTPUT_ROOT / f"chunk_{chunk}_model"))

        if better(record, best):
            best = record
            no_improvement = 0
            model.save(str(OUTPUT_ROOT / "best_model"))
            if model.replay_buffer is not None:
                model.save_replay_buffer(
                    str(OUTPUT_ROOT / "best_replay.pkl")
                )
            save_json(
                OUTPUT_ROOT / "best_evaluation.json",
                record,
            )
            save_json(
                OUTPUT_ROOT / "best_metadata.json",
                {
                    "source": f"CHUNK{chunk}_4K",
                    "chunk": chunk,
                },
            )
            print("STAGE2A17_BEST_UPDATED =", chunk)
        else:
            no_improvement += 1

        if record["gate"]:
            print("STAGE2A17_EARLY_STOP = GATE_PASS")
            break

        if (
            record["a3"]["success_rate"] < 0.76
            or record["bridge"]["success_rate"] < 0.76
            or record["max_collision"] > 0.15
        ):
            print(
                "STAGE2A17_EARLY_STOP = CAPABILITY_DEGRADATION"
            )
            break

        if no_improvement >= 2:
            print("STAGE2A17_EARLY_STOP = NO_IMPROVEMENT")
            break

    monitor_env.close()

    final_model = SAC.load(
        str(OUTPUT_ROOT / "best_model.zip"),
        device=device,
    )
    final = evaluate_candidate(
        final_model,
        label="FINAL_BEST_CONFIRM",
        seed_offset=90000,
    )
    save_focus_csv(
        final,
        OUTPUT_ROOT / "final_focus_trace.csv",
    )

    recommendation = (
        "ENTER_STAGE2B_LIGHT"
        if final["gate"]
        else "KEEP_STAGE2A17_BEST_AND_INSPECT_REMAINING_FAILURES"
    )
    report = {
        "source_model": str(SOURCE_MODEL),
        "root_cause": "terminal_two_step_action_chatter",
        "configure_report": configure_report,
        "records": records,
        "final": final,
        "recommendation": recommendation,
    }
    save_json(OUTPUT_ROOT / "final_report.json", report)

    print("\nSTAGE2A17 = FINISHED")
    print("BEST_MODEL =", OUTPUT_ROOT / "best_model.zip")
    print("BEST_EVALUATION =", OUTPUT_ROOT / "best_evaluation.json")
    print("FINAL_REPORT =", OUTPUT_ROOT / "final_report.json")
    print("RECOMMENDATION =", recommendation)


if __name__ == "__main__":
    main()
