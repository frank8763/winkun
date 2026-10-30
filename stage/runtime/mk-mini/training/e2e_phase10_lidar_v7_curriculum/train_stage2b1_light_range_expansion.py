from __future__ import annotations

import json
import math
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
    TASKS,
    save_json,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a13_terminal_braking_fix import (
    CAUTION_COLLISION_SEEDS,
    LIGHT_COLLISION_SEEDS,
    VEHICLE_CONFIG,
    evaluate_exact_seeds,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a17_terminal_chatter_fix import (
    FOCUS_SEED,
    Stage2A17ChatterEnv,
    evaluate_focus_trace,
)


SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "confirmed/stage2a17_pass_best_model.zip"
)
OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage2b1_light_range_expansion"
)

STAGE2B_TASKS: dict[str, dict[str, float | int]] = {
    "light_b1": {
        "goal_distance_min": 0.68,
        "goal_distance_max": 0.90,
        "goal_angle_deg": 40.0,
        "goal_clearance_margin": 0.34,
        "max_steps": 800,
    },
    "light_b2": {
        "goal_distance_min": 0.75,
        "goal_distance_max": 1.00,
        "goal_angle_deg": 45.0,
        "goal_clearance_margin": 0.32,
        "max_steps": 900,
    },
}


class Stage2B1Env(Stage2A17ChatterEnv):
    """Stage2B.1: retain Stage2A.17 safety and expand task geometry only."""

    def __init__(
        self,
        fixed_task: str | None = None,
        seed: int = 82100,
    ) -> None:
        super().__init__(seed=seed)
        self.fixed_task_stage2b = fixed_task
        self._stage2b_rng = np.random.default_rng(seed + 211)

    def _apply_task(self, task: str) -> None:
        if task in TASKS:
            OvernightObstacleEnv._apply_task(self, task)
            return

        if task not in STAGE2B_TASKS:
            raise ValueError(f"invalid Stage2B.1 task: {task}")

        config = STAGE2B_TASKS[task]
        self._active_task = task
        self.curriculum_stage = f"stage2b1_{task}"
        self.goal_distance_min = float(config["goal_distance_min"])
        self.goal_distance_max = float(config["goal_distance_max"])
        self.goal_angle_limit = math.radians(
            float(config["goal_angle_deg"])
        )
        self.goal_clearance_margin = float(
            config["goal_clearance_margin"]
        )
        self.max_steps = int(config["max_steps"])

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        forced = None if options is None else options.get("task_mode")

        if forced is not None:
            task = str(forced)
            selected_seed = int(
                seed
                if seed is not None
                else self._stage2b_rng.integers(
                    200000,
                    2_000_000_000,
                )
            )
            sampling_mode = "forced"
        elif self.fixed_task_stage2b is not None:
            task = self.fixed_task_stage2b
            selected_seed = int(
                seed
                if seed is not None
                else self._stage2b_rng.integers(
                    200000,
                    2_000_000_000,
                )
            )
            sampling_mode = "evaluation"
        else:
            task = str(
                self._stage2b_rng.choice(
                    [
                        "a3",
                        "bridge",
                        "caution",
                        "light",
                        "light_b1",
                        "light_b2",
                    ],
                    p=[
                        0.24,
                        0.20,
                        0.14,
                        0.17,
                        0.17,
                        0.08,
                    ],
                )
            )
            selected_seed = int(
                self._stage2b_rng.integers(
                    200000,
                    2_000_000_000,
                )
            )
            sampling_mode = (
                "retention"
                if task in ("a3", "bridge")
                else "stage2b"
            )

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

        info["task_mode"] = task
        info["sampling_mode"] = sampling_mode
        info["training_seed"] = selected_seed
        return observation, info


def configure_model(model: SAC) -> dict[str, Any]:
    actor = model.actor

    for parameter in actor.parameters():
        parameter.requires_grad = False

    actor.mu.weight.requires_grad = True
    actor.mu.bias.requires_grad = True

    if int(actor.mu.out_features) != 2:
        raise RuntimeError(
            f"expected two actions, got {actor.mu.out_features}"
        )

    set_learning_rate(model, 2.0e-6)

    return {
        "actor_trainable_parameters": [
            name
            for name, parameter in actor.named_parameters()
            if parameter.requires_grad
        ],
        "feature_layers_frozen": True,
        "speed_and_steering_outputs_trainable": True,
        "scheduled_learning_rate": 2.0e-6,
    }


def evaluate_task_stage2b(
    model: SAC,
    task: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    env = Stage2B1Env(
        fixed_task=task,
        seed=seed + 7000,
    )

    rows: list[dict[str, Any]] = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    try:
        for episode in range(int(episodes)):
            actual_seed = int(seed + episode)
            observation, reset_info = env.reset(seed=actual_seed)
            final_info: dict[str, Any] = {}
            minimum_clearance = float("inf")
            minimum_front = float("inf")
            steps = 0

            for step in range(env.max_steps):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                observation, _, terminated, truncated, info = env.step(
                    action
                )
                final_info = dict(info)
                steps = step + 1
                minimum_clearance = min(
                    minimum_clearance,
                    float(info.get("min_lidar_m", 6.0)),
                )
                minimum_front = min(
                    minimum_front,
                    float(info.get("front_clearance", 6.0)),
                )

                if terminated or truncated:
                    break

            success = bool(final_info.get("reach_goal", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(
                not success
                and not collision
                and steps >= env.max_steps
            )

            success_count += int(success)
            collision_count += int(collision)
            timeout_count += int(timeout)

            rows.append(
                {
                    "episode": episode + 1,
                    "seed": actual_seed,
                    "initial_goal_dist": float(
                        reset_info.get("goal_dist", 999.0)
                    ),
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "steps": steps,
                    "final_goal_dist": float(
                        final_info.get("goal_dist", 999.0)
                    ),
                    "minimum_clearance": minimum_clearance,
                    "minimum_front_clearance": minimum_front,
                }
            )
    finally:
        env.close()

    count = int(episodes)
    result = {
        "task": task,
        "episodes": count,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / count,
        "collision_rate": collision_count / count,
        "timeout_rate": timeout_count / count,
        "mean_final_goal_dist": float(
            np.mean([row["final_goal_dist"] for row in rows])
        ),
        "mean_minimum_clearance": float(
            np.mean([row["minimum_clearance"] for row in rows])
        ),
        "mean_minimum_front_clearance": float(
            np.mean(
                [
                    row["minimum_front_clearance"]
                    for row in rows
                ]
            )
        ),
        "results": rows,
    }

    print(f"\n========== EVAL {task.upper()} {count} ==========")
    print("SUCCESS_RATE =", result["success_rate"])
    print("COLLISION_RATE =", result["collision_rate"])
    print("TIMEOUT_RATE =", result["timeout_rate"])
    print(
        "MEAN_MINIMUM_CLEARANCE =",
        result["mean_minimum_clearance"],
    )
    return result


def evaluate_candidate(
    model: SAC,
    label: str,
    seed_offset: int,
) -> dict[str, Any]:
    episodes = {
        "a3": 50,
        "bridge": 50,
        "caution": 70,
        "light": 70,
        "light_b1": 90,
        "light_b2": 90,
    }
    seeds = {
        "a3": 10730,
        "bridge": 51000,
        "caution": 62000,
        "light": 63000,
        "light_b1": 72000,
        "light_b2": 73000,
    }

    tasks = {
        task: evaluate_task_stage2b(
            model,
            task=task,
            episodes=episodes[task],
            seed=seeds[task] + seed_offset,
        )
        for task in (
            "a3",
            "bridge",
            "caution",
            "light",
            "light_b1",
            "light_b2",
        )
    }

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
    focus = evaluate_focus_trace(model, seed=FOCUS_SEED)

    hard_collision_count = (
        int(hard_caution["collision_count"])
        + int(hard_light["collision_count"])
    )

    weights = {
        "a3": 0.16,
        "bridge": 0.16,
        "caution": 0.14,
        "light": 0.16,
        "light_b1": 0.22,
        "light_b2": 0.16,
    }
    weighted_success = sum(
        weights[name] * float(tasks[name]["success_rate"])
        for name in weights
    )
    weighted_collision = sum(
        weights[name] * float(tasks[name]["collision_rate"])
        for name in weights
    )
    max_collision = max(
        float(result["collision_rate"])
        for result in tasks.values()
    )
    retention_floor = min(
        float(tasks["a3"]["success_rate"]),
        float(tasks["bridge"]["success_rate"]),
    )

    acceptable = bool(
        hard_collision_count == 0
        and not bool(focus["collision"])
        and tasks["a3"]["success_rate"] >= 0.78
        and tasks["bridge"]["success_rate"] >= 0.76
        and tasks["caution"]["success_rate"] >= 0.75
        and tasks["light"]["success_rate"] >= 0.60
        and tasks["light_b1"]["success_rate"] >= 0.48
        and tasks["light_b2"]["success_rate"] >= 0.35
        and max_collision <= 0.08
    )

    gate = bool(
        hard_collision_count == 0
        and not bool(focus["collision"])
        and focus["speed_flip_ratio"] <= 0.15
        and focus["steer_flip_ratio"] <= 0.15
        and tasks["a3"]["success_rate"] >= 0.82
        and tasks["a3"]["collision_rate"] == 0.0
        and tasks["bridge"]["success_rate"] >= 0.80
        and tasks["bridge"]["collision_rate"] == 0.0
        and tasks["caution"]["success_rate"] >= 0.82
        and tasks["caution"]["collision_rate"] <= 0.025
        and tasks["light"]["success_rate"] >= 0.65
        and tasks["light"]["collision_rate"] <= 0.04
        and tasks["light_b1"]["success_rate"] >= 0.58
        and tasks["light_b1"]["collision_rate"] <= 0.05
        and tasks["light_b2"]["success_rate"] >= 0.45
        and tasks["light_b2"]["collision_rate"] <= 0.06
    )

    result = {
        "label": label,
        "tasks": tasks,
        "hard_caution": hard_caution,
        "hard_light": hard_light,
        "focus": focus,
        "hard_collision_count": hard_collision_count,
        "weighted_success": weighted_success,
        "weighted_collision": weighted_collision,
        "max_collision": max_collision,
        "retention_floor": retention_floor,
        "acceptable": acceptable,
        "gate": gate,
        "selection_key": [
            0 if acceptable else 1,
            hard_collision_count,
            1 if focus["collision"] else 0,
            max_collision,
            -retention_floor,
            -float(tasks["light_b2"]["success_rate"]),
            -float(tasks["light_b1"]["success_rate"]),
            -weighted_success,
        ],
    }

    print(f"\n========== STAGE2B1 {label} ==========")
    for task in (
        "a3",
        "bridge",
        "caution",
        "light",
        "light_b1",
        "light_b2",
    ):
        print(
            f"{task.upper()}_SUCCESS_RATE =",
            tasks[task]["success_rate"],
        )
        print(
            f"{task.upper()}_COLLISION_RATE =",
            tasks[task]["collision_rate"],
        )

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
    print("WEIGHTED_SUCCESS =", weighted_success)
    print("MAX_COLLISION =", max_collision)
    print("STAGE2B1_GATE =", "PASS" if gate else "FAIL")
    return result


def is_better(
    candidate: dict[str, Any],
    best: dict[str, Any],
) -> bool:
    return tuple(candidate["selection_key"]) < tuple(
        best["selection_key"]
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("SOURCE_MODEL =", SOURCE_MODEL)
    print("DEVICE =", device)
    print("STAGE = STAGE2B1_LIGHT_RANGE_EXPANSION")
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("VEHICLE_DYNAMICS_CHANGED = False")
    print("OBSERVATION_DIMENSION_CHANGED = False")
    print("REWARD_STRUCTURE_CHANGED = False")
    print("STAGE2A17_CHATTER_FIX_PRESERVED = True")
    print(
        "LIGHT_B1_CONFIG =",
        json.dumps(STAGE2B_TASKS["light_b1"]),
    )
    print(
        "LIGHT_B2_CONFIG =",
        json.dumps(STAGE2B_TASKS["light_b2"]),
    )

    source = SAC.load(str(SOURCE_MODEL), device=device)
    baseline = evaluate_candidate(
        source,
        label="SOURCE_STAGE2A17",
        seed_offset=0,
    )
    save_json(OUTPUT_ROOT / "baseline_evaluation.json", baseline)

    source.save(str(OUTPUT_ROOT / "best_model"))
    save_json(OUTPUT_ROOT / "best_evaluation.json", baseline)
    save_json(
        OUTPUT_ROOT / "best_metadata.json",
        {
            "source": "SOURCE_STAGE2A17",
            "chunk": 0,
        },
    )

    base_env = Stage2B1Env(seed=82100)
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
    configure_report = configure_model(model)
    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.asarray([0.004, 0.008], dtype=np.float32),
    )

    save_json(
        OUTPUT_ROOT / "configure_report.json",
        configure_report,
    )
    print("CONFIGURE_REPORT =", configure_report)
    print("ACTION_NOISE_SIGMA = [0.004, 0.008]")

    best = baseline
    records = [baseline]
    no_improvement = 0

    for chunk in range(1, 7):
        print("\nTRAIN_CHUNK =", chunk, "/ 6")
        print("CHUNK_TIMESTEPS = 5000")

        model.learn(
            total_timesteps=5_000,
            reset_num_timesteps=False,
            log_interval=4,
            progress_bar=False,
        )

        record = evaluate_candidate(
            model,
            label=f"CHUNK{chunk}_5K",
            seed_offset=chunk * 1000,
        )
        records.append(record)

        save_json(
            OUTPUT_ROOT / f"chunk_{chunk}_evaluation.json",
            record,
        )
        model.save(str(OUTPUT_ROOT / f"chunk_{chunk}_model"))

        if is_better(record, best):
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
                    "source": f"CHUNK{chunk}_5K",
                    "chunk": chunk,
                },
            )
            print("STAGE2B1_BEST_UPDATED =", chunk)
        else:
            no_improvement += 1

        if record["gate"]:
            print("STAGE2B1_EARLY_STOP = GATE_PASS")
            break

        if (
            record["hard_collision_count"] > 2
            or bool(record["focus"]["collision"])
            or record["max_collision"] > 0.15
            or record["tasks"]["a3"]["success_rate"] < 0.74
            or record["tasks"]["bridge"]["success_rate"] < 0.72
        ):
            print(
                "STAGE2B1_EARLY_STOP = SAFETY_OR_RETENTION_DEGRADATION"
            )
            break

        if no_improvement >= 2:
            print("STAGE2B1_EARLY_STOP = NO_IMPROVEMENT")
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

    recommendation = (
        "ENTER_STAGE2B2_OBSTACLE_COUNT_EXPANSION"
        if final["gate"]
        else "KEEP_STAGE2B1_BEST_AND_REVIEW_FAILURE_SEEDS"
    )

    report = {
        "source_model": str(SOURCE_MODEL),
        "stage2b_tasks": STAGE2B_TASKS,
        "configure_report": configure_report,
        "records": records,
        "final": final,
        "recommendation": recommendation,
    }
    save_json(OUTPUT_ROOT / "final_report.json", report)

    print("\nSTAGE2B1 = FINISHED")
    print("BEST_MODEL =", OUTPUT_ROOT / "best_model.zip")
    print("BEST_EVALUATION =", OUTPUT_ROOT / "best_evaluation.json")
    print("FINAL_REPORT =", OUTPUT_ROOT / "final_report.json")
    print("RECOMMENDATION =", recommendation)


if __name__ == "__main__":
    main()
