from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from training.e2e_phase10_lidar_v7_curriculum.train_stage2a11_overnight_autonomous import (
    OvernightObstacleEnv,
    VehicleConfig,
    evaluate_task,
    save_json,
)


SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage2a11_overnight_autonomous/best_model.zip"
)
OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage2a13_terminal_braking_fix"
)

VEHICLE_CONFIG = VehicleConfig(
    name="CONDIM3_PARALLEL",
    condim3=True,
    ackermann=False,
    front_free=False,
)

CAUTION_COLLISION_SEEDS = [
    142026,
    142085,
    142104,
    142119,
    142145,
]
LIGHT_COLLISION_SEEDS = [
    143004,
    143011,
    143029,
    143050,
    143052,
    143063,
    143097,
    143098,
]


class Stage2A13BrakingEnv(OvernightObstacleEnv):
    def __init__(self, seed: int = 61200) -> None:
        super().__init__(
            vehicle_config=VEHICLE_CONFIG,
            fixed_task=None,
            seed=seed,
        )
        self._sample_rng = np.random.default_rng(seed)
        self._last_front_clearance = float(self.lidar_max_range)
        self._actual_training_seed = -1
        self._sampling_mode = "normal"

        caution_neighborhood = [
            value
            for seed_value in CAUTION_COLLISION_SEEDS
            for value in range(seed_value - 2, seed_value + 3)
        ]
        light_neighborhood = [
            value
            for seed_value in LIGHT_COLLISION_SEEDS
            for value in range(seed_value - 2, seed_value + 3)
        ]
        self._hard_cases = (
            [
                ("caution", seed_value)
                for seed_value in CAUTION_COLLISION_SEEDS
                for _ in range(4)
            ]
            + [
                ("light", seed_value)
                for seed_value in LIGHT_COLLISION_SEEDS
                for _ in range(4)
            ]
            + [
                ("caution", value)
                for value in caution_neighborhood
                if value not in CAUTION_COLLISION_SEEDS
            ]
            + [
                ("light", value)
                for value in light_neighborhood
                if value not in LIGHT_COLLISION_SEEDS
            ]
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        draw = float(self._sample_rng.random())

        if draw < 0.35:
            task, selected_seed = self._hard_cases[
                int(self._sample_rng.integers(0, len(self._hard_cases)))
            ]
            self._sampling_mode = "hard"
        else:
            task = str(
                self._sample_rng.choice(
                    ["a3", "bridge", "caution", "light"],
                    p=[0.36, 0.29, 0.20, 0.15],
                )
            )
            selected_seed = int(
                self._sample_rng.integers(200000, 2_000_000_000)
            )
            self._sampling_mode = "normal"

        observation, info = super().reset(
            seed=selected_seed,
            options={"task_mode": task},
        )
        self._actual_training_seed = int(selected_seed)
        self._last_front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )
        info["sampling_mode"] = self._sampling_mode
        info["training_seed"] = self._actual_training_seed
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(
            action
        )

        front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )
        goal_dist = float(info.get("goal_dist", 999.0))
        # Stage2A.12 used an optional info field that was usually absent.
        # Compute speed risk from the actual SAC speed action.
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size < 1:
            raise RuntimeError("empty SAC action")
        speed_ratio = float(
            np.clip(abs(float(action_array[0])), 0.0, 1.0)
        )
        collision = bool(info.get("collision", False))

        terminal_zone = goal_dist <= 0.40
        critical_zone = front_clearance <= 0.275

        extra_reward = 0.0
        if terminal_zone and critical_zone:
            clearance_risk = float(
                np.clip(
                    (0.275 - front_clearance) / (0.275 - 0.165),
                    0.0,
                    1.0,
                )
            )
            goal_risk = float(
                np.clip(
                    (0.40 - goal_dist) / (0.40 - 0.18),
                    0.0,
                    1.0,
                )
            )
            risk = clearance_risk * (0.35 + 0.65 * goal_risk)

            clearance_drop = max(
                0.0,
                self._last_front_clearance - front_clearance,
            )
            normalized_approach = float(
                np.clip(clearance_drop / 0.018, 0.0, 1.0)
            )
            ttc_risk = clearance_risk * normalized_approach

            extra_reward -= 3.8 * risk * speed_ratio**2
            extra_reward -= 2.6 * ttc_risk * speed_ratio

            if speed_ratio <= 0.28:
                extra_reward += 0.22 * risk
            elif speed_ratio >= 0.60:
                extra_reward -= 0.60 * risk

        if collision and terminal_zone:
            extra_reward -= 32.0

        reward = float(reward + extra_reward)
        self._last_front_clearance = front_clearance

        info["stage2a13_extra_reward"] = float(extra_reward)
        info["stage2a13_speed_ratio"] = float(speed_ratio)
        info["sampling_mode"] = self._sampling_mode
        info["training_seed"] = self._actual_training_seed
        return observation, reward, terminated, truncated, info


def freeze_to_speed_output(model: SAC) -> dict[str, Any]:
    actor = model.actor

    for parameter in actor.parameters():
        parameter.requires_grad = False

    actor.mu.weight.requires_grad = True
    actor.mu.bias.requires_grad = True

    if int(actor.mu.out_features) != 2:
        raise RuntimeError(
            f"actor.mu.out_features must be 2, got {actor.mu.out_features}"
        )

    weight_mask = torch.zeros_like(actor.mu.weight)
    bias_mask = torch.zeros_like(actor.mu.bias)
    weight_mask[0, :] = 1.0
    bias_mask[0] = 1.0

    actor.mu.weight.register_hook(
        lambda gradient: gradient * weight_mask
    )
    actor.mu.bias.register_hook(
        lambda gradient: gradient * bias_mask
    )

    for group in actor.optimizer.param_groups:
        group["lr"] = 1.2e-5

    for group in model.critic.optimizer.param_groups:
        group["lr"] = 5.0e-6

    if model.ent_coef_optimizer is not None:
        for group in model.ent_coef_optimizer.param_groups:
            group["lr"] = 0.0

    trainable = [
        name
        for name, parameter in actor.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "actor_trainable_parameters": trainable,
        "actor_learning_rate": 1.2e-5,
        "critic_learning_rate": 5.0e-6,
        "steer_output_gradient": "frozen",
        "speed_output_gradient": "enabled",
    }


def evaluate_exact_seeds(
    model: SAC,
    task: str,
    seeds: list[int],
) -> dict[str, Any]:
    env = OvernightObstacleEnv(
        vehicle_config=VEHICLE_CONFIG,
        fixed_task=task,
        seed=79900,
    )
    rows: list[dict[str, Any]] = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    try:
        for value in seeds:
            observation, _ = env.reset(seed=int(value))
            final_info: dict[str, Any] = {}
            minimum_front = float("inf")
            minimum_clearance = float("inf")
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
                minimum_front = min(
                    minimum_front,
                    float(info.get("front_clearance", 6.0)),
                )
                minimum_clearance = min(
                    minimum_clearance,
                    float(info.get("min_lidar_m", 6.0)),
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
                    "seed": int(value),
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "steps": steps,
                    "final_goal_dist": float(
                        final_info.get("goal_dist", 999.0)
                    ),
                    "minimum_front_clearance": minimum_front,
                    "minimum_clearance": minimum_clearance,
                }
            )
    finally:
        env.close()

    count = len(seeds)
    return {
        "task": task,
        "episodes": count,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / count,
        "collision_rate": collision_count / count,
        "timeout_rate": timeout_count / count,
        "results": rows,
    }


def evaluate_candidate(
    model: SAC,
    label: str,
    seed_offset: int,
) -> dict[str, Any]:
    retention = evaluate_task(
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

    hard_collision_count = (
        int(hard_caution["collision_count"])
        + int(hard_light["collision_count"])
    )
    hard_success_count = (
        int(hard_caution["success_count"])
        + int(hard_light["success_count"])
    )

    normal_max_collision = max(
        float(retention["collision_rate"]),
        float(bridge["collision_rate"]),
        float(caution["collision_rate"]),
        float(light["collision_rate"]),
    )
    weighted_success = (
        0.25 * float(retention["success_rate"])
        + 0.25 * float(bridge["success_rate"])
        + 0.30 * float(caution["success_rate"])
        + 0.20 * float(light["success_rate"])
    )

    acceptable = bool(
        retention["success_rate"] >= 0.78
        and bridge["success_rate"] >= 0.78
        and caution["success_rate"] >= 0.68
        and light["success_rate"] >= 0.58
    )
    gate = bool(
        hard_collision_count == 0
        and retention["success_rate"] >= 0.80
        and retention["collision_rate"] == 0.0
        and bridge["success_rate"] >= 0.80
        and bridge["collision_rate"] == 0.0
        and caution["success_rate"] >= 0.70
        and caution["collision_rate"] <= 0.025
        and light["success_rate"] >= 0.60
        and light["collision_rate"] <= 0.05
    )

    result = {
        "label": label,
        "retention": retention,
        "bridge": bridge,
        "caution": caution,
        "light": light,
        "hard_caution": hard_caution,
        "hard_light": hard_light,
        "hard_collision_count": hard_collision_count,
        "hard_success_count": hard_success_count,
        "normal_max_collision": normal_max_collision,
        "weighted_success": weighted_success,
        "acceptable": acceptable,
        "gate": gate,
        "selection_key": [
            0 if acceptable else 1,
            hard_collision_count,
            normal_max_collision,
            -hard_success_count,
            -weighted_success,
        ],
    }

    print("\n========== STAGE2A13", label, "==========")
    print("A3_SUCCESS_RATE =", retention["success_rate"])
    print("A3_COLLISION_RATE =", retention["collision_rate"])
    print("BRIDGE_SUCCESS_RATE =", bridge["success_rate"])
    print("BRIDGE_COLLISION_RATE =", bridge["collision_rate"])
    print("CAUTION_SUCCESS_RATE =", caution["success_rate"])
    print("CAUTION_COLLISION_RATE =", caution["collision_rate"])
    print("LIGHT_SUCCESS_RATE =", light["success_rate"])
    print("LIGHT_COLLISION_RATE =", light["collision_rate"])
    print(
        "HARD_CAUTION_COLLISION_COUNT =",
        hard_caution["collision_count"],
    )
    print(
        "HARD_LIGHT_COLLISION_COUNT =",
        hard_light["collision_count"],
    )
    print("HARD_COLLISION_COUNT =", hard_collision_count)
    print("WEIGHTED_SUCCESS =", weighted_success)
    print("STAGE2A13_GATE =", "PASS" if gate else "FAIL")
    return result


def better(
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
    print("VEHICLE_CONFIG =", VEHICLE_CONFIG)
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("SPEED_OUTPUT_ONLY_UPDATE = True")
    print("STEER_OUTPUT_FROZEN = True")
    print("HARD_CAUTION_SEEDS =", CAUTION_COLLISION_SEEDS)
    print("HARD_LIGHT_SEEDS =", LIGHT_COLLISION_SEEDS)

    baseline_model = SAC.load(str(SOURCE_MODEL), device=device)
    baseline = evaluate_candidate(
        baseline_model,
        label="SOURCE_STAGE2A11",
        seed_offset=0,
    )
    save_json(OUTPUT_ROOT / "baseline_evaluation.json", baseline)

    baseline_model.save(str(OUTPUT_ROOT / "best_model"))
    save_json(OUTPUT_ROOT / "best_evaluation.json", baseline)
    save_json(
        OUTPUT_ROOT / "best_metadata.json",
        {
            "source": "SOURCE_STAGE2A11",
            "chunk": 0,
        },
    )

    base_env = Stage2A13BrakingEnv()
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
        ),
    )

    model = SAC.load(
        str(SOURCE_MODEL),
        env=monitor_env,
        device=device,
    )
    freeze_report = freeze_to_speed_output(model)
    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.asarray([0.008, 0.0], dtype=np.float32),
    )
    save_json(OUTPUT_ROOT / "freeze_report.json", freeze_report)
    print("FREEZE_REPORT =", freeze_report)
    print("ACTION_NOISE_SIGMA = [0.008, 0.0]")

    best = baseline
    no_improvement = 0
    records = [baseline]

    for chunk in range(1, 7):
        print("\nTRAIN_CHUNK =", chunk, "/ 6")
        print("CHUNK_TIMESTEPS = 6000")

        model.learn(
            total_timesteps=6_000,
            reset_num_timesteps=False,
            log_interval=4,
            progress_bar=False,
        )

        record = evaluate_candidate(
            model,
            label=f"CHUNK{chunk}_6K",
            seed_offset=chunk * 1000,
        )
        records.append(record)
        save_json(
            OUTPUT_ROOT / f"chunk_{chunk}_evaluation.json",
            record,
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
                    "source": f"CHUNK{chunk}_6K",
                    "chunk": chunk,
                },
            )
            print("STAGE2A13_BEST_UPDATED =", chunk)
        else:
            no_improvement += 1

        if record["gate"]:
            print("STAGE2A13_EARLY_STOP = GATE_PASS")
            break

        if (
            record["retention"]["success_rate"] < 0.76
            or record["bridge"]["success_rate"] < 0.76
            or record["normal_max_collision"] > 0.20
        ):
            print("STAGE2A13_EARLY_STOP = CAPABILITY_DEGRADATION")
            break

        if no_improvement >= 2:
            print("STAGE2A13_EARLY_STOP = NO_IMPROVEMENT")
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
        "ENTER_STAGE2B_LIGHT"
        if final["gate"]
        else "KEEP_STAGE2A13_BEST_AND_REVIEW_REMAINING_SEEDS"
    )
    report = {
        "source_model": str(SOURCE_MODEL),
        "vehicle_config": VEHICLE_CONFIG.__dict__,
        "freeze_report": freeze_report,
        "records": records,
        "final": final,
        "recommendation": recommendation,
    }
    save_json(OUTPUT_ROOT / "final_report.json", report)

    print("\nSTAGE2A13 = FINISHED")
    print("BEST_MODEL =", OUTPUT_ROOT / "best_model.zip")
    print("BEST_EVALUATION =", OUTPUT_ROOT / "best_evaluation.json")
    print("FINAL_REPORT =", OUTPUT_ROOT / "final_report.json")
    print("RECOMMENDATION =", recommendation)


if __name__ == "__main__":
    main()
