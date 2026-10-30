from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a3_terminal_env import (
    Phase10Stage1A3TerminalEnv,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a1_safety_reward import (
    Stage2A1SafetyEnv,
    save_json,
    set_learning_rate,
)


ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage2a11_overnight_autonomous"
)
SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage1a3_terminal_from_a2_20k/v7_stage1a3_best_model.zip"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")

ACTUATOR_NAMES = {
    "steer_left": "left_front_steer_act",
    "steer_right": "right_front_steer_act",
    "drive_left": "left_rear_wheel_act",
    "drive_right": "right_rear_wheel_act",
}
WHEEL_JOINT_NAMES = {
    "front_left": "left_front_wheel_joint",
    "front_right": "right_front_wheel_joint",
    "rear_left": "left_rear_wheel_joint",
    "rear_right": "right_rear_wheel_joint",
}


TASKS: dict[str, dict[str, float | int]] = {
    "a3": {
        "goal_distance_min": 0.45,
        "goal_distance_max": 0.65,
        "goal_angle_deg": 30.0,
        "goal_clearance_margin": 0.35,
        "max_steps": 500,
    },
    "bridge": {
        "goal_distance_min": 0.50,
        "goal_distance_max": 0.68,
        "goal_angle_deg": 30.0,
        "goal_clearance_margin": 0.43,
        "max_steps": 600,
    },
    "caution": {
        "goal_distance_min": 0.55,
        "goal_distance_max": 0.74,
        "goal_angle_deg": 32.0,
        "goal_clearance_margin": 0.40,
        "max_steps": 650,
    },
    "light": {
        "goal_distance_min": 0.60,
        "goal_distance_max": 0.82,
        "goal_angle_deg": 35.0,
        "goal_clearance_margin": 0.36,
        "max_steps": 700,
    },
}


@dataclass(frozen=True)
class VehicleConfig:
    name: str
    condim3: bool
    ackermann: bool
    front_free: bool


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    learning_rate: float
    speed_noise: float
    steer_noise: float
    warmup_mix: dict[str, float]
    main_mix: dict[str, float]


VEHICLE_CONFIGS = (
    VehicleConfig(
        name="BASELINE_CONDIM6_PARALLEL",
        condim3=False,
        ackermann=False,
        front_free=False,
    ),
    VehicleConfig(
        name="CONDIM3_PARALLEL",
        condim3=True,
        ackermann=False,
        front_free=False,
    ),
    VehicleConfig(
        name="CONDIM3_PARALLEL_FRONT_FREE",
        condim3=True,
        ackermann=False,
        front_free=True,
    ),
    VehicleConfig(
        name="CONDIM3_ACKERMANN",
        condim3=True,
        ackermann=True,
        front_free=False,
    ),
    VehicleConfig(
        name="CONDIM3_ACKERMANN_FRONT_FREE",
        condim3=True,
        ackermann=True,
        front_free=True,
    ),
)

CANDIDATE_CONFIGS = (
    CandidateConfig(
        name="conservative",
        learning_rate=8.0e-6,
        speed_noise=0.012,
        steer_noise=0.028,
        warmup_mix={"a3": 0.55, "bridge": 0.30, "caution": 0.15},
        main_mix={"a3": 0.40, "bridge": 0.25, "caution": 0.25, "light": 0.10},
    ),
    CandidateConfig(
        name="balanced",
        learning_rate=1.5e-5,
        speed_noise=0.020,
        steer_noise=0.045,
        warmup_mix={"a3": 0.50, "bridge": 0.30, "caution": 0.20},
        main_mix={"a3": 0.35, "bridge": 0.25, "caution": 0.30, "light": 0.10},
    ),
    CandidateConfig(
        name="exploration",
        learning_rate=2.5e-5,
        speed_noise=0.030,
        steer_noise=0.065,
        warmup_mix={"a3": 0.50, "bridge": 0.25, "caution": 0.25},
        main_mix={"a3": 0.30, "bridge": 0.25, "caution": 0.30, "light": 0.15},
    ),
)


def now_local() -> datetime:
    return datetime.now(TIMEZONE)


def next_morning_deadlines() -> tuple[datetime, datetime]:
    now = now_local()
    end = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if end <= now:
        end += timedelta(days=1)
    training_end = end - timedelta(minutes=30)
    return training_end, end


def object_id(
    model: mujoco.MjModel,
    object_type: Any,
    name: str,
) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return int(value)


def body_geom_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    if start < 0 or count <= 0:
        return []
    return list(range(start, start + count))


class VehicleDynamicsMixin:
    vehicle_config: VehicleConfig

    def _initialize_vehicle_patch(
        self,
        vehicle_config: VehicleConfig,
    ) -> None:
        self.vehicle_config = vehicle_config
        self._vehicle_actuator_ids = {
            key: object_id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            for key, name in ACTUATOR_NAMES.items()
        }

        wheel_positions: dict[str, np.ndarray] = {}
        wheel_collision_geoms: dict[str, int] = {}

        for wheel_name, joint_name in WHEEL_JOINT_NAMES.items():
            joint_id = object_id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            body_id = int(self.model.jnt_bodyid[joint_id])
            wheel_positions[wheel_name] = np.asarray(
                self.data.xpos[body_id],
                dtype=np.float64,
            ).copy()

            geoms = [
                geom_id
                for geom_id in body_geom_ids(self.model, body_id)
                if int(self.model.geom_contype[geom_id]) != 0
                and int(self.model.geom_conaffinity[geom_id]) != 0
                and int(self.model.geom_type[geom_id])
                in (
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                )
            ]
            if len(geoms) != 1:
                raise RuntimeError(
                    f"{wheel_name} wheel collision geom mismatch: {geoms}"
                )
            wheel_collision_geoms[wheel_name] = int(geoms[0])

        front_x = float(
            np.mean(
                [
                    wheel_positions["front_left"][0],
                    wheel_positions["front_right"][0],
                ]
            )
        )
        rear_x = float(
            np.mean(
                [
                    wheel_positions["rear_left"][0],
                    wheel_positions["rear_right"][0],
                ]
            )
        )
        left_y = float(
            np.mean(
                [
                    wheel_positions["front_left"][1],
                    wheel_positions["rear_left"][1],
                ]
            )
        )
        right_y = float(
            np.mean(
                [
                    wheel_positions["front_right"][1],
                    wheel_positions["rear_right"][1],
                ]
            )
        )
        self._vehicle_wheelbase = abs(front_x - rear_x)
        self._vehicle_track = abs(left_y - right_y)

        if vehicle_config.condim3:
            for geom_id in wheel_collision_geoms.values():
                self.model.geom_condim[geom_id] = 3
            for geom_id in range(self.model.ngeom):
                if int(self.model.geom_type[geom_id]) == int(
                    mujoco.mjtGeom.mjGEOM_PLANE
                ):
                    self.model.geom_condim[geom_id] = 3

            self.model.opt.cone = int(
                mujoco.mjtCone.mjCONE_PYRAMIDAL
            )
            self.model.opt.noslip_iterations = 0

        if vehicle_config.front_free:
            for joint_name in (
                "left_front_wheel_joint",
                "right_front_wheel_joint",
            ):
                joint_id = object_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                dof_id = int(self.model.jnt_dofadr[joint_id])
                self.model.dof_damping[dof_id] = 0.0
                self.model.dof_frictionloss[dof_id] = 0.0
                self.model.dof_armature[dof_id] = 1.0e-5

        mujoco.mj_forward(self.model, self.data)

    def _ackermann_commands(
        self,
        center_steer: float,
    ) -> tuple[float, float]:
        magnitude = abs(float(center_steer))
        if magnitude < 1.0e-8:
            return 0.0, 0.0

        radius = self._vehicle_wheelbase / math.tan(magnitude)
        inner = math.atan2(
            self._vehicle_wheelbase,
            max(radius - self._vehicle_track / 2.0, 1.0e-6),
        )
        outer = math.atan2(
            self._vehicle_wheelbase,
            radius + self._vehicle_track / 2.0,
        )

        if center_steer > 0.0:
            left, right = inner, outer
        else:
            left, right = -outer, -inner

        left_range = self.model.actuator_ctrlrange[
            self._vehicle_actuator_ids["steer_left"]
        ]
        right_range = self.model.actuator_ctrlrange[
            self._vehicle_actuator_ids["steer_right"]
        ]

        return (
            float(np.clip(left, left_range[0], left_range[1])),
            float(np.clip(right, right_range[0], right_range[1])),
        )

    def _write_vehicle_ctrl(
        self,
        steer: float,
        wheel: float,
    ) -> None:
        if not self.vehicle_config.ackermann:
            super()._write_vehicle_ctrl(steer, wheel)
            return

        left_steer, right_steer = self._ackermann_commands(steer)

        # ctrl belongs to MjData, not MjModel.
        self.data.ctrl[
            self._vehicle_actuator_ids["steer_left"]
        ] = left_steer
        self.data.ctrl[
            self._vehicle_actuator_ids["steer_right"]
        ] = right_steer
        self.data.ctrl[
            self._vehicle_actuator_ids["drive_left"]
        ] = float(wheel)
        self.data.ctrl[
            self._vehicle_actuator_ids["drive_right"]
        ] = float(wheel)


class TurnDiagnosticEnv(
    VehicleDynamicsMixin,
    Phase10Stage1A3TerminalEnv,
):
    def __init__(self, vehicle_config: VehicleConfig) -> None:
        super().__init__(max_steps=500)
        self._initialize_vehicle_patch(vehicle_config)


class OvernightObstacleEnv(
    VehicleDynamicsMixin,
    Stage2A1SafetyEnv,
):
    def __init__(
        self,
        vehicle_config: VehicleConfig,
        fixed_task: str | None = None,
        seed: int = 99001,
    ) -> None:
        super().__init__(stage="bridge")
        self._initialize_vehicle_patch(vehicle_config)

        self.fixed_task = fixed_task
        self._task_rng = np.random.default_rng(seed)
        self._task_names = ("a3", "bridge", "caution")
        self._task_probabilities = np.asarray(
            [0.55, 0.30, 0.15],
            dtype=np.float64,
        )
        self._active_task = "a3"

        # Conservative continuous safety reward. It starts before collision,
        # but does not overpower navigation progress.
        self.collision_penalty = 42.0
        self.proximity_distance = 0.38
        self.proximity_scale = 0.42
        self.safety_caution_distance = 0.38
        self.safety_danger_distance = 0.25
        self.safety_critical_distance = 0.18
        self.unsafe_progress_scale = 34.0
        self.risk_speed_scale = 0.32
        self.critical_speed_scale = 0.90
        self.approach_obstacle_scale = 11.0
        self.clearance_gain_scale = 7.0
        self.straight_risk_scale = 0.055
        self.safe_steer_scale = 0.004

    def set_mixture(self, mixture: dict[str, float]) -> None:
        names = tuple(mixture)
        values = np.asarray(
            [float(mixture[name]) for name in names],
            dtype=np.float64,
        )
        if np.any(values < 0.0) or float(values.sum()) <= 0.0:
            raise ValueError(f"invalid task mixture: {mixture}")
        values /= float(values.sum())
        self._task_names = names
        self._task_probabilities = values
        print(
            "TASK_MIXTURE =",
            json.dumps(
                {
                    name: float(value)
                    for name, value in zip(names, values)
                },
                ensure_ascii=False,
            ),
        )

    def _apply_task(self, task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"invalid task: {task}")
        config = TASKS[task]
        self._active_task = task
        self.curriculum_stage = f"stage2a11_{task}"
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
        forced_task = (
            None if options is None else options.get("task_mode")
        )
        if forced_task is not None:
            task = str(forced_task)
        elif self.fixed_task is not None:
            task = self.fixed_task
        elif seed is not None:
            selector = np.random.default_rng(int(seed) + 17011)
            task = str(
                selector.choice(
                    self._task_names,
                    p=self._task_probabilities,
                )
            )
        else:
            task = str(
                self._task_rng.choice(
                    self._task_names,
                    p=self._task_probabilities,
                )
            )

        self._apply_task(task)
        observation, info = super().reset(seed=seed, options=None)
        info["task_mode"] = task
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(
            action
        )
        info["task_mode"] = self._active_task
        return observation, reward, terminated, truncated, info


class DeadlineCallback(BaseCallback):
    def __init__(self, deadline: datetime) -> None:
        super().__init__(verbose=0)
        self.deadline = deadline

    def _on_step(self) -> bool:
        if self.n_calls % 250 == 0 and now_local() >= self.deadline:
            print("WALL_CLOCK_TRAINING_DEADLINE_REACHED = True")
            return False
        return True


def turn_test(
    vehicle_config: VehicleConfig,
    steer_action: float,
    steps: int = 350,
) -> dict[str, float | bool]:
    env = TurnDiagnosticEnv(vehicle_config)
    try:
        env.reset(seed=43001)

        # Keep the diagnostic target far away so goal success does not end it.
        pose = env._pose()
        x = float(pose[0])
        y = float(pose[1])
        yaw = float(pose[2])
        env.goal_xy = np.asarray([x + 50.0, y + 50.0], dtype=np.float64)
        env.prev_goal_dist = float(
            math.hypot(
                float(env.goal_xy[0]) - x,
                float(env.goal_xy[1]) - y,
            )
        )

        previous_x = float(x)
        previous_y = float(y)
        previous_yaw = float(yaw)
        path_length = 0.0
        yaw_total = 0.0
        collision = False

        action = np.asarray(
            [0.50, float(steer_action)],
            dtype=np.float32,
        )

        for _ in range(steps):
            _, _, terminated, truncated, info = env.step(action)
            pose = env._pose()
            x = float(pose[0])
            y = float(pose[1])
            yaw = float(pose[2])
            path_length += math.hypot(
                x - previous_x,
                y - previous_y,
            )
            yaw_total += (
                float(yaw) - previous_yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
            previous_x = float(x)
            previous_y = float(y)
            previous_yaw = float(yaw)
            collision = bool(info.get("collision", False))
            if terminated or truncated:
                break

        radius = (
            path_length / abs(yaw_total)
            if abs(yaw_total) > 1.0e-8
            else float("inf")
        )
        return {
            "path_length_m": float(path_length),
            "yaw_change_deg": float(math.degrees(yaw_total)),
            "turning_radius_m": float(radius),
            "collision": collision,
        }
    finally:
        env.close()


def diagnose_vehicle_configs() -> tuple[VehicleConfig, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for config in VEHICLE_CONFIGS:
        left = turn_test(config, 1.0)
        right = turn_test(config, -1.0)

        left_yaw = float(left["yaw_change_deg"])
        right_yaw = float(right["yaw_change_deg"])
        mean_yaw = 0.5 * (abs(left_yaw) + abs(right_yaw))
        mean_radius = 0.5 * (
            float(left["turning_radius_m"])
            + float(right["turning_radius_m"])
        )
        symmetry = min(abs(left_yaw), abs(right_yaw)) / max(
            max(abs(left_yaw), abs(right_yaw)),
            1.0e-8,
        )
        correct_signs = left_yaw > 0.0 and right_yaw < 0.0
        valid = bool(
            correct_signs
            and symmetry >= 0.45
            and float(left["path_length_m"]) >= 0.15
            and float(right["path_length_m"]) >= 0.15
            and not bool(left["collision"])
            and not bool(right["collision"])
        )

        row = {
            "config": config.__dict__,
            "left": left,
            "right": right,
            "mean_abs_yaw_deg": mean_yaw,
            "mean_radius_m": mean_radius,
            "symmetry": symmetry,
            "correct_signs": correct_signs,
            "valid": valid,
        }
        rows.append(row)

        print("\nVEHICLE_CONFIG =", config.name)
        print("LEFT_YAW_DEG =", left_yaw)
        print("RIGHT_YAW_DEG =", right_yaw)
        print("MEAN_TURN_RADIUS_M =", mean_radius)
        print("TURN_SYMMETRY =", symmetry)
        print("VEHICLE_CONFIG_VALID =", valid)

    valid_rows = [row for row in rows if bool(row["valid"])]
    if not valid_rows:
        raise RuntimeError(
            "No stable vehicle configuration passed the turn diagnostic."
        )

    best_row = min(
        valid_rows,
        key=lambda row: (
            float(row["mean_radius_m"]),
            -float(row["mean_abs_yaw_deg"]),
        ),
    )
    best_name = str(best_row["config"]["name"])
    selected = next(
        config for config in VEHICLE_CONFIGS if config.name == best_name
    )

    report = {
        "rows": rows,
        "selected": best_row,
    }
    save_json(ROOT / "vehicle_diagnostic.json", report)

    print("\nSELECTED_VEHICLE_CONFIG =", selected.name)
    print(
        "SELECTED_TURN_RADIUS_M =",
        best_row["mean_radius_m"],
    )
    return selected, report


def make_train_env(
    vehicle_config: VehicleConfig,
) -> tuple[OvernightObstacleEnv, Monitor]:
    base = OvernightObstacleEnv(vehicle_config=vehicle_config)
    monitor = Monitor(
        base,
        filename=str(ROOT / "monitor.csv"),
        info_keywords=(
            "is_success",
            "collision",
            "goal_dist",
            "goal_angle",
            "min_lidar_m",
            "front_clearance",
            "danger_zone",
            "critical_zone",
            "task_mode",
        ),
    )
    return base, monitor


def evaluate_task(
    model: SAC,
    vehicle_config: VehicleConfig,
    task: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    env = OvernightObstacleEnv(
        vehicle_config=vehicle_config,
        fixed_task=task,
        seed=seed + 9000,
    )
    rows: list[dict[str, Any]] = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    try:
        for episode in range(int(episodes)):
            observation, reset_info = env.reset(seed=seed + episode)
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
                    "seed": seed + episode,
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

    result = {
        "task": task,
        "episodes": int(episodes),
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / int(episodes),
        "collision_rate": collision_count / int(episodes),
        "timeout_rate": timeout_count / int(episodes),
        "mean_final_goal_dist": float(
            np.mean([row["final_goal_dist"] for row in rows])
        ),
        "mean_minimum_clearance": float(
            np.mean([row["minimum_clearance"] for row in rows])
        ),
        "mean_minimum_front_clearance": float(
            np.mean(
                [row["minimum_front_clearance"] for row in rows]
            )
        ),
        "results": rows,
    }

    print(f"\n========== EVAL {task.upper()} {episodes} ==========")
    print("SUCCESS_RATE =", result["success_rate"])
    print("COLLISION_RATE =", result["collision_rate"])
    print("TIMEOUT_RATE =", result["timeout_rate"])
    print(
        "MEAN_MINIMUM_CLEARANCE =",
        result["mean_minimum_clearance"],
    )
    return result


def evaluate_suite(
    model: SAC,
    vehicle_config: VehicleConfig,
    episodes: dict[str, int],
    seed_offset: int,
    label: str,
) -> dict[str, Any]:
    seeds = {
        "a3": 10730 + seed_offset,
        "bridge": 51000 + seed_offset,
        "caution": 52000 + seed_offset,
        "light": 53000 + seed_offset,
    }
    tasks = {
        task: evaluate_task(
            model,
            vehicle_config,
            task,
            episodes=episodes[task],
            seed=seeds[task],
        )
        for task in ("a3", "bridge", "caution", "light")
    }

    weights = {
        "a3": 0.25,
        "bridge": 0.25,
        "caution": 0.35,
        "light": 0.15,
    }
    weighted_success = sum(
        weights[name] * float(tasks[name]["success_rate"])
        for name in weights
    )
    weighted_collision = sum(
        weights[name] * float(tasks[name]["collision_rate"])
        for name in weights
    )
    weighted_timeout = sum(
        weights[name] * float(tasks[name]["timeout_rate"])
        for name in weights
    )
    max_collision = max(
        float(result["collision_rate"])
        for result in tasks.values()
    )

    acceptable = bool(
        tasks["a3"]["success_rate"] >= 0.70
        and tasks["bridge"]["success_rate"] >= 0.65
        and weighted_success >= 0.58
    )
    gate = bool(
        tasks["a3"]["success_rate"] >= 0.80
        and tasks["bridge"]["success_rate"] >= 0.75
        and tasks["caution"]["success_rate"] >= 0.65
        and tasks["light"]["success_rate"] >= 0.55
        and max_collision <= 0.03
    )

    result = {
        "label": label,
        "tasks": tasks,
        "weighted_success": weighted_success,
        "weighted_collision": weighted_collision,
        "weighted_timeout": weighted_timeout,
        "max_collision": max_collision,
        "acceptable": acceptable,
        "gate": gate,
        "selection_key": [
            0 if acceptable else 1,
            max_collision,
            weighted_collision,
            -weighted_success,
            weighted_timeout,
        ],
    }

    print("\nSUITE_LABEL =", label)
    print("WEIGHTED_SUCCESS =", weighted_success)
    print("WEIGHTED_COLLISION =", weighted_collision)
    print("MAX_COLLISION =", max_collision)
    print("SUITE_ACCEPTABLE =", acceptable)
    print("STAGE2A11_GATE =", "PASS" if gate else "FAIL")
    return result


def is_better(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
) -> bool:
    if best is None:
        return True
    return tuple(candidate["selection_key"]) < tuple(
        best["selection_key"]
    )


def save_global_best(
    model: SAC,
    suite: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    model.save(str(ROOT / "best_model"))
    if model.replay_buffer is not None and model.replay_buffer.size() > 0:
        model.save_replay_buffer(str(ROOT / "best_replay.pkl"))
    save_json(ROOT / "best_evaluation.json", suite)
    save_json(ROOT / "best_metadata.json", metadata)
    print("GLOBAL_BEST_UPDATED =", metadata)


def train_candidate(
    vehicle_config: VehicleConfig,
    candidate: CandidateConfig,
    repeat_index: int,
    training_deadline: datetime,
    global_best: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidate_name = f"{candidate.name}_r{repeat_index:02d}"
    candidate_dir = ROOT / "candidates" / candidate_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    base_env, train_env = make_train_env(vehicle_config)
    model = SAC.load(
        str(SOURCE_MODEL),
        env=train_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.set_random_seed(61000 + repeat_index)
    set_learning_rate(model, candidate.learning_rate)
    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.asarray(
            [candidate.speed_noise, candidate.steer_noise],
            dtype=np.float32,
        ),
    )

    records: list[dict[str, Any]] = []
    best_for_candidate: dict[str, Any] | None = None
    no_improvement = 0

    print("\nCANDIDATE_START =", candidate_name)
    print("LEARNING_RATE =", candidate.learning_rate)
    print(
        "ACTION_NOISE_SIGMA =",
        [candidate.speed_noise, candidate.steer_noise],
    )

    for chunk in range(1, 7):
        if now_local() >= training_deadline:
            print("TRAINING_DEADLINE_BEFORE_CHUNK = True")
            break

        mixture = (
            candidate.warmup_mix
            if chunk <= 2
            else candidate.main_mix
        )
        base_env.set_mixture(mixture)

        print("\nCANDIDATE =", candidate_name)
        print("TRAIN_CHUNK =", chunk, "/ 6")
        print("CHUNK_TIMESTEPS = 15000")

        model.learn(
            total_timesteps=15_000,
            reset_num_timesteps=False,
            log_interval=8,
            callback=DeadlineCallback(training_deadline),
            progress_bar=False,
        )

        if now_local() >= training_deadline:
            print("TRAINING_DEADLINE_AFTER_LEARN = True")
            break

        suite = evaluate_suite(
            model,
            vehicle_config,
            episodes={
                "a3": 30,
                "bridge": 30,
                "caution": 40,
                "light": 30,
            },
            seed_offset=repeat_index * 1000 + chunk * 100,
            label=f"{candidate_name}_chunk{chunk}",
        )
        records.append(suite)
        save_json(
            candidate_dir / f"chunk_{chunk}_evaluation.json",
            suite,
        )
        model.save(str(candidate_dir / f"chunk_{chunk}_model"))

        if is_better(suite, best_for_candidate):
            best_for_candidate = suite
            no_improvement = 0
        else:
            no_improvement += 1

        if is_better(suite, global_best):
            global_best = suite
            save_global_best(
                model,
                suite,
                {
                    "candidate": candidate_name,
                    "chunk": chunk,
                    "vehicle_config": vehicle_config.__dict__,
                    "time": now_local().isoformat(),
                },
            )

        if suite["gate"]:
            print("CANDIDATE_GATE_PASS = True")
            break

        if no_improvement >= 2:
            print("CANDIDATE_EARLY_STOP = NO_IMPROVEMENT")
            break

        if (
            suite["tasks"]["a3"]["success_rate"] < 0.65
            or suite["max_collision"] > 0.25
        ):
            print("CANDIDATE_EARLY_STOP = CAPABILITY_DEGRADATION")
            break

    save_json(
        candidate_dir / "candidate_report.json",
        {
            "candidate": candidate.__dict__,
            "repeat_index": repeat_index,
            "records": records,
            "best": best_for_candidate,
        },
    )
    train_env.close()
    return global_best, records


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    training_deadline, final_deadline = next_morning_deadlines()

    print("STABLE_BASELINES3_VERSION =", stable_baselines3.__version__)
    print("DEVICE =", "cuda" if torch.cuda.is_available() else "cpu")
    print("SOURCE_MODEL =", SOURCE_MODEL)
    print("START_TIME =", now_local().isoformat())
    print("TRAINING_DEADLINE =", training_deadline.isoformat())
    print("FINAL_DEADLINE =", final_deadline.isoformat())
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("A_STAR = False")
    print("PURE_PURSUIT = False")
    print("WAYPOINT = False")
    print("TEACHER = False")
    print("CAMERA = False")
    print("SAFETY_SHIELD = False")
    print("HER = False")
    print("MANUAL_QPOS_QVEL = False")

    vehicle_config, vehicle_report = diagnose_vehicle_configs()

    # Evaluate source A3 under the selected corrected dynamics.
    source_model = SAC.load(
        str(SOURCE_MODEL),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    baseline_suite = evaluate_suite(
        source_model,
        vehicle_config,
        episodes={
            "a3": 50,
            "bridge": 50,
            "caution": 100,
            "light": 50,
        },
        seed_offset=0,
        label="SOURCE_A3_CORRECTED_DYNAMICS",
    )
    save_json(ROOT / "source_baseline_evaluation.json", baseline_suite)

    global_best: dict[str, Any] | None = baseline_suite
    save_global_best(
        source_model,
        baseline_suite,
        {
            "candidate": "SOURCE_A3",
            "chunk": 0,
            "vehicle_config": vehicle_config.__dict__,
            "time": now_local().isoformat(),
        },
    )

    all_records: list[dict[str, Any]] = []
    repeat_index = 0

    while now_local() < training_deadline:
        candidate = CANDIDATE_CONFIGS[
            repeat_index % len(CANDIDATE_CONFIGS)
        ]
        global_best, records = train_candidate(
            vehicle_config=vehicle_config,
            candidate=candidate,
            repeat_index=repeat_index,
            training_deadline=training_deadline,
            global_best=global_best,
        )
        all_records.extend(records)
        repeat_index += 1

        if global_best is not None and bool(global_best["gate"]):
            # One additional confirmation candidate is enough after a pass.
            if repeat_index >= 2:
                print("OVERNIGHT_GATE_CONFIRMED = True")
                break

    print("\nFINAL_EVALUATION_START =", now_local().isoformat())
    best_model_path = ROOT / "best_model.zip"
    if not best_model_path.is_file():
        raise FileNotFoundError(best_model_path)

    final_model = SAC.load(
        str(best_model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    final_suite = evaluate_suite(
        final_model,
        vehicle_config,
        episodes={
            "a3": 100,
            "bridge": 100,
            "caution": 150,
            "light": 100,
        },
        seed_offset=90000,
        label="FINAL_BEST_LARGE_EVAL",
    )

    recommendation = (
        "ENTER_STAGE2B_LIGHT_OBSTACLES"
        if final_suite["gate"]
        else "KEEP_BEST_STAGE2A11_AND_REVIEW_FAILURE_SEEDS"
    )

    final_report = {
        "start_time": now_local().isoformat(),
        "training_deadline": training_deadline.isoformat(),
        "final_deadline": final_deadline.isoformat(),
        "vehicle_config": vehicle_config.__dict__,
        "vehicle_diagnostic": vehicle_report,
        "source_baseline": baseline_suite,
        "training_records": all_records,
        "final_best": final_suite,
        "recommendation": recommendation,
    }
    save_json(ROOT / "final_report.json", final_report)

    print("\nSTAGE2A11_OVERNIGHT = FINISHED")
    print("FINISH_TIME =", now_local().isoformat())
    print("BEST_MODEL =", best_model_path)
    print("BEST_REPLAY =", ROOT / "best_replay.pkl")
    print("FINAL_REPORT =", ROOT / "final_report.json")
    print("RECOMMENDATION =", recommendation)


if __name__ == "__main__":
    main()
