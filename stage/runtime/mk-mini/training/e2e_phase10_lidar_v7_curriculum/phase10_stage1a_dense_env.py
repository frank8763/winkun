from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from training.e2e_phase10_lidar_v7_curriculum.phase10_strict_mujoco_env import (
    Phase10StrictMuJoCoEnv,
)


class Phase10Stage1ADenseEnv(Phase10StrictMuJoCoEnv):
    """V7 Stage1-A strict end-to-end dense-reward environment.

    Observation is the original 112-D LiDAR/relative-goal/history vector.
    Action is rear-wheel speed plus front steering through real MuJoCo actuators.
    No planner, waypoint, teacher, camera, safety shield, or qpos/qvel motion is used.
    """

    def __init__(
        self,
        max_steps: int = 500,
        wheel_command_limit: float = 5.0,
        goal_distance_min: float = 0.45,
        goal_distance_max: float = 0.65,
        goal_angle_limit_deg: float = 30.0,
        goal_clearance_margin: float = 0.35,
    ) -> None:
        super().__init__(
            max_steps=max_steps,
            wheel_command_limit=wheel_command_limit,
        )

        self.curriculum_stage = "stage1a"
        self.success_radius = 0.20
        self.goal_distance_min = float(goal_distance_min)
        self.goal_distance_max = float(goal_distance_max)
        self.goal_angle_limit = math.radians(float(goal_angle_limit_deg))
        self.goal_clearance_margin = float(goal_clearance_margin)

        if not 0.20 < self.goal_distance_min < self.goal_distance_max:
            raise ValueError("Invalid Stage1-A goal-distance range")

        # Stage1-A teaches goal-directed motion first. Obstacle complexity is
        # deliberately deferred to Stage2; collision safety remains active.
        self.progress_scale = 80.0
        self.heading_motion_scale = 0.030
        self.distance_scale = 0.005
        self.step_penalty = 0.005
        self.success_bonus = 8.0
        self.collision_penalty = 12.0
        self.proximity_scale = 0.25
        self.proximity_distance = 0.28
        self.smoothness_scale = 0.005
        self.near_goal_speed_scale = 0.10
        self.near_goal_distance = 0.35
        self.stalled_penalty = 0.020

        self.lidar_forward_offset = 0.48
        self.lidar_max_range = 6.0
        self.lidar_sector_half_width = 2

        self.last_goal_sector_clearance = float("nan")
        self.last_goal_sensor_distance = float("nan")
        self.last_goal_relative_angle = float("nan")

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _lidar_geometry(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        x, y, yaw, _, _, _ = self._pose()
        center_xy = np.array([x, y], dtype=np.float64)
        sensor_xy = center_xy + self.lidar_forward_offset * np.array(
            [math.cos(yaw), math.sin(yaw)],
            dtype=np.float64,
        )
        lidar_m = self._lidar48().astype(np.float64) * self.lidar_max_range
        return center_xy, sensor_xy, lidar_m, float(yaw)

    def _candidate_is_safe(
        self,
        candidate_xy: np.ndarray,
        sensor_xy: np.ndarray,
        lidar_m: np.ndarray,
        yaw: float,
    ) -> tuple[bool, float, float, float]:
        sensor_delta = np.asarray(candidate_xy, dtype=np.float64) - sensor_xy
        sensor_distance = float(np.linalg.norm(sensor_delta))
        if sensor_distance < 1.0e-6:
            return False, 0.0, sensor_distance, 0.0

        relative_angle = self._wrap_angle(
            math.atan2(float(sensor_delta[1]), float(sensor_delta[0])) - yaw
        )
        ray_angles = np.linspace(
            -0.75 * math.pi,
            0.75 * math.pi,
            48,
            dtype=np.float64,
        )
        if not float(ray_angles[0]) <= relative_angle <= float(ray_angles[-1]):
            return False, 0.0, sensor_distance, relative_angle

        ray_index = int(np.argmin(np.abs(ray_angles - relative_angle)))
        lo = max(0, ray_index - self.lidar_sector_half_width)
        hi = min(len(lidar_m), ray_index + self.lidar_sector_half_width + 1)
        sector_clearance = float(np.min(lidar_m[lo:hi]))
        required = sensor_distance + self.goal_clearance_margin
        return (
            bool(sector_clearance >= required),
            sector_clearance,
            sensor_distance,
            relative_angle,
        )

    def _sample_stage1a_goal(self) -> np.ndarray:
        center_xy, sensor_xy, lidar_m, yaw = self._lidar_geometry()

        for _ in range(256):
            distance = float(
                self.np_random.uniform(
                    self.goal_distance_min,
                    self.goal_distance_max,
                )
            )
            relative_angle = float(
                self.np_random.uniform(
                    -self.goal_angle_limit,
                    self.goal_angle_limit,
                )
            )
            world_angle = yaw + relative_angle
            candidate = center_xy + distance * np.array(
                [math.cos(world_angle), math.sin(world_angle)],
                dtype=np.float64,
            )
            safe, clearance, sensor_distance, _ = self._candidate_is_safe(
                candidate,
                sensor_xy,
                lidar_m,
                yaw,
            )
            if safe:
                self.last_goal_sector_clearance = clearance
                self.last_goal_sensor_distance = sensor_distance
                self.last_goal_relative_angle = relative_angle
                return candidate

        candidates: list[tuple[float, np.ndarray, float, float, float]] = []
        for relative_angle in np.linspace(
            -self.goal_angle_limit,
            self.goal_angle_limit,
            41,
            dtype=np.float64,
        ):
            for distance in np.linspace(
                self.goal_distance_max,
                self.goal_distance_min,
                11,
                dtype=np.float64,
            ):
                world_angle = yaw + float(relative_angle)
                candidate = center_xy + float(distance) * np.array(
                    [math.cos(world_angle), math.sin(world_angle)],
                    dtype=np.float64,
                )
                safe, clearance, sensor_distance, _ = self._candidate_is_safe(
                    candidate,
                    sensor_xy,
                    lidar_m,
                    yaw,
                )
                if safe:
                    surplus = (
                        clearance
                        - sensor_distance
                        - self.goal_clearance_margin
                    )
                    candidates.append(
                        (
                            float(surplus),
                            candidate,
                            float(clearance),
                            float(sensor_distance),
                            float(relative_angle),
                        )
                    )

        if not candidates:
            raise RuntimeError("V7 Stage1-A cannot sample a safe LiDAR goal")

        _, candidate, clearance, sensor_distance, relative_angle = max(
            candidates,
            key=lambda item: item[0],
        )
        self.last_goal_sector_clearance = clearance
        self.last_goal_sensor_distance = sensor_distance
        self.last_goal_relative_angle = relative_angle
        return candidate

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        _, base_info = super().reset(seed=seed, options=options)
        self.goal_xy = self._sample_stage1a_goal()
        self.prev_action[:] = 0.0
        self.history.clear()
        _, self.prev_goal_dist, _ = self._goal_state()
        observation = self._observation(update_history=False)

        base_info.update(
            {
                "curriculum_stage": self.curriculum_stage,
                "goal_xy": self.goal_xy.astype(float).tolist(),
                "goal_dist": float(self.prev_goal_dist),
                "success_radius": self.success_radius,
                "goal_sector_clearance": float(
                    self.last_goal_sector_clearance
                ),
                "goal_sensor_distance": float(
                    self.last_goal_sensor_distance
                ),
                "goal_relative_angle": float(
                    self.last_goal_relative_angle
                ),
                "planner_used": False,
                "teacher_used": False,
                "camera_used": False,
                "safety_shield_used": False,
                "manual_kinematics": False,
            }
        )
        return np.asarray(observation, dtype=np.float32), base_info

    def _reward(
        self,
        previous_goal_dist: float,
        goal_dist: float,
        goal_angle: float,
        collision: bool,
        min_lidar_m: float,
        speed_ratio: float,
        action_delta_sq: float,
    ) -> tuple[float, dict[str, float]]:
        progress = float(
            np.clip(previous_goal_dist - goal_dist, -0.02, 0.02)
        )
        success = bool(goal_dist <= self.success_radius and not collision)
        proximity_ratio = float(
            np.clip(
                (self.proximity_distance - min_lidar_m)
                / self.proximity_distance,
                0.0,
                1.0,
            )
        )
        near_goal_ratio = float(
            np.clip(
                (self.near_goal_distance - goal_dist)
                / self.near_goal_distance,
                0.0,
                1.0,
            )
        )
        heading_forward = max(math.cos(float(goal_angle)), 0.0)
        stalled = bool(
            speed_ratio < 0.04
            and goal_dist > 0.28
            and not collision
        )

        terms = {
            "progress": self.progress_scale * progress,
            "heading_motion": (
                self.heading_motion_scale
                * speed_ratio
                * heading_forward
            ),
            "distance": -self.distance_scale * goal_dist,
            "step": -self.step_penalty,
            "success": self.success_bonus if success else 0.0,
            "collision": -self.collision_penalty if collision else 0.0,
            "proximity": (
                -self.proximity_scale
                * proximity_ratio
                * proximity_ratio
            ),
            "smoothness": -self.smoothness_scale * action_delta_sq,
            "near_goal_speed": (
                -self.near_goal_speed_scale
                * near_goal_ratio
                * speed_ratio
                * speed_ratio
            ),
            "stalled": -self.stalled_penalty if stalled else 0.0,
        }
        return float(sum(terms.values())), terms

    def step(self, action: np.ndarray):
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(2),
            -1.0,
            1.0,
        )
        previous_action = self.prev_action.copy()
        previous_goal_dist = float(self.prev_goal_dist)

        steer, wheel, command_v, command_w = self._policy_to_commands(action)
        self._write_vehicle_ctrl(steer, wheel)

        for _ in range(max(self.frame_skip, 1)):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        _, goal_dist, goal_angle = self._goal_state()
        min_lidar_m = float(np.min(self._lidar48()) * self.lidar_max_range)
        collision = bool(self._collision())
        success = bool(goal_dist <= self.success_radius and not collision)
        timeout = bool(self.step_count >= self.max_steps)
        terminated = bool(success or collision)
        truncated = bool(timeout and not terminated)

        raw_speed_ratio = 0.5 * (float(action[0]) + 1.0)
        speed_ratio = raw_speed_ratio * raw_speed_ratio
        action_delta_sq = float(np.sum((action - previous_action) ** 2))
        reward, terms = self._reward(
            previous_goal_dist=previous_goal_dist,
            goal_dist=float(goal_dist),
            goal_angle=float(goal_angle),
            collision=collision,
            min_lidar_m=min_lidar_m,
            speed_ratio=speed_ratio,
            action_delta_sq=action_delta_sq,
        )

        self.prev_action = action.copy()
        self.prev_goal_dist = float(goal_dist)
        observation = self._observation(update_history=True)

        info: dict[str, Any] = {
            "is_success": float(success),
            "reach_goal": success,
            "collision": collision,
            "goal_dist": float(goal_dist),
            "goal_angle": float(goal_angle),
            "goal_progress": float(previous_goal_dist - goal_dist),
            "min_lidar_m": min_lidar_m,
            "speed_ratio": float(speed_ratio),
            "action_delta_sq": action_delta_sq,
            "command_v": float(command_v),
            "command_w": float(command_w),
            "steer": float(steer),
            "wheel": float(wheel),
            "curriculum_stage": self.curriculum_stage,
            "planner_used": False,
            "teacher_used": False,
            "camera_used": False,
            "safety_shield_used": False,
            "manual_kinematics": False,
        }
        for name, value in terms.items():
            info[f"reward_{name}"] = float(value)

        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )
