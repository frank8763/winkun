from __future__ import annotations

import math
from typing import Any

import numpy as np

from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a_dense_env import (
    Phase10Stage1ADenseEnv,
)


class Phase10Stage1A2FinishEnv(Phase10Stage1ADenseEnv):
    """Stage1-A2: finish the final centimetres without changing task semantics.

    Observation, action, MuJoCo dynamics and goal sampling remain identical to
    Stage1-A. Only the dense reward is adjusted to remove the low-speed local
    optimum immediately outside the success radius.
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
            goal_distance_min=goal_distance_min,
            goal_distance_max=goal_distance_max,
            goal_angle_limit_deg=goal_angle_limit_deg,
            goal_clearance_margin=goal_clearance_margin,
        )
        self.curriculum_stage = "stage1a2_finish"

        # Keep the successful Stage1-A scale and change only the final approach.
        self.progress_scale = 80.0
        self.heading_motion_scale = 0.030
        self.distance_scale = 0.006
        self.step_penalty = 0.006
        self.success_bonus = 8.0
        self.collision_penalty = 12.0
        self.proximity_scale = 0.25
        self.proximity_distance = 0.28
        self.smoothness_scale = 0.005

        # Stage1-A slowed down too strongly near the goal and could settle just
        # outside 0.20 m. A2 keeps modest speed regularisation only very close
        # to the goal and adds pressure to cross the success boundary.
        self.near_goal_speed_scale = 0.015
        self.near_goal_distance = 0.26
        self.finish_zone_distance = 0.30
        self.finish_progress_scale = 60.0
        self.finish_linger_scale = 0.030
        self.finish_no_progress_penalty = 0.025
        self.finish_no_progress_threshold = 5.0e-5
        self.stalled_penalty = 0.035
        self.stalled_speed_ratio = 0.08

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
                / max(self.near_goal_distance, 1.0e-6),
                0.0,
                1.0,
            )
        )
        finish_ratio = float(
            np.clip(
                (self.finish_zone_distance - goal_dist)
                / max(
                    self.finish_zone_distance - self.success_radius,
                    1.0e-6,
                ),
                0.0,
                1.0,
            )
        )
        heading_forward = max(math.cos(float(goal_angle)), 0.0)

        stalled = bool(
            speed_ratio < self.stalled_speed_ratio
            and goal_dist > self.success_radius + 0.002
            and not collision
        )
        finish_no_progress = bool(
            finish_ratio > 0.0
            and goal_dist > self.success_radius
            and progress <= self.finish_no_progress_threshold
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
            "finish_progress": (
                self.finish_progress_scale
                * finish_ratio
                * progress
            ),
            "finish_linger": (
                -self.finish_linger_scale * finish_ratio
                if not success and not collision
                else 0.0
            ),
            "finish_no_progress": (
                -self.finish_no_progress_penalty
                if finish_no_progress
                else 0.0
            ),
            "stalled": -self.stalled_penalty if stalled else 0.0,
        }
        return float(sum(terms.values())), terms
