from __future__ import annotations

import math

import numpy as np

from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a2_finish_env import (
    Phase10Stage1A2FinishEnv,
)


class Phase10Stage1A3TerminalEnv(Phase10Stage1A2FinishEnv):
    """Stage1-A3: terminal approach coordination for strict end-to-end SAC.

    The observation, action, goal sampling and MuJoCo dynamics are unchanged.
    Only the reward and maximum wheel command are adjusted. The policy must
    still infer all terminal behaviour from LiDAR, relative goal and history.
    """

    def __init__(
        self,
        max_steps: int = 500,
        wheel_command_limit: float = 4.5,
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
        self.curriculum_stage = "stage1a3_terminal"

        self.progress_scale = 85.0
        self.heading_motion_scale = 0.025
        self.distance_scale = 0.006
        self.step_penalty = 0.006
        self.success_bonus = 10.0
        self.collision_penalty = 14.0
        self.proximity_scale = 0.28
        self.proximity_distance = 0.28
        self.smoothness_scale = 0.006

        self.finish_zone_distance = 0.32
        self.finish_progress_scale = 70.0
        self.finish_linger_scale = 0.025
        self.finish_no_progress_penalty = 0.030
        self.finish_no_progress_threshold = 5.0e-5

        # Adaptive terminal speed: high speed is allowed farther away, but the
        # allowed speed falls smoothly near 0.20 m and when heading is poor.
        self.terminal_speed_distance = 0.32
        self.terminal_speed_span = 0.12
        self.terminal_min_speed = 0.10
        self.terminal_max_speed = 0.78
        self.terminal_overspeed_scale = 0.18
        self.terminal_heading_scale = 0.060
        self.misaligned_motion_scale = 0.10

        self.stalled_penalty = 0.040
        self.stalled_speed_ratio = 0.06

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

        heading_forward = float(
            np.clip(math.cos(float(goal_angle)), 0.0, 1.0)
        )
        heading_error = float(1.0 - heading_forward)

        distance_ratio = float(
            np.clip(
                (goal_dist - self.success_radius)
                / self.terminal_speed_span,
                0.0,
                1.0,
            )
        )
        allowed_speed = (
            self.terminal_min_speed
            + (self.terminal_max_speed - self.terminal_min_speed)
            * distance_ratio
            * (0.25 + 0.75 * heading_forward)
        )
        overspeed = max(float(speed_ratio) - float(allowed_speed), 0.0)

        finish_no_progress = bool(
            finish_ratio > 0.0
            and goal_dist > self.success_radius
            and progress <= self.finish_no_progress_threshold
            and not collision
        )
        stalled = bool(
            speed_ratio < self.stalled_speed_ratio
            and goal_dist > self.success_radius + 0.003
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
            "terminal_heading": (
                -self.terminal_heading_scale
                * finish_ratio
                * heading_error
            ),
            "terminal_overspeed": (
                -self.terminal_overspeed_scale
                * finish_ratio
                * overspeed
                * overspeed
            ),
            "misaligned_motion": (
                -self.misaligned_motion_scale
                * finish_ratio
                * speed_ratio
                * heading_error
            ),
            "stalled": -self.stalled_penalty if stalled else 0.0,
        }
        return float(sum(terms.values())), terms
