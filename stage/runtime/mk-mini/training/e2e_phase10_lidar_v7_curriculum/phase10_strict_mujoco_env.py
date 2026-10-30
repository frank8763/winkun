from __future__ import annotations

import math

import numpy as np

from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import (
    Phase10SceneE2ELidarEnv,
)


class Phase10StrictMuJoCoEnv(Phase10SceneE2ELidarEnv):
    """
    Phase10 original-scene strict MuJoCo dynamics layer.

    The policy controls the real steering and rear-wheel actuators. No A*,
    Pure Pursuit, teacher action, path waypoint, camera, qpos/qvel movement,
    or manually imposed bicycle kinematics is used.
    """

    def __init__(
        self,
        max_steps: int = 700,
        wheel_command_limit: float = 5.0,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.max_wheel = float(wheel_command_limit)

    def _apply_phase10_vehicle_response(
        self,
        steer: float,
        wheel: float,
    ) -> None:
        # Strictly disable the manual Phase10 bicycle response. Vehicle motion
        # can only come from mujoco.mj_step() and the model actuators.
        del steer, wheel
        return None

    def _policy_to_commands(
        self,
        action: np.ndarray,
    ) -> tuple[float, float, float, float]:
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(2),
            -1.0,
            1.0,
        )

        # Squared mapping preserves the full speed range while giving the
        # policy much finer low-speed control near obstacles and the goal.
        raw_speed_ratio = 0.5 * (float(action[0]) + 1.0)
        speed_ratio = raw_speed_ratio * raw_speed_ratio
        wheel = speed_ratio * self.max_wheel
        steer = float(action[1]) * self.max_steer

        # Logging only. These values never update qpos/qvel.
        command_v = self.speed_gain * wheel
        command_w = (
            command_v
            / max(self.wheelbase, 1.0e-3)
            * math.tan(steer)
        )

        return (
            float(steer),
            float(wheel),
            float(command_v),
            float(command_w),
        )
