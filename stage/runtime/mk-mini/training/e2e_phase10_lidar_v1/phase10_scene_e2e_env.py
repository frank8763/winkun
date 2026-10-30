from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class Phase10SceneE2ELidarEnv(gym.Env):
    """
    Phase10原始截图场景 + MK-mini车辆 + 纯LiDAR端到端控制。

    运行时只加载复制出的MJB和状态文件，不导入Phase9/Phase10。
    无A*、无Pure Pursuit、无相机、无残差控制、无Safety Shield。
    """

    metadata = {"render_modes": []}

    OBS_DIM = 112
    CURRENT_DIM = 64
    HISTORY_FRAMES = 4
    HISTORY_FRAME_DIM = 12

    def __init__(self, max_steps: int = 600) -> None:
        super().__init__()

        root = Path(__file__).resolve().parent
        assets = root / "assets"
        model_path = assets / "phase10_original_scene.mjb"
        state_path = assets / "phase10_original_scene_state.npz"

        if not model_path.exists() or not state_path.exists():
            raise FileNotFoundError(
                "Phase10场景快照不存在，请先运行："
                "python -m training.e2e_phase10_lidar_v1."
                "snapshot_phase10_original_scene"
            )

        self.model = mujoco.MjModel.from_binary_path(
            str(model_path)
        )
        self.data = mujoco.MjData(self.model)

        state = np.load(state_path)
        self.initial_qpos = state["qpos"].astype(np.float64)
        self.initial_qvel = state["qvel"].astype(np.float64)
        self.initial_ctrl = state["ctrl"].astype(np.float64)
        self.goal_xy = state["goal_xy"].astype(np.float64)

        self.frame_skip = int(state["frame_skip"][0])
        self.max_steer = float(state["max_steer"][0])
        self.max_wheel = float(state["max_wheel"][0])
        self.speed_gain = float(
            state["bicycle_speed_gain"][0]
        )
        self.yaw_gain = float(
            state["bicycle_yaw_gain"][0]
        )
        self.wheelbase = float(
            state["bicycle_wheelbase"][0]
        )

        self.max_steps = int(max_steps)
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = 0.0
        self.history: deque[np.ndarray] = deque(
            maxlen=self.HISTORY_FRAMES
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        self.id_lf_steer = self._actuator_id(
            "left_front_steer_act"
        )
        self.id_rf_steer = self._actuator_id(
            "right_front_steer_act"
        )
        self.id_lr_wheel = self._actuator_id(
            "left_rear_wheel_act"
        )
        self.id_rr_wheel = self._actuator_id(
            "right_rear_wheel_act"
        )

        self.base_body_id = self._find_free_body()
        root_id = int(
            self.model.body_rootid[self.base_body_id]
        )
        self.robot_body_ids = {
            body_id
            for body_id in range(1, self.model.nbody)
            if int(self.model.body_rootid[body_id]) == root_id
        }

    def _actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if actuator_id < 0:
            raise RuntimeError(
                f"Phase10执行器不存在：{name}"
            )
        return int(actuator_id)

    def _find_free_body(self) -> int:
        for body_id in range(1, self.model.nbody):
            start = int(self.model.body_jntadr[body_id])
            count = int(self.model.body_jntnum[body_id])

            for joint_id in range(start, start + count):
                if (
                    int(self.model.jnt_type[joint_id])
                    == int(mujoco.mjtJoint.mjJNT_FREE)
                ):
                    return body_id

        raise RuntimeError("没有找到MK-mini自由基座")

    @staticmethod
    def _wrap(angle: float) -> float:
        return (
            angle + math.pi
        ) % (2.0 * math.pi) - math.pi

    def _pose(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        qpos = self.data.qpos
        qvel = self.data.qvel

        x = float(qpos[0])
        y = float(qpos[1])

        qw, qx, qy, qz = (
            float(qpos[3]),
            float(qpos[4]),
            float(qpos[5]),
            float(qpos[6]),
        )
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        vx = float(qvel[0]) if qvel.size > 0 else 0.0
        vy = float(qvel[1]) if qvel.size > 1 else 0.0
        yaw_rate = (
            float(qvel[5])
            if qvel.size > 5
            else 0.0
        )

        return x, y, yaw, vx, vy, yaw_rate

    def _goal_state(
        self,
    ) -> tuple[np.ndarray, float, float]:
        x, y, yaw, vx, vy, yaw_rate = self._pose()

        dx_world = float(self.goal_xy[0] - x)
        dy_world = float(self.goal_xy[1] - y)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        dx_body = (
            cos_yaw * dx_world
            + sin_yaw * dy_world
        )
        dy_body = (
            -sin_yaw * dx_world
            + cos_yaw * dy_world
        )

        goal_dist = math.hypot(
            dx_world,
            dy_world,
        )
        goal_angle = math.atan2(
            dy_body,
            dx_body,
        )

        body_vx = (
            cos_yaw * vx
            + sin_yaw * vy
        )
        body_vy = (
            -sin_yaw * vx
            + cos_yaw * vy
        )

        features = np.array(
            [
                np.clip(goal_dist / 10.0, 0.0, 2.0),
                math.sin(goal_angle),
                math.cos(goal_angle),
                np.clip(dx_body / 10.0, -2.0, 2.0),
                np.clip(dy_body / 10.0, -2.0, 2.0),
                np.clip(body_vx, -2.0, 2.0),
                np.clip(body_vy, -2.0, 2.0),
                np.clip(yaw_rate / 2.0, -2.0, 2.0),
                np.clip(math.hypot(vx, vy), 0.0, 2.0),
            ],
            dtype=np.float32,
        )

        return features, goal_dist, goal_angle

    def _lidar48(self) -> np.ndarray:
        x, y, yaw, _, _, _ = self._pose()

        angles = np.linspace(
            -0.75 * math.pi,
            0.75 * math.pi,
            48,
            dtype=np.float32,
        )
        max_range = 6.0
        output = np.ones(48, dtype=np.float32)

        origin = np.array(
            [
                x + 0.48 * math.cos(yaw),
                y + 0.48 * math.sin(yaw),
                0.36,
            ],
            dtype=np.float64,
        )

        geomgroup = np.ones(6, dtype=np.uint8)
        geomid = np.array([-1], dtype=np.int32)

        for index, relative in enumerate(angles):
            theta = yaw + float(relative)
            direction = np.array(
                [
                    math.cos(theta),
                    math.sin(theta),
                    0.0,
                ],
                dtype=np.float64,
            )

            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                geomgroup,
                1,
                self.base_body_id,
                geomid,
            )

            if distance < 0.0:
                distance = max_range

            output[index] = min(
                float(distance),
                max_range,
            ) / max_range

        return output

    @staticmethod
    def _clearance(lidar: np.ndarray) -> np.ndarray:
        angles = np.linspace(
            -0.75 * math.pi,
            0.75 * math.pi,
            lidar.size,
            dtype=np.float32,
        )

        masks = (
            np.ones(lidar.size, dtype=bool),
            np.abs(angles) <= math.radians(22.5),
            (
                (angles > math.radians(22.5))
                & (angles <= math.radians(110.0))
            ),
            (
                (angles < -math.radians(22.5))
                & (angles >= -math.radians(110.0))
            ),
        )

        return np.array(
            [
                float(np.min(lidar[mask]))
                if np.any(mask)
                else 1.0
                for mask in masks
            ],
            dtype=np.float32,
        )

    def _history_frame(
        self,
        lidar: np.ndarray,
        goal_dist: float,
        goal_angle: float,
    ) -> np.ndarray:
        sectors = np.array(
            [
                float(np.mean(chunk))
                for chunk in np.array_split(lidar, 8)
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                sectors,
                np.array(
                    [
                        np.clip(goal_dist / 10.0, 0.0, 2.0),
                        np.clip(goal_angle / math.pi, -1.0, 1.0),
                        self.prev_action[0],
                        self.prev_action[1],
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def _observation(
        self,
        update_history: bool,
    ) -> np.ndarray:
        lidar = self._lidar48()
        goal_features, goal_dist, goal_angle = (
            self._goal_state()
        )
        clearance = self._clearance(lidar)

        current = np.concatenate(
            [
                lidar,
                goal_features,
                self.prev_action,
                clearance,
                np.array(
                    [float(np.mean(lidar))],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        if current.shape != (64,):
            raise RuntimeError(
                f"当前观测不是64维：{current.shape}"
            )

        frame = self._history_frame(
            lidar,
            goal_dist,
            goal_angle,
        )

        if update_history:
            self.history.append(frame)

        if not self.history:
            for _ in range(self.HISTORY_FRAMES):
                self.history.append(frame.copy())

        while len(self.history) < self.HISTORY_FRAMES:
            self.history.appendleft(
                self.history[0].copy()
            )

        observation = np.concatenate(
            [current, *list(self.history)]
        ).astype(np.float32)

        observation = np.clip(
            np.nan_to_num(
                observation,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0,
            ),
            -10.0,
            10.0,
        ).astype(np.float32)

        if observation.shape != (112,):
            raise RuntimeError(
                f"最终观测不是112维：{observation.shape}"
            )

        return observation

    def _write_vehicle_ctrl(
        self,
        steer: float,
        wheel: float,
    ) -> None:
        self.data.ctrl[:] = self.initial_ctrl

        self.data.ctrl[self.id_lf_steer] = steer
        self.data.ctrl[self.id_rf_steer] = steer
        self.data.ctrl[self.id_lr_wheel] = wheel
        self.data.ctrl[self.id_rr_wheel] = wheel

    def _apply_phase10_vehicle_response(
        self,
        steer: float,
        wheel: float,
    ) -> None:
        dt = (
            float(self.model.opt.timestep)
            * max(self.frame_skip, 1)
        )

        wheel = max(0.0, float(wheel))
        v = self.speed_gain * wheel
        yaw_rate = (
            self.yaw_gain
            * v
            / max(self.wheelbase, 1e-3)
            * math.tan(float(steer))
        )

        qpos = self.data.qpos

        qw, qx, qy, qz = (
            float(qpos[3]),
            float(qpos[4]),
            float(qpos[5]),
            float(qpos[6]),
        )
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        yaw_new = self._wrap(
            yaw + yaw_rate * dt
        )

        qpos[0] += v * math.cos(yaw_new) * dt
        qpos[1] += v * math.sin(yaw_new) * dt
        qpos[3] = math.cos(0.5 * yaw_new)
        qpos[4] = 0.0
        qpos[5] = 0.0
        qpos[6] = math.sin(0.5 * yaw_new)

        if self.data.qvel.size >= 6:
            self.data.qvel[0] = v * math.cos(yaw_new)
            self.data.qvel[1] = v * math.sin(yaw_new)
            self.data.qvel[5] = yaw_rate

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def _collision(self) -> bool:
        floor_tokens = (
            "floor",
            "ground",
            "plane",
        )

        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            robot1 = body1 in self.robot_body_ids
            robot2 = body2 in self.robot_body_ids

            if robot1 == robot2:
                continue

            other_geom = geom2 if robot1 else geom1
            other_body = body2 if robot1 else body1

            geom_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    other_geom,
                )
                or ""
            )
            body_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    other_body,
                )
                or ""
            )

            combined = (
                geom_name + " " + body_name
            ).lower()

            if any(
                token in combined
                for token in floor_tokens
            ):
                continue

            if float(contact.dist) <= 0.002:
                return True

        return False

    def _policy_to_commands(
        self,
        action: np.ndarray,
    ) -> tuple[float, float, float, float]:
        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32,
            ).reshape(2),
            -1.0,
            1.0,
        )

        speed_ratio = 0.5 * (
            float(action[0]) + 1.0
        )
        wheel = speed_ratio * self.max_wheel
        command_v = self.speed_gain * wheel

        command_w = 1.2 * float(action[1])
        steer = math.atan2(
            self.wheelbase * command_w,
            max(command_v, 0.05),
        )
        steer = float(
            np.clip(
                steer,
                -self.max_steer,
                self.max_steer,
            )
        )

        return steer, wheel, command_v, command_w

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.step_count = 0
        self.prev_action[:] = 0.0
        self.history.clear()

        _, self.prev_goal_dist, _ = (
            self._goal_state()
        )

        observation = self._observation(
            update_history=False,
        )
        frame = self._history_frame(
            observation[:48],
            self.prev_goal_dist,
            math.atan2(
                float(observation[49]),
                float(observation[50]),
            ),
        )

        for _ in range(self.HISTORY_FRAMES):
            self.history.append(frame.copy())

        return self._observation(False), {
            "scene": "phase10_original_screenshot",
            "planner_used": False,
            "camera_used": False,
            "phase_runtime_dependency": False,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32,
            ).reshape(2),
            -1.0,
            1.0,
        )

        steer, wheel, command_v, command_w = (
            self._policy_to_commands(action)
        )

        self._write_vehicle_ctrl(
            steer,
            wheel,
        )

        for _ in range(max(self.frame_skip, 1)):
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self._apply_phase10_vehicle_response(
            steer,
            wheel,
        )

        self.step_count += 1
        previous_action = self.prev_action.copy()

        _, goal_dist, goal_angle = (
            self._goal_state()
        )
        progress = (
            self.prev_goal_dist - goal_dist
        )
        self.prev_goal_dist = goal_dist

        lidar = self._lidar48()
        min_lidar_m = float(
            np.min(lidar) * 6.0
        )
        collision = self._collision()
        reach_goal = goal_dist < 0.92
        timeout = self.step_count >= self.max_steps

        reward = 40.0 * progress
        reward += 0.10 * math.cos(goal_angle)
        reward += 0.03 * command_v
        reward -= 0.02
        reward -= 0.015 * abs(command_w)
        reward -= 0.02 * float(
            np.linalg.norm(
                action - previous_action
            )
        )

        self.prev_action = action.copy()

        if min_lidar_m < 0.65:
            reward -= 2.5 * (
                0.65 - min_lidar_m
            )

        if reach_goal:
            reward += 350.0

        if collision:
            reward -= 350.0

        terminated = bool(
            reach_goal or collision
        )
        truncated = bool(
            timeout and not terminated
        )

        observation = self._observation(
            update_history=True,
        )

        info = {
            "reach_goal": reach_goal,
            "collision": collision,
            "goal_dist": goal_dist,
            "goal_angle": goal_angle,
            "min_lidar_m": min_lidar_m,
            "command_v": command_v,
            "command_w": command_w,
            "steer": steer,
            "wheel": wheel,
            "planner_used": False,
            "camera_used": False,
            "phase_runtime_dependency": False,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        pass
