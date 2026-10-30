from pathlib import Path
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()
JOB = Path('/home/linghu/mkmini_stage_td3_t67_3m_nightrelay_20261027_kEPAW0/T67C058')

CORE = JOB / "td3_core_goal_contrast.py"

HELPER = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1/td3_smoke.py"
)

STAGE = H / (
    "mkmini_stage_c147_fixedgoal_sac_20261027_v1/"
    "GOAL_COMMIT_24000/train/train.py"
)

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_"
    "20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

EXPECTED = {
    CORE:
    "35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07",

    STAGE:
    "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",

    FROZEN:
    "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
}

TARGET = 134491
EP_CAP = 14000
WALL = 7500

WARMUP_FRAMES = 2048
ACTOR_LR = 3e-5
CRITIC_LR = 1e-4

TASKS = ((-17.3, 3.0), (13.6, 3.0), (18.8, 3.0), (-12.5, 3.0))
TRAIN_SEEDS = (2029653601, 2029653602, 2029654802, 2029654801)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(
            lambda: f.read(1048576),
            b""
        ):
            h.update(part)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def emit(name, obj):
    print(
        name,
        json.dumps(
            obj,
            ensure_ascii=False,
            default=str
        ),
        flush=True
    )


def save_json(path, data):
    temp = path.with_name(path.name + ".partial")

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        ) + "\n"
    )

    os.replace(temp, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path)
    )

    check(
        spec is not None and spec.loader is not None,
        "MODULE_SPEC_INVALID"
    )

    mod = importlib.util.module_from_spec(spec)

    sys.modules[name] = mod

    spec.loader.exec_module(mod)

    return mod


def preflight():
    for path, expected in EXPECTED.items():
        check(
            path.is_file(),
            "REQUIRED_FILE_MISSING:" + str(path)
        )

        check(
            sha(path) == expected,
            "SHA256_CHANGED:" + str(path)
        )

    check(
        HELPER.is_file(),
        "VERIFIED_T1_ENVIRONMENT_HELPER_MISSING"
    )

    import ast

    ast.parse(HELPER.read_text())
    ast.parse(CORE.read_text())

    source = HELPER.read_text()

    check(
        "def make_goal_task(" in source,
        "T1_GOAL_HELPER_MISSING"
    )

    manifest = {
        "experiment": "STAGE_TD3_T67C058",
        "algorithm": "TD3",
        "initialization": "FRESH",
        "source_core": sha(CORE),
        "source_stage": sha(STAGE),
        "source_helper": sha(HELPER),
        "frozen_reference": sha(FROZEN),
        "target_frames": TARGET,
        "warmup_frames": WARMUP_FRAMES,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "training_seed": 20261027,
        "goal_pair_dataset_sha256": sha(JOB / "goal_pairs.npz"),
        "goal_rank_margin": 0.4,
        "goal_rank_beta": 0.05,
        "planner": False,
        "waypoints": False,
        "automatic_promotion": False,
    }

    manifest_path = JOB / "manifest.json"

    if manifest_path.exists():
        check(
            json.loads(manifest_path.read_text())
            == manifest,
            "T6_MANIFEST_CHANGED"
        )
    else:
        save_json(manifest_path, manifest)

    emit("TD3_T6_PREFLIGHT=PASS", manifest)

    return manifest


def actor_probe(agent, probes):
    result = {}

    for side in ("LEFT", "RIGHT"):

        if side not in probes:
            continue

        obs = np.asarray(
            probes[side],
            dtype=np.float32
        )

        raw = agent.act(obs)

        result[side] = {
            "raw_throttle": float(raw[0]),
            "raw_steer": float(raw[1]),
            "throttle": float(
                (raw[0] + 1.0) / 2.0
            ),
            "steer": float(raw[1])
        }

    collapsed = False

    if len(result) == 2:

        a = result["LEFT"]
        b = result["RIGHT"]

        collapsed = (
            a["throttle"] > 0.98
            and b["throttle"] > 0.98
            and a["steer"] < -0.98
            and b["steer"] < -0.98
        ) or (
            a["throttle"] > 0.98
            and b["throttle"] > 0.98
            and a["steer"] > 0.98
            and b["steer"] > 0.98
        )

    return result, collapsed


def create_agent(core):
    agent = core.TD3(
        state_dim=112,
        action_dim=2,
        device="cpu",
        policy_noise=0.15,
        noise_clip=0.4,
        policy_delay=4
    )

    for group in agent.actor_optimizer.param_groups:
        group["lr"] = ACTOR_LR

    for group in agent.critic_optimizer.param_groups:
        group["lr"] = CRITIC_LR

    return agent




_T59_MODEL_CACHE = {}
_T59_COUNTER = 0
_T59_NEAR_COUNT = 0


def _t59_external_clearance_reward(env):
    import math
    import numpy as np
    import mujoco

    global _T59_COUNTER
    global _T59_NEAR_COUNT

    model = env.model
    data = env.data

    key = id(model)

    if key not in _T59_MODEL_CACHE:

        expected = {
            23: "mkmini_base",
            25: "left_front_wheel",
            27: "right_front_wheel",
            29: "left_rear_wheel",
            31: "right_rear_wheel",
        }

        for gid, expected_body in expected.items():

            if gid >= int(model.ngeom):
                raise RuntimeError(
                    "T59_GEOM_ID_MISSING:" + str(gid)
                )

            bid = int(model.geom_bodyid[gid])

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                bid
            )

            if name != expected_body:
                raise RuntimeError(
                    "T59_BODY_MISMATCH:"
                    + str(gid)
                    + ":"
                    + str(name)
                )

        chassis_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            23
        )

        if chassis_name != "chassis_collision":
            raise RuntimeError(
                "T59_CHASSIS_GEOM_CHANGED"
            )

        obstacles = []

        for gid in range(int(model.ngeom)):

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                gid
            )

            if name and name.startswith(
                "real_rect_obs_"
            ):
                obstacles.append(gid)

        obstacle3 = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "real_rect_obs_3"
        )

        if (
            obstacle3 != 4
            or len(obstacles) == 0
        ):
            raise RuntimeError(
                "T59_OBSTACLE_GEOMETRY_CHANGED"
            )

        _T59_MODEL_CACHE[key] = tuple(obstacles)

        print(
            "T59_GEOMETRY_GATE=PASS",
            "FRONT=(25,27)",
            "REAR=(29,31)",
            "CHASSIS=23",
            "RECT_OBSTACLES=",
            len(obstacles),
            flush=True
        )

    obstacles = _T59_MODEL_CACHE[key]

    max_range = 0.65
    fromto = np.zeros(6, dtype=np.float64)

    def distance(gid):

        nearest = max_range

        for oid in obstacles:

            d = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    int(gid),
                    int(oid),
                    max_range,
                    fromto
                )
            )

            # 防止某些MuJoCo版本在无近距离
            # 接触时返回极大浮点数。
            if math.isnan(d):
                raise RuntimeError(
                    "T59_DISTANCE_NAN"
                )

            if d == float("-inf"):
                raise RuntimeError(
                    "T59_DISTANCE_NEGATIVE_INFINITY"
                )

            d = min(max_range, d)

            nearest = min(
                nearest, d
            )

        return nearest

    front = min(
        distance(25),
        distance(27)
    )

    rear = min(
        distance(29),
        distance(31)
    )

    chassis = distance(23)

    def risk(d, margin):
        return max(
            0.0,
            (margin - d) / margin
        ) ** 2

    # 仅用于训练奖励，不修改Actor动作。
    # 左右前轮对称处理，重点避免前轮擦碰。
    penalty = (
        0.152000 * risk(front, 0.55)
        + 0.072000 * risk(rear, 0.46)
        + 0.120000 * risk(chassis, 0.56)
    )

    _T59_COUNTER += 1

    if _T59_COUNTER == 1:

        print(
            "T59_EXTERNAL_SAFETY_REWARD_ACTIVE=PASS",
            "FRONT_M=",
            round(front, 5),
            "REAR_M=",
            round(rear, 5),
            "CHASSIS_M=",
            round(chassis, 5),
            flush=True
        )

    if penalty > 0.0:

        _T59_NEAR_COUNT += 1

        if _T59_NEAR_COUNT == 1:
            print(
                "T59_EXTERNAL_NEAR_OBSTACLE=PASS",
                "FRONT_M=",
                round(front, 5),
                "REAR_M=",
                round(rear, 5),
                "CHASSIS_M=",
                round(chassis, 5),
                "PENALTY=",
                round(penalty, 7),
                flush=True
            )

    if _T59_COUNTER % 1000 == 0:

        print(
            "T59_SAFETY_PROGRESS",
            "FRAMES=",
            _T59_COUNTER,
            "NEAR_FRAMES=",
            _T59_NEAR_COUNT,
            "FRONT_M=",
            round(front, 4),
            "REAR_M=",
            round(rear, 4),
            "CHASSIS_M=",
            round(chassis, 4),
            flush=True
        )

    return float(penalty)

def main():
    JOB.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = preflight()

    torch.set_num_threads(2)
    torch.manual_seed(20261027)

    rng = np.random.default_rng(20261027)

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(
        HELPER,
        "stage_td3_t6_helper"
    )

    core = load_module(
        CORE,
        "stage_td3_t6_core"
    )

    legacy = load_module(
        STAGE,
        "stage_td3_t6_legacy_environment"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_INVALID:"
            + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_td3_t6_runner"
    )

    _, v, native_mod, prep, hashes = (
        runner.load_native()
    )

    check(
        isinstance(hashes, dict)
        and len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]

    check(
        callable(N.step_checked),
        "NATIVE_STEP_UNAVAILABLE"
    )

    make_base = legacy.make_short_task(
        runner.make_task
    )

    agent = create_agent(core)
    t12_data = np.load(JOB / "goal_pairs.npz", allow_pickle=False)
    t12_left = np.asarray(t12_data["left"], dtype=np.float32)
    t12_right = np.asarray(t12_data["right"], dtype=np.float32)
    check(t12_left.shape == (20, 112) and t12_right.shape == (20, 112), "T12_PAIR_SHAPE_INVALID")
    agent.t12_pairs = (torch.from_numpy(t12_left), torch.from_numpy(t12_right))
    emit("TD3_T12_GOAL_PAIR_GATE=PASS", {"pairs": len(t12_left), "margin": 0.4, "beta": 0.05})

    replay = deque(maxlen=20000)

    frames = 0
    decisions = 0
    episode_index = 0
    episodes = []
    probes = {}

    status_path = JOB / "status.json"

    if status_path.exists():

        old_status = json.loads(
            status_path.read_text()
        )

        if old_status["state"] in (
            "PILOT_COMPLETED_NONCOLLAPSED",
            "BLOCKED_POLICY_SATURATION",
            "BLOCKED_EXPLORATION_COVERAGE"
        ):
            emit(
                "TD3_T6_ALREADY_FINISHED=",
                old_status
            )
            return

        filename = old_status["checkpoint"]

        ck = JOB / "checkpoints" / filename

        check(
            ck.is_file()
            and sha(ck)
            == old_status["checkpoint_sha256"],
            "T6_RESUME_CHECKPOINT_INVALID"
        )

        state = torch.load(
            ck,
            map_location="cpu",
            weights_only=False
        )

        check(
            state["manifest"] == manifest,
            "T6_RESUME_MANIFEST_MISMATCH"
        )

        for key, net in (
            ("actor", agent.actor),
            ("actor_target", agent.actor_target),
            ("critic", agent.critic),
            ("critic_target", agent.critic_target)
        ):
            net.load_state_dict(state[key])

        agent.actor_optimizer.load_state_dict(
            state["actor_optimizer"]
        )

        agent.critic_optimizer.load_state_dict(
            state["critic_optimizer"]
        )

        agent.total_updates = int(
            state["updates"]
        )

        frames = int(state["frames"])
        decisions = int(state["decisions"])
        episode_index = int(state["episode_index"])

        episodes = list(state["episodes"])
        probes = dict(state["probes"])

        replay = deque(
            state["replay"],
            maxlen=20000
        )

        rng.bit_generator.state = (
            state["rng_state"]
        )

        torch.set_rng_state(
            state["torch_rng_state"]
        )

        check(
            episode_index == len(episodes)
            and 0 <= frames <= TARGET,
            "T6_RESUME_COUNTER_INVALID"
        )

        emit("TD3_T6_RESUME=PASS", {
            "frames": frames,
            "updates": agent.total_updates,
            "episodes": episode_index,
            "replay": len(replay)
        })

    else:

        emit("TD3_T6_FRESH_INITIALIZATION=PASS", {
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "warmup_frames": WARMUP_FRAMES,
            "parent_td3_checkpoint_loaded": False
        })

    deadline = time.monotonic() + WALL

    checkpoint_dir = JOB / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    blocked = False
    paused = False

    while frames < TARGET:

        if time.monotonic() >= deadline:
            paused = True
            break

        angle, distance = TASKS[
            episode_index % len(TASKS)
        ]

        cycle = episode_index // len(TASKS)

        seed = TRAIN_SEEDS[episode_index % len(TASKS)]

        env = None
        renderer = None

        try:

            env, renderer, obs, actual_seed = (
                helper.make_goal_task(
                    make_base,
                    v,
                    native_mod,
                    N,
                    seed,
                    angle,
                    distance
                )
            )
            env.max_steps = EP_CAP
            check(
                int(env.max_steps) == EP_CAP,
                "T26_ENV_HORIZON_INVALID"
            )
            print("T26_ENV_MAX_STEPS=", env.max_steps)

            side = (
                "LEFT"
                if angle < 0
                else "RIGHT"
            )

            if side not in probes:
                probes[side] = obs.copy()

            initial_raw = agent.act(obs)

            emit("TD3_T6_INITIAL_ACTOR=", {
                "episode": episode_index + 1,
                "side": side,
                "raw_action": initial_raw.tolist(),
                "goal_distance_m": distance,
                "obs_abs_max": float(
                    np.max(np.abs(obs))
                )
            })

            episode_frames = 0
            reward_sum = 0.0
            pos_steer = 0
            neg_steer = 0
            reached = False
            collision = False
            reason = "CAP"

            while (
                episode_frames < EP_CAP
                and frames < TARGET
            ):

                if time.monotonic() >= deadline:
                    reason = "WALL_BUDGET"
                    paused = True
                    break

                # 前2048帧：目标无关的双向均衡探索。
                # 不是导航控制器，不按目标方向打方向。
                # 后续仍保留20%的均衡随机探索。
                random_explore = (
                    frames < WARMUP_FRAMES
                    or rng.random() < 0.20
                )

                if random_explore:

                    block = decisions // 25

                    sign = (
                        1.0
                        if block % 2 == 0
                        else -1.0
                    )

                    raw = np.array(
                        [
                            rng.uniform(-0.4, 0.4),
                            sign * rng.uniform(0.3, 0.8)
                        ],
                        dtype=np.float32
                    )

                else:

                    raw = np.clip(
                        agent.act(obs)
                        + rng.normal(
                            0.0,
                            0.20,
                            2
                        ),
                        -1.0,
                        1.0
                    ).astype(np.float32)

                action = np.array(
                    [
                        (float(raw[0]) + 1.0) / 2.0,
                        float(raw[1])
                    ],
                    dtype=np.float32
                )

                before = float(env.data.time)

                _t66_before = float(env._goal_state()[1])
                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = N.step_checked(
                    env,
                    action
                )

                dt = float(
                    env.model.opt.timestep
                )

                nframe = int(round(
                    (
                        float(env.data.time)
                        - before
                    ) / dt
                ))

                check(
                    nframe == 1,
                    "PHYSICS_FRAME_DELTA_INVALID:"
                    + str(nframe)
                )

                next_obs = np.asarray(
                    next_obs,
                    dtype=np.float32
                ).reshape(-1)

                check(
                    next_obs.size == 112
                    and np.isfinite(next_obs).all()
                    and math.isfinite(float(reward)),
                    "TD3_T6_TRANSITION_INVALID"
                )

                collision = bool(
                    info.get("collision", False)
                )

                reached = bool(
                    info.get("reach_goal", False)
                )

                terminal = bool(
                    terminated
                    or collision
                    or reached
                )

                # 第6个字段仅用于审计左右经验覆盖。
                _t59_penalty = _t59_external_clearance_reward(env)
                reward = float(reward) - _t59_penalty * (0.8 + 2.4 * max(0.0, float(action[0]))**2)
                _t66_after = float(env._goal_state()[1])
                _t66_delta = max(-0.02, min(0.02, _t66_before - _t66_after))
                _t66_bonus = 22.0 * _t66_delta
                reward = float(reward) + _t66_bonus
                if not globals().get('_T66_REWARD_LOGGED', False):
                    print('T66_PROGRESS_REWARD_ACTIVE=PASS', 'BONUS=', round(_t66_bonus, 7), flush=True)
                    globals()['_T66_REWARD_LOGGED'] = True
                replay.append(
                    (
                        obs.copy(),
                        raw.copy(),
                        float(reward),
                        next_obs.copy(),
                        float(terminal),
                        side
                    )
                )

                pos_steer += int(
                    raw[1] > 0.25
                )

                neg_steer += int(
                    raw[1] < -0.25
                )

                obs = next_obs

                frames += 1
                decisions += 1
                episode_frames += 1

                reward_sum += float(reward)

                if (
                    len(replay) >= 256
                    and decisions % 2 == 0
                ):

                    # 先学Critic，避免早期Actor塌缩。
                    agent.policy_delay = (
                        1000000000
                        if frames < WARMUP_FRAMES
                        else 4
                    )

                    ids = rng.choice(
                        len(replay),
                        size=64,
                        replace=False
                    )

                    samples = [
                        replay[int(i)]
                        for i in ids
                    ]
                    # T43: 持久历史回放 + 真实3米成功轨迹。
                    # 只改变训练批次组成，不覆盖推理动作。
                    
                    if not hasattr(agent, "_t43_history"):
                        import hashlib as _t43_hashlib
                        import numpy as _t43_np
                        import torch as _t43_torch
                    
                        def _t43_sha(path):
                            h = _t43_hashlib.sha256()
                            with open(path, "rb") as f:
                                for chunk in iter(
                                    lambda: f.read(1048576), b""
                                ):
                                    h.update(chunk)
                            return h.hexdigest()
                    
                        if _t43_sha('/home/linghu/mkmini_stage_td3_t41_dual_replay_foundation_20261027_v1/t16_frozen_experience.pt') != '440e3887596e6900dde36e59c69d1823be6cf6948b3c92ac3e4c60e9eafb6de0':
                            raise RuntimeError("T43_'440e3887596e6900dde36e59c69d1823be6cf6948b3c92ac3e4c60e9eafb6de0'_FAILED")
                    
                        if _t43_sha('/home/linghu/mkmini_stage_td3_t42_verified_3m_replay_20261027_v1/verified_3m_success.npz') != '743c3b3099031a0f4104e2e41af22c12d491a70a6f3d780ca34477463cf9192b':
                            raise RuntimeError("T43_'743c3b3099031a0f4104e2e41af22c12d491a70a6f3d780ca34477463cf9192b'_FAILED")
                    
                        _history_obj = _t43_torch.load(
                            '/home/linghu/mkmini_stage_td3_t41_dual_replay_foundation_20261027_v1/t16_frozen_experience.pt',
                            map_location="cpu",
                            weights_only=False
                        )
                    
                        if (
                            _history_obj.get("parent_checkpoint_sha256")
                            != '21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a'
                            or not _history_obj.get(
                                "episode_segmentation_proven"
                            )
                        ):
                            raise RuntimeError("T43_HISTORY_PROVENANCE_INVALID")
                    
                        _old = _history_obj["replay"]
                        _episodes = _history_obj["episodes"]
                    
                        if len(_old) != 13491 or len(_episodes) != 8:
                            raise RuntimeError("T43_HISTORY_LENGTH_INVALID")
                    
                        _good_indices = []
                        _offset = 0
                    
                        for _episode in _episodes:
                            _count = int(_episode["physics_frames"])
                    
                            if (
                                _episode["success"]
                                and not _episode["collision"]
                            ):
                                _good_indices.extend(
                                    range(_offset, _offset + _count)
                                )
                    
                            _offset += _count
                    
                        if _offset != len(_old) or len(_good_indices) < 8:
                            raise RuntimeError("T43_HISTORY_SEGMENTS_INVALID")
                    
                        with _t43_np.load('/home/linghu/mkmini_stage_td3_t42_verified_3m_replay_20261027_v1/verified_3m_success.npz') as _archive:
                            _s = _t43_np.asarray(
                                _archive["states"],
                                dtype=_t43_np.float32
                            ).copy()
                    
                            _a = _t43_np.asarray(
                                _archive["raw_actions"],
                                dtype=_t43_np.float32
                            ).copy()
                    
                            _r = _t43_np.asarray(
                                _archive["rewards"],
                                dtype=_t43_np.float32
                            ).copy()
                    
                            _ns = _t43_np.asarray(
                                _archive["next_states"],
                                dtype=_t43_np.float32
                            ).copy()
                    
                            _d = _t43_np.asarray(
                                _archive["dones"],
                                dtype=_t43_np.float32
                            ).copy()
                    
                            _offsets = _archive[
                                "episode_offsets"
                            ].tolist()
                    
                        if (
                            _s.shape != (20319, 112)
                            or _a.shape != (20319, 2)
                            or _r.shape != (20319,)
                            or _ns.shape != (20319, 112)
                            or _d.shape != (20319,)
                            or _offsets != [0, 10612, 20319]
                            or _t43_np.flatnonzero(_d).tolist()
                                != [10611, 20318]
                            or not all(
                                _t43_np.isfinite(x).all()
                                for x in (_s, _a, _r, _ns, _d)
                            )
                        ):
                            raise RuntimeError("T43_3M_TRANSITIONS_INVALID")
                    
                        agent._t43_history = _old
                        agent._t43_old_success_indices = (
                            _t43_np.asarray(
                                _good_indices,
                                dtype=_t43_np.int64
                            )
                        )
                    
                        agent._t43_3m = (
                            _s, _a, _r, _ns, _d
                        )
                    
                        agent._t43_batches = 0
                    
                        print(
                            "T43_ARCHIVES_VERIFIED=PASS",
                            "OLD_TRANSITIONS=13491",
                            "OLD_SUCCESS_TRANSITIONS=",
                            len(_good_indices),
                            "THREE_M_TRANSITIONS=20319",
                            flush=True
                        )
                    
                    # 使用独立、可由更新计数重建的随机源。
                    # 不额外消耗原训练器的探索随机数。
                    _mix_rng = _t43_np.random.default_rng(
                        202610274300000 + int(agent.total_updates)
                    )
                    
                    _hist = agent._t43_history
                    _good = agent._t43_old_success_indices
                    
                    _old_random_ids = _mix_rng.choice(
                        len(_hist),
                        size=8,
                        replace=False
                    )
                    
                    _old_good_ids = _mix_rng.choice(
                        _good,
                        size=8,
                        replace=False
                    )
                    
                    # 两侧真实3米成功轨迹各8条。
                    _left_ids = _mix_rng.choice(
                        10612,
                        size=8,
                        replace=False
                    )
                    
                    _right_ids = _mix_rng.choice(
                        _t43_np.arange(10612, 20319),
                        size=8,
                        replace=False
                    )
                    
                    _s, _a, _r, _ns, _d = agent._t43_3m
                    
                    def _three_m_transition(i):
                        i = int(i)
                        return (
                            _s[i],
                            _a[i],
                            float(_r[i]),
                            _ns[i],
                            float(_d[i]),
                            "LEFT" if i < 10612 else "RIGHT"
                        )
                    
                    # 保留32条滚动经验；
                    # 将其余32条替换为独立历史样本。
                    samples[32:40] = [
                        _hist[int(i)] for i in _old_good_ids
                    ]
                    
                    samples[40:48] = [
                        _hist[int(i)] for i in _old_random_ids
                    ]
                    
                    samples[48:56] = [
                        _three_m_transition(i) for i in _left_ids
                    ]
                    
                    samples[56:64] = [
                        _three_m_transition(i) for i in _right_ids
                    ]
                    
                    if len(samples) != 64:
                        raise RuntimeError("T43_MIXED_BATCH_INVALID")
                    
                    agent._t43_batches += 1
                    
                    if agent._t43_batches == 1:
                        print(
                            "T43_MIXED_REPLAY_ACTIVE=PASS",
                            "ONLINE=32",
                            "OLD_SUCCESS=8",
                            "OLD_ALL=8",
                            "THREE_M_LEFT=8",
                            "THREE_M_RIGHT=8",
                            "BATCH=64",
                            flush=True
                        )
                    
                    if agent._t43_batches % 2000 == 0:
                        print(
                            "T43_MIXED_BATCH_COUNT=",
                            agent._t43_batches,
                            flush=True
                        )

                    batch = tuple(
                        np.asarray(
                            [
                                row[k]
                                for row in samples
                            ],
                            dtype=np.float32
                        )
                        for k in range(5)
                    )

                    loss = agent.update(
                        batch
                    )

                    check(
                        math.isfinite(
                            loss["critic_loss"]
                        ),
                        "TD3_T6_CRITIC_NONFINITE"
                    )

                    if loss["actor_loss"] is not None:
                        check(
                            math.isfinite(
                                loss["actor_loss"]
                            ),
                            "TD3_T6_ACTOR_NONFINITE"
                        )

                if episode_frames % 250 == 0:

                    emit("TD3_T6_PROGRESS=", {
                        "frames": frames,
                        "episode_frames": episode_frames,
                        "updates": agent.total_updates,
                        "remaining_m": round(
                            float(
                                env._goal_state()[1]
                            ),
                            4
                        ),
                        "positive_steer": pos_steer,
                        "negative_steer": neg_steer,
                        "actor_updates_enabled":
                            frames >= WARMUP_FRAMES
                    })

                if terminal or truncated:
                    reason = (
                        "GOAL"
                        if reached
                        else "CONTACT"
                        if collision
                        else "TRUNCATED"
                    )
                    break

            row = {
                "episode": episode_index + 1,
                "seed": actual_seed,
                "side": side,
                "angle": angle,
                "distance_m": distance,
                "physics_frames": episode_frames,
                "remaining_m": float(
                    env._goal_state()[1]
                ),
                "reward_sum": reward_sum,
                "positive_steer": pos_steer,
                "negative_steer": neg_steer,
                "success": bool(reached),
                "collision": bool(collision),
                "end": reason
            }

            check(
                episode_frames > 0,
                "EMPTY_T6_EPISODE"
            )

            episodes.append(row)

            episode_index += 1

            emit("TD3_T6_EPISODE=", row)

        finally:

            if renderer is not None:
                renderer.close()

            if env is not None:
                env.close()

        actor_results, collapsed = actor_probe(
            agent,
            probes
        )

        if len(actor_results) == 2:
            left = actor_results["LEFT"]
            right = actor_results["RIGHT"]

            same_sign = (
                left["steer"] * right["steer"] > 0
            )

            severe_saturation = (
                left["throttle"] > 0.95
                and right["throttle"] > 0.95
                and abs(left["steer"]) > 0.85
                and abs(right["steer"]) > 0.85
                and same_sign
                and abs(
                    left["steer"] - right["steer"]
                ) < 0.20
            )

            collapsed = (
                collapsed or severe_saturation
            )

            emit("TD3_T9_STRICT_GATE=", {
                "frames": frames,
                "severe_saturation":
                    severe_saturation,
                "actor_left": left,
                "actor_right": right
            })

        emit("TD3_T6_ACTOR_GATE=", {
            "frames": frames,
            "left_right_probe": actor_results,
            "saturated": collapsed
        })

        if (
            frames >= WARMUP_FRAMES
            and collapsed
        ):
            blocked = True

        state = {
            "algorithm": "TD3",
            "manifest": manifest,
            "actor": agent.actor.state_dict(),
            "actor_target":
                agent.actor_target.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target":
                agent.critic_target.state_dict(),
            "actor_optimizer":
                agent.actor_optimizer.state_dict(),
            "critic_optimizer":
                agent.critic_optimizer.state_dict(),
            "updates": agent.total_updates,
            "frames": frames,
            "decisions": decisions,
            "episode_index": episode_index,
            "episodes": episodes,
            "replay": list(replay),
            "probes": probes,
            "rng_state":
                rng.bit_generator.state,
            "torch_rng_state":
                torch.get_rng_state(),
            "automatic_promotion": False
        }

        filename = (
            f"episode_{episode_index:04d}.pt"
        )

        ck = checkpoint_dir / filename
        temp = ck.with_suffix(".pt.partial")

        check(
            not ck.exists(),
            "CHECKPOINT_ALREADY_EXISTS"
        )

        torch.save(state, temp)

        os.replace(temp, ck)

        summary = {
            "state": (
                "BLOCKED_POLICY_SATURATION"
                if blocked
                else "PILOT_COMPLETED_NONCOLLAPSED"
                if frames >= TARGET
                else "PAUSED_WALL_BUDGET"
                if paused
                else "RUNNING"
            ),
            "frames": frames,
            "target_frames": TARGET,
            "updates": agent.total_updates,
            "episodes": episode_index,
            "success": sum(
                int(x["success"])
                for x in episodes
            ),
            "collisions": sum(
                int(x["collision"])
                for x in episodes
            ),
            "checkpoint": filename,
            "checkpoint_sha256": sha(ck),
            "actor_probe": actor_results,
            "collapsed": collapsed,
            "automatic_promotion": False,
            "scientific_performance_confirmed": False
        }

        save_json(
            status_path,
            summary
        )

        emit("TD3_T6_CHECKPOINT=", summary)

        if blocked or paused:
            break

    if not status_path.is_file():
        raise RuntimeError(
            "T6_NO_COMPLETED_EPISODE"
        )

    final = json.loads(
        status_path.read_text()
    )

    # 核查每一方向是否有足够的正负探索。
    coverage = {}

    for side in ("LEFT", "RIGHT"):

        data = [
            float(row[1][1])
            for row in replay
            if row[5] == side
        ]

        if data:
            x = np.asarray(data)

            coverage[side] = {
                "count": len(x),
                "positive_fraction": float(
                    np.mean(x > 0.25)
                ),
                "negative_fraction": float(
                    np.mean(x < -0.25)
                )
            }

    emit("TD3_T6_EXPLORATION_COVERAGE=", coverage)

    if (
        final["frames"] >= TARGET
        and not final["collapsed"]
    ):

        balanced = all(
            side in coverage
            and coverage[side]["count"] >= 100
            and coverage[side]["positive_fraction"] >= 0.1
            and coverage[side]["negative_fraction"] >= 0.1
            for side in ("LEFT", "RIGHT")
        )

        if not balanced:
            final["state"] = (
                "BLOCKED_EXPLORATION_COVERAGE"
            )
            save_json(status_path, final)

    check(
        sha(FROZEN) == EXPECTED[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    emit("TD3_T6_FINAL=", final)

    print(
        "FROZEN_MODEL_MODIFIED=FALSE",
        flush=True
    )


if __name__ == "__main__":
    try:

        main()

    except Exception as error:

        emit("TD3_T6_BLOCKED=", {
            "error": repr(error)
        })

        traceback.print_exc()

    finally:

        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
