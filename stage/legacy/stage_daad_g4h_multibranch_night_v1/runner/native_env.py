#!/usr/bin/env python3
"""Isolated Stage-only goal-conditioned native MJCF/MJB task factory.

Never rewrites a live world geom or a user project file. The existing Stage
Phase10 (historical naming) pure end-to-end MuJoCo base remains the motion owner.
"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import native_factory_utils as U

HOME=Path.home()
ROOT_G1=HOME/'mkmini_stage_daad_native_layout_recovery_v4_2'/'run'
ROOT_G2C=HOME/'mkmini_stage_daad_g2c_exact_factory_entry_v1'/'run'
ROOT=HOME/'mkmini_stage_daad_g2e_overnight_gate_v1'/'run'
BIN=ROOT/'models'
LAYOUTS=('reference','peripheral_shift','peripheral_pair')
TRAIN_LAYOUTS=('reference','peripheral_shift')
TEST_LAYOUT='peripheral_pair'
PARENT=U.PARENT
PARENT_SHA=U.PARENT_SHA


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')
    temporary.replace(path)


def env_setup(project):
    project=Path(project).resolve()
    for item in (project,project/'training/e2e_phase10_lidar_v7_curriculum'):
        if str(item) not in sys.path:sys.path.insert(0,str(item))
    import mujoco
    from training.e2e_phase10_lidar_v7_curriculum.stage8_full_plant_patch import install_full_plant_patch
    from training.e2e_phase10_lidar_v7_curriculum.stage9_runtime_fullplant_fix_v5_2 import install_runtime_fullplant_fix_v5_2
    install_full_plant_patch()
    install_runtime_fullplant_fix_v5_2()
    from training.e2e_phase10_lidar_v1 import phase10_scene_e2e_env as B
    from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a_dense_env import Phase10Stage1ADenseEnv
    from training.e2e_phase10_lidar_v7_curriculum.train_stage3_full_indoor_auto import FullIndoorEnv
    if U.sha(Path(B.__file__))!=U.ORIGINAL_BASE_SHA:raise RuntimeError('BASE_SOURCE_SHA_DRIFT')
    return mujoco,B,Phase10Stage1ADenseEnv,FullIndoorEnv


def binary_inventory(mj,need_create=False):
    g1file=ROOT_G1/'result.json';g2cfile=ROOT_G2C/'report.json'
    if not g1file.is_file() or not g2cfile.is_file():raise RuntimeError('G1_G2C_EVIDENCE_MISSING')
    g1=json.loads(g1file.read_text());g2c=json.loads(g2cfile.read_text())
    if g1.get('gate')!='ISOLATED_LAYOUTS_PASS_NATIVE_ENV_INTEGRATION_PENDING':raise RuntimeError('G1_CONTRACT_NOT_PASS')
    if g2c.get('gate')!='G2C_EXACT_FACTORY_TRACE_PASS_LAYOUT_NOT_ATTACHED':raise RuntimeError('G2C_TRACE_NOT_PASS')
    BIN.mkdir(parents=True,exist_ok=True)
    rows={}
    for name in LAYOUTS:
        xml=ROOT_G1/(name+'.xml')
        if U.sha(xml)!=g2c.get('g1_layout_sha256',{}).get(name):raise RuntimeError('G1_XML_CHANGED:'+name)
        expected=mj.MjModel.from_xml_path(str(xml))
        file=BIN/(name+'.mjb')
        if not file.is_file():
            if not need_create:raise RuntimeError('MJB_MISSING_PREFLIGHT_REQUIRED:'+name)
            mj.mj_saveModel(expected,str(file))
        real=mj.MjModel.from_binary_path(str(file))
        U.common_model_checks(real,expected,mj)
        if len(file.read_bytes())<1024:raise RuntimeError('MJB_BAD:'+name)
        rows[name]={'path':str(file),'sha256':U.sha(file),'geom_signature':hashlib.sha256(json.dumps(U.geom_signature(real,mj),sort_keys=True).encode()).hexdigest()}
    if len({r['geom_signature'] for r in rows.values()})!=3:raise RuntimeError('LAYOUT_GEOMETRIES_NOT_DISTINCT')
    return rows


def create(project, layout, difficulty, seed):
    if layout not in LAYOUTS or difficulty not in ('short','medium','indoor_near'):
        raise ValueError('INVALID_STAGE_LAYOUT_OR_TASK')
    mj,B,Stage1A,FullIndoor=env_setup(project)
    binary=BIN/(layout+'.mjb')
    if not binary.is_file():raise RuntimeError('MJB_NOT_COMPILED:'+layout)
    expected=mj.MjModel.from_binary_path(str(binary))
    events=[];env=None
    try:
        with U.selected_binary_factory(B.Phase10SceneE2ELidarEnv,B,binary,events):
            if difficulty=='short':
                env=Stage1A(max_steps=700,goal_distance_min=.45,goal_distance_max=.65,goal_angle_limit_deg=35,goal_clearance_margin=.28)
            elif difficulty=='medium':
                env=Stage1A(max_steps=1100,goal_distance_min=.75,goal_distance_max=1.15,goal_angle_limit_deg=55,goal_clearance_margin=.30)
            else:
                env=FullIndoor(fixed_task='indoor_near',seed=int(seed))
        if len(events)!=1:raise RuntimeError('BASE_BINARY_FACTORY_NOT_CALLED_EXACTLY_ONCE:'+repr(events))
        U.common_model_checks(env.model,expected,mj)
        if tuple(env.observation_space.shape)!=(112,) or tuple(env.action_space.shape)!=(2,):
            raise RuntimeError('POLICY_INTERFACE_NOT_112x2')
        obs,info=env.reset(seed=int(seed))
        obs=np.asarray(obs,dtype=np.float32).reshape(-1)
        if obs.shape!=(112,) or not np.isfinite(obs).all():raise RuntimeError('RESET_OBS_INVALID')
        U.common_model_checks(env.model,expected,mj)
        if int(mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_7'))<0:raise RuntimeError('STATIC_GEOM_MISSING')
        if not hasattr(env,'_lidar48'):raise RuntimeError('NATIVE_LIDAR_NOT_FOUND')
        raw=np.asarray(env._lidar48(),dtype=np.float32).reshape(-1)
        if raw.shape!=(48,) or not np.isfinite(raw).all() or not np.allclose(raw,obs[:48],atol=1e-5):
            raise RuntimeError('NATIVE_LIDAR112_DESYNC')
        goal=np.asarray(env.goal_xy,dtype=float).reshape(2)
        if not np.isfinite(goal).all():raise RuntimeError('INVALID_SAMPLED_GOAL')
        if bool(env._collision()):raise RuntimeError('NATIVE_INITIAL_CONTACT')
        return env,obs,dict(info),goal
    except Exception:
        if env is not None:env.close()
        raise


def step_checked(env,action):
    import mujoco
    from training.e2e_phase10_lidar_v7_curriculum.stage9_runtime_fullplant_fix_v5_2 import get_last_step_record
    a=np.asarray(action,dtype=np.float32).reshape(2)
    if not np.isfinite(a).all() or np.any(np.abs(a)>1.00001):raise RuntimeError('BAD_STAGE_ACTION')
    obs,reward,term,trunc,info=env.step(a)
    ob=np.asarray(obs,dtype=np.float32).reshape(-1)
    if ob.shape!=(112,) or not np.isfinite(ob).all() or not math.isfinite(float(reward)):
        raise RuntimeError('NATIVE_STEP_NONFINITE')
    for key in ('goal_dist','collision','reach_goal'):
        if key not in info:raise RuntimeError('MISSING_NATIVE_EVENT:'+key)
    distance=float(info['goal_dist']);collision=bool(info['collision']);goal=bool(info['reach_goal'])
    if not math.isfinite(distance) or distance<0:raise RuntimeError('BAD_GOAL_DISTANCE')
    if goal and collision:raise RuntimeError('GOAL_AND_CONTACT_SAME_STEP')
    if (goal or collision) and not term:raise RuntimeError('EVENT_MUST_TERMINATE')
    # Preserve dynamic actuation witness: real MuJoCo steps, not a qpos kinematic override.
    record=get_last_step_record(env.data)
    if not record or int(record.get('physics_calls',0))<1:
        raise RuntimeError('ACKERMANN_FULLPLANT_STEP_NOT_CONFIRMED')
    return ob,float(reward),bool(term),bool(trunc),dict(info)
