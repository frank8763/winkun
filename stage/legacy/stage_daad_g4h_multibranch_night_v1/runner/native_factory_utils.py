#!/usr/bin/env python3
"""STAGE ONLY: deterministic isolated MJB snapshots + EXACT low-level factory hook.

No project edits, no actor training, no Phase writes. This is an attachment
and sensor/goal *smoke pilot*, not a full goal randomization success gate.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap
import traceback
import types
from contextlib import contextmanager

PROJECT_G1 = Path.home()/'mkmini_stage_daad_native_layout_recovery_v4_2'/'run'
PROJECT_G2C = Path.home()/'mkmini_stage_daad_g2c_exact_factory_entry_v1'/'run'
PARENT = Path.home()/'mkmini_stage32e5_b0d3t12a0m10_heldout_long_v1'/'strictly_improved_research_best.zip'
PARENT_SHA = '35c301288bb4bd2fabdaffd4d17d5fd03b5d6788ea35a03a169d2c30316a3f5a'
ORIGINAL_BASE_SHA = 'b01165f5d03b760a3ac770ed31e84df0544c36764e625aaa4c2fea5970aedf91'
LAYOUTS = ('reference', 'peripheral_shift', 'peripheral_pair')
PASS = 'G2D_BINARY_FACTORY_ATTACHMENT_PASS_GOAL_LIDAR_GENERALIZATION_PENDING'
OUT = Path.home()/'mkmini_stage_daad_g2d_binary_factory_attach_v1'/'run'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def replace_exact(src, old, new):
    count=src.count(old)
    if count!=1:raise RuntimeError('SOURCE_CONSTRUCTOR_DRIFT_REPLACEMENT_COUNT:'+str(count)+':'+old)
    return src.replace(old,new,1)


def isolated_constructor(original, module, selected_mjb, events):
    """Copy exactly the known base constructor *in memory*, alter only MJB path.

    This avoids unsupported MjModel class patching and does not mutate code on
    disk or change any already-constructed MjModel/MjData. The function is scoped
    to each new Stage env construction and original is always restored.
    """
    src=textwrap.dedent(inspect.getsource(original))
    src=replace_exact(src,'super().__init__()','gym.Env.__init__(self)')
    src=replace_exact(src,'model_path = assets / "phase10_original_scene.mjb"',
                       'model_path = _stage_g2d_selected_mjb')
    assert 'self.model = mujoco.MjModel.from_binary_path(' in src
    assert 'state_path = assets / "phase10_original_scene_state.npz"' in src
    g=vars(module).copy()
    g['_stage_g2d_selected_mjb'] = Path(selected_mjb)
    exec(compile(src,'<stage_g2d_isolated_base_constructor>','exec'),g)
    result=g['__init__']
    def counted(self,*args,**kwargs):
        events.append({'selected_mjb':str(selected_mjb),'name':type(self).__name__})
        return result(self,*args,**kwargs)
    return counted


@contextmanager
def selected_binary_factory(base, module, mjb, events):
    original=base.__init__
    base.__init__=isolated_constructor(original,module,mjb,events)
    try:
        yield
    finally:
        base.__init__=original
        if base.__init__ is not original:
            raise RuntimeError('BASE_CONSTRUCTOR_RESTORE_FAIL')


def owners(env, mujoco):
    queue=[('outer',env)]
    seen=set(); rows=[]; ms=[]; ds=[]
    while queue and len(rows)<40:
        path,obj=queue.pop(0)
        if id(obj) in seen:continue
        seen.add(id(obj))
        row={'path':path,'cls':type(obj).__name__}
        for name in ('model','mj_model'):
            try: value=getattr(obj,name,None)
            except Exception:continue
            if isinstance(value,mujoco.MjModel):
                row['model_id']=id(value);row['model_attr']=name;ms.append(id(value));break
        for name in ('data','mj_data'):
            try:value=getattr(obj,name,None)
            except Exception:continue
            if isinstance(value,mujoco.MjData):
                row['data_id']=id(value);row['data_attr']=name;ds.append(id(value));break
        rows.append(row)
        for attr in ('env','task','base_env','_env','raw_env'):
            try:value=getattr(obj,attr,None)
            except Exception:continue
            if value is not None and not isinstance(value,(dict,list,tuple,int,float,str)):
                queue.append((path+'.'+attr,value))
    if not ms or len(set(ms))!=1 or not ds or len(set(ds))!=1:
        raise RuntimeError('MODEL_DATA_OWNER_SPLIT_OR_MISSING:'+str(rows))
    return rows


def geom_signature(model, mujoco):
    d={}
    for i in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if name.startswith(('real_rect_obs_','real_circle_obs_')):
            d[name]={'pos':[round(float(x),6) for x in model.geom_pos[i]],
                     'size':[round(float(x),6) for x in model.geom_size[i]],
                     'contype':int(model.geom_contype[i]),'conaffinity':int(model.geom_conaffinity[i])}
    return d


def common_model_checks(model, expected, mujoco):
    fields=('nq','nv','nu','ngeom','njnt')
    if any(int(getattr(model,k))!=int(getattr(expected,k)) for k in fields):
        raise RuntimeError('MODEL_DIMS_DRIFT:'+repr({k:(getattr(model,k),getattr(expected,k)) for k in fields}))
    if abs(float(model.opt.timestep)-0.002)>1e-9:raise RuntimeError('TIMESTEP_DRIFT')
    a=geom_signature(model,mujoco);b=geom_signature(expected,mujoco)
    if a!=b:
        dif=[n for n in (a.keys()|b.keys()) if a.get(n)!=b.get(n)]
        raise RuntimeError('ATTACHED_GEOMETRY_NOT_FROM_SELECTED_SNAPSHOT:'+repr(dif[:12]))
    return a


def check_state_file(state, model):
    import numpy as np
    with np.load(state,allow_pickle=False) as s:
        expected={'qpos':model.nq,'qvel':model.nv,'ctrl':model.nu,'goal_xy':2}
        for name,n in expected.items():
            if name not in s:raise RuntimeError('SNAPSHOT_STATE_MISSING:'+name)
            a=np.asarray(s[name]).reshape(-1)
            if a.size!=n or not np.isfinite(a).all():raise RuntimeError('SNAPSHOT_STATE_DIMENSION_BAD:'+name)
        for name in ('frame_skip','max_steer','max_wheel','bicycle_speed_gain','bicycle_yaw_gain','bicycle_wheelbase'):
            if name not in s or not np.isfinite(s[name]).all():raise RuntimeError('SNAPSHOT_STATE_FIELD_BAD:'+name)
    return True


def make_binaries(mujoco, out, g1):
    binaries={}; models={}; hashes={}; signatures={}
    for name in LAYOUTS:
        xml=PROJECT_G1/(name+'.xml')
        rows=[r for r in g1['layouts'] if r.get('layout')==name and r.get('status')=='ISOLATED_COMPILE_PASS']
        if len(rows)!=1 or Path(rows[0].get('xml','')).resolve()!=xml.resolve():
            raise RuntimeError('G1_LAYOUT_ROW_NOT_VERIFIED:'+name)
        if not xml.is_file():raise RuntimeError('G1_XML_MISSING:'+str(xml))
        m=mujoco.MjModel.from_xml_path(str(xml))
        path=out/(name+'.mjb')
        mujoco.mj_saveModel(m,str(path))
        if not path.is_file() or path.stat().st_size<1024:raise RuntimeError('MJB_WRITE_FAIL:'+name)
        reloaded=mujoco.MjModel.from_binary_path(str(path))
        common_model_checks(reloaded,m,mujoco)
        binaries[name]=path;models[name]=reloaded
        hashes[name]=sha(path)
        signatures[name]=geom_signature(m,mujoco)
    if len(set(hashes.values()))!=3:raise RuntimeError('MJB_VARIANTS_NOT_DISTINCT')
    if len({json.dumps(v,sort_keys=True) for v in signatures.values()})!=3:
        raise RuntimeError('THREE_LAYOUT_GEOMETRY_NOT_DISTINCT')
    return binaries, models, hashes


def perform(root, report, out):
    import mujoco
    import numpy as np
    for x in (root,root/'training/e2e_phase10_lidar_v7_curriculum'):
        if str(x) not in sys.path:sys.path.insert(0,str(x))
    from training.e2e_phase10_lidar_v1 import phase10_scene_e2e_env as B
    from training.e2e_phase10_lidar_v7_curriculum import t12a0m27_frontwheel_risk_sac as M27
    srcfile=Path(inspect.getsourcefile(B.Phase10SceneE2ELidarEnv.__init__)).resolve()
    if sha(srcfile)!=ORIGINAL_BASE_SHA:raise RuntimeError('ORIGINAL_BASE_SOURCE_SHA_DRIFT:'+str(srcfile))
    snapshot_state=srcfile.parent/'assets'/'phase10_original_scene_state.npz'
    if not snapshot_state.is_file():raise RuntimeError('ORIGINAL_STATE_SNAPSHOT_MISSING:'+str(snapshot_state))
    rt=M27.runtime(root)
    report['exact_base_constructor_file']=str(srcfile)
    g1file=PROJECT_G1/'result.json';g2cfile=PROJECT_G2C/'report.json'
    if not g1file.is_file() or not g2cfile.is_file():raise RuntimeError('G1_OR_G2C_REPORT_MISSING')
    g1=json.loads(g1file.read_text());g2c=json.loads(g2cfile.read_text())
    if g1.get('gate')!='ISOLATED_LAYOUTS_PASS_NATIVE_ENV_INTEGRATION_PENDING':raise RuntimeError('G1_GATE_NOT_PASS')
    if g2c.get('gate')!='G2C_EXACT_FACTORY_TRACE_PASS_LAYOUT_NOT_ATTACHED':raise RuntimeError('G2C_GATE_NOT_PASS')
    expected_hashes=g2c.get('g1_layout_sha256',{})
    for name in LAYOUTS:
        xml=PROJECT_G1/(name+'.xml')
        if sha(xml)!=expected_hashes.get(name):raise RuntimeError('G1_XML_HASH_DRIFT:'+name)
    binaries,models,model_hashes=make_binaries(mujoco,out,g1)
    report['binary_hashes']=model_hashes
    report['binary_files']={k:str(v) for k,v in binaries.items()}
    report['reference_layout_model_dims']={k:int(getattr(models['reference'],k)) for k in ('nq','nv','nu','ngeom','njnt')}
    report['native_state_shape_pass']=check_state_file(snapshot_state,models['reference'])
    # Initial pose and actuator arrays from original Stage snapshot; no edits to qpos/qvel.
    before_init=B.Phase10SceneE2ELidarEnv.__init__
    for name in LAYOUTS:
        events=[];env=None; row={'layout':name,'status':'PENDING','attached':False}
        report['layout_trials'].append(row)
        try:
            with selected_binary_factory(B.Phase10SceneE2ELidarEnv,B,binaries[name],events):
                env=rt['make_env'](-2.17)
            if B.Phase10SceneE2ELidarEnv.__init__ is not before_init:raise RuntimeError('FACTORY_NOT_RESTORED')
            if len(events)!=1:raise RuntimeError('EXACT_BINARY_FACTORY_CALL_COUNT:'+str(len(events)))
            obs,reset_info=env.reset(seed=20261012)
            M27.physics_lock(env,rt)
            rows=owners(env,mujoco)
            live=env.model
            sig=common_model_checks(live,models[name],mujoco)
            z=np.asarray(obs,dtype=np.float32).reshape(-1)
            if z.shape!=(112,) or not np.isfinite(z).all():raise RuntimeError('112_OBSERVATION_INVALID')
            # Walk the actual model-owning inheritance chain: `outer.env.env.env.task.env`.
            # Do not assume every wrapper exposes the same `.task` attribute.
            obj=env;visited=set();a=None
            while obj is not None and id(obj) not in visited:
                visited.add(id(obj))
                raw_lidar=getattr(obj,'_lidar48',None)
                if callable(raw_lidar):
                    a=np.asarray(raw_lidar(),dtype=np.float32).reshape(-1)
                    break
                nested=getattr(obj,'env',None)
                if nested is None:nested=getattr(obj,'task',None)
                obj=nested
            if a is None or a.shape!=(48,) or not np.isfinite(a).all():
                raise RuntimeError('NATIVE_RAW_LIDAR48_NOT_FOUND_OR_INVALID')
            _,reward,terminated,truncated,info=env.step(np.zeros(2,dtype=np.float32))
            for key in ('stage27_goal_m','stage27_raw_vehicle_obstacle_contact','stage27_success'):
                if key not in info:raise RuntimeError('NATIVE_METRIC_MISSING:'+key)
            if not math.isfinite(float(info['stage27_goal_m'])):raise RuntimeError('NATIVE_GOAL_NONFINITE')
            if not math.isfinite(float(reward)):raise RuntimeError('REWARD_NONFINITE')
            if bool(info['stage27_raw_vehicle_obstacle_contact']):
                raise RuntimeError('INITIAL_STEP_RAW_CONTACT')
            if bool(info['stage27_success']):
                raise RuntimeError('IMMEDIATE_UNEXPECTED_GOAL_SUCCESS')
            row.update({'status':'ATTACHED_SMOKE_PASS','attached':True,
                'model_geom_signature':hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest(),
                'binary_mjb_sha256':model_hashes[name],
                'model_owner_paths':rows,
                'exact_constructor_calls':len(events),
                'reset_obs_sha256':hashlib.sha256(z.tobytes()).hexdigest(),
                'reset_lidar48_sha256':hashlib.sha256(a.tobytes()).hexdigest() if a is not None else None,
                'native_goal_m':float(info['stage27_goal_m']),
                'native_success':bool(info['stage27_success']),
                'native_contact':bool(info['stage27_raw_vehicle_obstacle_contact']),
                'reward':float(reward),
                'terminated':bool(terminated),'truncated':bool(truncated)})
        except Exception as ex:
            row.update({'status':'BLOCKED','error':repr(ex),'traceback':traceback.format_exc(limit=9)})
            print('G2D_LAYOUT',name,'BLOCKED',repr(ex),flush=True)
        finally:
            if env is not None:
                try:env.close()
                except Exception:pass
            if B.Phase10SceneE2ELidarEnv.__init__ is not before_init:
                B.Phase10SceneE2ELidarEnv.__init__=before_init
                raise RuntimeError('BASE_CONSTRUCTOR_LEAK_PREVENTED')
        print('G2D_LAYOUT',name,'ATTACHED',row['attached'],'STATUS',row['status'],flush=True)
    if all(x['attached'] for x in report['layout_trials']):
        sigs={x['model_geom_signature'] for x in report['layout_trials']}
        if len(sigs)==3: report['gate']=PASS
        else: report['gate']='BLOCKED_G2D_RUNTIME_GEOMETRY_NOT_DISTINCT'
    else:report['gate']='BLOCKED_G2D_EXACT_BINARY_FACTORY'
    lidar={x.get('reset_lidar48_sha256') for x in report['layout_trials']}
    report['initial_raw_lidar_differs_across_layouts']=len(lidar)==3
    report['goal_randomization_validated']=False
    report['goal_lidar_full_gate_pass']=False
    report['training_gate_pass']=False


def selftest():
    sample='''def __init__(self, max_steps: int = 600) -> None:\n    super().__init__()\n    model_path = assets / "phase10_original_scene.mjb"\n    state_path = assets / "phase10_original_scene_state.npz"\n    self.model = mujoco.MjModel.from_binary_path(str(model_path))\n'''
    sample=replace_exact(sample,'super().__init__()','gym.Env.__init__(self)')
    sample=replace_exact(sample,'model_path = assets / "phase10_original_scene.mjb"','model_path = _stage_g2d_selected_mjb')
    compile(sample,'<selftest>','exec')
    try:replace_exact('abc','not-present','xx')
    except RuntimeError:pass
    else:raise AssertionError('DRIFT_GATE_FAIL')
    print('SELFTEST=PASS EXACT_BINARY_LOADER_SOURCE=PASS RESTORE_SCOPE=PASS NO_PROJECT_WRITES=PASS')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(Path.home()/'catkin_ws/src/mk-mini'))
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:selftest();return 0
    out=OUT
    out.mkdir(parents=True,exist_ok=True)
    dest=out/'report.json'
    if dest.is_file():
        old=json.loads(dest.read_text())
        if old.get('gate')==PASS:
            print('G2D_VERIFIED_REPORT_EXISTS_NO_DUPLICATE='+str(dest));return 0
        print('G2D_EXISTING_BLOCKED_REPORT_WILL_NOT_BE_OVERWRITTEN='+str(dest))
        return 3
    root=Path(args.root).expanduser().resolve()
    init=sha(PARENT) if PARENT.is_file() else None
    report={'gate':'BLOCKED_G2D_PREREQUISITE','stage_only':True,'planner_used':False,
            'training_started':False,'training_gate_pass':False,'phase_write':False,
            'parent_promoted':False,'parent_sha256_before':init,'layout_trials':[]}
    try:
        if init!=PARENT_SHA:raise RuntimeError('PARENT_FROZEN_HASH_MISMATCH')
        perform(root,report,out)
    except Exception as e:
        report['gate']='BLOCKED_G2D_FATAL'
        report['error']=repr(e)
        report['traceback']=traceback.format_exc(limit=13)
        print('G2D_FATAL',repr(e),flush=True)
    finally:
        final=sha(PARENT) if PARENT.is_file() else None
        report['parent_sha256_after']=final
        report['parent_unchanged']=init==final==PARENT_SHA
        if not report['parent_unchanged']:report['gate']='BLOCKED_G2D_PARENT_INTEGRITY'
        dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('GATE='+report['gate'])
        print('RESULT='+str(dest))
        print('NO_TRAINING=TRUE PHASE_WRITE=FALSE PARENT_FROZEN='+str(report['parent_unchanged']))
    return 0 if report['gate']==PASS else 3

# Embedded as Stage-only helper; do not run the old G2D experiment.
if __name__=='__main__':raise SystemExit(main())
