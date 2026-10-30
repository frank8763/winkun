"""Inspect and measure the actual inherited Stage action conversion; no training."""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ['C98_JOB'] = os.environ['TD3_JOB'] = str(ROOT)
os.environ['C22_SCREEN_ONLY'] = '0'
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('t149_probe_train', ROOT.parent / 'stage_t148/train.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
helper = t.load_module(t.HELPER, 't149_probe_helper')
legacy = t.load_module(t.STAGE, 't149_probe_legacy')
for p, h in legacy.EXPECTED.items():
    assert t.sha(p) == h
runner = legacy.module(legacy.SRC, 't149_probe_runner')
_, v, mod, prep, _ = runner.load_native()
N = prep[3]
make = legacy.make_short_task(runner.make_task)
env = renderer = None
try:
    env, renderer, obs, seed = helper.make_goal_task(make,v,mod,N,2029653602,13.6,3.0)
    sources = {}
    for label, fn in [('policy_to_commands', env._policy_to_commands), ('native_step_checked', N.step_checked)]:
        sources[label] = {'file':inspect.getsourcefile(fn),'source':inspect.getsource(fn)}
    report = {'experiment':'STAGE_T149_ACTUAL_ACTION_INTERFACE','training':False,
        'environment_class':str(type(env)), 'sources':sources,
        'frame_skip':env.frame_skip,'physics_dt':float(env.model.opt.timestep),'commands':[]}
    qpos, qvel, ctrl = env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy()
    prev = env.prev_action.copy()
    for raw in (-1.0,-.5,0.0,.5,1.0):
        mapped = np.array([(raw+1)/2,0],dtype=np.float32)
        result = tuple(map(float, env._policy_to_commands(mapped)))
        report['commands'].append({'actor_raw':raw,'trainer_action':mapped.tolist(),
            'steer_rad':result[0],'wheel_command':result[1],'command_v':result[2],'command_w':result[3]})
    assert np.array_equal(qpos,env.data.qpos) and np.array_equal(qvel,env.data.qvel) and np.array_equal(ctrl,env.data.ctrl) and np.array_equal(prev,env.prev_action)
    # One real native step at the trainer's minimum raw speed.
    time0 = float(env.data.time)
    _, _, term, trunc, info = N.step_checked(env,np.array([0,0],dtype=np.float32))
    report['one_native_step_at_minimum'] = {
        'elapsed_s':float(env.data.time)-time0,
        'info':{k:float(info[k]) for k in ['speed_ratio','command_v','command_w','wheel','steer'] if k in info},
        'terminated':bool(term),'truncated':bool(trunc)}
    (ROOT/'interface_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
finally:
    if renderer is not None: renderer.close()
    if env is not None: env.close()
