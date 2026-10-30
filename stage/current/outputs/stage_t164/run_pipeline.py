"""One bounded two-arm Stage pilot; no restart and no automatic promotion."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PY='/home/linghu/miniconda3/envs/mk_mujoco/bin/python'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(obj):
    p=ROOT/'pipeline_status.json';tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)

def main():
    lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'pipeline_status.json').exists(), 'Do not duplicate/restart'
    master=json.loads((ROOT/'plan.json').read_text())
    env=dict(os.environ);env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
        'MUJOCO_GL':'egl','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PYTHONPATH':'/home/linghu/catkin_ws/src/mk-mini'})
    procs={};logs=[]
    for name,arm in master['arms'].items():
        d=Path(arm['directory']);assert sha(d/'plan.json')==arm['plan_sha256']
        assert not (d/'status.json').exists()
        log=(d/'train.log').open('ab',buffering=0);logs.append(log)
        procs[name]=subprocess.Popen([PY,'-u',str(d/'train_curriculum.py')],env=env,
            cwd='/home/linghu/catkin_ws/src/mk-mini',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    save({'state':'TRAINING','train_pids':{k:v.pid for k,v in procs.items()},'started_unix':time.time()})
    results={name:p.wait() for name,p in procs.items()}
    for log in logs:log.close()
    if any(results.values()):
        save({'state':'TRAINING_FAILED','exit_codes':results});return
    out=ROOT/'frozen_eval';plan=json.loads((out/'declaration.json').read_text())
    for name,arm in master['arms'].items():
        s=json.loads((Path(arm['directory'])/'status.json').read_text())
        if s['state']!='COMPLETED' or s['new_physics_frames']!=master['new_frames_per_arm']:
            save({'state':'BUDGET_NOT_COMPLETED','arm':name,'status':s});return
        assert sha(s['checkpoint'])==s['checkpoint_sha256']
        plan['models']['T164_'+name]={'checkpoint':s['checkpoint'],'sha256':s['checkpoint_sha256']}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save({'state':'FROZEN_EVALUATION','models':plan['models']})
    with (out/'run.log').open('ab',buffering=0) as log:
        result=subprocess.run([PY,'-u',str(ROOT.parent/'stage_t153/distance_eval.py'),'--plan',str(out/'plan.json')],
            env=env,cwd='/home/linghu/catkin_ws/src/mk-mini',stdout=log,stderr=subprocess.STDOUT)
    save({'state':'COMPLETE' if result.returncode==0 else 'EVALUATION_FAILED','exit_code':result.returncode,
        'automatic_promotion':False})

if __name__=='__main__': main()
