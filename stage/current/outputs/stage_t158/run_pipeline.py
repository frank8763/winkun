from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parent
PY='/home/linghu/miniconda3/envs/mk_mujoco/bin/python'

def save(obj):
    p=ROOT/'pipeline_status.json';tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)

def main():
    lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ);env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
        'MUJOCO_GL':'egl','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PYTHONPATH':'/home/linghu/catkin_ws/src/mk-mini'})
    save({'state':'TRAINING'})
    with (ROOT/'train.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([PY,'-u',str(ROOT/'train_curriculum.py')],env=env,
            cwd='/home/linghu/catkin_ws/src/mk-mini',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save({'state':'TRAINING','train_pid':p.pid});rc=p.wait()
    if rc!=0:
        save({'state':'TRAINING_FAILED','exit_code':rc});return
    status=json.loads((ROOT/'status.json').read_text())
    if status['state']!='COMPLETED':
        save({'state':'STOPPED_BUDGET_NOT_COMPLETED','training_status':status});return
    ck=Path(status['checkpoint']);sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    assert sha==status['checkpoint_sha256']
    out=ROOT/'frozen_eval'
    plan=json.loads((out/'declaration.json').read_text())
    plan['models']['T158_final']={'checkpoint':str(ck),'sha256':sha}
    plan['new_training_frames']=status['new_physics_frames']
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save({'state':'FROZEN_EVALUATION','checkpoint':str(ck)})
    with (out/'run.log').open('ab',buffering=0) as log:
        r=subprocess.run([PY,'-u',str(ROOT.parent/'stage_t153/distance_eval.py'),
            '--plan',str(out/'plan.json')],env=env,cwd='/home/linghu/catkin_ws/src/mk-mini',stdout=log,stderr=subprocess.STDOUT)
    save({'state':'COMPLETE' if r.returncode==0 else 'FROZEN_EVALUATION_FAILED','exit_code':r.returncode})

if __name__=='__main__':main()
