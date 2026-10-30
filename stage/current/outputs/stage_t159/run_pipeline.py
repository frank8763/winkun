from pathlib import Path
import fcntl
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
    env=dict(os.environ);env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','MUJOCO_GL':'egl',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PYTHONPATH':'/home/linghu/catkin_ws/src/mk-mini'})
    save({'state':'COLLECTION_AND_DISTILLATION'})
    with (ROOT/'train.log').open('ab',buffering=0) as f:
        p=subprocess.Popen([PY,'-u',str(ROOT/'consolidate.py')],env=env,cwd='/home/linghu/catkin_ws/src/mk-mini',stdout=f,stderr=subprocess.STDOUT)
        save({'state':'COLLECTION_AND_DISTILLATION','worker_pid':p.pid});rc=p.wait()
    if rc!=0:save({'state':'TRAINING_FAILED','exit_code':rc});return
    status=json.loads((ROOT/'status.json').read_text());out=ROOT/'frozen_eval'
    plan=json.loads((out/'declaration.json').read_text())
    plan['models']['T159_student']={'checkpoint':status['checkpoint'],'sha256':status['checkpoint_sha256']}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save({'state':'FROZEN_EVALUATION'})
    with (out/'run.log').open('ab',buffering=0) as f:
        r=subprocess.run([PY,'-u',str(ROOT.parent/'stage_t153/distance_eval.py'),'--plan',str(out/'plan.json')],
            env=env,cwd='/home/linghu/catkin_ws/src/mk-mini',stdout=f,stderr=subprocess.STDOUT)
    save({'state':'COMPLETE' if r.returncode==0 else 'FROZEN_EVALUATION_FAILED','exit_code':r.returncode})

if __name__=='__main__':main()
