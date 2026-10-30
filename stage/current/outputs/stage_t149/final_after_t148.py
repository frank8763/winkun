"""Run the predeclared final-budget comparison once the existing T148 pipeline exits."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / 'stage_t148'
OUT = ROOT / 'final_paired'

def save(name, obj):
    path = OUT / name
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(tmp,path)

def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None

def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'waiter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():
        return
    declaration = json.loads((OUT/'declaration.json').read_text())
    pid = declaration['upstream_pid']
    save('queue_status.json',{'state':'WAITING_EXISTING_T148_PIPELINE','pid':pid})
    while identity(pid) == declaration['upstream_start_ticks']:
        time.sleep(10)
    status = json.loads((TRAIN/'status.json').read_text())
    if status.get('collapsed') or status['frames'] < 8000000:
        save('queue_status.json',{'state':'NOT_RUN_TARGET_NOT_MET_OR_COLLAPSED','upstream_status':status,
            'note':'Do not silently select a partial or best-looking checkpoint.'})
        return
    ck = TRAIN/'checkpoints'/status['checkpoint']
    actual = hashlib.sha256(ck.read_bytes()).hexdigest()
    assert actual == status['checkpoint_sha256']
    plan = declaration['plan_template']
    plan['models']['T148_final']={'checkpoint':str(ck),'sha256':actual}
    plan['final_frames']=status['frames']
    plan['final_checkpoint_selection']='last checkpoint after predeclared 8M cumulative frame budget'
    save('plan.json',plan)
    save('queue_status.json',{'state':'RUNNING_FINAL_PAIRED_EVAL','checkpoint':str(ck),'sha256':actual})
    env = dict(os.environ)
    env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','MUJOCO_GL':'egl',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PYTHONPATH':'/home/linghu/catkin_ws/src/mk-mini'})
    with (OUT/'run.log').open('ab',buffering=0) as log:
        r = subprocess.run(['/home/linghu/miniconda3/envs/mk_mujoco/bin/python','-u',
            str(ROOT/'paired_eval.py'),'--plan',str(OUT/'plan.json')],
            env=env,cwd='/home/linghu/catkin_ws/src/mk-mini',stdout=log,stderr=subprocess.STDOUT)
    save('queue_status.json',{'state':'COMPLETE' if r.returncode==0 else 'FAILED',
         'exit_code':r.returncode,'checkpoint':str(ck)})

if __name__=='__main__':
    main()
