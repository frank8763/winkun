"""Bounded continuation of the two saved T164 arms; final frozen paired screen."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'stage/runs/t165_resume'
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)

def main():
    lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'pipeline_status.json').exists(),'No duplicate/restart'
    protocol=json.loads((HERE/'protocol.json').read_text())
    for path,expected in protocol['source_hashes'].items():assert digest(path)==expected,path
    processes={};logs=[]
    for arm in protocol['arms']:
        folder=OUT/arm;assert not (folder/'status.json').exists()
        env=dict(os.environ);env['STAGE_RUN_DIR']=str(folder)
        log=(folder/'train.log').open('ab',buffering=0);logs.append(log)
        processes[arm]=subprocess.Popen([str(REPO/'run_stage.sh'),'-u',str(HERE/'train.py')],
            cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    save('pipeline_status.json',{'state':'TRAINING','pids':{k:p.pid for k,p in processes.items()},'started_unix':time.time()})
    exits={arm:p.wait() for arm,p in processes.items()}
    for log in logs:log.close()
    if any(exits.values()):save('pipeline_status.json',{'state':'TRAINING_FAILED','exits':exits});return
    plan=json.loads((HERE/'evaluation_declaration.json').read_text())
    for arm in protocol['arms']:
        s=json.loads((OUT/arm/'status.json').read_text())
        if s['state']!='COMPLETED' or s['new_physics_frames']!=256000:
            save('pipeline_status.json',{'state':'BUDGET_NOT_COMPLETED','arm':arm,'status':s});return
        assert digest(s['checkpoint'])==s['checkpoint_sha256']
        plan['models']['T165_'+arm]={'checkpoint':s['checkpoint'],'sha256':s['checkpoint_sha256']}
    evaluation=OUT/'frozen_eval';evaluation.mkdir()
    (evaluation/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save('pipeline_status.json',{'state':'FROZEN_EVALUATION','models':plan['models']})
    with (evaluation/'run.log').open('ab',buffering=0) as log:
        rc=subprocess.run([str(REPO/'run_stage.sh'),'-u',str(REPO/'stage/current/outputs/stage_t153/distance_eval.py'),
            '--plan',str(evaluation/'plan.json')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:save('pipeline_status.json',{'state':'EVALUATION_FAILED','exit_code':rc});return
    results=json.loads((evaluation/'summary.json').read_text())
    scores={}
    for name in plan['models']:
        rows=[r for r in results['cases'] if r['model']==name]
        old=[r for r in rows if r['distance_m']<5];new=[r for r in rows if r['distance_m']==5]
        scores[name]={'old_safe_success':sum(r['safe_success'] for r in old),'old_n':len(old),
            '5m_safe_success':sum(r['safe_success'] for r in new),'5m_n':len(new),
            '5m_collisions':sum(r['collision'] for r in new),'eligible_fixed_course':all(r['safe_success'] for r in rows)}
    save('acceptance.json',{'scores':scores,'automatic_promotion':False,
        'selection_rule':'No promotion without all six old tasks and both 5m tasks passing; review all outcomes. Single training history, no superiority or generalization claim.'})
    removed=[]
    for arm in protocol['arms']:
        previous=OUT/arm/'checkpoint_previous.pt'
        if previous.exists():
            removed.append({'path':str(previous),'sha256':digest(previous),'bytes':previous.stat().st_size,
                'reason':'Owned rolling predecessor, not selected by fixed-final-budget protocol'})
            previous.unlink()
    save('storage_receipt.json',{'discarded_owned_predecessors':removed,'retained_final_checkpoints':2,
        'run_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'cap_bytes':1024**3})
    save('pipeline_status.json',{'state':'COMPLETE','exits':exits,'automatic_promotion':False})

if __name__=='__main__':main()
