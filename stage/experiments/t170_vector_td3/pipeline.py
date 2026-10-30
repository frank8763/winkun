from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess

REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=REPO/'stage/runs/t170_vector_td3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
    p=OUT/n;tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def main():
    lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'pipeline_status.json').exists()
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    protocol=json.loads((HERE/'protocol.json').read_text())
    for p,h in protocol['input_hashes'].items():assert sha(p)==h,p
    with (OUT/'train.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([str(REPO/'run_stage.sh'),'-u',str(HERE/'train.py')],cwd=REPO,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save('pipeline_status.json',{'state':'TRAINING','train_pid':process.pid,'n_envs':8,'device':'cuda'});rc=process.wait()
    if rc:save('pipeline_status.json',{'state':'TRAINING_FAILED','exit_code':rc});return
    status=json.loads((OUT/'status.json').read_text())
    assert sha(status['checkpoint'])==status['checkpoint_sha256']
    plan=json.loads((HERE/'evaluation_declaration.json').read_text())
    plan['models']['T170_final']={'checkpoint':status['checkpoint'],'sha256':status['checkpoint_sha256'],'control_hold_physics_steps':25}
    plan['actual_new_training_frames']=status['physics_frames'];plan['training_budget_completed']=status['state']=='COMPLETED'
    out=OUT/'frozen_eval';out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save('pipeline_status.json',{'state':'FROZEN_EVALUATION','training_state':status['state'],'models':plan['models']})
    with (out/'run.log').open('ab',buffering=0) as log:
        rc=subprocess.run([str(REPO/'run_stage.sh'),'-u',str(REPO/'stage/experiments/t167_smdp/cadence_eval.py'),
            '--plan',str(out/'plan.json')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:save('pipeline_status.json',{'state':'EVALUATION_FAILED','exit_code':rc});return
    results=json.loads((out/'summary.json').read_text());scores={}
    for name in plan['models']:
        scores[name]={}
        for group in sorted({r['group'] for r in results['cases']}):
            rows=[r for r in results['cases'] if r['model']==name and r['group']==group]
            scores[name][group]={'safe_success':sum(r['safe_success'] for r in rows),'n':len(rows),'collisions':sum(r['collision'] for r in rows)}
    save('acceptance.json',{'scores':scores,'actual_new_physics_frames':status['physics_frames'],
        'planned_physics_frames':1000000,'training_end_state':status['state'],'automatic_promotion':False,
        'scope':'One training run, one geometric layout; prospective cases are same-map goal/start tests, not cross-map or statistical superiority evidence.'})
    previous=OUT/'checkpoint_previous.pt'
    if previous.exists():
        save('discarded_predecessor.json',{'path':str(previous),'bytes':previous.stat().st_size,'sha256':sha(previous),'reason':'owned rolling predecessor, final stopping-rule checkpoint retained'});previous.unlink()
    save('pipeline_status.json',{'state':'COMPLETE','training_end_state':status['state'],'automatic_promotion':False})
if __name__=='__main__':main()
