"""Validate frozen cadence, then run one bounded SMDP TD3 engineering pilot."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time

REPO=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=REPO/'stage/runs/t167_smdp'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)
def main():
    lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'pipeline_status.json').exists(),'No duplicate'
    protocol=json.loads((HERE/'protocol.json').read_text())
    for p,h in protocol['source_hashes'].items():assert sha(p)==h,p
    save('pipeline_status.json',{'state':'WAITING_DECLARED_CADENCE_GATE'})
    deadline=time.monotonic()+1200
    while not (OUT/'cadence_eval/summary.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Cadence gate incomplete; no training')
        time.sleep(5)
    gate=json.loads((OUT/'cadence_eval/summary.json').read_text());eligible=[];scores={}
    for hold in [25,10]:
        rows=[r for r in gate['cases'] if r['model']==f'T159_hold{hold}' and r['distance_m']<5.]
        assert len(rows)==6
        scores[str(hold)]={'old_safe_success':sum(r['safe_success'] for r in rows),'old_n':6}
        if all(r['safe_success'] for r in rows):eligible.append(hold)
    save('cadence_gate.json',{'scores':scores,'eligible_holds':eligible,'selection':'Largest preserving all six old tasks'})
    if not eligible:
        save('pipeline_status.json',{'state':'CADENCE_GATE_FAILED_NO_TRAINING'});return
    hold=max(eligible);folder=OUT/'learn';folder.mkdir()
    actor=REPO/'stage/current/outputs/stage_t159/consolidated_actor.pt'
    anchors=REPO/'stage/current/outputs/stage_t159/six_task_success.npz'
    files=[actor,anchors,HERE/'train.py',HERE/'task_bank.json',REPO/'stage/experiments/t165_resume/resume_support.py',
        REPO/'stage/current/outputs/stage_t148/train.py',REPO/'stage/current/outputs/stage_t148/td3_core_goal_contrast.py']
    plan={**protocol['next_training_predeclaration'],'seed':2026103067,'control_hold_physics_steps':hold,
        'physics_budget':256000,'wall_seconds':1200,'actor_parent':str(actor),'retention_states':str(anchors),
        'task_bank':json.loads((HERE/'task_bank.json').read_text())['tasks'],
        'input_hashes':{str(p):sha(p) for p in files},'critic_and_replay':'fresh SMDP semantics, no old one-step Critic/replay reuse',
        'cap_all_new_runs_bytes':1024**3,'automatic_promotion':False}
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ);env['STAGE_RUN_DIR']=str(folder)
    with (folder/'train.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([str(REPO/'run_stage.sh'),'-u',str(HERE/'train.py')],cwd=REPO,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save('pipeline_status.json',{'state':'TRAINING','train_pid':p.pid,'control_hold_steps':hold});rc=p.wait()
    if rc:
        save('pipeline_status.json',{'state':'TRAINING_FAILED','exit_code':rc});return
    status=json.loads((folder/'status.json').read_text())
    if status['state']!='COMPLETED':save('pipeline_status.json',{'state':'BUDGET_NOT_COMPLETED','training_status':status});return
    assert sha(status['checkpoint'])==status['checkpoint_sha256']
    evaluation=OUT/'final_eval';evaluation.mkdir()
    plan=json.loads((HERE/'evaluation_declaration.json').read_text())
    plan['models']={'T159_same_cadence':{'checkpoint':str(actor),'sha256':sha(actor),'control_hold_physics_steps':hold},
        'T167_final':{'checkpoint':status['checkpoint'],'sha256':status['checkpoint_sha256'],'control_hold_physics_steps':hold}}
    (evaluation/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save('pipeline_status.json',{'state':'FROZEN_EVALUATION','control_hold_steps':hold,'models':plan['models']})
    with (evaluation/'run.log').open('ab',buffering=0) as log:
        rc=subprocess.run([str(REPO/'run_stage.sh'),'-u',str(HERE/'fixed_goal_eval.py'),
            '--plan',str(evaluation/'plan.json')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:save('pipeline_status.json',{'state':'EVALUATION_FAILED','exit_code':rc});return
    summary=json.loads((evaluation/'summary.json').read_text());scores={}
    for name in plan['models']:
        rows=[r for r in summary['cases'] if r['model']==name]
        scores[name]={}
        for group in sorted({r['group'] for r in rows}):
            subset=[r for r in rows if r['group']==group]
            scores[name][group]={'safe_success':sum(r['safe_success'] for r in subset),'n':len(subset),'collisions':sum(r['collision'] for r in subset)}
    save('acceptance.json',{'scores':scores,'automatic_promotion':False,'generalization_claim':False,
        'study':'Single-seed multi-change engineering pilot; fixed-course outcomes do not establish novel method effectiveness'})
    previous=folder/'checkpoint_previous.pt'
    if previous.exists():
        save('discarded_predecessor.json',{'path':str(previous),'sha256':sha(previous),'bytes':previous.stat().st_size,
            'reason':'owned rolling predecessor, not final-budget model'});previous.unlink()
    save('pipeline_status.json',{'state':'COMPLETE','automatic_promotion':False})

if __name__=='__main__':main()
