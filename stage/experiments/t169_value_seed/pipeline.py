from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess

REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=REPO/'stage/runs/t169_value_seed'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
    p=OUT/n;tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def main():
    lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'pipeline_status.json').exists()
    protocol=json.loads((HERE/'protocol.json').read_text())
    for p,h in protocol['source_hashes'].items():assert sha(p)==h,p
    verified=json.loads((OUT/'demonstration_verification.json').read_text());assert verified['state']=='PASS' and verified['safe_success']==6
    demo=OUT/'demonstrations.npz';assert sha(demo)==verified['sha256']
    folder=OUT/'learn';folder.mkdir()
    actor=REPO/'stage/current/outputs/stage_t159/consolidated_actor.pt'
    paths=[actor,demo,HERE/'train.py',HERE/'value_pretrain.py',HERE/'reward_contract.py',REPO/'stage/experiments/t165_resume/resume_support.py',REPO/'stage/current/outputs/stage_t148/train.py',REPO/'stage/current/outputs/stage_t148/td3_core_goal_contrast.py']
    plan={'seed':2026103069,'control_hold_physics_steps':25,'discount_efold_seconds':60.,'physics_budget':128000,'wall_seconds':900,
        'actor_parent':str(actor),'retention_states':str(demo),'demonstration_file':str(demo),
        'task_bank':json.loads((REPO/'stage/experiments/t167_smdp/task_bank.json').read_text())['tasks'],
        'input_hashes':{str(p):sha(p) for p in paths},'protocol_sha256':sha(HERE/'protocol.json')}
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ);env['STAGE_RUN_DIR']=str(folder)
    with (folder/'train.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([str(REPO/'run_stage.sh'),'-u',str(HERE/'train.py')],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save('pipeline_status.json',{'state':'VALUE_PREFIT_AND_ONLINE_TRAINING','pid':p.pid});rc=p.wait()
    if rc:save('pipeline_status.json',{'state':'TRAINING_FAILED','exit_code':rc});return
    status=json.loads((folder/'status.json').read_text())
    if status['state']!='COMPLETED':save('pipeline_status.json',{'state':'BUDGET_NOT_COMPLETED','status':status});return
    assert sha(status['checkpoint'])==status['checkpoint_sha256']
    plan=json.loads((HERE/'evaluation_declaration.json').read_text())
    plan['models']={'T159_same_cadence':{'checkpoint':str(actor),'sha256':sha(actor),'control_hold_physics_steps':25},
        'T169_final':{'checkpoint':status['checkpoint'],'sha256':status['checkpoint_sha256'],'control_hold_physics_steps':25}}
    evaluation=OUT/'frozen_eval';evaluation.mkdir();(evaluation/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save('pipeline_status.json',{'state':'FROZEN_EVALUATION','models':plan['models']})
    with (evaluation/'run.log').open('ab',buffering=0) as log:
        rc=subprocess.run([str(REPO/'run_stage.sh'),'-u',str(REPO/'stage/experiments/t167_smdp/cadence_eval.py'),
            '--plan',str(evaluation/'plan.json')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:save('pipeline_status.json',{'state':'EVALUATION_FAILED','exit_code':rc});return
    summary=json.loads((evaluation/'summary.json').read_text());scores={}
    for name in plan['models']:
        scores[name]={}
        for group in ['old_3m_4m_retention','5m_primary_46s','5m_secondary_93s']:
            rows=[r for r in summary['cases'] if r['model']==name and r['group']==group]
            scores[name][group]={'safe_success':sum(r['safe_success'] for r in rows),'n':len(rows),'collisions':sum(r['collision'] for r in rows)}
    save('acceptance.json',{'scores':scores,'automatic_promotion':False,'generalization_claim':False,
        'secondary_limit':'93s safe success is reported separately and does not pass46.668s primary acceptance',
        'novelty':'Reward/discount/demonstrations/value calibration/retention changed together; engineering pilot only, controlled ablations and repeated seeds still needed.'})
    previous=folder/'checkpoint_previous.pt'
    if previous.exists():
        save('discarded_predecessor.json',{'path':str(previous),'bytes':previous.stat().st_size,'sha256':sha(previous),'reason':'owned rolling predecessor, final-budget checkpoint retained'});previous.unlink()
    save('pipeline_status.json',{'state':'COMPLETE','automatic_promotion':False})
if __name__=='__main__':main()
