from pathlib import Path
import hashlib
import json
import fcntl
import subprocess
import torch

REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=REPO/'stage/runs/t168_distance_ablation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(OUT/n).write_text(json.dumps(d,indent=2)+'\n')
def evaluate(plan,folder):
    folder.mkdir();(folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (folder/'run.log').open('ab',buffering=0) as log:
        rc=subprocess.run([str(REPO/'run_stage.sh'),'-u',str(REPO/'stage/experiments/t167_smdp/cadence_eval.py'),
            '--plan',str(folder/'plan.json')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0
    return json.loads((folder/'summary.json').read_text())
def main():
    lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists()
    protocol=json.loads((HERE/'protocol.json').read_text());assert sha(protocol['parent'])==protocol['parent_sha256']
    torch.set_num_threads(1);parent=torch.load(protocol['parent'],map_location='cpu',weights_only=False)
    models={'parent':{'checkpoint':protocol['parent'],'sha256':protocol['parent_sha256'],'control_hold_physics_steps':25}}
    for scale in protocol['scales']:
        actor={k:v.clone() for k,v in parent['actor'].items()}
        actor['net.0.weight'][:,protocol['columns']]*=scale
        for key,value in actor.items():
            if key!='net.0.weight':assert torch.equal(value,parent['actor'][key])
        other=[i for i in range(112) if i not in protocol['columns']]
        assert torch.equal(actor['net.0.weight'][:,other],parent['actor']['net.0.weight'][:,other])
        name='scale'+str(scale).replace('.','p');p=OUT/(name+'.pt')
        torch.save({'actor':actor,'scale':scale,'parent_sha256':protocol['parent_sha256'],'t148_resume_compatible':False},p)
        models[name]={'checkpoint':str(p),'sha256':sha(p),'control_hold_physics_steps':25}
    plan=json.loads((REPO/'stage/runs/t167_smdp/cadence_eval/plan.json').read_text())
    fullcases=plan['cases'];plan.update(experiment=protocol['experiment'],models=models,
        cases=[c for c in fullcases if c['distance_m']==5.],wall_limit_seconds=600)
    save('status.json',{'state':'FROZEN_ABLATION_SCREEN'})
    summary=evaluate(plan,OUT/'five_meter');scores={}
    for name in models:
        rows=[r for r in summary['cases'] if r['model']==name]
        scores[name]={'safe_success':sum(r['safe_success'] for r in rows),'n':len(rows),'collisions':sum(r['collision'] for r in rows)}
    eligible=[scale for scale in protocol['scales'] if scores['scale'+str(scale).replace('.','p')]['safe_success']==2]
    regression=None
    if eligible:
        scale=max(eligible);name='scale'+str(scale).replace('.','p')
        plan.update(models={name:models[name]},cases=[c for c in fullcases if c['distance_m']<5.])
        regression=evaluate(plan,OUT/'old_regression')
    save('decision.json',{'five_meter_scores':scores,'old_regression_safe_success':sum(r['safe_success'] for r in regression['cases']) if regression else None,
        'eligible_scales':eligible,'automatic_promotion':False,'generalization_claim':False})
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert used<protocol['disk_cap_bytes']
    save('status.json',{'state':'COMPLETE','run_bytes':used})
if __name__=='__main__':main()
