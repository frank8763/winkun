"""A small declared search over actual neural weights; frozen native evaluation."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import fcntl
import torch

REPO=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=REPO/'stage/runs/t166_steering_probe'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def main():
    lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate'
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert sha(protocol['parent_checkpoint'])==protocol['parent_sha256']
    parent=torch.load(protocol['parent_checkpoint'],map_location='cpu',weights_only=False)
    base={k:v.clone() for k,v in parent['actor'].items()}
    models={'T159_parent':{'checkpoint':protocol['parent_checkpoint'],'sha256':protocol['parent_sha256']}}
    audit=[]
    for delta in protocol['steering_bias_deltas']:
        actor={k:v.clone() for k,v in base.items()};actor['net.4.bias'][1]+=delta
        changed=[k for k,v in actor.items() if not torch.equal(v,base[k])]
        assert changed==['net.4.bias'] and torch.equal(actor['net.4.bias'][:1],base['net.4.bias'][:1])
        name='steer_'+('plus' if delta>0 else 'minus')+str(abs(delta)).replace('.','p')
        path=OUT/(name+'.pt')
        torch.save({'actor':actor,'steering_bias_delta':delta,'parent_sha256':protocol['parent_sha256'],
            't148_resume_compatible':False,'study':'declared neural parameter probe'},path)
        models[name]={'checkpoint':str(path),'sha256':sha(path)}
        audit.append({'model':name,'delta':delta,'changed_keys':changed,'throttle_parameter_unchanged':True})
    save('parameter_audit.json',audit)
    plan=json.loads((REPO/'stage/current/outputs/stage_t164/frozen_eval/declaration.json').read_text())
    plan.update(experiment=protocol['experiment'],models=models,
        cases=[c for c in plan['cases'] if c['distance_m']==5.],wall_limit_seconds=600,
        selection_rule=protocol['acceptance'],purpose=protocol['training_type'])
    (OUT/'frozen_eval').mkdir();save('frozen_eval/plan.json',plan);save('status.json',{'state':'FROZEN_PARAMETER_SCREEN'})
    with (OUT/'frozen_eval/run.log').open('ab',buffering=0) as log:
        rc=subprocess.run([str(REPO/'run_stage.sh'),'-u',str(REPO/'stage/current/outputs/stage_t153/distance_eval.py'),
            '--plan',str(OUT/'frozen_eval/plan.json')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0
    summary=json.loads((OUT/'frozen_eval/summary.json').read_text());scores={}
    for name in models:
        rows=[r for r in summary['cases'] if r['model']==name]
        scores[name]={'safe_success':sum(r['safe_success'] for r in rows),'n':len(rows),
            'collisions':sum(r['collision'] for r in rows)}
    eligible=[a for a in audit if scores[a['model']]['safe_success']==2]
    eligible.sort(key=lambda a:(abs(a['delta']),a['delta']))
    save('decision.json',{'scores':scores,'eligible_for_old_task_regression':eligible[:1],
        'automatic_promotion':False,'generalization_claim':False,'all_search_trials_reported':True})
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert used<protocol['disk_cap_bytes']
    save('status.json',{'state':'COMPLETE','run_bytes':used,'disk_cap_bytes':protocol['disk_cap_bytes']})

if __name__=='__main__':main()
