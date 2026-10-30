"""Verify completed Stage HER contract and SAC pilot without new physics."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import ast,hashlib,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'stage/docs/results_20261030_her_contract'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    torch.set_num_threads(1);metrics={'state':'PASS','protocols':{},'automatic_promotion':False,'incumbent':'T159'}
    for name in ['t191_her_contract_audit','t192_sac_her']:
        source=ROOT/'stage/experiments'/name;out=ROOT/'stage/runs'/name
        plan=read(source/'protocol.json');summary=read(out/'summary.json');assert summary['state']==read(out/'status.json')['state']=='COMPLETE'
        for p,h in plan['input_hashes'].items():assert sha(p)==h,p
        assert sha(source/'protocol.json')==sha(out/'protocol.json')==read(out/'launch.json')['protocol_sha256']
        for p in source.glob('*.py'):ast.parse(p.read_text())
        if (out/'workers.json').exists():
            process=read(out/'workers.json');pids=[process['parent']]+process['pids']
        else:pids=[read(out/'launch.json')['pid']]
        for pid in pids:assert not Path('/proc',str(pid)).exists(),pid
        metrics['protocols'][name]={'sha256':sha(source/'protocol.json'),'inputs_verified':len(plan['input_hashes'])}
        write(DOC/(name+'_summary.json'),summary)
    audit=read(ROOT/'stage/runs/t191_her_contract_audit/summary.json');assert audit['optimizer_updates']==0 and audit['actor_unchanged']
    assert audit['native_physics_steps']==sum(r['physics_steps'] for r in audit['cases'])
    assert audit['max_original_observation_rebuild_error']==0
    for row in audit['cases']:assert sha(row['path'])==row['sha256']
    metrics['T191']={k:audit[k] for k in ['native_physics_steps','optimizer_updates','relabel_cases','eligible_relabel_cases','max_original_observation_rebuild_error','interior_hold_first_hits','original5m_safe_success']}
    out=ROOT/'stage/runs/t192_sac_her';summary=read(out/'summary.json');plan=read(ROOT/'stage/experiments/t192_sac_her/protocol.json');training=rows(out/'episodes.jsonl');evaluation=rows(out/'evaluation.jsonl')
    assert len(training)==128 and len(evaluation)==28
    for row in training:
        assert sha(row['path'])==row['sha256']
        with np.load(row['path'],allow_pickle=False) as z:
            assert len(z['states'])==row['decisions'];assert int(z['native_counts'].sum())==row['physics_steps'];assert len(z['physical_xy'])==row['physics_steps'];assert all(np.isfinite(z[k]).all() for k in z.files)
    metrics['T192']={'training':{},'scores':summary['scores'],'evaluation_physics':summary['evaluation_physics']}
    for arm,spec in summary['training'].items():
        assert sha(spec['checkpoint'])==spec['sha256']
        ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==sha(ROOT/'stage/experiments/t192_sac_her/protocol.json')
        rr=[r for r in training if r['arm']==arm];assert len(rr)==64 and sum(r['physics_steps'] for r in rr)==spec['physics_steps'] and sum(r['decisions'] for r in rr)==spec['decisions']
        assert spec['actor_updates']==4000 and spec['critic_updates']==5500
        assert not (out/arm/'checkpoint_previous.pt').exists()
        for receipt in read(out/arm/'replay_receipts.json'):assert sha(receipt['path'])==receipt['sha256']
        if arm=='no_her':assert spec['her_replay_transitions']==0
        else:assert spec['her_replay_transitions']>0
        metrics['T192']['training'][arm]=spec
    assert sum(r['physics_steps'] for r in evaluation)==summary['evaluation_physics']
    # Same parent and noise stream before Actor updates in either arm.
    a=[r for r in training if r['arm']=='no_her'][:8];b=[r for r in training if r['arm']=='her'][:8]
    for x,y in zip(a,b):
        assert x['physics_steps']==y['physics_steps'] and x['success']==y['success'] and x['collision']==y['collision']
        with np.load(x['path']) as p,np.load(y['path']) as q:
            assert all(np.array_equal(p[k],q[k]) for k in p.files),x['id']
    metrics['initial_training_batch_exactly_matched']=True
    previous=read(ROOT/'stage/manifests/head_curriculum_20261030_files.json')
    for row in previous['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
    metrics['previous_archived_files_unchanged']=len(previous['files'])
    metrics['new_online_training_physics']=sum(r['physics_steps'] for r in training)
    metrics['new_training_decisions']=sum(r['decisions'] for r in training)
    metrics['actor_updates']=sum(r['actor_updates'] for r in summary['training'].values());metrics['critic_updates']=sum(r['critic_updates'] for r in summary['training'].values())
    metrics['all_new_runs_bytes']=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert metrics['all_new_runs_bytes']<1024**3
    write(DOC/'verification_and_metrics.json',metrics);print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
