"""Verify the first retained deterministic RIGHT5m development candidate."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import ast,hashlib,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'stage/docs/results_20261030_success_consolidation'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    torch.set_num_threads(1);metrics={'state':'PASS','protocols':{},'automatic_promotion':False,'incumbent':'T159'}
    for name in ['t193_success_consolidation','t194_drive_sensitivity','t195_selected_verification']:
        source=ROOT/'stage/experiments'/name;out=ROOT/'stage/runs'/name;plan=read(source/'protocol.json');summary=read(out/'summary.json')
        assert summary['state']==read(out/'status.json')['state']=='COMPLETE'
        for p,h in plan['input_hashes'].items():assert sha(p)==h,p
        assert sha(source/'protocol.json')==sha(out/'protocol.json')==read(out/'launch.json')['protocol_sha256']
        for p in source.glob('*.py'):ast.parse(p.read_text())
        if (out/'workers.json').exists():
            ps=read(out/'workers.json');pids=[ps['parent']]+ps['pids']
        else:pids=[read(out/'launch.json')['pid']]
        for pid in pids:assert not Path('/proc',str(pid)).exists(),pid
        for row in rows(out/'evaluation.jsonl'):
            filename=(row['case']+'_frozen_trace.npz') if name.startswith('t193') else row['id']+'.npz'
            assert sha(out/filename)==row['trace_sha256']
        metrics['protocols'][name]={'sha256':sha(source/'protocol.json'),'verified_inputs':len(plan['input_hashes'])}
        write(DOC/(name+'_summary.json'),summary)
    run193=ROOT/'stage/runs/t193_success_consolidation';s193=read(run193/'summary.json')
    replay=read(run193/'data_replay_verification.json');assert replay['state']=='PASS'
    assert all(r['safe_goal'] and r['max113_observation_error']==r['max_physics_xy_error']==0 for r in replay['rows'])
    assert s193['supervised_updates']==5000 and s193['rl_updates']==s193['new_training_physics']==0
    assert sha(run193/'final_policy.pt')==s193['checkpoint_sha256']
    run194=ROOT/'stage/runs/t194_drive_sensitivity';s194=read(run194/'summary.json');plan=read(ROOT/'stage/experiments/t194_drive_sensitivity/protocol.json');r194=rows(run194/'evaluation.jsonl');assert len(r194)==32
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];selected=torch.load(s194['selected']['path'],map_location='cpu',weights_only=False)['policy']
    expected={k:v.clone() for k,v in parent.items()};expected['native_head.bias'][0]+=.2
    assert s194['selected']['delta']==.2 and all(torch.equal(selected[k],v) for k,v in expected.items())
    assert sha(s194['selected']['path'])==s194['selected']['sha256']
    run195=ROOT/'stage/runs/t195_selected_verification';s195=read(run195/'summary.json');r195=rows(run195/'evaluation.jsonl');assert len(r195)==34 and s195['primary_reproduction_exact']
    right=next(r for r in r195 if r['group']=='primary' and r['case']=='5m_2029654802_RIGHT');left=next(r for r in r195 if r['group']=='primary' and r['case']=='5m_2029653601_LEFT')
    assert right['success'] and not right['collision'] and left['collision'] and not left['success']
    grid=[right]+[r for r in r195 if r['group']=='right_perturbation'];assert len(grid)==27 and all(r['success'] and not r['collision'] for r in grid)
    for row in [r for r in r195 if r['group']=='primary']:
        old=next(r for r in r194 if r['delta']==.2 and r['case']==row['case'])
        for key in ['trace_sha256','physics_steps','success','collision','timeout','nearest_m']:assert row[key]==old[key]
    previous=read(ROOT/'stage/manifests/her_contract_20261030_files.json')
    for row in previous['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
    metrics.update(previous_archived_files_unchanged=len(previous['files']),right5m_seconds=right['physics_steps']*.002,right5m_remaining_m=right['remaining_m'],right_reset_grid=s195['scores']['right_reset_grid'],left5m_remaining_m=left['remaining_m'],left5m_failure='collision',old6=s195['scores']['old6'],original5m=s195['scores']['original5m'],candidate=s194['selected'],supervised_updates=5000,new_rl_updates=0,new_training_physics=0,data_replay_physics=s193['data_replay_physics'],T193_evaluation_physics=s193['frozen_evaluation_physics'],T194_evaluation_physics=s194['physics_steps'],T195_evaluation_physics=s195['physics_steps'],scope='RIGHT fixed task and local reset development grid only. The same canonical reproduction is not independent success evidence; no generalized navigation, novelty or complete LEFT/RIGHT5m pass.')
    metrics['all_new_runs_bytes']=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert metrics['all_new_runs_bytes']<1024**3
    write(DOC/'verification_and_metrics.json',metrics);print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
