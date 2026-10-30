"""Verify matched residual placement, true behavior actions and compact models."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys,re,ast,hashlib,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'stage/experiments/t200_residual_placement';OUT=ROOT/'stage/runs/t200_residual_placement';DOC=ROOT/'stage/docs/results_20261030_sac_gradient'
sys.path.insert(0,str(SRC))
from common import Policy,ParentPolicy
from replay import build
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    metrics={'state':'PASS','protocols':{},'automatic_promotion':False,'incumbent':'T159'}
    for name in ['t199_sac_mean_gradient','t200_residual_placement']:
        source=ROOT/'stage/experiments'/name;out=ROOT/'stage/runs'/name;plan=read(source/'protocol.json');summary=read(out/'summary.json')
        assert summary['state']==read(out/'status.json')['state']=='COMPLETE'
        assert sha(source/'protocol.json')==sha(out/'protocol.json')==read(out/'launch.json')['protocol_sha256']
        for p,h in plan['input_hashes'].items():assert sha(p)==h,p
        for p in source.glob('*.py'):ast.parse(p.read_text())
        if (out/'workers.json').exists():
            ps=read(out/'workers.json');pids=[ps['parent']]+ps['pids']
        else:pids=[read(out/'launch.json')['pid']]
        for pid in pids:assert not Path('/proc',str(pid)).exists(),pid
        metrics['protocols'][name]={'sha256':sha(source/'protocol.json'),'inputs_verified':len(plan['input_hashes'])}
        write(DOC/(name+'_summary.json'),summary)
    audit=read(ROOT/'stage/runs/t199_sac_mean_gradient/summary.json');assert audit['native_physics_steps']==audit['optimizer_updates']==0 and max(audit['initial_function_errors'].values())==0
    assert sha(ROOT/'stage/runs/t199_sac_mean_gradient/initial_residual_policy.pt')==audit['initial_checkpoint_sha256']
    plan=read(SRC/'protocol.json');summary=read(OUT/'summary.json');training=[json.loads(s) for s in (OUT/'episodes.jsonl').read_text().splitlines()];evaluation=[json.loads(s) for s in (OUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(training)==64 and len(evaluation)==48
    cache={};max_error=0.
    for row in training:
        assert row['placement']==row['arm'] and sha(row['path'])==row['sha256'] and sha(row['behavior_checkpoint'])==row['behavior_sha256']
        with np.load(row['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        assert all(np.isfinite(a[k]).all() for k in a);assert len(a['states'])==row['decisions'] and int(a['native_counts'].sum())==row['physics_steps']
        a['end_flags']=a['end_flags'].astype(float);build(a)
        path=row['behavior_checkpoint']
        if path not in cache:
            with np.load(path,allow_pickle=False) as z:cache[path]={k:torch.tensor(z[k]) for k in z.files}
        p=Policy(row['placement']).eval().requires_grad_(False);p.load_state_dict(cache[path]);batch,index=map(int,re.search(r'_b(\d+)_w(\d+)$',row['id']).groups());seed=plan['seed']+batch*100+index
        with torch.no_grad():mu=torch.cat([p.latent_mean(x) for x in torch.tensor(a['states']).split(64)]).numpy();sigma=p.logstd.clamp(-3,0).exp().numpy()
        expected=np.tanh(mu+sigma*np.random.default_rng(seed).normal(size=mu.shape)).astype(np.float32);error=float(np.abs(expected-a['actions']).max());max_error=max(max_error,error);assert error<2e-5,(row['id'],error)
    first=[r for r in training if r['arm']=='native'][:8];second=[r for r in training if r['arm']=='latent'][:8]
    for a,b in zip(first,second):
        with np.load(a['path']) as x,np.load(b['path']) as y:assert all(np.array_equal(x[k],y[k]) for k in x.files)
    with np.load(ROOT/'stage/runs/t196_left_exploration/left_baseline_trace.npz') as z:common=torch.tensor(z['states'],dtype=torch.float32)
    metrics['training']={}
    for arm,spec in summary['training'].items():
        rr=[r for r in training if r['arm']==arm];assert len(rr)==32 and sum(r['snapshot_side'] is None for r in rr)==24
        assert sum(r['physics_steps'] for r in rr)==spec['physics_steps']<=plan['max_physics_per_arm'] and sum(r['decisions'] for r in rr)==spec['decisions']
        assert spec['actor_updates']==2000 and spec['critic_updates']==3500 and sha(spec['checkpoint'])==spec['sha256']
        ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==sha(SRC/'protocol.json') and not ck['complete_optimizer_resume_supported']
        assert 'actor_optimizer' not in ck and 'critic_optimizer' not in ck
        receipt=read(OUT/arm/'optimizer_cleanup_receipt.json');assert receipt['all_actor_and_twin_critic_weights_exact'] and receipt['compact_sha256']==spec['sha256']
        assert not (OUT/arm/'checkpoint_previous.pt').exists()
        p=Policy(arm).eval();p.load_state_dict(ck['policy'])
        with torch.no_grad():
            h=p.body.net[:4](common);base=ParentPolicy.forward(p,common);clip_input=base+p.extra_head(h) if arm=='native' else base
            fractions=(clip_input.abs()>=.999999).float().mean(0).tolist();action_range=[p(common).min(0).values.tolist(),p(common).max(0).values.tolist()]
        metrics['training'][arm]={**spec,'commonstate_base_clip_fractions':fractions,'commonstate_native_mean_range':action_range,'extra_mean_path_available_even_if_base_clipped':arm=='latent'}
    assert sum(r['physics_steps'] for r in evaluation)==summary['evaluation_physics']
    previous=read(ROOT/'stage/manifests/timeout_progress_20261030_files.json')
    for row in previous['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
    metrics.update(first_behavior_batch_exactly_equal=True,max_reconstructed_action_error=max_error,previous_archived_files_unchanged=len(previous['files']),new_training_physics=sum(r['physics_steps'] for r in training),new_training_decisions=sum(r['decisions'] for r in training),actor_updates=sum(r['actor_updates'] for r in summary['training'].values()),critic_updates=sum(r['critic_updates'] for r in summary['training'].values()),frozen_evaluation_physics=summary['evaluation_physics'],scores=summary['scores'],own_final_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),all_new_runs_bytes=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file()))
    assert metrics['own_final_bytes']<plan['max_bytes'] and metrics['all_new_runs_bytes']<1024**3
    write(DOC/'verification_and_metrics.json',metrics);print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
