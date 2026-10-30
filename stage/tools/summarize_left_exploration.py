"""Verify true off-policy actions, fixed parameter draws and bounded T196 counts."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys,ast,re,json,hashlib,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'stage/experiments/t196_left_exploration';OUT=ROOT/'stage/runs/t196_left_exploration';DOC=ROOT/'stage/docs/results_20261030_left_exploration'
sys.path.insert(0,str(SRC))
from common import Policy
from collect import fold
from replay import build
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    plan=read(SRC/'protocol.json');summary=read(OUT/'summary.json');assert summary['state']==read(OUT/'status.json')['state']=='COMPLETE'
    assert sha(SRC/'protocol.json')==sha(OUT/'protocol.json')==read(OUT/'launch.json')['protocol_sha256']
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    for p in SRC.glob('*.py'):ast.parse(p.read_text())
    process=read(OUT/'workers.json')
    for pid in [process['parent']]+process['pids']+[read(OUT/'diagnostic_launch.json')['pid']]:assert not Path('/proc',str(pid)).exists(),pid
    diagnostic=read(OUT/'diagnostic.json');assert diagnostic['state']=='PASS' and diagnostic['physics_max_error']==diagnostic['observation_max_error']==0
    assert sha(diagnostic['snapshot_path'])==diagnostic['snapshot_sha256']
    assert any(r['other_geom']=='real_circle_obs_9' and r['robot_geom']=='chassis_collision' for r in diagnostic['contacts'])
    training=[json.loads(s) for s in (OUT/'episodes.jsonl').read_text().splitlines()];evaluation=[json.loads(s) for s in (OUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(training)==96 and len(evaluation)==48
    with np.load(plan['parameter_basis']) as z:basis={k:z[k].copy() for k in z.files}
    cache={};max_error=0.;metrics={'state':'PASS','protocol_sha256':sha(SRC/'protocol.json'),'inputs_verified':len(plan['input_hashes']),'training':{},'automatic_promotion':False,'incumbent':'T159'}
    for row in training:
        assert sha(row['path'])==row['sha256'] and sha(row['behavior_checkpoint'])==row['behavior_sha256']
        with np.load(row['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        assert len(a['states'])==row['decisions'] and int(a['native_counts'].sum())==row['physics_steps']
        assert all(np.isfinite(a[k]).all() for k in a)
        a['end_flags']=a['end_flags'].astype(float);transitions=build(a);assert np.isfinite(transitions['rewards']).all()
        path=row['behavior_checkpoint']
        if path not in cache:
            with np.load(path,allow_pickle=False) as z:cache[path]={k:torch.tensor(z[k]) for k in z.files}
        policy=Policy().eval().requires_grad_(False);policy.load_state_dict(cache[path])
        weight=np.zeros(600,dtype=np.float32);bias=np.float32(0)
        if row['parameter_noise']:
            coeff,weight,bias=fold(policy,basis,row['parameter_seed'],plan['parameter_scale']);assert np.array_equal(coeff,np.asarray(row['parameter_coefficients']))
        else:assert row['arm']=='action_noise' and np.count_nonzero(row['parameter_coefficients'])==0
        assert hashlib.sha256(weight.tobytes()+bias.tobytes()).hexdigest()==row['delta_weight_bias_sha256']
        batch,index=map(int,re.search(r'_b(\d+)_w(\d+)$',row['id']).groups());seed=plan['seed']+batch*100+index
        with torch.no_grad():mu=torch.cat([policy.latent_mean(x) for x in torch.tensor(a['states']).split(64)]).numpy();sigma=policy.logstd.clamp(-3,0).exp().numpy()
        expected=np.tanh(mu+sigma*np.random.default_rng(seed).normal(size=mu.shape)).astype(np.float32)
        error=float(np.abs(expected-a['actions']).max());max_error=max(max_error,error);assert error<2e-5,(row['id'],error)
    for arm,stats in summary['training'].items():
        rr=[r for r in training if r['arm']==arm];assert len(rr)==48 and sum(r['snapshot_side'] is None for r in rr)==36
        assert sum(r['physics_steps'] for r in rr)==stats['physics_steps']<=plan['max_physics_per_arm']
        assert sum(r['decisions'] for r in rr)==stats['decisions']
        assert stats['actor_updates']==3000 and stats['critic_updates']==4500
        assert sha(stats['checkpoint'])==stats['sha256'];ck=torch.load(stats['checkpoint'],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==sha(SRC/'protocol.json')
        assert not (OUT/arm/'checkpoint_previous.pt').exists()
        metrics['training'][arm]=stats
    assert sum(r['physics_steps'] for r in evaluation)==summary['evaluation_physics']
    previous=read(ROOT/'stage/manifests/success_consolidation_20261030_files.json')
    for row in previous['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
    metrics.update(max_reconstructed_behavior_action_error=max_error,previous_archived_files_unchanged=len(previous['files']),new_training_physics=sum(r['physics_steps'] for r in training),new_training_decisions=sum(r['decisions'] for r in training),actor_updates=sum(r['actor_updates'] for r in summary['training'].values()),critic_updates=sum(r['critic_updates'] for r in summary['training'].values()),diagnostic_physics=diagnostic['diagnostic_physics'],frozen_evaluation_physics=summary['evaluation_physics'],scores=summary['scores'],own_run_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),all_new_runs_bytes=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file()))
    assert metrics['own_run_bytes']<plan['max_bytes'] and metrics['all_new_runs_bytes']<1024**3
    write(DOC/'t196_summary.json',summary);write(DOC/'diagnostic.json',diagnostic);write(DOC/'verification_and_metrics.json',metrics);print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
