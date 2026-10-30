"""Read-only evidence checks for T186--T188."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import json,hashlib,ast,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'stage/docs/results_20261030_control_robustness'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
metrics={'state':'PASS','protocols':{},'automatic_promotion':False,'incumbent':'T159'}
for name in ['t186_anchor_robustness','t187_local_robustness','t188_direct_action_head']:
 src=ROOT/'stage/experiments'/name;out=ROOT/'stage/runs'/name;plan=read(src/'protocol.json');status=read(out/'status.json');summary=read(out/'summary.json');assert status['state']==summary['state']=='COMPLETE'
 for p,h in plan['input_hashes'].items():assert sha(p)==h,p
 assert read(out/'launch.json')['protocol_sha256']==sha(src/'protocol.json')
 for p in src.glob('*.py'):ast.parse(p.read_text())
 metrics['protocols'][name]={'sha256':sha(src/'protocol.json'),'verified_inputs':len(plan['input_hashes'])}
 (DOC/(name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
train=ROOT/'stage/runs/t187_local_robustness';summary=read(train/'summary.json');plan=read(ROOT/'stage/experiments/t187_local_robustness/protocol.json');rows=[json.loads(x) for x in (train/'episodes.jsonl').read_text().splitlines()];ev=[json.loads(x) for x in (train/'evaluation.jsonl').read_text().splitlines()];assert len(rows)==128 and len(ev)==258
metrics['training']={};metrics['local_scores']={};metrics['paired_local_outcomes']={}
initial=torch.load(plan['initial_policy_checkpoint'],weights_only=False,map_location='cpu')['policy']
for arm,stats in summary['training'].items():
 assert sha(stats['checkpoint'])==stats['sha256'];ck=torch.load(stats['checkpoint'],weights_only=False,map_location='cpu');assert ck['protocol_sha256']==sha(ROOT/'stage/experiments/t187_local_robustness/protocol.json')
 rr=[r for r in rows if r['arm']==arm];assert len(rr)==64 and sum(r['steps'] for r in rr)==stats['physics_steps'];assert not (train/arm/'checkpoint_previous.pt').exists()
 n=0
 for p in (train/arm).glob('batch_*.npz'):
  with np.load(p,allow_pickle=False) as z:assert all(np.isfinite(z[k]).all() for k in z.files);n+=len(z['states'])
 w=ck['policy']['native_head.weight'];b=ck['policy']['native_head.bias']
 metrics['training'][arm]={**stats,'decisions':n,'max_actor_parameter_change':max(float((v-ck['policy'][k]).abs().max()) for k,v in initial.items() if k!='logstd'),'analytic_mean_lower_bound':(b-w.abs().sum(1)).tolist()}
for name in ['initial_embedding','fixed_source','randomized_reset']:
 metrics['local_scores'][name]={}
 for group in ['development_grid','fresh_holdout']:
  metrics['local_scores'][name][group]={}
  for side in ['LEFT','RIGHT']:
   subset=[r for r in ev if r['model']==name and r['group']==group and r['id'].startswith(('dev_'+side) if group=='development_grid' else ('holdout_'+side))]
   assert len(subset)==(27 if group=='development_grid' else 12)
   metrics['local_scores'][name][group][side]={'n':len(subset),'success':sum(r['success'] for r in subset)}
for arm in ['fixed_source','randomized_reset']:
 base={r['id']:r for r in ev if r['model']=='initial_embedding' and r['group'] in ['development_grid','fresh_holdout']};new={r['id']:r for r in ev if r['model']==arm and r['id'] in base};assert base.keys()==new.keys()
 assert all(base[k]['initial_reset_signature']==new[k]['initial_reset_signature'] for k in base)
 metrics['paired_local_outcomes'][arm]={g:{'gained':[k for k in base if base[k]['group']==g and not base[k]['success'] and new[k]['success']],'lost':[k for k in base if base[k]['group']==g and base[k]['success'] and not new[k]['success']]} for g in ['development_grid','fresh_holdout']}
metrics['training_physics_steps']=sum(r['physics_steps'] for r in summary['training'].values());metrics['training_decisions']=sum(r['decisions'] for r in metrics['training'].values());metrics['optimizer_updates']=sum(r['actor_updates'] for r in summary['training'].values());metrics['T187_frozen_evaluation_physics_steps']=sum(r['steps'] for r in ev);metrics['T186_diagnostic_physics_steps']=read(ROOT/'stage/runs/t186_anchor_robustness/summary.json')['physics_steps'];metrics['max_cpu_gpu_behavior_logp_error']=summary['max_cpu_gpu_behavior_logp_error']
audit=read(ROOT/'stage/runs/t188_direct_action_head/summary.json');assert audit['initial_max_native_output_error']==0 and audit['optimizer_updates']==audit['native_physics_steps']==0
metrics['action_head_initial_output_error']=audit['initial_max_native_output_error'];metrics['direct_head_trained']=False;metrics['primary_scores']=summary['scores'];metrics['all_new_runs_bytes']=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert metrics['all_new_runs_bytes']<1024**3
(DOC/'verification_and_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
