"""Verify completed Stage T180--T183 and summarize fixed-policy comparisons."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import json,hashlib,ast
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'stage/docs/results_20261030_onpolicy_course'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocols=[('t180_policy_sensitivity','protocol.json','t180_policy_sensitivity'),('t181_lateral_start','protocol.json','t181_lateral_start'),('t182_success_course_preflight','protocol.json','t182_success_course_preflight'),('t182_success_course_preflight','retry_protocol.json','t182b_success_course_preflight'),('t182_success_course_preflight','verified_protocol.json','t182c_success_course_preflight'),('t183_onpolicy_course','protocol.json','t183_onpolicy_course'),('t184_course_connection','protocol.json','t184_course_connection'),('t185_inference_contract','protocol.json','t185_inference_contract')]
metrics={'state':'PASS','input_hash_checks':{},'training':{},'paired_course_outcomes':{},'automatic_promotion':False,'incumbent':'T159'}
for folder,name,runname in protocols:
 src=ROOT/'stage/experiments'/folder;out=ROOT/'stage/runs'/runname;p=read(src/name)
 for f,h in p['input_hashes'].items():assert sha(f)==h,f
 assert read(out/'launch.json')['protocol_sha256']==sha(src/name)
 for f in src.glob('*.py'):ast.parse(f.read_text())
 metrics['input_hash_checks'][runname]={'inputs':len(p['input_hashes']),'protocol_sha256':sha(src/name),'state':read(out/'status.json')['state']}
 if (out/'summary.json').exists():(DOC/(runname+'_summary.json')).write_text(json.dumps(read(out/'summary.json'),indent=2)+'\n')
trainout=ROOT/'stage/runs/t183_onpolicy_course';s=read(trainout/'summary.json');assert s['state']=='COMPLETE';metrics['scores']=s['scores'];metrics['max_behavior_logp_error']=s['max_cpu_gpu_behavior_logp_error']
rows=[json.loads(x) for x in (trainout/'episodes.jsonl').read_text().splitlines()];evaluation=[json.loads(x) for x in (trainout/'evaluation.jsonl').read_text().splitlines()]
assert len(rows)==192 and len(evaluation)==96
initial=torch.load(ROOT/'stage/runs/t177_frontier_onpolicy/initial_policy.pt',weights_only=False,map_location='cpu')['policy']
for arm,stats in s['training'].items():
 p=Path(stats['checkpoint']);assert sha(p)==stats['sha256'];ck=torch.load(p,weights_only=False,map_location='cpu');assert ck['protocol_sha256']==sha(ROOT/'stage/experiments/t183_onpolicy_course/protocol.json')
 assert not (trainout/arm/'checkpoint_previous.pt').exists()
 a=[r for r in rows if r['arm']==arm];assert len(a)==96 and sum(r['steps'] for r in a)==stats['physics_steps']
 n=0
 for batch in (trainout/arm).glob('batch_*.npz'):
  with np.load(batch,allow_pickle=False) as z:
   assert all(np.isfinite(z[k]).all() for k in z.files);n+=len(z['states'])
 metrics['training'][arm]={**stats,'decisions':n,'training_timeouts':sum(r['timeout'] for r in a),'by_course_bin':{},'max_actor_parameter_change_vs_initial':max(float((v-ck['policy'][k]).abs().max()) for k,v in initial.items() if k!='logstd')}
 for b in [0,1,2,3,None]:
  subset=[r for r in a if r['course_bin']==b];metrics['training'][arm]['by_course_bin'][str(b)]={'n':len(subset),'success':sum(r['success'] for r in subset),'collisions':sum(r['collision'] for r in subset)}
 base={r['id']:r for r in evaluation if r['model']=='initial_embedding' and r['group']=='course_reset'};candidate={r['id']:r for r in evaluation if r['model']==arm and r['group']=='course_reset'};assert base.keys()==candidate.keys()
 assert all(base[k]['initial_reset_signature']==candidate[k]['initial_reset_signature'] for k in base)
 metrics['paired_course_outcomes'][arm]={'both_success':[k for k in base if base[k]['success'] and candidate[k]['success']],'new_success':[k for k in base if not base[k]['success'] and candidate[k]['success']],'lost_success':[k for k in base if base[k]['success'] and not candidate[k]['success']],'per_bin':{}}
 for b in range(4):
  rr=[r for r in candidate.values() if r['course_bin']==b];metrics['paired_course_outcomes'][arm]['per_bin'][str(b)]={'n':len(rr),'success':sum(r['success'] for r in rr)}
# The completed prefix is consumed once, not re-counted as new continuation physics.
pre=read(ROOT/'stage/runs/t182c_success_course_preflight/summary.json');assert pre['near_success_gate']=='PASS'
metrics['diagnostic_physics_steps_recorded_unique']=sum(read(ROOT/'stage/runs'/n/'summary.json')['physics_steps'] for n in ['t180_policy_sensitivity','t181_lateral_start','t182c_success_course_preflight','t184_course_connection','t185_inference_contract'])
metrics['incomplete_attempt_unrecorded_physics']={'exact':None,'upper_bound':373344,'reason':'Two interrupted preflights; at most8 in-flight capped episodes each. Their logs preserved; never assume zero or double-count recorded9caseprefix.'}
prior_success={r['case_id'] for r in map(json.loads,(ROOT/'stage/runs/t172_frozen_frontier/cases.jsonl').read_text().splitlines()) if r['model']=='T159' and r['safe_success']}
for arm in ['keep10','keep1']:
 candidate={r['source'] for r in evaluation if r['model']==arm and r['group']=='course_reset' and r['success']}
 metrics['paired_course_outcomes'][arm]['comparison_to_historical_T172_20Hz']={'new_success':sorted(candidate-prior_success),'lost_success':sorted(prior_success-candidate),'caveat':'HistoricalT172differentcaps,variableGPUbatch andmissingfullintegrationstate. This setcomparison is provenance, not a controlled algorithm/cadence comparison.'}
contract=read(ROOT/'stage/runs/t185_inference_contract/summary.json');assert contract['matched_initial_integration_state_and_observation']=='PASS'
metrics['inference_contract']={'left_all8_failed':all(not v['success'] for h in contract['scores']['LEFT'].values() for v in h.values()),'right_all8_success':all(v['success'] for h in contract['scores']['RIGHT'].values() for v in h.values()),'max_same_state_native_action_difference':max(max(v['max_same_state_action_difference']) for side in contract['scores'].values() for h in side.values() for v in h.values())}
metrics['training_physics_steps']=sum(x['physics_steps'] for x in metrics['training'].values())
metrics['training_decisions']=sum(x['decisions'] for x in metrics['training'].values())
metrics['actor_optimizer_updates']=sum(x['actor_updates'] for x in metrics['training'].values())
metrics['T183_frozen_evaluation_physics_steps']=sum(x['steps'] for x in evaluation)
metrics['all_new_runs_bytes']=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert metrics['all_new_runs_bytes']<1024**3
(DOC/'verification_and_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(9.3,3.5))
for ax,side in zip(axes,['LEFT','RIGHT']):
 rows=[json.loads(x) for x in (ROOT/'stage/runs/t185_inference_contract/cases.jsonl').read_text().splitlines()]
 for impl in ['cpu112','cuda112_batch1','cuda112_batch8','cpu113_embedding']:
  row=next(v for v in rows if v['case']['side']==side and v['hold_steps']==25 and v['implementation']==impl)
  with np.load(ROOT/'stage/runs/t185_inference_contract'/(row['id']+'.npz')) as z:
   d=np.linalg.norm(z['poses'][:,:2]-np.asarray(row['goal_xy']),axis=1);ax.plot(z['steps']*.002,d,label=impl)
 ax.axhline(.2,color='black',ls=':',lw=1);ax.set(title=side+' anchor, 20 Hz',xlabel='Simulation time (s)',ylabel='Distance to final goal (m)');ax.grid(alpha=.2)
axes[1].legend(fontsize=7);fig.suptitle('T185: same full integration state, frozen T159',fontsize=11);fig.tight_layout();fig.savefig(DOC/'inference_contract.png',dpi=150);plt.close(fig)
