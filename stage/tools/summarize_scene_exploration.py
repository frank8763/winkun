"""Read-only verification and compact report data for completed T176--T178."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import ast
import hashlib
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'stage/docs/results_20261030_scene_exploration'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
result={'state':'PASS','protocols':{},'training':{},'new_training_physics_steps':0,'automatic_promotion':False,'incumbent':'T159'}
for name in ['t176_scene_controls','t177_frontier_onpolicy','t178_exploration_time']:
 src=ROOT/'stage/experiments'/name;out=ROOT/'stage/runs'/name;protocol=read(src/'protocol.json');summary=read(out/'summary.json')
 assert read(out/'status.json')['state']=='COMPLETE'
 for p,h in protocol['input_hashes'].items():assert sha(p)==h,p
 for p in src.glob('*.py'):ast.parse(p.read_text())
 launch=read(out/'launch.json');assert launch['protocol_sha256']==sha(src/'protocol.json')
 result['protocols'][name]={'sha256':sha(src/'protocol.json'),'verified_inputs':len(protocol['input_hashes'])}
 (DOC/(name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
 if name.startswith('t176'):continue
 episodes=[json.loads(s) for s in (out/'episodes.jsonl').read_text().splitlines()]
 initial=torch.load(ROOT/'stage/runs/t177_frontier_onpolicy/initial_policy.pt',weights_only=False,map_location='cpu')['policy']
 for arm,stats in summary['training'].items():
  p=Path(stats['checkpoint']);assert sha(p)==stats['sha256'];checkpoint=torch.load(p,weights_only=False,map_location='cpu')
  assert checkpoint['protocol_sha256']==sha(src/'protocol.json') and not (out/arm/'checkpoint_previous.pt').exists()
  delta=max(float((v-checkpoint['policy'][k]).abs().max()) for k,v in initial.items() if k!='logstd');assert delta>0
  rows=[r for r in episodes if r['arm']==arm];assert len(rows)==48 and sum(r['steps'] for r in rows)==stats['physics_steps']
  assert sum(r['success'] for r in rows)==stats['training_success']
  batches=[]
  for p in sorted((out/arm).glob('batch_*.npz')):
   with np.load(p,allow_pickle=False) as z:
    assert all(np.isfinite(z[k]).all() for k in z.files);batches.append(z['latents'].copy())
  latent=np.concatenate(batches);native=latent.clip(-1,1);native[:,0]=latent[:,0].clip(0 if arm=='legacy_range' else -1,1)
  result['training'][name+'/'+arm]={**stats,'training_timeouts':sum(r['timeout'] for r in rows),'max_policy_parameter_change':delta,
      'decision_count':len(latent),'clipped_decision_fraction':float(np.any(abs(latent-native)>1e-7,axis=1).mean()),
      'negative_native_throttle_fraction':float((native[:,0]<0).mean()),'frozen_scores':summary['scores'][arm]}
  result['new_training_physics_steps']+=stats['physics_steps']
scene=read(ROOT/'stage/runs/t176_scene_controls/summary.json');assert scene['actor_unchanged'] and scene['paired_reset_hashes']=='PASS'
result['scene_scores']=scene['scores'];result['scene_physics_steps']=scene['physics_steps'];result['scene_empty_nearest']={}
for row in scene['cases']:
 if row['kind']=='scene_control' and row['scene']=='empty':
  with np.load(ROOT/'stage/runs/t176_scene_controls'/(row['id']+'.npz'),allow_pickle=False) as z:
   result['scene_empty_nearest'][row['id']]={'sampled_nearest_m':float(np.linalg.norm(z['poses'][:,:2]-np.asarray(row['goal_xy']),axis=1).min()),'sampling_physics_steps':25}
result['all_new_runs_bytes']=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
assert result['all_new_runs_bytes']<1024**3
(DOC/'verification_and_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['training','scene_scores']},indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
for ax,side in zip(axes,['LEFT','RIGHT']):
 for kind,color in [('original','#b23a48'),('empty','#277da1'),('culprit_removed','#43aa8b')]:
  row=next(r for r in scene['cases'] if r['kind']=='scene_control' and r['scene']==kind and r['case']['side']==side)
  with np.load(ROOT/'stage/runs/t176_scene_controls'/(row['id']+'.npz')) as z:
   ax.plot(z['steps']*.002,np.linalg.norm(z['poses'][:,:2]-row['goal_xy'],axis=1),label=kind.replace('_',' '),color=color)
  ax.scatter([row['steps']*.002],[row['remaining_m']],color=color,s=18)
 ax.axhline(.2,color='black',ls=':',lw=1,label='goal radius')
 ax.set(title=side+' 5 m: frozen T159',xlabel='Simulation time (s)',xlim=(0,47),ylim=(0,5.2));ax.grid(alpha=.2)
axes[0].set_ylabel('Distance to final goal (m)');axes[1].legend(fontsize=8)
fig.suptitle('T176 scene controls: matched physical reset, shared history refill',fontsize=11)
fig.tight_layout();fig.savefig(DOC/'scene_controls.png',dpi=160);plt.close(fig)
