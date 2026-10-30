from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[3];SRC=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t179_mechanism_audit';DOC=ROOT/'stage/docs/results_20261030_mechanism_audit';DOC.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=read(OUT/'summary.json');audit=read(OUT/'gradient_audit.json');recheck=read(OUT/'contact_recheck/summary.json')
assert summary['state']==audit['state']==recheck['state']=='COMPLETE'
for name,folder in [('protocol.json',OUT),('contact_recheck_protocol.json',OUT/'contact_recheck')]:
 plan=read(SRC/name)
 for p,h in plan['input_hashes'].items():assert sha(p)==h,p
 assert read(folder/'launch.json')['protocol_sha256']==sha(SRC/name)
for r in recheck['cases']:
 old=next(v for v in summary['cases'] if v['id']==r['id'])
 for k in ['steps','initial_obs_sha256','scene_hash','reset_hash','final_pose','safe_success']:assert r[k]==old[k],(k,r['id'])
 assert any('real_circle_obs_9' in [v['geom1'],v['geom2']] and 'chassis_collision' in [v['geom1'],v['geom2']] for v in r['terminal_contacts'])
old=read(ROOT/'stage/runs/t176_scene_controls/summary.json')
for scale,key in [(0,'culprit_removed_5m_2029653601_LEFT'),(1,'original_5m_2029653601_LEFT')]:
 a=next(r for r in summary['cases'] if r['scale']==scale and r['control_hold_steps']==25);b=next(r for r in old['cases'] if r['id']==key)
 for k in ['steps','remaining_m','final_pose']:assert a[k]==b[k],k
angle=np.linspace(-.75*np.pi,.75*np.pi,48,dtype=np.float32);visibility=[]
for r in summary['cases']:
 if r['scale']<=0 or r['history']!='legacy':continue
 with np.load(OUT/(r['id']+'.npz')) as z:poses=z['poses'];states=z['states'];steps=z['steps']
 center=np.array([.899999976158142,-3.09999990463257]);radius=.25*r['scale'];yaw=poses[:,2]
 origin=poses[:,:2]+.48*np.stack([np.cos(yaw),np.sin(yaw)],axis=1);d=center-origin;theta=yaw[:,None]+angle[None,:]
 proj=d[:,0,None]*np.cos(theta)+d[:,1,None]*np.sin(theta);disc=radius**2-(np.sum(d*d,axis=1)[:,None]-proj**2)
 entry=proj-np.sqrt(np.maximum(disc,0));exit=proj+np.sqrt(np.maximum(disc,0));expected=np.where(entry>=0,entry,exit)
 matched=(disc>=0)&(expected>=0)&(expected<=6)&(abs(states[:,:48]*6-expected)<1e-4)
 window=(steps*.002>=r['simulation_s']-12)&(steps*.002<=r['simulation_s']-6);counts=matched.sum(1)[window]
 visibility.append({'id':r['id'],'ray_match_count_12to6s_before_contact_min':int(counts.min()),'max':int(counts.max()),'mean':float(counts.mean()),'scope':'Analytic vertical-cylinder intersections matched to saved ray ranges within1e-4m, not stored native hit-ID telemetry. Different radius policies follow different physical trajectories; not a matched-pose visibility-effect estimate.'})
metrics={'state':'PASS','main_cases':12,'main_physics_steps':summary['physics_steps'],'contact_rechecks':3,'contact_recheck_physics_steps':recheck['physics_steps'],'gradient_updates':0,'t176_reference_replays':'exact_steps_final_pose_and_distance','contact_recheck_replays':'exact_initial_obs_scene_reset_steps_final_pose','native_owner':read(OUT/'native_inspection.json')['native_command_owner'],'visibility_diagnostic':visibility,'prior_training_accounting':{}}
for name in ['t177_frontier_onpolicy','t178_exploration_time']:
 prev=read(ROOT/'stage/runs'/name/'summary.json')
 for arm,s in prev['training'].items():
  n=0
  for p in (ROOT/'stage/runs'/name/arm).glob('batch_*.npz'):
   with np.load(p) as z:n+=len(z['states'])
  metrics['prior_training_accounting'][name+'/'+arm]={'physics_steps':s['physics_steps'],'decisions':n,'episodes':s['training_episodes'],'updates':s['actor_updates']}
for name in ['summary.json','gradient_audit.json','native_inspection.json']:(DOC/name).write_text(json.dumps(read(OUT/name),indent=2)+'\n')
(DOC/'contact_recheck_summary.json').write_text(json.dumps(recheck,indent=2)+'\n');(DOC/'verification.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
