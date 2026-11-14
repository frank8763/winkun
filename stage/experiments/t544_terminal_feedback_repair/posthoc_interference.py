"""Bounded saved-model inference, no native step or optimizer update.

Read-only original evidence. Compare identical actual histories, separately
from closed-loop native outcomes. Adding this analysis does not alter the
registered T544 scientific source/protocol or its original failed gate.
"""
from pathlib import Path
import copy,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t544_runtime as R
start=time.time();p=R.configuration(R.read(R.OUT/'protocol.json'));R.verify(p,True);s=R.read(R.OUT/'summary.json');assert R.read(R.OUT/'audit.json')['state']=='PASS';R.G.stopped(R.OUT);assert not Path('/proc',str(R.read(R.OUT/'launch.json')['pid'])).exists()
R.torch.set_num_threads(1);R.torch.use_deterministic_algorithms(True)
rows=R.G.rows(R.OUT);source=R.G.rows(Path(p['audited_source_folder']))[:20];old=next(r for r in source if r['job_id']==p['probe_job']['id']);student=next(r for r in rows if r['role']=='PURE_STUDENT_ALL32' and r['job_id']==old['job_id']);teacher=rows[0];a=R.E.N.S.arrays(old['path']);b=R.E.N.S.arrays(teacher['path']);c=R.E.N.S.arrays(student['path']);k=int(R.read(R.OUT/'derived_prefix.json')['teacher_first']);q0=R.L.A.frozen(p,p['initial']['path']);q1=R.L.A.frozen(p,s['result']['model']['path']);assert R.sha(s['result']['model']['path'])==s['result']['model']['sha256']
assert old['array_hashes']['initial_native_integration']==student['array_hashes']['initial_native_integration']==teacher['array_hashes']['initial_native_integration'];assert old['array_hashes']['task_goal_xy']==student['array_hashes']['task_goal_xy']
assert R.np.array_equal(a['states'][:k],b['states'][:k]) and R.np.array_equal(a['locations'][:k],b['locations'][:k]);stats=[]
with R.torch.no_grad():
 x=R.torch.tensor(b['states']);z0=q0.sequence(x).numpy();z1=q1.sequence(x).numpy();assert R.np.abs(z0-b['locations']).max()<1e-10
 delta=R.np.abs(z1-z0);executed0=R.np.clip(z0,-1,1).astype('f4');executed1=R.np.clip(z1,-1,1).astype('f4');ix=R.np.flatnonzero(R.np.any(executed0[:k]!=executed1[:k],axis=1));assert len(ix)>0
 assert R.np.array_equal(executed1[0],c['actions'][0]) and R.np.array_equal(executed0[0],a['actions'][0]);assert ix[0]==0
 prefix=dict(holds=k,seconds=float(b['native_counts'][:k].sum()*.002),first_policy_action_change_hold=int(ix[0]),first_parent_action=executed0[0].tolist(),first_updated_action=executed1[0].tolist(),initial_native_and_goal_hash_identical=True,same_actual_prefix_location_RMSE=float(R.np.sqrt(R.np.mean((z1[:k]-z0[:k])**2))),same_actual_prefix_location_max=float(delta[:k].max()),same_actual_prefix_executed_max=float(R.np.abs(executed1[:k]-executed0[:k]).max()),first_updated_vs_parent_location_delta=(z1[0]-z0[0]).tolist())
 owner=R.torch.tensor(b['ownership'].astype(bool));targets=R.torch.tensor(b['teacher_actions'],dtype=R.torch.float64)
 losses=[float(R.L.P.action_loss(R.torch.tensor(z)[owner],targets[owner]).mean()) for z in [z0,z1]]
 refs=[r for r in source if r['success'] and not r['collision']];assert len(refs)==19 and old['job_id'] not in [r['job_id'] for r in refs]
 for r in refs:
  data=R.E.N.S.arrays(r['path']);pred=q1.sequence(R.torch.tensor(data['states'])).numpy();target=data['locations'];d=pred-target;selected=sum(r['job_id'] in t['ranking']['selected'] for t in s['result']['update']['trace']);stats.append(dict(id=r['job_id'],RMSE=float(R.np.sqrt(R.np.mean(d*d))),max_abs=float(R.np.abs(d).max()),MIR_selected_updates=selected,initial_memory_loss=s['result']['update']['trace'][0]['selected_rehearsal_loss']))
params=[];sd0=q0.state_dict();sd1=q1.state_dict()
for group in ['encoder.','core.','head.','drive_adapter.','steer_adapter.']:
 keys=[name for name in sd0 if name.startswith(group)];norm=float(sum((sd1[name]-sd0[name]).square().sum() for name in keys).sqrt());params.append(dict(module=group,changed=any(not R.torch.equal(sd0[name],sd1[name]) for name in keys),delta_L2=norm))
assert R.torch.equal(sd0['mean'],sd1['mean']) and R.torch.equal(sd0['scale'],sd1['scale']);assert time.time()-start<180
result=dict(state='SAVED_INTERFERENCE_CONFIRMED',new_native=0,new_optimizer=0,pid=os.getpid(),seconds=time.time()-start,source_protocol_sha256=R.sha(R.OUT/'protocol.json'),candidate_model=s['result']['model'],native_old_collision_seconds=student['physics']*.002,teacher_start_seconds=prefix['seconds'],prefix=prefix,recovery_executed_action_MSE_before_after=losses,all19_successes=stats,module_changes=params,initial_normalization_buffers_unchanged=True,findings=['Updated Actor changed first executed action at identical native reset and goal, not just after teacherstart34s.','Failing target excluded from19success replay; its physically safe pre-correction history had no explicit functional preservation objective.','All shared modules changed; offline identical-history action changes plus observed earlier collision support prefix interference, not unique full-trajectory causal attribution.','MIR loss proxy and selected sequence rehearsals do not guarantee native success.','No new native reconstruction of updated failing full120 is claimed; updated failure persisted compact only.'],next_single_factor='Add teacher-owned0 actual prefix location-preservation term to every current recovery objective incl virtual update; keep same initial/recovery data/MIR19/16updates/lr/gates. This is a soft constraint, initially zero-gradient, not a safety certificate.')
R.save(R.DOC/'T544_POSTHOC_INTERFERENCE.json',result);print('POSTHOC_PREFIX',prefix);print('RECOVERY_MSE',losses);print('LOST_FIELD',next(r for r in stats if r['id']=='t502_field_sector2_offset2'));print('MODULES',params)
