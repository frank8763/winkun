"""One-factor preservation of the failed task pre-intervention prefix.

Reuse the audited20 full histories from T543 and qualified successful
T544 recovery history. No new teacher rollout.
Final evaluation has no teacher at runtime.
"""
from pathlib import Path
import copy,fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t545_protected_prefix_repair';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(HERE));import t545_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t542_offline_field_relay'));import t542_runtime as Base
F,G,E,P,np,torch=Base.F,L.G,Base.E,L.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
BASE_PAYLOAD=E.payload

def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration(root):
 p=read(root['actor_configuration']);p.update(root);p['out']=str(OUT);p['evaluation_jobs']=p['old8_jobs']+p['field_jobs'];assert p['out']==str(OUT)
 return p
Base.OUT=OUT;Base.HERE=HERE;Base.DOC=DOC;Base.verify=verify

def setup():
 Base.setup(OUT);F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify;P.frozen=L.A.frozen

def feedback(p,pose,goal,parameters):
 import math
 speed_gain,max_wheel,wheelbase,yaw_gain,max_steer=map(float,parameters)
 assert speed_gain>0 and max_wheel>0 and wheelbase>0 and yaw_gain>0 and 0<max_steer<math.pi/2
 d=float(np.linalg.norm(np.asarray(goal)-np.asarray(pose)[:2]));angle=math.atan2(goal[1]-pose[1],goal[0]-pose[0]);e=math.atan2(math.sin(angle-pose[2]),math.cos(angle-pose[2]))
 steer=float(np.clip(math.atan(p['physical_wheelbase_m']*2*math.sin(e)/max(d,p['curvature_distance_floor_m'])),-max_steer,max_steer));vmax=speed_gain*max_wheel
 desired=min(p['nominal_speed_m_s'],vmax)*min(1.,d/p['slow_distance_m'])*max(p['minimum_heading_speed_fraction'],math.cos(e))
 u=np.asarray([np.clip(2*math.sqrt(desired/vmax)-1,-1,1),np.clip(steer/max_steer,-1,1)],np.float32)
 return u,steer,desired


def recovery_audit(p,row,a,baseline):
 k=p['teacher_first']
 for name in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:assert np.array_equal(a[name][:k],baseline[name][:k]),name
 for name in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[name][k],baseline[name][k]),name
 owner=np.zeros(len(a['states']),np.uint8);owner[k:]=1;assert np.array_equal(owner,a['ownership']) and not a['stochastic'].any();assert np.array_equal(a['initial_native_integration'],baseline['initial_native_integration'])
 for i,x in enumerate(a['poses']):
  u,_,_=feedback(p,x,a['task_goal_xy'],a['teacher_vehicle_parameters']);assert np.array_equal(u,a['teacher_actions'][i])
 F.M.audit_full_history(p,row,a)

def assess(p,rows):
 c=Base.assess(p,rows);ids=set(c['field_success_ids']);before=set(p['initial_check']['field_success_ids']);c.update(lost_prior_field_ids=sorted(before-ids),gained_vs_T507=sorted(ids-set(p['accepted_baseline_field_ids'])),panel_repaired=c['old8']['success']==8 and before<=ids)
 c['accepted']=bool(c['panel_repaired'] and c['field24']['contact']<=p['max_accepted_field_contacts'] and c['field24']['invalid']==c['field24']['censored']==c['old8']['invalid']==c['old8']['censored']==0)
 return c

def main():
 root=read(OUT/'protocol.json');p=configuration(root);setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  source=G.rows(Path(p['audited_source_folder']));assert len(source)==21
  successful=[r for r in source[:20] if r['success'] and not r['collision']];assert len(successful)==19
  teacher=p['recovery_row'];assert sha(teacher['path'])==teacher['sha256'] and teacher['success'] and not teacher['collision'];a=E.N.S.arrays(teacher['path']);assert all(digest(v)==teacher['array_hashes'][k] for k,v in a.items());p['teacher_first']=int(np.flatnonzero(a['ownership'])[0]);assert p['teacher_first']==170
  eligible=True;derived=p['fixed_intervention'];result=dict(source_model=p['initial'],reused_audited_success_histories=19,reused_full_native_teacher=True,derived_prefix=derived,teacher_qualified=True,teacher_result=dict(success=True,contact=False,timeout=False,remaining_m=teacher['remaining_m']),Actor_Adam=0,Value_updates=0,new_policy_gradient_updates=0,update=None,model=None,check=None,accepted=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
  if eligible:
   obj=L.make(p);shadow=L.make(p);before=L.state(obj);refs=L.references(successful);rt.role='FEEDBACK_MIR_IMITATION'
   aa=L.fit(obj,a,p,refs,lambda:rt.guard(True),Base.event('live'));bb=L.fit(shadow,a,p,refs,lambda:rt.guard(True),Base.event('shadow'));assert aa==bb and digest(L.state(obj))==digest(L.state(shadow))
   model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.A.frozen(p,model['path']).phash()==model['policy_hash']
   rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
   vals=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_ALL32');er=[r for a,r in vals];rt.request(p['evaluation_jobs'],model,'EXACT_ALL32',reference=er);check=assess(p,er)
   result.update(update=aa,Actor_Adam=aa['Actor_Adam'],virtual_discarded_Adam=aa['virtual_discarded_Adam'],model=model,check=check,accepted=check['accepted'])
   if not result['accepted']:L.load(obj,before);L.load(shadow,before)
   result['retained_joint_hash']=digest(L.state(obj))
  result['reason']='JOINT_REPAIR_RESEARCH_GATE_PASSED' if result['accepted'] else 'REPAIR_REJECTED_ROLLBACK' if eligible else 'TERMINAL_FEEDBACK_NOT_QUALIFIED_ZERO_LEARNING';state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for o in [obj,shadow]:L.load(o,before)
  save(OUT/'failure.json',dict(error=error,all_raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();G.stopped(OUT)
 rows=G.rows(OUT);rep=G.rows(OUT,'replays.jsonl');source=G.rows(Path(p['audited_source_folder']))[:20]
 for r in source:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());F.M.audit_full_history(p,r,a)
 teacher=p['recovery_row'];a=E.N.S.arrays(teacher['path']);assert sha(teacher['path'])==teacher['sha256'] and teacher['success'] and not teacher['collision'];assert all(digest(v)==teacher['array_hashes'][k] for k,v in a.items());p['teacher_first']=int(np.flatnonzero(a['ownership'])[0]);assert p['teacher_first']==170
 baseline=E.N.S.arrays(source[19]['path']);recovery_audit(p,teacher,a,baseline)
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics']
  if r['full']:
   F.M.audit_full_history(p,r,a)
 by={r['path']:r for r in rows}
 for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 result=s['result'];assert result['teacher_qualified'];steps=virtual=0
 if result['teacher_qualified']:
  obj=L.make(p);before=L.state(obj);u=L.fit(obj,E.N.S.arrays(teacher['path']),p,L.references([r for r in source if r['success']]),lambda:None,lambda *_:None);steps=u['Actor_Adam'];virtual=u['virtual_discarded_Adam'];assert u==result['update'] and digest(L.state(obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256'];check=assess(p,rows);assert check==result['check'] and len(rows)==32
  if not result['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==result['retained_joint_hash']
 else:assert len(rows)==0 and not G.rows(OUT,'optimizer_calls.jsonl')
 expected=[[who,stage,k] for who in ['live','shadow'] for k in range(steps) for stage in ['VIRTUAL_BEFORE','VIRTUAL_ROLLED_BACK','BEFORE','AFTER']];assert G.rows(OUT,'optimizer_calls.jsonl')==expected
 assert len(rows)==len(rep);assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'];assert s['commands']==len(rows)+len(rep)+4<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Actor_Adam=steps,discarded_virtual_Adam=virtual,teacher_actual_feedback_commands_and_intervention_rebuilt=True,all_saved_full_training_density_rewards_hidden_rebuilt=True,Actor_Adam_and_virtual_MIR_ranking_rollback_exact=True,compact_evaluation_full120_NOT_reconstructed_offline=True,all_native_replay_full_hashes_and_integration_exact=True,physics=s['physics'],commands=s['commands']));print('T545_PREFIX_REPAIR_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
