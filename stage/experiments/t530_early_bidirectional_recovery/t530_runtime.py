"""Validated old-skill recovery, actual new-skill reconstruction, then native gate."""
from pathlib import Path
import fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t529_bidirectional_recovery'));import t529_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t525_student_state_recovery'));import t525_runtime as Recovery
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t530_early_bidirectional_recovery'
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def evaluate(rt,p,model,reference=None):
 rt.role='EXACT_FULL32' if reference else 'AUTONOMOUS_FULL32';rows=[]
 for off in range(0,32,2):
  jobs=p['evaluation_jobs'][off:off+2];rt.guard();assert sum(rt.counts)+sum(j['case']['step_cap'] for j in jobs)<=p['max_physics'] and rt.commands+4<=p['max_commands'];rt.commands+=2;req=[dict(model=model['path'],job=j,seed=None,query=False,full_path=None) for j in jobs]
  for i,((a,r),j,request) in enumerate(zip(rt.pool.request(req),jobs,req)):
   assert r['policy_hash']==model['policy_hash'];r.update(job_id=j['id'],role=rt.role,seed=None,request=request,loaded_model_sha256=sha(model['path']),full=False,full_arrays_generated_but_not_saved=True);r['field_detail']=F.detail(p,r,a) if j['id'].startswith('t502_') else None
   if reference:
    old=reference[off+i];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail'];r.update(reference_path=old['path'],exact=True,independent_new_sample=False);rt.replays.append(r);name='replays.jsonl'
   else:
    path=Path(p['out'])/'episodes'/('eval_'+str(off+i)+'.npz');path.parent.mkdir(exist_ok=True);E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},p['compact_bytes']);r.update(path=str(path),sha256=sha(path));rt.rows.append(r);name='cases.jsonl'
   with (Path(p['out'])/name).open('a') as h:h.write(json.dumps(r)+'\n')
   rows.append(r)
 return rows

def assess(p,rows):
 assert len(rows)==32;old=G.R.M.tally(rows[:8]);field=G.R.M.tally(rows[8:]);ids={r['job_id'] for r in rows[8:] if r['field_detail']['field_qualified_success']};previous=set(p['baseline_field_success_ids']);target=p['new_success_job']['id'];ok=old['success']==8 and previous<=ids and target in ids and len(ids)>=12 and field['contact']<=9 and field['invalid']==field['censored']==old['invalid']==old['censored']==0
 return dict(old8=old,field24=field,field_qualified_goals=len(ids),field_success_ids=sorted(ids),lost_baseline_ids=sorted(previous-ids),new_target_retained=target in ids,accepted=ok)

def main(phase):
 folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);P.frozen=L.A.frozen;E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None;obj=shadow=before=None
 try:
  p['baseline_row']=p['old_control_row'];own=p['new_control_row']
  rt.pool=Recovery.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);job=dict(p['probe_job'],teacher_holds=-1);vv=rt.request([job],p['initial'],'VALIDATE_PARENT_ON_ACTUAL_OLD_FAILURE_STATE',full=True);a,row=vv[0];Recovery.check_prefix(p,a);detail=F.detail(p,row,a);rt.request([job],p['initial'],'EXACT_PARENT_RECOVERY',full=True,references=[row]);rt.pool.close();rt.pool=None;rt.commands+=2
  eligible=detail['field_qualified_success'];result=dict(recovery_teacher_qualified=eligible,old_recovery_detail=detail,new_goal_control=G.R.M.tally([own]),learning_eligible=eligible,Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,old8=None,field24=None,accepted=False,full_old275_six54_eight54_NOT_executed=True,fresh16=None,reload78=None,under8_gate=False)
  if eligible:
   obj=L.make(p);shadow=L.make(p);before=P.state(*obj);assert digest(before)==p['initial']['joint_hash'];refs=L.references(p,own)
   rt.role='BIDIRECTIONAL_RECOVERY_WITH_ACTUAL_NEW_SUCCESS_KEEP';aa=L.fit(obj,a,p,refs,lambda:rt.guard(True),F.R.emit('live'));bb=L.fit(shadow,a,p,refs,lambda:rt.guard(True),F.R.emit('shadow'));assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
   model=L.A.export(obj,folder/'proposal.pt.gz',p['model_bytes']);rt.commands+=2;rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);rows=evaluate(rt,p,model);evaluate(rt,p,model,rows);result.update(assess(p,rows),model=model,update=aa,Actor_Adam=aa['Actor_Adam'])
   if not result['accepted']:
    for z in [obj,shadow]:P.load(z,before)
   result['final_joint_hash']=digest(P.state(*obj))
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for z in [obj,shadow]:P.load(z,before)
  save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(phase):
 folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder);rows=G.rows(folder);rep=G.rows(folder,'replays.jsonl');result=s['result'];p['baseline_row']=p['old_control_row'];P.frozen=L.A.frozen
 for row in rows:
  a=E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256']
  if row['full']:
   assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
   if row['teacher']:Recovery.audit_recovery(p,row,a)
   else:M.audit_full_history(p,row,a)
  else:
   assert set(a)=={'actions','native_counts','end_flags'} and int(a['native_counts'].sum())==row['physics']
   for k,v in a.items():assert digest(v)==row['array_hashes'][k]
   assert a['end_flags'][-1].tolist()==[int(row['collision']),int(row['success']),int(row['timeout'])] and not row['administrative_censor']
 by={r['path']:r for r in rows}
 for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 teacher=rows[0];a=E.N.S.arrays(teacher['path']);assert F.detail(p,teacher,a)==result['old_recovery_detail'];eligible=result['old_recovery_detail']['field_qualified_success'];assert eligible==result['learning_eligible'];steps=0
 if eligible:
  obj=L.make(p);before=P.state(*obj);u=L.fit(obj,a,p,L.references(p,p['new_control_row']),lambda:None,lambda *_:None);steps=u['Actor_Adam'];assert u==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
  for k,v in assess(p,rows[1:]).items():assert result[k]==v
  if not result['accepted']:P.load(obj,before)
  assert digest(P.state(*obj))==result['final_joint_hash']
 assert len(rows)==len(rep)==(33 if eligible else 1);assert G.rows(folder,'optimizer_calls.jsonl')==[[w,st,k] for w in ['live','shadow'] for k in range(steps) for st in ['BEFORE','AFTER']]
 assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==len(rows)+len(rep)+(8 if eligible else 4)
 save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,historical_discarded_Actor_Adam=steps,Value_Adam=0,historical_T529_exact_old_failure_and_new_success_reused_not_new_samples=True,validated_parent_recovery_history_and_teacher_labels_rebuilt=True,all20_success_rehearsal_and_whole_ActorAdam_rebuilt=True if eligible else None,all_native_fullarray_hashes_and_integrations_replay_exact=True,compact_evaluation_NOT_full120_rebuilt_offline=True,physics=s['physics'],commands=s['commands']));print('T530_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
