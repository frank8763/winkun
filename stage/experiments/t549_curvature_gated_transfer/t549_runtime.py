"""One verified terminal recovery, protected readout, then complete32/exact."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t549_curvature_gated_transfer';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(HERE));import t549_actor as L
sys.path.insert(0,str(ROOT/'stage/experiments/t547_frozen_complete_panel'));import t547_runtime as Prior
sys.path.insert(0,str(ROOT/'stage/experiments/t544_terminal_feedback_repair'));import t544_runtime as Teacher
Base,F,G,E,P,np,torch=Prior.Base,Prior.F,L.G,Prior.E,L.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
NORMAL_PAYLOAD=Prior.Previous.E.payload

def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def configuration(root):
 p=read(root['actor_configuration']);p.update(root);p['out']=str(OUT);return p

def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

Base.OUT=OUT;Base.HERE=HERE;Base.DOC=DOC;Base.verify=verify

def setup():
 Base.setup(OUT);F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify;P.frozen=L.frozen;E.payload=NORMAL_PAYLOAD
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def worker(pipe,index,p,counts):
 G.T.Session=Teacher.RecoverySession if p.get('recovery_teacher') else F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Teacher.payload if p.get('recovery_teacher') else NORMAL_PAYLOAD;G.cpu_worker(pipe,index,p,counts)

class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
   f=OUT/'worker_generations.json';old=read(f) if f.exists() else [];old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),recovery_teacher=bool(p.get('recovery_teacher')),CPU=[14,15],GPU=False));save(f,old)
  except BaseException:self.close();raise
 def receive(self,i):
  value=super().receive(i)
  if self.p.get('recovery_teacher') and isinstance(value,tuple):value[1].update(teacher=True,teacher_first=self.p['teacher_first'],training_recovery_NOT_autonomous=True)
  return value

def assess(p,rows):
 by={r['job_id']:r for r in rows};assert len(rows)==len(by)==32;ordered=[by[j['id']] for j in p['evaluation_jobs']];c=Base.assess(p,ordered);now=set(c['field_success_ids']);prior=set(p['protected_field_ids']);c.update(lost_prior_field_ids=sorted(prior-now),gained_field_ids=sorted(now-prior),protected20_retained=c['old8']['success']==8 and prior<=now,target_safe=by[p['probe_job']['id']]['success'] and not by[p['probe_job']['id']]['collision']);c['strict_research_gate_pass']=c['protected20_retained'] and len(now)>len(prior) and c['field24']['contact']<=9 and c['field24']['invalid']==c['field24']['censored']==0;return c

def preserved_parent20(p,rows):
 by={r['job_id']:r for r in rows};result=[]
 for old in p['parent20_rows']:
  current=by[old['job_id']];assert old['policy_hash']==p['initial']['policy_hash'] and sha(old['path'])==old['sha256'];result.append(dict(id=old['job_id'],parent_path=old['path'],new_path=current['path'],all_array_hashes_equal=current['array_hashes']==old['array_hashes'],integration_equal=current['integration_sha256']==old['integration_sha256'],new_safe=bool(current['success'] and not current['collision']),not_independent_same_candidate_repeat=True))
 return result

def main():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None
 result=dict(reason='NOT_EVALUATED',teacher_qualified=False,qualification=None,model=None,check=None,new_Adam=0,new_RL=0,new_Value=0,accepted=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
 try:
  baseline=E.N.S.arrays(p['baseline_row']['path']);rt.pool=Pool(dict(p,phase='student',recovery_teacher=True,native_stop_epoch=rt.deadline),rt.counts,rt.guard);first=rt.request([p['probe_job']],p['initial'],'VERIFIED_TRAINING_TERMINAL_RECOVERY',full=True);a,row=first[0];Teacher.recovery_audit(p,row,a,baseline);rt.request([p['probe_job']],p['initial'],'EXACT_TRAINING_RECOVERY',full=True,reference=[row]);eligible=bool(row['success'] and not row['collision']);result.update(teacher_qualified=eligible,teacher_result=dict(success=row['success'],contact=row['collision'],timeout=row['timeout'],remaining_m=row['remaining_m']))
  rt.pool.close();rt.pool=None;rt.commands+=2
  if eligible:
   try:
    q,metric=L.fit(p,a,lambda stage,value:save(OUT/'fit_progress.json',dict(stage=stage,metrics=value,epoch=time.time())));shadow,other=L.fit(p,a);assert metric==other and q.phash()==shadow.phash();rt.guard(True);model=L.export(q,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==q.phash();result.update(qualification=metric,model=model)
   except AssertionError as e:result.update(reason='FIXED_FUNCTION_QUALIFICATION_REJECTED',qualification_failure=str(e),qualification_trace=traceback.format_exc());save(OUT/'qualification_rejection.json',dict(error=result['qualification_trace'],no_retry=True))
   if result['model']:
    rt.pool=Pool(dict(p,phase='student',recovery_teacher=False,native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
    target=rt.request([p['probe_job']],model,'PURE_ACTOR_NEW_TARGET_FULL',full=True);restjobs=[j for j in p['evaluation_jobs'] if j['id']!=p['probe_job']['id']];rest=rt.request(restjobs,model,'PURE_ACTOR_COMPLETE_REMAINING31');rr=[r for a,r in target+rest];rt.request([p['probe_job']],model,'EXACT_NEW_TARGET_FULL',full=True,reference=rr[:1]);otherjobs=[j for j in p['unprotected_jobs'] if j['id']!=p['probe_job']['id']];byid={r['job_id']:r for r in rr};rt.request(otherjobs,model,'EXACT_OTHER11_PREVIOUS_FAILURES',reference=[byid[j['id']] for j in otherjobs]);identity=preserved_parent20(p,rr);save(OUT/'parent20_identity.json',identity);check=assess(p,rr);check['all20_parent_native_identity']=all(x['all_array_hashes_equal'] and x['integration_equal'] and x['new_safe'] for x in identity);check['protected20_retained']=check['protected20_retained'] and check['all20_parent_native_identity'];check['strict_research_gate_pass']=check['strict_research_gate_pass'] and check['all20_parent_native_identity'];result.update(check=check,strict_research_gate_pass=check['strict_research_gate_pass'],all32_safe=check['all_field24_pass'],new_same_candidate_replays=12,parent_matched_native_cases=20,all32_same_candidate_replays_NOT_executed=True);result['reason']='PROTECTED_DEVELOPMENT_IMPROVEMENT' if check['protected20_retained'] and check['gained_field_ids'] else 'CANDIDATE_REJECTED_NO_PROTECTED_GAIN'
  if not eligible:result['reason']='RECOVERY_TEACHER_NOT_QUALIFIED_ZERO_FIT'
  assert sha(p['initial']['path'])==p['initial']['sha256'];state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();G.stopped(OUT);rows=G.rows(OUT);rep=G.rows(OUT,'replays.jsonl');by={r['path']:r for r in rows};assert rows and (len(rows)==len(rep)==1 or (len(rows)==33 and len(rep)==13))
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['full']:
   if r.get('teacher'):Teacher.recovery_audit(p,r,a,E.N.S.arrays(p['baseline_row']['path']))
   else:F.M.audit_full_history(p,r,a)
 for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 result=s['result'];assert result['teacher_qualified']==bool(rows[0]['success'] and not rows[0]['collision']);rebuild=0
 if result['teacher_qualified']:
  a=E.N.S.arrays(rows[0]['path'])
  if result['model']:
   q,metric=L.fit(p,a);assert metric==result['qualification'] and q.phash()==result['model']['policy_hash'] and digest(dict(actor=q.state_dict()))==result['model']['joint_hash'];assert sha(result['model']['path'])==result['model']['sha256'];assert len(rows)==33;check=assess(p,rows[1:]);identity=preserved_parent20(p,rows[1:]);assert identity==read(OUT/'parent20_identity.json');check['all20_parent_native_identity']=all(x['all_array_hashes_equal'] and x['integration_equal'] and x['new_safe'] for x in identity);check['protected20_retained']=check['protected20_retained'] and check['all20_parent_native_identity'];check['strict_research_gate_pass']=check['strict_research_gate_pass'] and check['all20_parent_native_identity'];assert check==result['check'];rebuild=1
  else:
   try:L.fit(p,a)
   except AssertionError as e:assert str(e)==result['qualification_failure']
   else:raise AssertionError('Rejected fit unexpectedly passes')
 else:assert len(rows)==1
 assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'];assert s['commands']==len(rows)+len(rep)+(8 if result['model'] else 4)<=p['max_commands'];assert sha(p['initial']['path'])==p['initial']['sha256']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_Adam=0,independent_closed_form_rebuilds=rebuild,all_teacher_prefix_and_action_labels_verified=True,protected20_and_original_model_arithmetic_preservation_verified=bool(rebuild),new_target_full_history_density_hidden_reward_verified=bool(rebuild),all12_same_candidate_native_replays_verified=bool(rebuild),parent20_native_identity_verified=bool(rebuild),all32_same_candidate_replays_NOT_executed=True,physics=s['physics'],commands=s['commands'],accepted=False));print('T549_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
