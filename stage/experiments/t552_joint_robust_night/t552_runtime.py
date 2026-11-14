"""Bounded offline joint-retention child, genuine own-policy samples only for RL."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t552_joint_robust_night';DOC=ROOT/'stage/docs/overnight_20261114'
sys.path.insert(0,str(HERE));import t552_actor as L
sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'));import t550_runtime as Prior
Base,F,G,E,P,np,torch=Prior.Base,Prior.F,Prior.G,Prior.E,Prior.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file() and not x.is_symlink())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def plan(folder):
 root=read(OUT/'protocol.json');c=read(Path(folder)/'protocol.json');assert c['_root_sha256']==sha(OUT/'protocol.json');p=read(root['actor_configuration']);p.update(root);p.update(c);p['out']=str(folder);return p

def setup(folder):
 Base.OUT=OUT;Base.HERE=HERE;Base.verify=verify;Base.setup(folder);Base.ACTIVE=folder;F.OUT=folder;F.HERE=HERE;F.verify=verify;F.R.OUT=folder;F.R.HERE=HERE;F.R.verify=verify;G.B.ACTIVE=folder;G.B.verify=verify;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def worker(pipe,index,p,counts):
 G.T.Session=F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Prior.NORMAL;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready);save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise

def assess(p,rows):
 assert [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']];old=G.R.M.tally(rows[:8]);lost=G.R.M.tally(rows[8:14]);field=G.R.M.tally(rows[14:]);ids=sorted(r['job_id'] for r in rows if r['success'] and not r['collision']);fieldids=sorted(r['job_id'] for r in rows[14:] if r['field_detail']['field_qualified_success']);return dict(old8=old,restore6=lost,field24=field,success_ids=ids,field_success_ids=fieldids,distance_sum=sum(min(20.,r['remaining_m']) for r in rows),full38_safe=len(ids)==38,not_full275_retention=True)
def accepts(before,after):return bool(set(before['success_ids'])<=set(after['success_ids']) and len(after['success_ids'])>len(before['success_ids']) and after['field24']['contact']<=before['field24']['contact'])
def event(w):return lambda stage,k:Base.append('optimizer_calls.jsonl',[w,stage,k])

def main(child):
 folder=OUT/child;p=plan(folder);setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  if p['kind']=='round':
   data,pairs=L.cached_dataset(p);obj=L.make(p);shadow=L.make(p);before=L.state(obj);rt.role='JOINT_BC';b=L.bc(obj,data,pairs,p,lambda:rt.guard(True),event('live'));b2=L.bc(shadow,data,pairs,p,lambda:rt.guard(True),event('shadow'));assert b==b2 and digest(L.state(obj))==digest(L.state(shadow));behavior=L.export(obj,p,folder/'behavior.pt.gz');assert L.frozen(p,behavior['path']).phash()==obj[0].phash()
   rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['training_jobs'],behavior,'FRESH_CURRENT_POST_BC_POLICY',seeds=p['training_seeds'],full=True);sr=[r for a,r in vv];rt.request(p['training_jobs'],behavior,'EXACT_CURRENT_POST_BC_POLICY',seeds=p['training_seeds'],full=True,reference=sr);arrays=[E.N.S.arrays(r['path']) for r in sr];rt.role='REAL_RL' if p['arm']!='supervised' else 'MATCHED_SUPERVISED_CONTROL';u=L.rl_or_control(obj,arrays,data,pairs,p,lambda:rt.guard(True),event('live'));v=L.rl_or_control(shadow,arrays,data,pairs,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow));proposal=L.export(obj,p,folder/'proposal.pt.gz');assert L.frozen(p,proposal['path']).phash()==obj[0].phash()
   vv=rt.request(p['evaluation_jobs'],proposal,'PURE_STUDENT_DEVELOPMENT38');rs=[r for a,r in vv];rt.request(p['evaluation_jobs'],proposal,'EXACT_PURE_STUDENT_DEVELOPMENT38',reference=rs);check=assess(p,rs);accepted=accepts(p['before_check'],check)
   if not accepted:L.load(obj,before);L.load(shadow,before)
   result=dict(kind='round',BC_trace=b,update=u,behavior=behavior,proposal=proposal,training=G.R.M.tally(sr),check=check,accepted=accepted,model=proposal if accepted else p['initial'],retained_joint_hash=digest(L.state(obj)),live_Adam=len(b)+len(u['trace']),shadow_Adam=len(b2)+len(v['trace']),old275=None,wide6=None,wide8=None,fresh16=None,reload78=None)
  elif p['kind']=='engineering':
   rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['probe_jobs'],p['initial'],'ENGINEERING_SHORT_CAP',seeds=p['probe_seeds'],full=True);rs=[r for a,r in vv];rt.request(p['probe_jobs'],p['initial'],'ENGINEERING_EXACT',seeds=p['probe_seeds'],full=True,reference=rs)
   for a,r in vv:F.M.audit_full_history(p,r,a)
   result=dict(kind='engineering',new_Adam=0,new_RL=0,not_navigation_success=True)
  elif p['kind']=='final':
   rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['final_jobs'],p['initial'],'FINAL_SAME_MODEL407');rs=[r for a,r in vv];result=dict(kind='final',model=p['initial'],old275=G.R.M.tally(rs[:275]),wide6=G.R.M.tally(rs[275:329]),wide8=G.R.M.tally(rs[329:383]),field24=G.R.M.tally(rs[383:]),all407safe=all(r['success'] and not r['collision'] for r in rs),fresh16=None,reload78=None,full_stability_gate=False)
  else:raise AssertionError(p['kind'])
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for o in [obj,shadow]:L.load(o,before)
  save(folder/'failure.json',dict(error=error,rollback=digest(L.state(obj)) if obj else None,no_restart=True,all_raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(child):
 folder=OUT/child;p=plan(folder);setup(folder);verify(p,True);G.stopped(folder);assert not (folder/'audit.json').exists();s=read(folder/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl');by={r['path']:r for r in rows}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['full']:F.M.audit_full_history(p,r,a)
 for r in reps:
  old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
 r=s['result'];steps=0
 if p['kind']=='round':
  assert len(rows)==len(reps)==42;obj=L.make(p);before=L.state(obj);data,pairs=L.cached_dataset(p);events=[];bc=L.bc(obj,data,pairs,p,lambda:None,lambda a,b:events.append([a,b]));assert bc==r['BC_trace'] and digest(L.state(obj))==r['behavior']['joint_hash'];u=L.rl_or_control(obj,[E.N.S.arrays(x['path']) for x in rows[:4]],data,pairs,p,lambda:None,lambda a,b:events.append([a,b]));assert u==r['update'] and digest(L.state(obj))==r['proposal']['joint_hash'];steps=len(bc)+len(u['trace']);assert r['live_Adam']==r['shadow_Adam']==steps;assert G.rows(folder,'optimizer_calls.jsonl')==[[who,a,b] for who,ev in [('live',events[:2*len(bc)]),('shadow',events[:2*len(bc)]),('live',events[2*len(bc):]),('shadow',events[2*len(bc):])] for a,b in ev];assert assess(p,rows[4:])==r['check'];assert accepts(p['before_check'],r['check'])==r['accepted'];assert sha(r['behavior']['path'])==r['behavior']['sha256'] and sha(r['proposal']['path'])==r['proposal']['sha256'];assert L.frozen(p,r['proposal']['path']).phash()==obj[0].phash()
  if not r['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==r['retained_joint_hash']
 elif p['kind']=='engineering':assert len(rows)==len(reps)==2
 elif p['kind']=='final':assert len(rows)==407 and not reps
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==len(rows)+len(reps)+4<=p['max_commands']
 save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Adam=steps,all_live_shadow_and_rollback_rebuilt=True,full_training_history_hidden_density_reward_rebuilt=True,compact_eval_only_all_array_hash_and_integration_replayed=True,physics=s['physics'],commands=s['commands'],not_global_stability_certificate=True));print('T552_AUDIT_PASS',child,flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
