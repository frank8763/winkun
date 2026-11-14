"""Complete the missing19 for exactly the frozen T546 candidate; zero fitting."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t547_frozen_complete_panel';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(ROOT/'stage/experiments/t546_functional_residual'));import t546_runtime as Previous
Base,F,G,E,P,np,torch,A=Previous.Base,Previous.F,Previous.G,Previous.E,Previous.P,Previous.np,Previous.torch,Previous.A
read,save,sha,digest=G.read,G.save,G.sha,G.digest

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
 Base.setup(OUT);F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify;P.frozen=A.frozen
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def worker(pipe,index,p,counts):
 G.T.Session=F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=A.frozen;G.cpu_worker(pipe,index,p,counts)

class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
   save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise

def cached(p):
 rows=G.rows(Path(p['cached_folder']));reps=G.rows(Path(p['cached_folder']),'replays.jsonl');assert len(rows)==len(reps)==13 and read(Path(p['cached_folder'])/'audit.json')['state']=='PASS'
 for row,rep in zip(rows,reps):
  assert row['policy_hash']==p['initial']['policy_hash'] and row['loaded_model_sha256']==p['initial']['sha256'];assert sha(row['path'])==row['sha256'];assert rep['reference_path']==row['path'] and rep['exact'] and rep['array_hashes']==row['array_hashes'] and rep['integration_sha256']==row['integration_sha256'];assert row['success'] and not row['collision']
 return rows

def combine(p,new):
 rows=cached(p)+new;assert len(rows)==32 and len({r['job_id'] for r in rows})==32
 by={r['job_id']:r for r in rows};ordered=[by[j['id']] for j in p['evaluation_jobs']]
 assert all(r['policy_hash']==p['initial']['policy_hash'] for r in rows)
 for j,r in zip(p['evaluation_jobs'],ordered):assert r['request']['job']==j
 check=Base.assess(p,ordered);current=set(check['field_success_ids']);before=set(p['protected_field_ids']);assert before<=current
 check.update(lost_prior_field_ids=sorted(before-current),known13_same_model=True,old8_safe=check['old8']['success']==8,field24_complete=True,all32_safe=check['old8']['success']==8 and len(current)==24,research_nonregression_pass=check['old8']['success']==8 and before<=current and check['field24']['contact']<=9)
 return check,ordered

def main():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();cached(p);rt=Base.Runtime(p);error=None;state='FAILED';result=None
 try:
  q=A.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash'];del q
  rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  old=rt.request(p['missing_old_jobs'],p['initial'],'FROZEN_MISSING_OLD7');field=rt.request(p['missing_field_jobs'],p['initial'],'FROZEN_MISSING_FIELD12_FULL',full=True)
  first=[r for a,r in old+field];rt.request(p['missing_old_jobs'],p['initial'],'EXACT_MISSING_OLD7',reference=first[:7]);rt.request(p['missing_field_jobs'],p['initial'],'EXACT_MISSING_FIELD12_FULL',full=True,reference=first[7:])
  check,ordered=combine(p,first);save(OUT/'combined_development32.json',dict(policy=p['initial'],cached13_not_independent_new_samples=[r['path'] for r in cached(p)],new19=[r['path'] for r in first],ordered_rows=ordered,check=check));assert sha(p['initial']['path'])==p['initial']['sha256']
  result=dict(reason='COMPLETE_FROZEN32_ALL_SAFE' if check['all32_safe'] else 'COMPLETE_FROZEN32_REMAINING_FAILURES',check=check,new_native_cases=19,reused_same_hash_cases=13,new_exact_replays=19,new_Actor_Adam=0,new_RL=0,new_Value=0,new_fit=0,weights_immutable=True,accepted=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None);state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();G.stopped(OUT);rows=G.rows(OUT);rep=G.rows(OUT,'replays.jsonl');assert len(rows)==len(rep)==19;by={r['path']:r for r in rows}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert r['policy_hash']==p['initial']['policy_hash']
  if r['full']:F.M.audit_full_history(p,r,a)
 for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 check,ordered=combine(p,rows);assert check==s['result']['check'] and ordered==read(OUT/'combined_development32.json')['ordered_rows'];assert sha(p['initial']['path'])==p['initial']['sha256']
 assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==42<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_Adam=0,new_fit=0,frozen_policy_hash_verified=True,cached13_same_model_and_exact_replays_verified=True,new19_and19exact_verified=True,all12_new_full_history_hidden_density_reward_verified=True,compact7_full120_not_persisted=True,physics=s['physics'],commands=s['commands'],accepted=False));print('T547_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
