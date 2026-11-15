"""Own-state replay extraction and bounded whole-set native policy acceptance."""
from pathlib import Path
import sys,os,time,fcntl,traceback,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t576_nonlinear_plasticity_pair';DOC=ROOT/'stage/docs/day_20261115/t576'
sys.path.insert(0,str(HERE));import t576_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t569_frozen_complete_panel'));import t569_runtime as Old
Base,F,G,E,P,np,torch=Old.Base,Old.F,Old.G,Old.E,Old.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file() and not f.is_symlink())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration(arm):
 assert arm in ['linear','nonlinear'];p=read(OUT/'protocol.json');cfg=read(p['actor_configuration']);cfg.update(p);cfg.update(arm=arm,out=str(OUT/arm),max_physics=p['per_arm_max_physics'],max_commands=38,main_seconds=p['per_arm_main_seconds'],learning_seconds=p['per_arm_learning_seconds']);return cfg

def setup(folder):
 Old.OUT=folder;Old.HERE=HERE;Old.verify=verify;Old.setup();Base.ACTIVE=folder;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def worker(pipe,index,p,counts):
 G.T.Session=F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Old.Old.Prior.NORMAL;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(len(self.children))];assert all(v['state']=='READY' for v in self.ready);folder=Path(p['out']);save(folder/'worker_generations.json',(read(folder/'worker_generations.json') if (folder/'worker_generations.json').exists() else [])+[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],CPUQuota='200%',GPU=False)])
  except BaseException:self.close();raise

def assess(p,rows):
 assert len(rows)==32 and [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']];ids={r['job_id'] for r in rows if r['success'] and not r['collision'] and (not r['job_id'].startswith('t502_') or r['field_detail']['field_qualified_success'])};current=set(p['protected_success_ids']);reference=set(p['reference23_success_ids']);targets=set(p['restoration_target_ids']);lost=current-ids
 return dict(old8=G.R.M.tally(rows[:8]),field24=G.R.M.tally(rows[8:]),success_ids=sorted(ids),lost_current21=sorted(lost),restored_targets=sorted(targets&ids),missing_reference23=sorted(reference-ids),accepted=not lost and reference<=ids,complete32_safe=len(ids)==32,not_full407_or_unseen_qualification=True)
def event(w):return lambda s,k:Base.append('optimizer_calls.jsonl',[w,s,k])
def main(arm):
 p=configuration(arm);folder=Path(p['out']);setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);rt.commands=0;state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  data=L.dataset(p);obj=L.make(p);shadow=L.make(p);before=L.state(obj);rt.role='MATCHED_'+arm.upper()+'_HEAD_ADAM';u=L.fit(obj,data,p,lambda:rt.guard(True),event('live'));v=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow));rt.guard(True);model=L.export(obj,p,folder/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash();save(folder/'LEARNING_COMPLETE.json',dict(epoch=time.time(),learning_stop=p['absolute_learning_stop'],model=model))
  rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_COMPLETE32_'+arm);rows=[r for a,r in vv];keys=[i for i,j in enumerate(p['evaluation_jobs']) if j['id'] in p['key_replay_ids']];assert len(keys)==2;rt.request([p['evaluation_jobs'][i] for i in keys],model,'EXACT_KEY2_'+arm,reference=[rows[i] for i in keys]);check=assess(p,rows)
  if not check['accepted']:
   for o in [obj,shadow]:L.load(o,before)
  result=dict(fit=u,model=model,check=check,accepted=check['accepted'],retained_research_model=model if check['accepted'] else p['initial'],retained_joint_hash=digest(L.state(obj)),live_Adam=u['steps'],shadow_Adam=v['steps'],new_RL=0,new_Critic=0,old275=None,six54=None,eight54=None,fresh16=None,reload78=None,globally_promoted=False);state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for o in [obj,shadow]:L.load(o,before)
  save(folder/'failure.json',dict(error=error,no_restart=True,all_raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(arm):
 p=configuration(arm);folder=Path(p['out']);setup(folder);verify(p,True);G.stopped(folder);assert not (folder/'audit.json').exists();s=read(folder/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl');assert len(rows)==32 and len(reps)==2;by={r['path']:r for r in rows}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
 for r in reps:
  x=by[r['reference_path']];assert r['exact'] and r['array_hashes']==x['array_hashes'] and r['integration_sha256']==x['integration_sha256']
 obj=L.make(p);before=L.state(obj);u=L.fit(obj,L.dataset(p),p,lambda:None,lambda *_:None);r=s['result'];assert u==r['fit'] and digest(L.state(obj))==r['model']['joint_hash'] and sha(r['model']['path'])==r['model']['sha256'];assert assess(p,rows)==r['check']
 if not r['accepted']:L.load(obj,before)
 assert digest(L.state(obj))==r['retained_joint_hash'];assert s['physics']==sum(x['physics'] for x in rows+reps)<=p['max_physics'];assert s['commands']==38;assert read(folder/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop'];assert G.rows(folder,'optimizer_calls.jsonl')==[[who,phase,k] for who in ['live','shadow'] for k in range(p['fit_steps']) for phase in ['BEFORE','AFTER']]
 save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_RL=0,historical_discarded_Adam=p['fit_steps'],exact_key_native_replays=2,full32_independent_native_replay=False,actual_dataset_and_ActorAdam_rollback_rebuilt=True,physics=s['physics'],commands=s['commands']));print('T576_AUDIT_PASS',arm,flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
