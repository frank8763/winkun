"""Frozen matched-model attribution and one-step execution perturbations.

No learning, optimizer, teacher or candidate selection. All histories are
physical; perturbations are explicit off-policy interventions, never PPO data.
"""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback,copy
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t551_retention_recovery_probe';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'));import t550_runtime as Old
L,Base,F,G,E,P,np,torch=Old.L,Old.Base,Old.F,Old.G,Old.E,Old.P,Old.np,Old.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
NORMAL=Old.NORMAL

def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file() and not f.is_symlink())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration(root):
 p=read(root['actor_configuration']);p.update(root);p['out']=str(OUT);return p

def setup():
 Base.OUT=OUT;Base.HERE=HERE;Base.verify=verify;Base.setup(OUT);Base.ACTIVE=OUT;F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify;G.B.ACTIVE=OUT;G.B.verify=verify;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def perturbed_command(command,sign):
 assert sign in [-1,1];u=command.clone();assert u.shape==(1,2) and abs(float(u[0,1]))<=.9+1e-6;u[0,1]+=sign*.1;assert float(u.abs().max())<=1 and torch.equal(u[:,0],command[:,0]);return u
class PulseSession(F.FieldSession):
 def __init__(self,job,*args,**kwargs):
  super().__init__(job,*args,**kwargs);self.pulse_hold=int(job['pulse_hold']);self.pulse_sign=int(job['pulse_sign']);assert self.pulse_hold>0
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0 and actor_active;use=len(self.rows)==self.pulse_hold;original=self.q.step
  if use:
   def intervention(*args,**kwargs):
    result=list(original(*args,**kwargs));result[0]=perturbed_command(result[0],self.pulse_sign);return tuple(result)
   self.q.step=intervention
  try:return super().step(owner=int(use),actor_active=True)
  finally:
   if use:self.q.step=original

def payload(session,tape=None):
 a=NORMAL(session,tape);a['intervention_hold']=np.asarray(session.pulse_hold,np.int64);a['intervention_sign']=np.asarray(session.pulse_sign,np.int8);return a

def worker(pipe,index,p,counts):
 G.T.Session=PulseSession if p.get('pulse') else F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload if p.get('pulse') else NORMAL;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready);f=OUT/'worker_generations.json';old=read(f) if f.exists() else [];old.append(dict(parent=os.getpid(),workers=[x.pid for x in self.children],epoch=time.time(),pulse=bool(p.get('pulse')),CPU=[14,15],GPU=False));save(f,old)
  except BaseException:self.close();raise
 def receive(self,i):
  v=super().receive(i)
  if isinstance(v,tuple):v[1].update(teacher=False,controlled_intervention=bool(self.p.get('pulse')),not_on_policy_training=bool(self.p.get('pulse')))
  return v

def audit_pulse(p,r,a,baseline):
 q=L.frozen(p,r['request']['model']);q.sample=False;q.reset_noise();h=None;k=int(a['intervention_hold']);sign=int(a['intervention_sign']);n=len(a['states']);assert 0<k<n and not a['stochastic'].any();owner=np.zeros(n,np.uint8);owner[k]=1;assert np.array_equal(owner,a['ownership']);assert r['controlled_intervention'] and not r['teacher']
 for name in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times','hidden_before','locations']:assert np.array_equal(a[name][:k],baseline[name][:k]),name
 for name in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[name][k],baseline[name][k]),name
 assert np.array_equal(a['initial_native_integration'],baseline['initial_native_integration'])
 with torch.no_grad():
  for i,x in enumerate(a['states']):
   hb=np.zeros(64) if h is None else h[0,0].numpy().copy();u,loc,h,_=q.step(torch.tensor(x[None]),h);assert np.array_equal(hb,a['hidden_before'][i])
   for key,value in q.tape[-1].items():assert np.array_equal(value,a[key][i]),(i,key)
   command=perturbed_command(u,sign) if i==k else u;assert np.array_equal(command[0].numpy(),a['actions'][i]);assert np.abs(E.N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
   count=int(a['native_counts'][i]);z=E.N.K.R.GAMMA**count;flags=a['end_flags'][i];d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy']);reward=-.000002*(1-z)/(1-E.N.K.R.GAMMA)+.03*d0-z*.03*d1*(not flags.any())
   if flags.any():reward+=E.N.K.R.GAMMA**(count-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
   assert abs(reward-a['rewards'][i])<1e-12
 assert np.array_equal(h[0,0].numpy(),a['last_hidden_after'])

def compare(p,rows):
 by={name:{r['job_id']:r for r in rows if r['role']=='FROZEN_'+name} for name in p['models']};lost=p['lost6_ids'];out=[]
 for ident in [j['id'] for j in p['jobs']]:
  entry=dict(id=ident,models={})
  for name in p['models']:
   r=by[name][ident];a=E.N.S.arrays(r['path']);entry['models'][name]=dict(success=bool(r['success'] and not r['collision']),contact=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],first_action=a['actions'][0].tolist(),seconds=r['physics']*.002,initial_native_integration_sha256=r['array_hashes']['initial_native_integration'],path=r['path'])
  assert len({v['initial_native_integration_sha256'] for v in entry['models'].values()})==1
  entry['first_known_loss_stage']='ALREADY_LOST_BY_T549' if entry['models']['T507']['success'] and not entry['models']['T549']['success'] else 'BETWEEN_T549_AND_T550' if entry['models']['T549']['success'] and not entry['models']['T550']['success'] else 'NO_T549_TO_T550_SUCCESS_LOSS'
  out.append(entry)
 pulse=[]
 for r in rows:
  if r['role']=='CONTROLLED_PULSE_NOT_POLICY_SAMPLE':
   a=E.N.S.arrays(r['path']);pulse.append(dict(id=r['job_id'],baseline_id=r['request']['job']['baseline_id'],hold=int(a['intervention_hold']),sign=int(a['intervention_sign']),success=bool(r['success'] and not r['collision']),contact=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],seconds=r['physics']*.002))
 return dict(matched8=out,pulse4=pulse,new_Adam=0,new_RL=0,old275=None,wide6_54=None,wide8_54=None,fresh16=None,cross_site_NOT_tested=True,robustness_only4_controlled_development_probes=True,no_candidate_selected=True)

def main():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None
 try:
  for index,(name,model) in enumerate(p['models'].items()):
   if index:rt.commands+=2
   rt.pool=Pool(dict(p,phase='student',pulse=False,native_stop_epoch=rt.deadline),rt.counts,rt.guard);values=rt.request(p['jobs'],model,'FROZEN_'+name,full=True);rr=[r for a,r in values]
   for row in rr:
    ref=p['historical_rows'].get(name,{}).get(row['job_id'])
    if ref:assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256'],(name,row['job_id'],'HISTORICAL_RECONSTRUCTION_DIFFERENCE')
   rt.request(p['jobs'],model,'EXACT_'+name,full=True,reference=rr);rt.pool.close();rt.pool=None;rt.commands+=2
  canonical={r['job_id']:r for r in rt.rows if r['role']=='FROZEN_T550'};rt.commands+=2;rt.pool=Pool(dict(p,phase='student',pulse=True,native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['pulse_jobs'],p['models']['T550'],'CONTROLLED_PULSE_NOT_POLICY_SAMPLE',full=True)
  for a,row in vv:audit_pulse(p,row,a,E.N.S.arrays(canonical[row['request']['job']['baseline_id']]['path']))
  rt.request(p['pulse_jobs'],p['models']['T550'],'EXACT_CONTROLLED_PULSE',full=True,reference=[r for a,r in vv]);result=compare(p,rt.rows);save(OUT/'comparison.json',result);state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==28;by={r['path']:r for r in rows};canonical={r['job_id']:r for r in rows if r['role']=='FROZEN_T550'}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['controlled_intervention']:audit_pulse(p,r,a,E.N.S.arrays(canonical[r['request']['job']['baseline_id']]['path']))
  else:F.M.audit_full_history(p,r,a)
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert compare(p,rows)==read(OUT/'comparison.json')==s['result'];assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==72<=p['max_commands']
 for m in p['models'].values():assert sha(m['path'])==m['sha256']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,all28_full_histories_and28_native_replays_exact=True,controlled_intervention_not_teacher_not_onpolicy=True,all_models_unchanged=True,physics=s['physics'],commands=s['commands'],no_candidate_selected=True));print('T551_AUDIT_PASS')
if __name__=='__main__':globals()[sys.argv[1]]()
