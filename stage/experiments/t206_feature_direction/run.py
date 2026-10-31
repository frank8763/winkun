"""Fixed neural readout sensitivity diagnostic, single network all episode."""
import os
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU','0'))})
from pathlib import Path
import sys,json,time,shutil,fcntl,multiprocessing as mp,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t206_feature_direction'
sys.path.insert(0,str(ROOT/'stage/experiments/t203_native_exploration'))
from runtime import *
class BudgetStop(Exception):pass

def main():
 lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate T206'
 plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline_unix=time.time()+plan['wall_seconds'];deadline=start+plan['wall_seconds'];spent=episodes=0;results=[];pipes=[];workers=[];tasks=[];candidates=[];selected=None;reason=None;inflight=0;final=False;replays=[];saved_gate=[];robust=[];invalid=[];stable=False
 def write(name,obj):C.save(OUT/name,obj)
 def append(name,obj):
  with (OUT/name).open('a') as f:f.write(json.dumps(obj)+'\n')
 def used(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
 def safe(r):return r['complete'] and r['success'] and not r['collision']
 def status(state):write('status.json',{'state':state,'physics_steps':spent,'episodes':episodes,'evaluated_candidates':len(candidates),'selected':selected,'elapsed_s':time.monotonic()-start,'training_physics_steps':0,'optimizer_updates':0})
 def guard(n=0,physics=0):
  if time.monotonic()+(0 if final else plan['final_reserve_seconds'])>=deadline:raise BudgetStop('WALL_RESERVE')
  if spent+physics>plan['max_physics'] or episodes+n>plan['max_episodes']:raise BudgetStop('PHYSICS_OR_EPISODE_RESERVE')
  if used(OUT)+plan['pending_bytes']>=plan['final_bytes'] or used(ROOT/'stage/runs')+plan['pending_bytes']>=1024**3 or shutil.disk_usage(ROOT).free<=40*1024**3:raise BudgetStop('STORAGE_RESERVE')
 def receive(pipe):
  if not pipe.poll(max(.1,min(120,deadline-time.monotonic()+2))):raise BudgetStop('WORKER_RESPONSE_TIMEOUT')
  r=pipe.recv();assert 'error' not in r,r;return r
 def task(spec):
  guard();pipes[0].send(('make_task',spec));r=receive(pipes[0]);append('tasks.jsonl',{**spec,**r});return {**r,'case':spec['case']}
 def job(t,path,id,retain=False):return {'id':id,'task':t['id'],'case':t['case'],'exact_snapshot':t['snapshot'],'heads':str(path),'seed':0,'mode':'deterministic','summary_only':not retain,'role':'frozen_neural_policy_probe','initial_pose':t['pose']}
 def batch(jobs):
  nonlocal spent,episodes,inflight
  rr=[]
  for offset in range(0,len(jobs),plan['workers']):
   sub=jobs[offset:offset+plan['workers']];need=sum(j['case']['step_cap'] for j in sub);guard(len(sub),need);inflight=need
   for pipe,j in zip(pipes,sub):pipe.send(('rollout',j))
   rows=[receive(pipe) for pipe in pipes[:len(sub)]];spent+=sum(r['physics_steps'] for r in rows);episodes+=len(rows);inflight=0
   for r in rows:append('rollouts.jsonl',r)
   results.extend(rows);rr.extend(rows);status('EVALUATING')
   if not all(r['complete'] for r in rows):raise BudgetStop('WALL_CENSORED_EPISODES')
  return rr
 try:
  for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);parent=Policy().eval().requires_grad_(False);weights=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];parent.load_state_dict(weights);(OUT/'heads').mkdir();parent_heads=OUT/'heads/parent.npz';save_heads(parent_heads,parent)
  with np.load(plan['failure_trace']) as z:x=torch.tensor(z['states']);steps=z['steps']
  with np.load(plan['retention_targets']) as z:old=torch.tensor(z['states']);targets=torch.tensor(z['actions']);offsets=z['offsets']
  with torch.no_grad():
   f=torch.cat([parent.body.net[:4](x),torch.ones((len(x),1))],1).double();g=torch.cat([parent.body.net[:4](old),torch.ones((len(old),1))],1).double();assert torch.equal(parent(old),targets)
  lo,hi=plan['failure_step_window'];focus=f[(steps>=lo)&(steps<=hi)];A=focus.T@focus/len(focus);B=sum(g[offsets[j]:offsets[j+1]].T@g[offsets[j]:offsets[j+1]]/(offsets[j+1]-offsets[j]) for j in range(7))/7;ridge=float(B.trace()/len(B))*plan['ridge_multiplier'];v,u=torch.linalg.eigh(B+ridge*torch.eye(len(B),dtype=torch.float64));W=(u*v.rsqrt())@u.T;ev,vec=torch.linalg.eigh(W@A@W);direction=W@vec[:,-1];direction/=torch.sqrt(torch.mean((focus@direction)**2))
  if float((focus@direction).mean())<0:direction=-direction
  assert torch.isfinite(direction).all();np.savez_compressed(OUT/'direction.npz',direction=direction.numpy(),failure_projection=(f@direction).numpy(),retention_projection=(g@direction).numpy(),eigenvalues=ev.numpy())
  write('offline_direction.json',{'state':'PASS','feature_dim':len(direction),'generalized_eigenvalue':float(ev[-1]),'failure_RMS':float(torch.sqrt(torch.mean((focus@direction)**2))),'retention_native_RMS':float(torch.sqrt(direction@B@direction)),'ridge':ridge,'direction_sha256':C.sha(OUT/'direction.npz'),'optimizer_updates':0,'physics_steps':0})
  proposals=[]
  for spec in plan['grid']:
   p=Policy().eval().requires_grad_(False);p.load_state_dict(weights)
   with torch.no_grad():
    for channel,key in enumerate(['throttle','steer']):p.direct_head.weight[channel].add_(float(spec[key])*direction[:-1].float());p.direct_head.bias[channel].add_(float(spec[key])*direction[-1].float())
    errors=(p(old)-targets).square().mean(1);rmse=float(torch.sqrt(sum(errors[offsets[j]:offsets[j+1]].mean() for j in range(7))/7))
   assert all(torch.equal(v,p.state_dict()[k]) for k,v in weights.items() if k not in ['direct_head.weight','direct_head.bias'])
   path=OUT/'heads'/(spec['id']+'.npz');save_heads(path,p);row={**spec,'heads':str(path),'offline_old7_RMSE':rmse,'offline_gate':rmse<=plan['max_offline_old7_RMSE']};append('proposals.jsonl',row)
   if row['offline_gate']:proposals.append(row)
  ctx=mp.get_context('spawn')
  for i in range(plan['workers']):
   a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,str(OUT),plan['checkpoint'],deadline_unix));w.start();b.close();pipes.append(a);workers.append(w)
  os.environ['STAGE_PROCESS_CPU']='0';write('workers.json',{'parent':os.getpid(),'workers':[w.pid for w in workers]});status('PARENT_GATES')
  for case in plan['primary_cases']:
   t=task({'id':'original_'+case['id'],'case':case});assert t['valid'];tasks.append(t)
  original=batch([job(t,parent_heads,f'parent_{i}') for i,t in enumerate(tasks)]);assert all(safe(original[i]) for i in [0,1,2,3,4,5,7]) and original[6]['collision']
  tested=batch([job(tasks[6],r['heads'],r['id']+'_LEFT',True) for r in proposals])
  for spec,r in zip(proposals,tested):
   row={**spec,'LEFT':r,'seven_gate':[],'eligible':False}
   if safe(r):
    gate=batch([job(tasks[i],spec['heads'],spec['id']+f'_old{i}') for i in [0,1,2,3,4,5,7]]);row['seven_gate']=gate;row['eligible']=all(safe(q) for q in gate)
   candidates.append(row);append('candidates.jsonl',row)
  eligible=[r for r in candidates if r['eligible']]
  if eligible:
   final=True;pick=min(eligible,key=lambda r:(abs(r['steer'])+abs(r['throttle']),abs(r['steer']),r['id']));p=Policy().eval();p.load_state_dict(weights);set_heads(p,read_heads(pick['heads']));model=OUT/'selected_policy.pt';torch.save({'policy':p.state_dict(),'protocol_sha256':C.sha(HERE/'protocol.json'),'parent_sha256':C.sha(plan['checkpoint']),'selection':pick['id'],'origin':'Frozen low-rank neural parameter sensitivity; zero optimizer updates','automatic_promotion':False},model)
   reload=Policy().eval();reload.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy']);assert all(torch.equal(v,reload.state_dict()[k]) for k,v in p.state_dict().items());reloaded=OUT/'heads/saved_reload.npz';save_heads(reloaded,reload);selected={'id':pick['id'],'model':str(model),'sha256':C.sha(model)}
   guard(physics=pick['LEFT']['physics_steps']);pipes[0].send(('replay',pick['LEFT']));check=receive(pipes[0]);spent+=check['physics_steps'];replays.append(check);write('replay_checks.json',replays);assert check['verified']
   saved_gate=batch([job(t,reloaded,f'saved_original_{i}',i==6) for i,t in enumerate(tasks)])
   perturbed=[]
   for i in [6,7]:
    for dx in plan['robustness_xy']:
     for dy in plan['robustness_xy']:
      for deg in plan['robustness_yaw']:
       t=task({'id':f'grid_{i}_{len(perturbed)+len(invalid)}','case':tasks[i]['case'],'near_snapshot':tasks[i]['snapshot'],'delta':[dx,dy,deg]})
       if t['valid']:perturbed.append(t)
       else:invalid.append(t)
   robust=batch([job(t,reloaded,f'saved_grid_{i}') for i,t in enumerate(perturbed)]);stable=len(saved_gate)==8 and all(safe(r) for r in saved_gate) and len(robust)==54 and not invalid and all(safe(r) for r in robust)
   reason='FIXED_GRID_COMPLETE_SELECTED'
  else:reason='NO_OLD7_PRESERVING_LEFT_GOAL'
 except BudgetStop as error:reason=str(error)
 except BaseException:
  reason='FAILED';write('failure.json',{'state':'FAILED','physics_steps':spent,'unrecorded_inflight_physics_upper_bound':inflight,'optimizer_updates':0,'traceback':traceback.format_exc()});raise
 finally:
  for pipe in pipes:
   try:pipe.send(None)
   except OSError:pass
  for w in workers:
   w.join(3)
   if w.is_alive():w.terminate();w.join(3)
  if reason!='FAILED':
   for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
   summary={'state':'COMPLETE','stop_reason':reason,'physics_steps':spent,'episodes':episodes,'training_physics_steps':0,'optimizer_updates':0,'candidates':len(candidates),'LEFT_safe_GOAL':sum(safe(r['LEFT']) for r in candidates),'old7_preserving_LEFT_GOAL':sum(r['eligible'] for r in candidates),'selected':selected,'saved_old8_completed':len(saved_gate),'saved_old8_safe_GOAL':sum(safe(r) for r in saved_gate),'robustness_completed':len(robust),'robustness_safe_GOAL':sum(safe(r) for r in robust),'invalid_robustness':len(invalid),'stable_5m_gate_passed':stable,'all_workers_stopped':True,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),'all_new_runs_bytes':used(ROOT/'stage/runs'),'automatic_promotion':False,'scope':plan['acceptance_contract']}
   assert used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3 and spent<=plan['max_physics'];write('summary.json',summary);write('status.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
