"""Fixed same-state temporal intervention diagnostic. Zero learning/planning."""
import os
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU','0'))})
from pathlib import Path
import sys,json,time,hashlib,math,multiprocessing as mp,fcntl,shutil,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t205_temporal_intervention'
sys.path.insert(0,str(ROOT/'stage/experiments/t200_residual_placement'))
import common as C
import numpy as np
import torch
import mujoco
GAMMA=math.exp(-.002/60)
def digest(*arrays):
 h=hashlib.sha256()
 for a in arrays:h.update(np.asarray(a).tobytes())
 return h.hexdigest()
def integration(env):
 v=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(env.model,env.data,v,mujoco.mjtState.mjSTATE_INTEGRATION);return v
def contacts(env):
 rows=[]
 for c in env.data.contact[:env.data.ncon]:
  a,b=int(c.geom1),int(c.geom2);ra=int(env.model.geom_bodyid[a]) in env.robot_body_ids;rb=int(env.model.geom_bodyid[b]) in env.robot_body_ids
  if ra==rb or float(c.dist)>.002:continue
  other=b if ra else a;name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,other) or ''
  if any(x in name.lower() for x in ['floor','ground','plane']):continue
  rows.append({'robot_geom':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,a if ra else b),'other_geom':name,'distance_m':float(c.dist)})
 return rows

def simulate(create,native,p,plan,job,deadline,source=False):
 env=None
 try:
  env,obs,actual=create(plan['case']);env.max_steps=plan['case']['step_cap'];cap=env.max_steps
  if job['start']:
   snap=C.read_snapshot(OUT/f'anchor_{job["start"]}.npz');obs=C.restore(env,snap);assert np.array_equal(integration(env),snap['physics'])
  assert env.success_radius==.2 and env.frame_skip==1 and abs(env.model.opt.timestep-.002)<1e-12
  steps=job['start'];initial=float(env._goal_state()[1]);nearest=initial;goal=env.goal_xy.copy();model=C.model_hash(env);term=trunc=False;info={};records=[];snapshots=[];source_futures={}
  addresses=[int(env.model.jnt_qposadr[int(env.model.actuator_trnid[getattr(env,n),0])]) for n in ['id_lf_steer','id_rf_steer']]
  while not(term or trunc) and steps<cap and time.monotonic()<deadline:
   if source and steps in plan['start_steps']:
    file=OUT/f'anchor_{steps}.npz';assert not file.exists();C.write_snapshot(file,C.capture(env,obs));snapshots.append({'steps':steps,'path':str(file),'sha256':C.sha(file),'pose':list(env._pose()),'remaining_m':float(env._goal_state()[1])})
   before=steps;state=C.observation(obs,1-steps/cap);pose=np.asarray(env._pose());d0=float(env._goal_state()[1]);steer_before=np.asarray(env.data.qpos[addresses]).copy()
   with torch.no_grad():base=p(torch.tensor(state[None])).numpy()[0].astype(np.float32)
   action=base.copy();active=steps-job['start']<job['hold_steps']
   if active:
    action[1]=np.clip(action[1]+job['steer_delta'],-1.,1.)
    if job['throttle_cap'] is not None:action[0]=min(action[0],job['throttle_cap'])
   for _ in range(min(100,cap-steps)):
    obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
    if term or trunc:break
   n=steps-before;discount=GAMMA**n;d1=float(env._goal_state()[1]);terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
   reward=-.000002*(1-discount)/(1-GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(GAMMA**(n-1)*event if terminal else 0)
   records.append((before,state,base,action.copy(),n,C.observation(obs,1-steps/cap),pose,np.asarray(env._pose()),[info['collision'],info['reach_goal'],trunc],reward,active,steer_before,np.asarray(env.data.qpos[addresses]).copy()))
  complete=bool(term or trunc);assert model==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
  keys=['steps','states','base_actions','actions','native_counts','next_states','poses','next_poses','end_flags','rewards','intervention_active','steer_before','steer_after'];data={k:np.asarray([r[i] for r in records]) for i,k in enumerate(keys)}
  data['goal_xy']=goal;data['final_integration']=integration(env)
  path=OUT/'traces'/(job['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();np.savez_compressed(path,**data);assert path.stat().st_size<320*1024
  reward_sum=float(np.sum(GAMMA**(data['steps']-job['start'])*data['rewards'])) if complete else None
  if complete:
   event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc);n=steps-job['start'];expected=-.000002*(1-GAMMA**n)/(1-GAMMA)+.03*initial+GAMMA**(n-1)*event;assert abs(reward_sum-expected)<1e-8
  intervention_rows=data['intervention_active'].astype(bool);change=data['actions']-data['base_actions']
  row={**job,'complete':complete,'success':bool(info.get('reach_goal',False)),'collision':bool(info.get('collision',False)),'timeout':bool(trunc),'physics_steps':steps-job['start'],'absolute_end_steps':steps,'initial_goal_distance_m':initial,'nearest_m':nearest,'remaining_m':float(env._goal_state()[1]),'final_pose':list(env._pose()),'contacts':contacts(env) if info.get('collision') else [],'environment_return':reward_sum,'actual_decision_count':len(records),'intervention_decisions':int(intervention_rows.sum()),'intervention_action_change_RMS':np.sqrt(np.mean(change[intervention_rows]**2,axis=0)).tolist() if intervention_rows.any() else [0.,0.],'final_integration_sha256':digest(data['final_integration']),'path':str(path),'sha256':C.sha(path),'snapshots':snapshots,'training_physics_steps':0,'optimizer_updates':0}
  return row
 finally:
  if env is not None:env.close()

def worker(pipe,index,plan,deadline):
 folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('w',buffering=1);sys.stdout=sys.stderr=log
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=C.OriginalPolicy().eval().requires_grad_(False);p.load_state_dict(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']);native,create=C.native_setup(folder)
  while True:
   job=pipe.recv()
   if job is None:break
   pipe.send(simulate(create,native,p,plan,job,deadline))
 except EOFError:pass
 except BaseException:
  try:pipe.send({'error':traceback.format_exc()})
  except OSError:pass
 finally:pipe.close();log.close()

def main():
 lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate diagnostic'
 plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];spent=0;results=[];pipes=[];workers=[];inflight=0;reason=None
 for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
 def used(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
 def status(state):C.save(OUT/'status.json',{'state':state,'physics_steps':spent,'completed_cases':len(results),'training_physics_steps':0,'optimizer_updates':0,'elapsed_s':time.monotonic()-start})
 def guard(need):
  assert spent+need<=plan['max_physics'],'Physics reservation'
  assert used(ROOT/'stage/runs')+plan['pending_bytes']<1024**3 and used(OUT)+plan['pending_bytes']<plan['peak_bytes']
  assert shutil.disk_usage(ROOT).free>40*1024**3
 def batch(jobs):
  nonlocal spent,inflight
  rr=[]
  for offset in range(0,len(jobs),plan['workers']):
   if deadline-time.monotonic()<plan['batch_reserve_seconds']:return rr,False
   sub=jobs[offset:offset+plan['workers']];need=sum(plan['case']['step_cap']-j['start'] for j in sub);guard(need);inflight=need
   for pipe,j in zip(pipes,sub):pipe.send(j)
   replies=[]
   for pipe in pipes[:len(sub)]:
    assert pipe.poll(max(.1,min(120,deadline-time.monotonic()+3))),'Worker deadline';r=pipe.recv();assert 'error' not in r,r;replies.append(r)
   spent+=sum(r['physics_steps'] for r in replies);inflight=0
   for r in replies:
    with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   results.extend(replies);rr.extend(replies);status('DIAGNOSING')
   if not all(r['complete'] for r in replies):return rr,False
  return rr,True
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=C.OriginalPolicy().eval().requires_grad_(False);weights=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p.load_state_dict(weights)
  native,create=C.native_setup(OUT/'source_native');status('SOURCE_REPRODUCTION');guard(plan['case']['step_cap'])
  job={'id':'source','start':0,'hold_steps':0,'throttle_cap':None,'steer_delta':0.,'kind':'source'};source=simulate(create,native,p,plan,job,deadline,source=True);spent+=source['physics_steps'];C.save(OUT/'source.json',source)
  assert source['complete'] and source['collision'] and source['absolute_end_steps']==plan['baseline_expected_steps']
  with np.load(source['path']) as z, np.load(plan['baseline_reference']) as ref:
   assert all(np.array_equal(z[k],ref[k]) for k in ['steps','states','actions','poses'])
  assert abs(source['remaining_m']-plan['baseline_expected_remaining_m'])<1e-12
  ctx=mp.get_context('spawn')
  for i in range(plan['workers']):
   a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,plan,deadline));w.start();b.close();pipes.append(a);workers.append(w)
  os.environ['STAGE_PROCESS_CPU']='0';C.save(OUT/'workers.json',{'parent':os.getpid(),'workers':[w.pid for w in workers]})
  controls,complete=batch([j for j in plan['probes'] if j['kind']=='baseline']);assert complete and len(controls)==2
  gates=[]
  for r in controls:
   with np.load(source['path']) as z,np.load(r['path']) as sub:
    mask=z['steps']>=r['start'];fields=['steps','states','base_actions','actions','native_counts','next_states','poses','next_poses','end_flags','rewards','steer_before','steer_after'];assert all(np.array_equal(z[k][mask],sub[k]) for k in fields);assert np.array_equal(z['final_integration'],sub['final_integration'])
   gates.append({'start':r['start'],'suffix_all_fields_exact':True,'final_integration_exact':True})
  C.save(OUT/'preflight.json',{'state':'PASS','original_trace_exact':True,'controls':gates,'physics_steps_so_far':spent,'training_physics_steps':0,'optimizer_updates':0})
  interventions,complete=batch([j for j in plan['probes'] if j['kind']=='intervention']);reason='COMPLETE_FIXED_GRID' if complete else 'WALL_RESERVE_OR_CENSORING'
  assert all(torch.equal(v,p.state_dict()[k]) for k,v in weights.items())
 except BaseException:
  reason='FAILED';C.save(OUT/'failure.json',{'state':'FAILED','physics_steps':spent,'unrecorded_inflight_physics_upper_bound':inflight,'training_physics_steps':0,'optimizer_updates':0,'traceback':traceback.format_exc()});raise
 finally:
  for pipe in pipes:
   try:pipe.send(None)
   except OSError:pass
  for w in workers:
   w.join(3)
   if w.is_alive():w.terminate();w.join(3)
  if reason!='FAILED':
   for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
   rr=[r for r in results if r['kind']=='intervention'];groups=[]
   for hold in [100,2000]:
    subset=[r for r in rr if r['hold_steps']==hold];groups.append({'hold_steps':hold,'n':len(subset),'safe_GOAL':sum(r['complete'] and r['success'] and not r['collision'] for r in subset),'CONTACT':sum(r['collision'] for r in subset),'TIMEOUT':sum(r['timeout'] for r in subset),'partial':sum(not r['complete'] for r in subset)})
   summary={'state':'COMPLETE','stop_reason':reason,'physics_steps':spent,'training_physics_steps':0,'optimizer_updates':0,'cases':len(results),'groups':groups,'intervention_successes':[r for r in rr if r['complete'] and r['success'] and not r['collision']],'all_workers_stopped':True,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),'new_runs_bytes':used(ROOT/'stage/runs'),'automatic_training_or_promotion':False,'scope':plan['contract']}
   assert spent<=plan['max_physics'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3
   C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
