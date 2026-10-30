"""Frozen full-state success-anchor reproduction and fixed perturbation grid."""
from pathlib import Path
import os
CPU_POOL=[int(v) for v in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL))
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import sys,json,time,math,hashlib,fcntl,shutil,traceback,multiprocessing as mp
from multiprocessing.connection import wait
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t186_anchor_robustness'
sys.path.insert(0,str(ROOT/'stage/experiments/t177_frontier_onpolicy'))
import common as C

def physics(env):
 import mujoco
 a=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
 mujoco.mj_getState(env.model,env.data,a,mujoco.mjtState.mjSTATE_INTEGRATION);return a

def state_hash(env,obs):return hashlib.sha256(physics(env).tobytes()+obs.tobytes()+env.goal_xy.tobytes()).hexdigest()

def policy_load(plan):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 p=C.Policy().eval().requires_grad_(False)
 p.load_state_dict(torch.load(plan['checkpoint'],weights_only=False,map_location='cpu')['policy']);return p

def command(policy,obs,elapsed,cap):
 with torch.no_grad():a=policy(torch.tensor(C.observation(obs,1-elapsed/cap).reshape(1,-1))).numpy()[0]
 return C.action_clip(a,'full_range')

def init_pose(env,source,perturb=None):
 import mujoco
 task=source['reset_task'];assert np.array_equal(env.goal_xy,np.asarray(task['goal_xy']))
 env.data.qpos[:2]=task['start_pose'][:2];env.data.qpos[3:7]=task['quaternion'];env.data.qvel[:]=0.;env.data.ctrl[:]=0.
 mujoco.mj_forward(env.model,env.data);env.prev_action=np.zeros(2,dtype=np.float32);env.step_count=0;env.history.clear();env.prev_goal_dist=float(env._goal_state()[1])
 return np.asarray(env._observation(update_history=False),dtype=np.float32)

def perturb(env,obs,job):
 import mujoco
 dx,dy,yaw_delta=job['delta_x_m'],job['delta_y_m'],job['delta_yaw_deg']
 if dx==dy==yaw_delta==0:return obs
 yaw=float(env._pose()[2])+math.radians(yaw_delta)
 env.data.qpos[:2]+=[dx,dy];env.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
 mujoco.mj_forward(env.model,env.data);env.prev_action=np.zeros(2,dtype=np.float32);env.step_count=0;env.history.clear();env.prev_goal_dist=float(env._goal_state()[1])
 return np.asarray(env._observation(update_history=False),dtype=np.float32)

def rollout(env,obs,native,policy,plan,deadline,path=None):
 cap=plan['episode_step_cap'];env.max_steps=cap;goal=env.goal_xy.copy();model=C.model_hash(env);steps=0;done=False;rows=[];nearest=float(env._goal_state()[1]);start_hash=state_hash(env,obs)
 while not done:
  assert time.monotonic()<deadline
  action=command(policy,obs,steps,cap);before=steps;state=C.observation(obs,1-steps/cap);pose=np.asarray(env._pose())
  for _ in range(min(plan['control_hold_steps'],cap-steps)):
   t0=float(env.data.time);obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1
   assert abs(float(env.data.time)-t0-.002)<1e-7;nearest=min(nearest,float(env._goal_state()[1]));done=bool(term or trunc)
   if done:break
  if path is not None:rows.append((before,state,action.copy(),pose,steps-before))
 assert C.model_hash(env)==model and np.array_equal(env.goal_xy,goal)
 if path is not None:
  np.savez_compressed(path,**{k:np.asarray([r[i] for r in rows]) for i,k in enumerate(['steps','states','native_actions','poses','native_counts'])})
 result={'steps':steps,'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc and not term),'remaining_m':float(env._goal_state()[1]),'nearest_m':nearest,'decisions':math.ceil(steps/plan['control_hold_steps']),'initial_fullstate_obs_goal_sha256':start_hash,'model_hash':model,'final_pose':list(env._pose()),'goal_xy':goal.tolist()}
 if path is not None:result['trajectory_sha256']=C.sha(path)
 return result

def prepare(plan,deadline):
 native,create=C.native_setup(OUT/'prepare');policy=policy_load(plan);rows=[];spent=0
 for side,source in plan['anchors'].items():
  env=clone=replica=None
  try:
   env,_,_=create(source['case']);env.max_steps=plan['episode_step_cap'];obs=init_pose(env,source)
   assert not env._collision();snapshot=C.capture(env,obs);path=OUT/(side+'_fullstate.npz');C.write_snapshot(path,snapshot)
   clone,_,_=create(source['case']);obs2=C.restore(clone,C.read_snapshot(path));assert state_hash(env,obs)==state_hash(clone,obs2)
   maxphysics=maxobs=0.;a=b=None
   for step in range(plan['restore_future_steps']):
    assert time.monotonic()<deadline
    if step%plan['control_hold_steps']==0:a=command(policy,obs,step,plan['episode_step_cap']);b=command(policy,obs2,step,plan['episode_step_cap'])
    obs,_,term,trunc,_=native.step_checked(env,a);obs2,_,term2,trunc2,_=native.step_checked(clone,b);spent+=2
    obs=np.asarray(obs,dtype=np.float32);obs2=np.asarray(obs2,dtype=np.float32)
    maxphysics=max(maxphysics,float(abs(physics(env)-physics(clone)).max()));maxobs=max(maxobs,float(abs(obs-obs2).max()))
    assert not(term or trunc or term2 or trunc2)
   assert maxphysics==maxobs==0
   replica,_,_=create(source['case']);obs3=C.restore(replica,snapshot)
   rep=rollout(replica,obs3,native,policy,plan,deadline,OUT/(side+'_source_reproduction.npz'));spent+=rep['steps']
   rows.append({'side':side,'snapshot_path':str(path),'snapshot_sha256':C.sha(path),'physics_restore_max_error':maxphysics,'observation_restore_max_error':maxobs,'reproduction':rep})
  finally:
   for e in [env,clone,replica]:
    if e is not None:e.close()
 prep={'state':'PASS' if all(r['reproduction']['success'] for r in rows) else 'FAIL','sources':rows,'physics_steps':spent,'scope':'New full-state source reproduction and paired independent200step futures, frozen T183keep10'}
 C.save(OUT/'preparation.json',prep);return prep

def worker(pipe,index,deadline):
 folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None
 try:
  plan=json.loads((HERE/'protocol.json').read_text());policy=policy_load(plan);native,create=C.native_setup(folder);prep=json.loads((OUT/'preparation.json').read_text());snapshots={r['side']:C.read_snapshot(r['snapshot_path']) for r in prep['sources']}
  while True:
   job=pipe.recv()
   if job is None:break
   if env is not None:env.close()
   env,_,_=create(plan['anchors'][job['side']]['case']);obs=C.restore(env,snapshots[job['side']]);obs=perturb(env,obs,job)
   invalid=bool(env._collision() or not np.isfinite(obs).all() or np.any(abs(env.data.qpos[:2])>7.5))
   if invalid:row={**job,'invalid_reset':True,'success':False,'collision':False,'timeout':False,'steps':0,'decisions':0}
   else:row={**job,'invalid_reset':False,**rollout(env,obs,native,policy,plan,deadline,OUT/(job['id']+'.npz'))}
   pipe.send(row)
 except (EOFError,SystemExit):pass
 except BaseException:
  try:pipe.send({'error':traceback.format_exc()})
  except OSError:pass
  raise
 finally:
  if env is not None:env.close()
  pipe.close();log.close()

def main():
 lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate or implicit resume'
 plan=json.loads((HERE/'protocol.json').read_text())
 for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
 assert shutil.disk_usage(OUT).free>40*1024**3
 assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())+plan['max_run_bytes']<1024**3
 start=time.monotonic();deadline=start+plan['wall_seconds'];C.save(OUT/'status.json',{'state':'PREPARING','optimizer_updates':0});prep=prepare(plan,deadline)
 if prep['state']!='PASS':
  result={'state':'COMPLETE','training_gate':'FAIL_SOURCE_REPRODUCTION','preparation':prep,'optimizer_updates':0,'automatic_promotion':False};C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);return
 ctx=mp.get_context('spawn');pipes=[];workers=[];rows=[];jobs=plan['jobs'].copy()
 try:
  for i in range(8):
   a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,i,deadline));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);workers.append(p)
  C.save(OUT/'workers.json',{'parent':os.getpid(),'pids':[p.pid for p in workers]});active=set(pipes)
  for pipe in pipes:pipe.send(jobs.pop(0))
  while active:
   assert time.monotonic()<deadline
   for pipe in wait(list(active),timeout=1):
    row=pipe.recv();assert 'error' not in row,row;rows.append(row)
    with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    frames=prep['physics_steps']+sum(r['steps'] for r in rows);assert frames<=plan['physics_budget']
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<plan['max_run_bytes']
    C.save(OUT/'status.json',{'state':'RUNNING','cases':len(rows),'physics_steps':frames,'optimizer_updates':0})
    if jobs:pipe.send(jobs.pop(0))
    else:pipe.send(None);active.remove(pipe)
  scores={}
  for side in plan['anchors']:
   subset=[r for r in rows if r['side']==side];source=next(r for r in prep['sources'] if r['side']==side)
   zero=next(r for r in subset if r['delta_x_m']==r['delta_y_m']==r['delta_yaw_deg']==0)
   for key in ['steps','initial_fullstate_obs_goal_sha256','final_pose']:assert zero[key]==source['reproduction'][key],key
   scores[side]={'n':len(subset),'safe_success':sum(r['success'] for r in subset),'invalid':sum(r['invalid_reset'] for r in subset),'collisions':sum(r['collision'] for r in subset),'timeouts':sum(r['timeout'] for r in subset)}
  result={'state':'COMPLETE','scores':scores,'training_gate':'PASS' if all(r['safe_success']>=plan['min_success_per_side'] for r in scores.values()) else 'FAIL_ROBUSTNESS','physics_steps':prep['physics_steps']+sum(r['steps'] for r in rows),'optimizer_updates':0,'source_checkpoint_sha256':C.sha(plan['checkpoint']),'automatic_promotion':False,'incumbent':'T159','scope':'Frozen T183keep10, two consumed near-goal anchors. Fixed27 perturbations each, not original5m or generalization. No automatic training launch.'}
  C.save(OUT/'summary.json',result);C.save(OUT/'status.json',{**result,'elapsed_s':time.monotonic()-start});print(json.dumps(result),flush=True)
 finally:
  for pipe in pipes:
   try:pipe.send(None)
   except (OSError,EOFError):pass
  for p in workers:
   p.join(5)
   if p.is_alive():p.terminate();p.join(5)
  for p in pipes:pipe.close()
if __name__=='__main__':main()
