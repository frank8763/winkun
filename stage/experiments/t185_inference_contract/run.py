"""Frozen exact-state inference implementation comparison; no policy training."""
from pathlib import Path
import os
CPU_POOL=[int(x) for x in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL));os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import sys,time,json,hashlib,fcntl,shutil,traceback,multiprocessing as mp
from multiprocessing.connection import wait
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t185_inference_contract'
sys.path.insert(0,str(ROOT/'stage/experiments/t177_frontier_onpolicy'))
import common as C

def contacts(env,mj):
 rows=[]
 for con in env.data.contact[:env.data.ncon]:
  g1,g2=int(con.geom1),int(con.geom2);r1=int(env.model.geom_bodyid[g1]) in env.robot_body_ids;r2=int(env.model.geom_bodyid[g2]) in env.robot_body_ids
  if r1==r2 or float(con.dist)>.002:continue
  other=g2 if r1 else g1;body=int(env.model.geom_bodyid[other]);name=mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_GEOM,other) or '';bn=mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_BODY,body) or ''
  if any(x in (name+' '+bn).lower() for x in ['floor','ground','plane']):continue
  rows.append({'robot_geom':mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_GEOM,g1 if r1 else g2),'other_geom':name,'dist_m':float(con.dist)})
 return rows

def worker(pipe,index,deadline):
 folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None
 try:
  import mujoco
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  plan=json.loads((HERE/'protocol.json').read_text());old=torch.load(plan['actor_checkpoint'],map_location='cpu',weights_only=False)['actor']
  torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.15)
  actor=C.CORE.Actor().eval().requires_grad_(False);gpu=C.CORE.Actor().cuda().eval().requires_grad_(False);gpu.load_state_dict(old)
  embedding=C.Policy().eval().requires_grad_(False);embedding.load_state_dict(torch.load(plan['initial_policy_checkpoint'],weights_only=False,map_location='cpu')['policy'])
  native,create=C.native_setup(folder)
  from training.e2e_phase10_lidar_v7_curriculum.stage9_runtime_fullplant_fix_v5_2 import get_last_step_record
  while True:
   job=pipe.recv()
   if job is None:break
   if env is not None:env.close()
   actor.load_state_dict(old)
   expected={k:v.clone() for k,v in old.items()}
   assert all(torch.equal(expected[k],v) for k,v in actor.state_dict().items())
   env,obs,actual=create(job['case']);env.max_steps=job['case']['step_cap'];goal=env.goal_xy.copy();modelhash=C.model_hash(env)
   reset=job['reset_task'];assert np.array_equal(goal,np.asarray(reset['goal_xy']))
   env.data.qpos[:2]=reset['start_pose'][:2];env.data.qpos[3:7]=reset['quaternion'];env.data.qvel[:]=0.;env.data.ctrl[:]=0.
   mujoco.mj_forward(env.model,env.data);env.prev_action=np.zeros(2,dtype=np.float32);env.prev_goal_dist=float(env._goal_state()[1]);env.history.clear();env.step_count=0
   obs=np.asarray(env._observation(update_history=False),dtype=np.float32);assert not env._collision()
   snapshot=C.capture(env,obs);integration_hash=hashlib.sha256(snapshot['physics'].tobytes()+obs.tobytes()+goal.tobytes()).hexdigest();max_action_differences=np.zeros(3)

   initialhash=hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()+goal.tobytes()+obs.tobytes()).hexdigest();steps=0;term=trunc=False;data=[];nearest=float(env._goal_state()[1]);travel=0.;previous=np.asarray(env._pose()[:2]);lastcontacts=[]
   while not(term or trunc):
    assert time.monotonic()<deadline
    with torch.no_grad():
     x=torch.tensor(obs.reshape(1,-1));raw=actor(x).numpy()[0];g1=gpu(x.cuda()).cpu().numpy()[0];g8=gpu(x.cuda().repeat(8,1)).cpu().numpy()[0]
     embed=embedding(torch.tensor(C.observation(obs,1-steps/env.max_steps).reshape(1,-1))).numpy()[0]
    cpu=np.array([.5*(raw[0]+1),raw[1]],dtype=np.float32)
    cuda1=np.array([.5*(g1[0]+1),g1[1]],dtype=np.float32);cuda8=np.array([.5*(g8[0]+1),g8[1]],dtype=np.float32)
    max_action_differences=np.maximum(max_action_differences,[float(abs(cuda1-cpu).max()),float(abs(cuda8-cpu).max()),float(abs(embed-cpu).max())])
    native_action=C.action_clip({'cpu112':cpu,'cuda112_batch1':cuda1,'cuda112_batch8':cuda8,'cpu113_embedding':embed}[job['implementation']],'full_range');state=obs.copy();pose=np.asarray(env._pose());before=steps
    for n in range(min(job['hold_steps'],env.max_steps-steps)):
     t0=float(env.data.time);obs,_,term,trunc,info=native.step_checked(env,native_action);obs=np.asarray(obs,dtype=np.float32);steps+=1
     assert abs(float(env.data.time)-t0-.002)<1e-7;d=float(env._goal_state()[1]);nearest=min(nearest,d);p=np.asarray(env._pose()[:2]);travel+=float(np.linalg.norm(p-previous));previous=p.copy()
     if term or trunc:break
    record=get_last_step_record(env.data);assert record is not None
    applied=record['last_applied'];ctrl=np.array([applied[k] for k in ['lf','rf','lr','rr']])
    data.append((before,state,raw.copy(),native_action.copy(),pose,steps-before,ctrl))
   assert np.array_equal(env.goal_xy,goal) and C.model_hash(env)==modelhash
   assert all(torch.equal(expected[k],v) for k,v in actor.state_dict().items())
   path=OUT/(job['id']+'.npz');np.savez_compressed(path,**{k:np.asarray([r[i] for r in data]) for i,k in enumerate(['steps','states','raw_actions','native_actions','poses','native_counts','applied_lf_rf_lr_rr'])})
   row={**job,'actual_seed':int(actual),'steps':steps,'decisions':len(data),'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc),'remaining_m':float(env._goal_state()[1]),'nearest_m':nearest,'travel_m':travel,'simulation_s':steps*.002,'terminal_contacts':contacts(env,mujoco),'final_pose':list(env._pose()),'goal_xy':goal.tolist(),'initial_hash':initialhash,'model_hash':modelhash,'trajectory_sha256':C.sha(path),'actor_weights_unchanged':True,'integration_state_hash':integration_hash,'same_state_max_native_action_difference_cuda1_cuda8_embedding_vs_cpu':max_action_differences.tolist()}
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
 lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists()
 plan=json.loads((HERE/'protocol.json').read_text())
 for p,h in plan['input_hashes'].items():assert C.sha(p)==h,p
 assert shutil.disk_usage(OUT).free>40*1024**3
 assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())+plan['max_run_bytes']<1024**3
 ctx=mp.get_context('spawn');pipes=[];workers=[];jobs=plan['jobs'].copy();rows=[];start=time.monotonic();deadline=start+plan['wall_seconds']
 try:
  for i in range(4):
   a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,i,deadline));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);workers.append(p)
  C.save(OUT/'workers.json',{'parent':os.getpid(),'pids':[p.pid for p in workers]});active=set(pipes)
  for pipe in pipes:pipe.send(jobs.pop(0))
  while active:
   assert time.monotonic()<deadline
   for pipe in wait(list(active),timeout=1):
    row=pipe.recv();assert 'error' not in row,row;rows.append(row)
    with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    frames=sum(r['steps'] for r in rows);assert frames<=plan['physics_budget']
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['max_run_bytes']
    C.save(OUT/'status.json',{'state':'RUNNING','cases':len(rows),'physics_steps':frames,'optimizer_updates':0})
    if jobs:pipe.send(jobs.pop(0))
    else:pipe.send(None);active.remove(pipe)
  for side in ['LEFT','RIGHT']:
   subset=[r for r in rows if r['case']['side']==side];assert len({r['integration_state_hash'] for r in subset})==1
  scores={}
  for side in ['LEFT','RIGHT']:
   scores[side]={}
   for hold in [25,100]:
    subset=[r for r in rows if r['case']['side']==side and r['hold_steps']==hold]
    scores[side][str(hold)]={r['implementation']:{'success':r['success'],'collision':r['collision'],'steps':r['steps'],'nearest_m':r['nearest_m'],'max_same_state_action_difference':r['same_state_max_native_action_difference_cuda1_cuda8_embedding_vs_cpu']} for r in subset}
  summary={'state':'COMPLETE','scores':scores,'physics_steps':sum(r['steps'] for r in rows),'decisions':sum(r['decisions'] for r in rows),'optimizer_updates':0,'matched_initial_integration_state_and_observation':'PASS','parent_checkpoint_unchanged':C.sha(plan['actor_checkpoint'])==plan['input_hashes'][plan['actor_checkpoint']],'automatic_promotion':False,'incumbent':'T159','scope':'Two already-consumed anchor states, four numerically equivalent inference implementations, two decisioncadences. Outcomes diagnose brittleness/contract sensitivity, not novel capability, geometry infeasibility or generalization.'}
  C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',{**summary,'elapsed_s':time.monotonic()-start});print(json.dumps(summary),flush=True)
 finally:
  for p in pipes:
   try:p.send(None)
   except (OSError,EOFError):pass
  for w in workers:
   w.join(5)
   if w.is_alive():w.terminate();w.join(5)
  for p in pipes:p.close()
if __name__=='__main__':main()
