from pathlib import Path
import importlib.util,json,os,math
import numpy as np,torch,mujoco
REPO=Path(__file__).resolve().parents[3];OUT=REPO/'stage/runs/reward_contract_audit';OUT.mkdir(exist_ok=False)
os.environ['C98_JOB']=os.environ['TD3_JOB']=str(OUT);os.environ['C22_SCREEN_ONLY']='0';torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('reward_audit_source',REPO/'stage/current/outputs/stage_t148/train.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
h=t.load_module(t.HELPER,'reward_helper');l=t.load_module(t.STAGE,'reward_legacy');runner=l.module(l.SRC,'reward_runner');_,v,mod,prep,_=runner.load_native();N=prep[3]
env=renderer=None
try:
 env,renderer,obs,_=h.make_goal_task(l.make_short_task(runner.make_task),v,mod,N,2029653601,-17.3,5.)
 agent=t.create_agent(t.load_module(t.CORE,'reward_core'));agent.actor.load_state_dict(torch.load(REPO/'stage/current/outputs/stage_t159/consolidated_actor.pt',map_location='cpu',weights_only=False)['actor']);agent.actor.eval()
 rows=[]
 for step in range(4001):
  if step%500==0:
   distances=[]
   for i in range(env.model.ngeom):
    name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i)
    if not name or not name.startswith('real_rect_obs_'):continue
    witness=np.full(6,np.nan)
    d=float(mujoco.mj_geomDistance(env.model,env.data,23,i,.65,witness))
    distances.append({'obstacle':name,'distance':d,'witness':witness.tolist(),'world_position':env.data.geom_xpos[i].tolist(),'half_sizes':env.model.geom_size[i].tolist()})
   rows.append({'step':step,'time':float(env.data.time),'pose':list(env._pose()),'qpos':env.data.qpos.tolist(),
    'chassis_center':env.data.geom_xpos[23].tolist(),'chassis_rotation':env.data.geom_xmat[23].reshape(3,3).tolist(),
    'chassis_half_sizes':env.model.geom_size[23].tolist(),'collision':bool(env._collision()),
    'extra_reward_penalty':float(t._t59_external_clearance_reward(env)),'box_distances':distances})
  if step==4000:break
  raw=agent.act(obs);obs,_,terminal,trunc,info=N.step_checked(env,np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32))
  assert not terminal and not trunc
 report={'experiment':'STAGE_T59_CHASSIS_DISTANCE_NUMERIC_AUDIT','native_physics_steps':4000,'training_updates':0,'policy':'frozen T159, no exploration','snapshots':rows}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print([(r['step'],r['pose'][:2],r['collision'],min(x['distance'] for x in r['box_distances']),r['extra_reward_penalty']) for r in rows])
finally:
 if renderer is not None:renderer.close()
 if env is not None:env.close()
