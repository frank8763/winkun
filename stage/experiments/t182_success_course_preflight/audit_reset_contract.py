from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t182b_success_course_preflight'
sys.path.insert(0,str(ROOT/'stage/experiments/t177_frontier_onpolicy'));import common as C
plan=json.loads((Path(__file__).with_name('retry_protocol.json')).read_text());native,create=C.native_setup(OUT/'contract_audit');rows=[];env=None
try:
 for job in plan['jobs']:
  if env is not None:env.close()
  env,_,_=create(job['case']);reset=job['reset_task'];yaw=reset['start_pose'][2];env.data.qpos[:2]=reset['start_pose'][:2]
  found=[]
  for turns in [-1,0,1]:
   a=yaw+turns*2*math.pi
   env.data.qpos[3:7]=[math.cos(a/2),0.,0.,math.sin(a/2)]
   sig=hashlib.sha256(np.round(np.r_[env.data.qpos,env.goal_xy],8).tobytes()).hexdigest()
   if sig==reset['signature']:found.append({'turns':turns,'quaternion':env.data.qpos[3:7].tolist()})
  rows.append({'id':job['id'],'matches':found})
 C.save(OUT/'reset_contract_audit.json',{'state':'PASS' if all(r['matches'] for r in rows) else 'FAIL','physics_steps':0,'rows':rows});print(json.dumps(rows,indent=2))
finally:
 if env is not None:env.close()
