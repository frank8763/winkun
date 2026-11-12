"""Read-only raw-history comparison after T497; zero native/optimizer steps."""
from pathlib import Path
import json,sys,time,os,shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'))
import speed_actor as S
G,np,torch=S.G,S.np,S.torch
read,sha,digest=G.read,G.sha,G.digest
OUT=ROOT/'stage/runs/t497_success_union_student';DOC=ROOT/'stage/docs/review_20261112'
def metrics(a,r):
 dt=a['native_counts']*.002;t=np.r_[0.,np.cumsum(dt)];pose=np.vstack([a['poses'],a['next_poses'][-1]]);d=np.linalg.norm(pose[:,:2]-a['task_goal_xy'],axis=1)
 speed=np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1)/dt;radial=(d[:-1]-d[1:])/dt
 angle=np.arctan2(a['task_goal_xy'][1]-a['poses'][:,1],a['task_goal_xy'][0]-a['poses'][:,0]);heading=np.abs(np.arctan2(np.sin(angle-a['poses'][:,2]),np.cos(angle-a['poses'][:,2])))*180/np.pi
 def seg(mask):
  avg=lambda x:float(np.average(x[mask],weights=dt[mask]));return dict(speed_m_s=avg(speed),heading_deg=avg(heading),drive=avg(a['actions'][:,0]),steer=avg(a['actions'][:,1]),away_seconds=float(dt[mask & (radial<-.001)].sum()))
 return dict(success=r['success'],timeout=r['timeout'],contact=r['collision'],seconds=float(t[-1]),remaining_m=float(d[-1]),min_decision_distance_m=float(d.min()),closest_seconds=float(t[d.argmin()]),first20=seg(t[:-1]<20),last20=seg(t[:-1]>=t[-1]-20),whole=seg(np.ones(len(dt),dtype=bool)),distance_by_second={str(k):float(np.interp(k,t,d)) for k in [0,10,20,40,60,80,90,100]},time=t,pose=pose)
def main():
 start=time.time();p=read(OUT/'protocol.json');summary=read(OUT/'student/summary.json');assert read(OUT/'student/audit.json')['state']=='PASS'
 sealed=read(OUT/'SEALED.json');assert all(not Path('/proc',str(pid)).exists() for pid in sealed['scientific_pids'])
 failed=[r for r in G.rows(OUT/'student') if not r['success']];assert len(failed)==1;job=failed[0]['job_id']
 baseline=next(r for r in G.rows(OUT/'teacher') if r['job_id']==job)
 pulse=next(r for r in G.rows(ROOT/'stage/runs/t496_neural_speed_profile/teacher') if r['job_id']==job)
 raw={};result={};sources={}
 for name,row in [('baseline_success',baseline),('pulse_failure',pulse),('student_failure',failed[0])]:
  assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items());raw[name]=a;result[name]=metrics(a,row);sources[name]=dict(path=row['path'],sha256=row['sha256'],array_digest=row['array_digest'])
 q=S.frozen(p,summary['result']['model']['path']);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 a=raw['baseline_success'];x=torch.tensor(a['states']);t=result['baseline_success']['time'][:-1]
 with torch.no_grad():
  predicted=q.sequence(x).clamp(-1,1).float().numpy();target=a['actions'];err=predicted-target
  assert np.array_equal(predicted[:,1],target[:,1]);fit={}
  for name,mask in [('all',np.ones(len(t),bool)),('first20',t<20),('last20',t>=t[-1]-20)]:fit[name]=dict(drive_RMSE=float(np.sqrt(np.mean(err[mask,0]**2))),drive_mean_error=float(err[mask,0].mean()),drive_max_abs_error=float(abs(err[mask,0]).max()))
  offsets=[dict(seconds=float(t[i]),target_drive=float(target[i,0]),student_drive_on_success_history=float(predicted[i,0]),error=float(err[i,0])) for i in [int(abs(t-k).argmin()) for k in [0,5,10,20,40,60,80,90]]]
 b=result['baseline_success'];v=result['student_failure'];grid=np.arange(0,min(b['time'][-1],v['time'][-1]),.2);xyb=np.stack([np.interp(grid,b['time'],b['pose'][:,k]) for k in [0,1]],axis=1);xyv=np.stack([np.interp(grid,v['time'],v['pose'][:,k]) for k in [0,1]],axis=1);deviation=np.linalg.norm(xyb-xyv,axis=1)
 differences={str(th):float(grid[np.flatnonzero(deviation>th)[0]]) if (deviation>th).any() else None for th in [.01,.05,.1,.2]}
 for item in result.values():item.pop('time');item.pop('pose')
 receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T497.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 save=dict(state='POSTHOC_READ_ONLY_COMPLETE',job=job,source_files=sources,trajectories=result,student_actions_on_original_success_history=fit,drive_examples=offsets,first_XY_deviation_seconds=differences,native=0,Adam=0,causal_intervention_tested=False,nearest_distance_is_decision_endpoints_not_continuous_native_minimum=True,global_logical_bytes=G.M.used(),free_bytes=shutil.disk_usage(ROOT).free,elapsed_s=time.time()-start)
 (DOC/'T497_SINGLE_FAILURE_REVIEW.json').write_text(json.dumps(save,ensure_ascii=False,indent=2)+'\n');print(json.dumps(save,ensure_ascii=False))
if __name__=='__main__':main()
