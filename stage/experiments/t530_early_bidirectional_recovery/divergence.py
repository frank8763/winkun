"""Read-only deterministic trajectory discrepancy; no native/Adam."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t529_bidirectional_recovery'));import t529_runtime as R
G,np,read,save,sha=R.G,R.np,R.read,R.save,R.sha
DOC=ROOT/'stage/docs/day_20261113/T530_DIVERGENCE.json'
def compute():
 p=read(ROOT/'stage/runs/t529_bidirectional_recovery/protocol.json');c=read(ROOT/'stage/runs/t529_bidirectional_recovery/pilot/control_rows.json');old=c['old_failure'];parent=next(r for r in p['keep_rows'] if r['job_id']==old['job_id']);a=G.E.N.S.arrays(old['path']);b=G.E.N.S.arrays(parent['path']);assert all(sha(r['path'])==r['sha256'] for r in [old,parent,c['new_success']]);n=min(len(a['states']),len(b['states']))-1;assert np.array_equal(a['native_counts'][:n],b['native_counts'][:n]);times=np.r_[0,np.cumsum(a['native_counts'][:n-1])]*.002
 xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1);dy=a['poses'][:n,2]-b['poses'][:n,2];yaw=np.abs(np.arctan2(np.sin(dy),np.cos(dy)))*180/np.pi;bad=(xy>=.05)|(yaw>=2.);starts=[i for i in range(n-4) if bad[i:i+5].all()];assert starts;k=starts[0];assert 0<k<151
 # Switch one decision before first persistent discrepancy. No simulation-based selection.
 first=max(1,k-1);assert np.sum(a['native_counts'][:first])==100*first
 return dict(native=0,Adam=0,old_row=old,parent_row=parent,new_success_row=c['new_success'],criterion='First five consecutive native decision-endpoint poses with XY>=0.05m OR abs wrapped yaw>=2deg versus successful T507 at equal physical clock; switch one decision before first such run. Posthoc source-data rule frozen before ANY new simulation; not collision/reachability/stability proof.',first_persistent_discrepancy_index=k,first_persistent_discrepancy_seconds=float(times[k]),teacher_first=first,prefix_physics_steps=int(a['native_counts'][:first].sum()),switch_seconds=.002*int(a['native_counts'][:first].sum()),xy_at_switch_m=float(xy[first]),yaw_at_switch_deg=float(yaw[first]),xy_at_old_30p2_m=float(xy[151]),yaw_at_old_30p2_deg=float(yaw[151]),old_failure_remain_m=old['remaining_m'],samples=[dict(seconds=float(times[i]),xy_m=float(xy[i]),yaw_deg=float(yaw[i])) for i in sorted(set([0,first,k,50,100,151,n-1]))])
if __name__=='__main__':
 d=compute();assert not DOC.exists();save(DOC,d);print({k:v for k,v in d.items() if k not in ['old_row','parent_row','new_success_row','samples']},flush=True)
