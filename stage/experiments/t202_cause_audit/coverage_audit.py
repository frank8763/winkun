"""Post-hoc source coverage: distinguish successful suffixes from full LEFT."""
import os
os.sched_setaffinity(0,{11})
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t202_cause_audit'
sys.path.insert(0,str(ROOT/'stage/experiments/t200_residual_placement'))
import common as C
import numpy as np

def main():
    assert not (OUT/'coverage.json').exists()
    with np.load(ROOT/'stage/runs/t196_left_exploration/left_baseline_trace.npz') as z:
        frontier=z['poses'][z['steps']==14000][0,:2];last=z['poses'][-1,:2];start=z['poses'][0,:2]
    plan=json.loads((ROOT/'stage/experiments/t200_residual_placement/protocol.json').read_text())
    near=np.asarray(plan['anchor_sources']['LEFT']['reset_task']['start_pose'][:2]);goal=np.asarray(plan['anchor_sources']['LEFT']['reset_task']['goal_xy']);result={};hashes={}
    for arm in ['native','latent']:
        rows=json.loads((ROOT/f'stage/runs/t200_residual_placement/{arm}/replay_receipts.json').read_text());counts={};successful=[]
        for r in rows:
            assert C.sha(r['path'])==r['sha256'];hashes[r['path']]=r['sha256']
            with np.load(r['path']) as z:
                a={k:z[k].copy() for k in z.files}
            left=np.max(np.abs(a['goal_xy']-goal))<1e-7;distance=float(np.linalg.norm(a['poses'][0,:2]-a['goal_xy']));group='original_LEFT' if left and distance>4.9 else 'near_LEFT' if left else 'other'
            if 'end_flags' in a:flag=a['end_flags'][-1]
            elif 'physical_flags' in a:flag=a['physical_flags'][-1]
            else:flag=a['physical'][-1,7:10]
            event='GOAL' if flag[1] else 'CONTACT' if flag[0] else 'TIMEOUT';entry=counts.setdefault(group,{'episodes':0,'decisions':0,'GOAL':0,'CONTACT':0,'TIMEOUT':0});entry['episodes']+=1;entry['decisions']+=len(a['states']);entry[event]+=1
            if left and flag[1]:successful.append(np.concatenate([a['poses'][:,:2],a['next_poses'][-1:,:2]]))
        points=np.concatenate(successful)
        result[arm]={'replay_groups':counts,'nearest_successful_LEFT_path_point_to_parent_frontier_m':float(np.linalg.norm(points-frontier,axis=1).min()),'nearest_successful_LEFT_path_point_to_parent_last_decision_m':float(np.linalg.norm(points-last,axis=1).min())}
    data={'state':'COMPLETE','analysis_type':'POST_HOC_SAVED_TRAJECTORY_ONLY','physics_steps':0,'optimizer_updates':0,'input_hashes':hashes,'original_LEFT_start_xy':start.tolist(),'parent_frontier_at_28s_xy':frontier.tolist(),'parent_last_decision_xy':last.tolist(),'near_LEFT_start_xy':near.tolist(),'goal_xy':goal.tolist(),'near_start_to_parent_frontier_m':float(np.linalg.norm(near-frontier)),'near_start_to_parent_last_decision_m':float(np.linalg.norm(near-last)),'models':result,
          'scope':'Euclidean spatial gaps on consumed fixed task, not proof of unreachable state components or a planner/path construction. Successful suffix resets are not full5m successes. No updates/new native interactions. All source trajectories and failures preserved.'}
    C.save(OUT/'coverage.json',data);print(json.dumps({k:v for k,v in data.items() if k!='input_hashes'},indent=2))

if __name__=='__main__':main()
