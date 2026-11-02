"""Posthoc recorded-decision geometry; no native, model, threshold or update."""
from pathlib import Path
import hashlib,json,math,numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t294_paired_action_value';DEST=ROOT/'stage/docs/distance_20261102/T294_RECORDED_GEOMETRY.json'


def main():
    assert not DEST.exists();audit=json.loads((DEST.parent/'T294_AUDIT.json').read_text());assert audit['state']=='PASS'
    rows=json.loads((OUT/'cases.json').read_text());result=[];arrays={}
    for r in rows:
        f=Path(r['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
        with np.load(f) as z:
            states=z['states'];nxt=np.concatenate([states[1:],z['last_next_state']]) if 'last_next_state' in z.files else z['next_states']
            pos=np.vstack([z['poses'][0,:2],z['next_poses'][:,:2]]);tt=np.r_[0,np.cumsum(z['native_counts'].astype(np.int64))]*.002
        d=np.array([math.hypot(*(p-np.array(r['goal_xy']))) for p in pos]);k=int(np.argmin(d));arrays[(r['source_reset_id'],r['arm'])]=(pos,tt)
        result.append(dict(task=r['source_reset_id'],arm=r['arm'],nearest_recorded_decision_goal_m=float(d[k]),nearest_recorded_decision_time_s=float(tt[k]),
            first_hold_next_LF_RF_steer_rad=(nxt[0,113:115].astype(np.float64)*.5).tolist(),first_hold_next_LF_RF_steer_rate_rad_s=(nxt[0,115:117].astype(np.float64)*3).tolist()))
    for r in result:
        p,t=arrays[(r['task'],r['arm'])];base,bt=arrays[(r['task'],'zero')];assert np.array_equal(t,bt)
        r['max_recorded_XY_deviation_from_zero_m']=float(np.linalg.norm(p-base,axis=1).max())
    DEST.write_text(json.dumps(dict(state='POSTHOC_DESCRIPTION',native_added=0,Actor_updates=0,optimizer_calls=0,rows=result,
        limitation='Only original and decision endpoints, not continuous native minimum or path length. Steering normalized packet columns multiplied by documented .5rad and3rad/s scales; no actuator-clipping saturation occurs in reported first-hold values. No new success threshold or selection.'),indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
