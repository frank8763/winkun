"""Read only: time/goal alignment from the four already completed probes."""
from t278_native import ROOT,C,np
import json,math
OUT=ROOT/'stage/runs/t278_fixed_goal_starts'
DEST=ROOT/'stage/docs/long_distance_20261101/T278_BEHAVIOR.json'


def main():
    assert not DEST.exists()
    p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    assert s['state']=='COMPLETE'
    gg={r['id']:r for r in map(json.loads,(OUT/'initials.jsonl').read_text().splitlines())}
    rr=[r for r in map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()) if not r['replay']]
    results=[]
    for r in rr:
        assert C.sha(r['path'])==r['sha256'];g=gg[r['job_id']];goal=np.array(g['goal_xy'])
        with np.load(r['path']) as z:
            poses=np.concatenate([z['poses'][:1],z['next_poses']]);times=np.r_[0,np.cumsum(z['native_counts'].astype(np.int64))*.002]
            acts=z['actions'];drive=acts[:,0];steer=acts[:,1];start=z['states'][0]
            states_clock=[float(start[112]),float(z['next_states'][-1,112])]
        distances=np.linalg.norm(poses[:,:2]-goal,axis=1);idx=int(distances.argmin())
        error=math.atan2(goal[1]-poses[0,1],goal[0]-poses[0,0])-poses[0,2]
        error=(error+math.pi)%(2*math.pi)-math.pi
        samples={}
        for request in [1.,2.,5.,10.,20.,40.,56.]:
            k=int(np.searchsorted(times,request))
            if k<len(times):samples[str(request)]=dict(actual_s=float(times[k]),goal_distance_m=float(distances[k]))
        results.append(dict(id=r['job_id'],side=r['side'],start_distance_m=float(distances[0]),
            min_recorded_endpoint_distance_m=float(distances[idx]),min_endpoint_time_s=float(times[idx]),
            min_is_hold_endpoint_only_not_all_native_steps=True,final_distance_m=float(distances[-1]),
            initial_goal_heading_error_deg=math.degrees(error),first_action=acts[0].tolist(),
            first_5holds_mean_drive=float(drive[:5].mean()),first_5holds_mean_steer=float(steer[:5].mean()),
            state_clock_start_and_end=states_clock,endpoint_samples=samples,
            timeout=r['timeout'],collision=r['collision'],native_success=r['success']))
    C.save(DEST,dict(state='READ_ONLY_COMPLETED',physics_added=0,optimizer_added=0,
        original_raw_files_unchanged=True,model_sha256=p['checkpoint_sha256'],results=results,
        limitations='Four new reset states, not original6m starts or inherited along-path suffixes; new positions, fresh history and clock create distribution shift. No causal claim that distance or clock alone caused failure, no proof all possible short starts fail.'))
    print(json.dumps(dict(state='PASS',results=results)),flush=True)


if __name__=='__main__':main()
