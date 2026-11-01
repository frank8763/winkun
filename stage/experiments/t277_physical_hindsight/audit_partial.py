"""Audit an ended partial recovery without changing its FAILED summary."""
from t277_native import *
from t277_labels import fixed_targets,first_hit_record
import hashlib,json,math,os,shutil,time
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t277_physical_hindsight'
DEST=ROOT/'stage/docs/long_distance_20261101/T277_PARTIAL_AUDIT.json'


def main():
    assert not DEST.exists();s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['state']=='FAILED' and s['inflight'] and s['all_workers_stopped'] and s['all_native_envs_closed']
    fail=json.loads((OUT/'failure.json').read_text());assert fail['physics_steps']==s['physics_steps'] and fail['commands']==s['commands']
    for pid in {json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']}:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Process exists or PID reused',pid))
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()))
    assert len(rows)==s['exact_old_trajectories']==3 and s['commands']==4 and s['physics_steps']==79297
    complete=sum(r['physics_steps'] for r in rows);assert complete==65397
    inflight=s['physics_steps']-complete;assert inflight==13900
    assert not (OUT/'hindsight_first_hits.json').exists(),'Original pipeline unexpectedly persisted a full file'
    labels=[];proof=[]
    for row,ref in zip(rows,p['references']):
        assert row['id']==ref['id'] and C.sha(row['path'])==row['sha256']
        assert row['native_array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        assert C.sha(ref['path'])==ref['sha256'] and row['old_actions_delivered_unchanged']
        with np.load(row['path']) as z:
            assert set(z.files)=={'physical_xy','physical_flags'};xy=z['physical_xy'].copy();flags=z['physical_flags'].copy()
        assert xy.dtype==np.float64 and xy.shape==(row['physics_steps'],2) and np.isfinite(xy).all()
        assert flags.dtype==np.uint8 and flags.shape==(len(xy),) and np.all(flags[:-1]==0)
        assert int(flags[-1])==int(row['collision'])+2*int(row['success'])+4*int(row['timeout'])
        assert S.digest(xy)==row['physical_xy_array_sha256'] and S.digest(flags)==row['physical_flags_array_sha256']
        with np.load(ref['path']) as z:source={k:z[k].copy() for k in z.files}
        ends=source['native_counts'].astype(np.int64).cumsum();initial=source['poses'][0,:2]
        assert np.array_equal(xy[ends-1],source['next_poses'][:,:2])
        for chosen in fixed_targets(source):
            r=first_hit_record(xy,flags,initial,source['next_poses'][chosen,:2],ends,int(chosen));r.update(source_id=row['id']);labels.append(r)
            if not r['eligible']:continue
            goal=r['goal_xy'];d=np.fromiter((math.hypot(float(goal[0])-float(q[0]),float(goal[1])-float(q[1])) for q in xy),dtype=np.float64,count=len(xy))
            candidate=np.flatnonzero((d<=.2)&((flags&1)==0));first=int(candidate[0])+1
            assert first==r['first_goal_step'] and not np.any(flags[:first]&1)
            initial_distance=math.hypot(float(goal[0])-float(initial[0]),float(goal[1])-float(initial[1]));assert initial_distance>.2
            assert np.all(d[:first-1]>.2) and d[first-1]<=.2
            k=int(np.searchsorted(ends,first,side='left'));assert k==r['terminal_transition']
            all_d=np.r_[initial_distance,d];total=0.;discount=1.
            for t in range(k+1):
                begin=0 if t==0 else int(ends[t-1]);end=min(int(ends[t]),first);n=end-begin;disc=R.GAMMA**n;terminal=t==k
                reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*float(all_d[begin])-disc*.03*float(all_d[end])*(not terminal)+(R.GAMMA**(n-1) if terminal else 0.)
                total+=discount*reward;discount*=disc
            exact=.03*initial_distance-.000002*(1-R.GAMMA**first)/(1-R.GAMMA)+R.GAMMA**(first-1)
            assert abs(total-exact)<1e-12 and r['bootstrap_discount']==0
            proof.append(dict(source_id=row['id'],chosen_decision=int(chosen),first_goal_step=first,
                interior_hold=r['inside_original_hold'],relabeled_return=total,closed_form=exact,error=abs(total-exact)))
    eligible=[r for r in labels if r['eligible']]
    assert len(labels)==s['future_goal_requests']==24 and len(eligible)==s['eligible_future_labels']==21
    assert sum(r['inside_original_hold'] for r in eligible)==s['interior_hold_first_hits']==21
    assert all(not r['success'] for r in rows) and s['original_goal_safe']==0
    assert s['optimizer_calls']==s['Actor_updates']==s['independent_new_navigation_episodes']==0
    assert s['elapsed_s']<=p['wall_seconds']+1 and s['physics_steps']<=p['max_physics']
    assert S.used(OUT)<p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    result=dict(state='PARTIAL_EVIDENCE_RECONCILED_NOT_FULL_RECOVERY',all_processes_absent=True,physics_added=0,optimizer_added=0,
        original_summary_state='FAILED',complete_old_trajectories=3,inflight_censored_trajectory=1,not_started_trajectories=4,
        raw_complete_physics_verified=complete,additional_inflight_physics_reported_not_XY_reconstructed=inflight,
        total_physics_charged_as_reported=s['physics_steps'],inflight_XY_payload_not_persisted_by_original_runner=True,
        all3_saved_physical_payload_SHA_and_original8_native_array_integration_hashes_verified=True,
        recovered_fixed_requests=24,eligible=21,invalid=3,inside_hold_firsthits=21,
        independent_native_math_hypot_firsthit_and_collision_priority_verified=True,
        relabeled_return_MC_closed_form_max_error=max(r['error'] for r in proof),
        recovered_label_records=labels,return_proofs=proof,
        original_hindsight_file_missing_recovered_in_this_derived_document_not_original_summary=True,
        original_goal_safe=0,new_independent_navigation_successes=0,
        full8_recovery_passed=False,ready_onpolicy_or_Actor120_relabel_replay=False,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),
        original_summary_sha256=C.sha(OUT/'summary.json'),original_failure_sha256=C.sha(OUT/'failure.json'),audit_epoch=time.time())
    C.save(DEST,result);print(json.dumps(dict(state=result['state'],complete=3,inflight=1,unrun=4,eligible=21)),flush=True)


if __name__=='__main__':main()
