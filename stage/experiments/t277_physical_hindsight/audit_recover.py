from t277_native import *
from t277_labels import fixed_targets,first_hit_record
import json,os,time
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t277_physical_hindsight'
DEST=ROOT/'stage/docs/long_distance_20261101/T277_AUDIT.json'


def main():
    assert not DEST.exists();s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['all_native_envs_closed'] and not s['inflight']
    for pid in {json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']}:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Process exists or PID reused',pid))
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()));labels=json.loads((OUT/'hindsight_first_hits.json').read_text())
    assert len(rows)==s['commands']==8 and len(labels)==s['future_goal_requests']==64
    assert sum(r['physics_steps'] for r in rows)==s['physics_steps']==201980
    data={};rebuilt=[];returns=[]
    for row,ref in zip(rows,p['references']):
        assert row['id']==ref['id'] and C.sha(row['path'])==row['sha256']
        assert row['native_array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        assert row['old_actions_delivered_unchanged'] and row['original_source_sha256']==ref['sha256']
        with np.load(row['path']) as z:
            assert set(z.files)=={'physical_xy','physical_flags'};xy=z['physical_xy'].copy();flags=z['physical_flags'].copy()
        assert xy.dtype==np.float64 and xy.shape==(row['physics_steps'],2) and np.isfinite(xy).all()
        assert flags.dtype==np.uint8 and flags.shape==(len(xy),) and np.all(flags[:-1]==0)
        assert int(flags[-1])==int(row['collision'])+2*int(row['success'])+4*int(row['timeout'])
        assert S.digest(xy)==row['physical_xy_array_sha256'] and S.digest(flags)==row['physical_flags_array_sha256']
        with np.load(ref['path']) as z:source={k:z[k].copy() for k in z.files}
        ends=source['native_counts'].astype(np.int64).cumsum();initial=source['poses'][0,:2]
        assert np.array_equal(xy[ends-1],source['next_poses'][:,:2])
        data[row['id']]=(xy,flags,source,ends)
        for chosen in fixed_targets(source):
            r=first_hit_record(xy,flags,initial,source['next_poses'][chosen,:2],ends,int(chosen));r.update(source_id=row['id']);rebuilt.append(r)
    assert rebuilt==labels
    for r in labels:
        if not r['eligible']:continue
        xy,flags,source,ends=data[r['source_id']];goal=np.array(r['goal_xy']);first=r['first_goal_step']
        # Independently scan every native point, including original initial,
        # rather than inferring first-hit from later decision endpoints.
        d=np.linalg.norm(np.vstack([source['poses'][0,:2],xy])-goal,axis=1)
        assert np.all(d[:first]>.2) and d[first]<=.2 and not np.any(flags[:first]&1)
        k=int(np.searchsorted(ends,first,side='left'));assert k==r['terminal_transition']
        total=0.;discount=1.
        for t in range(k+1):
            begin=0 if t==0 else int(ends[t-1]);end=min(int(ends[t]),first);n=end-begin
            d0=float(d[begin]);d1=float(d[end]);disc=R.GAMMA**n;terminal=t==k
            reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1) if terminal else 0.)
            total+=discount*reward;discount*=disc
        exact=.03*float(d[0])-.000002*(1-R.GAMMA**first)/(1-R.GAMMA)+R.GAMMA**(first-1)
        assert abs(total-exact)<1e-12
        returns.append(dict(source_id=r['source_id'],chosen_decision=r['chosen_decision'],first_goal_step=first,
            interior_hold=r['inside_original_hold'],relabeled_return=total,closed_form_return=exact,error=abs(total-exact),
            original_goal_success_unchanged=True,not_Actor120_relabel_or_ready_training_replay=True))
    eligible=[r for r in labels if r['eligible']]
    assert len(eligible)==s['eligible_future_labels'] and sum(r['inside_original_hold'] for r in eligible)==s['interior_hold_first_hits']
    assert s['optimizer_calls']==s['Actor_updates']==s['independent_new_navigation_episodes']==0 and s['original_goal_safe']==0
    assert s['elapsed_s']<p['wall_seconds'] and S.used(OUT)<p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes']
    meta=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and f.suffix!='.npz');assert meta<=p['metadata_bytes']
    C.save(DEST,dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_added=0,optimizer_added=0,all_processes_absent=True,
        original8_native_arrays_and_final_integration_exact=True,all8_physical_payload_sha_verified=True,
        all201980_native_float64_XY_and_packed_flags_verified=True,all64_fixed_requests_and_firsthit_records_rebuilt_exact=True,
        original_goal_safe=0,new_independent_navigation_successes=0,eligible_future_labels=len(eligible),
        interior_hold_first_hits=s['interior_hold_first_hits'],all_eligible_firsthit_collision_priority_and_zero_terminal_bootstrap_verified=True,
        relabeled_reward_MC_matches_closed_form_max_error=max(r['error'] for r in returns),future_return_contracts=returns,
        Actor120_goalhistory_reconstruction_not_claimed=True,ready_training_replay=False,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),metadata_bytes=meta,
        summary_sha256=C.sha(OUT/'summary.json'),audit_epoch=time.time()))
    print(json.dumps(dict(state='PASS',physics=201980,eligible=len(eligible),interior=s['interior_hold_first_hits'])),flush=True)


if __name__=='__main__':main()
