"""Independent recorded endpoint/clock/context check, no model/native update."""
from future_data import *
import os,time
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t275_future_target_contract'
DEST=ROOT/'stage/docs/long_distance_20261101/T275_AUDIT.json'


def main():
    assert not DEST.exists()
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['native_environments_created']==0
    pid=json.loads((OUT/'workers.json').read_text())['parent']
    try:os.kill(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError(('Contract process exists or PID reused',pid))
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    assert sha(OUT/'future_pairs.npz')==s['report']['future_pairs_sha256']
    with np.load(OUT/'future_pairs.npz') as z:a={k:z[k].copy() for k in z.files}
    checked=0;support=[]
    for e,r in enumerate(p['records']):
        ix=np.flatnonzero(a['episode']==e)
        with np.load(r['path']) as z:
            end=z['native_counts'].astype(np.int64).cumsum();start=np.r_[0,end[:-1]]
            k=a['anchor'][ix].astype(np.int64);j=a['endpoint'][ix].astype(np.int64)
            assert np.all(k%4==0) and np.all(j>=k) and np.all(z['end_flags'][j,0]==0)
            assert np.array_equal(a['delta_steps'][ix],end[j]-start[k])
            assert np.all(end[j]<=r['cap']) and np.all(a['train'][ix]==(r['split']=='train'))
            expected=relative(z['poses'][k],z['next_poses'][j,:2]).astype(np.float32)
            assert np.array_equal(expected,a['goal_body_m'][ix]);checked+=len(ix)
            support.append(dict(id=r['id'],side=r['side'],distance_m=r['distance_m'],split=r['split'],
                maximum_recorded_query_distance_m=float(np.linalg.norm(expected,axis=1).max()),
                max_delta_seconds=float(a['delta_steps'][ix].max())*.002,
                initial_anchor_max_recorded_query_distance_m=float(np.linalg.norm(expected[k==0],axis=1).max())))
    assert checked==s['report']['pairs']==26395
    b,stats=build(p);assert all(np.array_equal(a[k],v) for k,v in b.items())
    assert s['physics_steps']==s['optimizer_calls']==s['Actor_updates']==0
    assert s['elapsed_s']<p['wall_seconds'] and used(OUT)<p['final_bytes'] and used(ROOT/'stage/runs')<p['global_bytes']
    save(DEST,dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_added=0,optimizer_added=0,all_processes_absent=True,
        all26395_endpoint_indices_times_coordinates_split_recomputed_exact=True,all_source_sha_verified=True,
        all_collision_endpoints_excluded=True,all_pairs_strictly_future_within_observed_original_cap=True,
        all_arrays_rebuilt_bit_exact=True,train_pairs=stats['train_pairs'],test_pairs=stats['test_pairs'],
        original_six_m_success_added=False,original120_goal_relabel_contract_not_claimed=True,
        first_hit_or_shortest_time_probability_not_claimed=True,query_support_by_episode=support,
        own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),summary_sha256=sha(OUT/'summary.json'),audit_epoch=time.time()))
    print(json.dumps(dict(state='PASS',pairs=checked)),flush=True)


if __name__=='__main__':main()
