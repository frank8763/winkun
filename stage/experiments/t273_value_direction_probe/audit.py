"""Independent raw count, exact restore/replay and delivered-action audit."""
from t273_native import *
from proposals import propose
import os,time
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t273_value_direction_probe'
DEST=ROOT/'stage/docs/long_distance_20261101/T273_AUDIT.json'


def main():
    assert not DEST.exists()
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['state']=='COMPLETE' and s['all_native_envs_closed'] and s['all_workers_stopped']
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live or reused PID',pid))
    for path,h in p['input_hashes'].items():assert C.sha(path)==h,path
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    assert propose(p)==p['proposals']
    rr=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()));assert len(rr)==s['commands']==6
    assert sum(r['physics_steps'] for r in rr)==s['physics_steps']<=p['max_physics']
    arrays={}
    for r in rr:
        assert C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:
            a={k:z[k].copy() for k in z.files};assert set(a)==set(r['array_hashes'])
            assert all(S.digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        assert np.array_equal(a['states'][0],np.array(p['proposals']['context120'],dtype=np.float32))
        assert np.array_equal(a['actions'][0],np.array(p['proposals']['actions'][r['proposal']],dtype=np.float32))
        assert r['absolute_end_step']<=28000 and r['start_step']==18000
        arrays[r['id']]=a
    for a,b in zip(rr[::2],rr[1::2]):
        assert a['array_hashes']==b['array_hashes'] and a['integration_sha256']==b['integration_sha256']
        assert all(np.array_equal(v,arrays[b['id']][k]) for k,v in arrays[a['id']].items())
    assert rr[0]['array_hashes']==p['reference']['array_hashes'] and rr[0]['integration_sha256']==p['reference']['integration_sha256']
    assert json.loads((OUT/'restoration_gate.json').read_text())['no_new_intervention_before_gate']
    assert s['optimizer_calls']==s['Actor_updates']==0 and s['elapsed_s']<p['wall_seconds']
    assert S.used(OUT)<p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes']
    u={r['proposal']:r for r in rr[::2]};a,b,c=[u[k]['actual_score16'] for k in ['up','control','down']]
    screen=a>=b>=c and a>c;assert screen==s['empirical_direction_screen_passed']
    C.save(DEST,dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_added=0,optimizer_added=0,
        all_processes_absent=True,total_physics_from_raw=s['physics_steps'],commands=6,
        original_control_and_three_full10array_integration_replays_exact=True,
        six_full120_payload_sha_verified=True,all_three_first_actions_and_frozen120_context_verified=True,
        scorer_gradient_and_native_rounding_rebuilt_exact=True,original_Actor_and_event_head_files_unchanged=True,
        empirical_direction_screen_passed=screen,unique_action_results=[{k:r[k] for k in ['proposal','predicted_score16','actual_score16','success','collision','timeout','remaining_m','absolute_end_step']} for r in rr[::2]],
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),summary_sha256=C.sha(OUT/'summary.json'),audit_epoch=time.time(),
        six_m_navigation_gate=False,automatic_Actor_coupling=False))
    print(json.dumps(dict(state='PASS',direction_screen=screen)),flush=True)


if __name__=='__main__':main()
