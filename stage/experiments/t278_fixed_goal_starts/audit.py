"""Stopped-run audit: fixed goal/reset geometry, full states and frozen actions."""
from t278_native import ROOT,C,S,R,L,np,torch,policy
from pathlib import Path
import json,os,shutil,time
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t278_fixed_goal_starts'
DEST=ROOT/'stage/docs/long_distance_20261101/T278_AUDIT.json'


def readrows(name):
    f=OUT/name
    return list(map(json.loads,f.read_text().splitlines())) if f.exists() else []


def main():
    assert not DEST.exists()
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['all_native_envs_closed']
    w=json.loads((OUT/'workers.json').read_text());assert w['CPU_affinity']==[2] and not w['workers']
    for pid in [w['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live or reused PID',pid))
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    q=policy(p).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
    gg=readrows('initials.jsonl');rr=readrows('cases.jsonl');ii=readrows('invalid.jsonl')
    assert len(gg)==4 and len(ii)==s['invalid_initial_states']
    assert s['commands']==len(gg)+len(rr)//2<=p['max_commands']
    assert sum(r['physics_steps'] for r in rr)==s['physics_steps']<=p['max_physics']
    assert len(rr)==2*sum(g['valid'] for g in gg) and not any(r['error'] or not r['complete'] for r in rr)
    actions_checked=0;results=[]
    for job,g in zip(p['jobs'],gg):
        assert job['id']==g['id'] and job['side']==g['side']
        goal=np.array(g['goal_xy']);old=np.array(g['original_pose']);new=np.array(g['reset_pose'])
        expected=goal-(goal-old[:2])/np.linalg.norm(goal-old[:2])*job['remaining_distance_m']
        assert np.array_equal(new[:2],expected) and new[2]==old[2]
        assert g['original_zero_translation120_exact'] and g['only_native_qpos_XY_changed'] and g['history_fresh_at_new_pose']
        if not g['valid']:
            assert any(r['id']==job['id'] for r in ii) and not any(r['job_id']==job['id'] for r in rr)
            results.append(dict(id=job['id'],valid=False));continue
        pair=[r for r in rr if r['job_id']==job['id']];a,b=pair
        assert not a['replay'] and b['replay']
        assert a['array_hashes']==b['array_hashes'] and a['integration_sha256']==b['integration_sha256']
        assert C.sha(a['path'])==a['sha256']
        snap=C.read_snapshot(OUT/(job['id']+'_snapshot.npz'))
        assert R.arrays_hash(snap['physics'])==g['integration_sha256'] and snap['model_hash']==g['model_sha256']
        assert np.array_equal(snap['attrs']['goal_xy'],goal) and snap['attrs']['step_count']==0
        with np.load(a['path']) as z:
            arrays={k:z[k].copy() for k in z.files}
        assert all(S.digest(v)==a['array_hashes'][k] for k,v in arrays.items()) and len(arrays)==10
        x=arrays['states'];nx=arrays['next_states'];acts=arrays['actions'];n=arrays['native_counts'].astype(np.int64)
        assert x.shape==nx.shape and x.shape[1]==120 and int(n.sum())==a['physics_steps']==b['physics_steps']
        assert np.all((n>0)&(n<=100)) and S.digest(x[0])==g['initial120_sha256']
        assert np.array_equal(x[1:],nx[:-1]) and np.array_equal(x[1:,117:119],acts[:-1])
        assert np.array_equal(x[1:,119],n[:-1].astype(np.float32)/100)
        assert np.array_equal(x[0,117:],np.zeros(3,dtype=np.float32)) and x[0,112]==1
        assert np.array_equal(arrays['means'],np.zeros_like(acts)) and np.all(arrays['logp']==0)
        for row,action in zip(x,acts):
            with torch.no_grad():actual=q.step(torch.tensor(row[None]))[0][0].numpy()
            assert np.array_equal(actual,action);actions_checked+=1
        flags=arrays['end_flags'];assert not flags[:-1].any() and flags[-1].any()
        assert bool(flags[-1,0])==a['collision'] and bool(flags[-1,1])==a['success'] and bool(flags[-1,2])==a['timeout']
        remaining=float(np.linalg.norm(goal-arrays['next_poses'][-1,:2]));assert abs(remaining-a['remaining_m'])<1e-12
        assert bool(a['success'])==(remaining<=.2 and not a['collision'])
        results.append(dict(id=job['id'],valid=True,success=a['success'],collision=a['collision'],timeout=a['timeout'],remaining_m=remaining,physics_steps=a['physics_steps']))
    assert s['optimizer_calls']==s['Actor_updates']==0 and s['elapsed_s']<p['wall_seconds']
    assert sum(r.get('success',False) and not r.get('collision',False) for r in results)==s['safe']
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(DEST,dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_added=0,optimizer_added=0,all_processes_absent=True,
        original_Actor_SHA_unchanged=True,all_input_protocol_and_payload_SHA_verified=True,
        four_fixed_goal_reset_geometries_verified=True,all_valid_native_snapshots_and_full120_reloads_exact=True,
        all_delivered_actions_frozen_safe_T255_bit_exact=actions_checked,full_global_scan_calls=s['full_global_scan_calls'],
        commands=s['commands'],physics_steps=s['physics_steps'],own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),
        results=results,summary_sha256=C.sha(OUT/'summary.json'),audit_epoch=time.time(),
        original6m_from_original_start_success_added=0,forgetting_or_curriculum_causal_benefit_tested=False,automatic_training=False))
    print(json.dumps(dict(state='PASS',results=results)),flush=True)


if __name__=='__main__':main()
