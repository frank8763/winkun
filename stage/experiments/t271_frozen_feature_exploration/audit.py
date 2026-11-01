"""Stopped-run evidence audit; zero simulator/optimizer calls."""
from pathlib import Path
import json
import gzip
import os
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from t271_runtime import *
OUT=ROOT/'stage/runs/t271_frozen_feature_exploration'
DEST=ROOT/'stage/docs/long_distance_20261101/T271_AUDIT.json'


def main():
    assert not DEST.exists(),'Never overwrite a completed audit'
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['all_workers_stopped'] and s['state']=='COMPLETE'
    w=json.loads((OUT/'workers.json').read_text())
    for pid in [w['parent'],*w['workers']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Process still exists or PID reused',pid))
    for path,h in p['input_hashes'].items():assert C.sha(path)==h,path
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    rows=lambda name:list(map(json.loads,(OUT/name).read_text().splitlines()))
    gg,cc,ee,rr=[rows(n) for n in ['geometry.jsonl','controls.jsonl','exploration.jsonl','replays.jsonl']]
    assert [len(a) for a in [gg,cc,ee,rr]]==[4,4,32,32]
    assert all(r['valid'] and r['physics_steps']==0 for r in gg)
    assert sum(r['physics_steps'] for r in cc+ee+rr)==s['physics_steps']
    assert sum(map(len,[gg,cc,ee,rr]))==s['commands']==72
    q=policy(p).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
    replay={r['id']:r for r in rr};feature_rows=0;max_density_error=0.
    for row in cc+ee:
        assert row['policy_hash']==p['policy_hash'] and row['complete']
        assert C.sha(row['path'])==row['sha256']
        with np.load(row['path']) as z:a={k:z[k].copy() for k in z.files}
        assert all(S.digest(v)==row['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==row['physics_steps']
        if row in cc:
            ref=p['control_references'][row['source_reset_id']]
            assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        else:
            r=replay[row['id']]
            assert r['array_hashes']==row['array_hashes'] and r['integration_sha256']==row['integration_sha256']
            assert r['physics_steps']==row['physics_steps'] and r['job']==row['job']
        if row['mode']=='feature_copula':
            assert set(a)==set(row['array_hashes']) and a['states'].shape[1]==120
            for k,x in enumerate(a['states']):
                with torch.no_grad():_,mean,h,_=q.step(torch.tensor(x[None]))
                assert np.array_equal(mean[0].numpy(),a['means'][k])
                f=F.unit_feature(h);assert np.array_equal(f.numpy(),a['feature_vectors'][k])
                z=F.latent_sample(f,torch.tensor(a['block_weights'][k//F.BLOCK]),a['independent_innovations'][k])
                assert np.array_equal(z.numpy(),a['latent_draws'][k])
                assert np.array_equal(F.AR.sample(mean[0],z).numpy(),a['actions'][k])
            lp=F.sequence_logp(torch.tensor(a['means']),torch.tensor(a['actions']),torch.tensor(a['feature_vectors']))
            error=float((lp-torch.tensor(a['logp'])).abs().max());assert error==0
            max_density_error=max(max_density_error,error);feature_rows+=len(a['states'])
    for k in range(0,32,2):
        a,b=ee[k:k+2]
        assert a['mode']=='conditional_AR' and b['mode']=='feature_copula'
        assert a['source_reset_id']==b['source_reset_id'] and a['seed']==b['seed']
        for key in ['case','delta_x_m','delta_y_m','delta_yaw_deg','exact_snapshot']:
            assert a['job'][key]==b['job'][key]
    assert s['optimizer_calls']==s['new_navigation_RL_updates']==0
    assert s['physics_steps']<=p['max_physics'] and s['elapsed_s']<p['wall_seconds']+10
    own=S.used(OUT);used=S.used(ROOT/'stage/runs')
    assert own<p['final_bytes'] and used<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    # Hash equality means reuse/repetition, never another independent sample.
    unique=len({(r['integration_sha256'],tuple(sorted(r['array_hashes'].items()))) for r in ee})
    core=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards']
    signature=lambda r:(r.get('integration_sha256'),tuple(r.get('array_hashes',{}).get(k) for k in core))
    history={}
    for directory in ['t257_distance_policy_gradient','t259_closed_loop_recovery','t263_adaptive_goal_course',
                      't267_causal_credit_course','t268_storage_and_frozen_gate','t269_same_state_controls']:
        folder=ROOT/'stage/runs'/directory
        for f in sorted([*folder.glob('*.jsonl'),*folder.glob('*.jsonl.gz')]):
            content=gzip.decompress(f.read_bytes()).decode() if f.suffix=='.gz' else f.read_text()
            for line in content.splitlines():
                r=json.loads(line)
                if r.get('integration_sha256') and all(k in r.get('array_hashes',{}) for k in core):
                    history.setdefault(signature(r),[]).append(dict(file=str(f.relative_to(ROOT)),id=r.get('id')))
    overlaps=[dict(id=r['id'],matches=history[signature(r)]) for r in ee if signature(r) in history]
    result=dict(state='POSTRUN_EVIDENCE_RECONCILED',all_processes_absent=True,physics_steps_added=0,
        optimizer_calls_added=0,total_physics_from_raw=s['physics_steps'],commands=s['commands'],
        four_zero_physics_geometry_valid=True,four_deterministic_controls_exact_T268=True,
        exact_full_array_and_integration_replays=32,matched_pairs=16,unique_exploration_hash_sets=unique,
        historical_exact10array_and_integration_overlaps=overlaps,
        feature_rows_actor_mean_feature_latent_action_rebuilt=feature_rows,conditional_density_max_error=max_density_error,
        all36_saved_payload_sha_verified=True,source_and_frozen_checkpoint_sha_verified=True,
        original_summary_sha256=C.sha(OUT/'summary.json'),protocol_sha256=C.sha(OUT/'protocol.json'),
        own_bytes=own,global_bytes=used,audit_epoch=time.time(),six_m_deterministic_gate=False,
        interpretation='Fixed exploration draws on four known starts; no trained-policy improvement or population superiority established.')
    C.save(DEST,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
