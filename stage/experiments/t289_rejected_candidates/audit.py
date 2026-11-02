from run import *


def main_audit():
    dest=ROOT/'stage/docs/distance_20261102/T289_AUDIT.json';assert not dest.exists()
    p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE'
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Process live or reused',pid))
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    rows=json.loads((OUT/'cases.json').read_text());replays=json.loads((OUT/'replays.json').read_text());assert len(rows)==len(replays)==16 and len(rows)+len(replays)==s['commands']==32
    assert sum(r['physics_steps'] for r in rows+replays)==s['physics_steps']<=p['max_physics'];index={r['id']:r for r in rows}
    for r in rows:
        assert r['complete'] and not r['error'] and C.sha(r['path'])==r['sha256'];a=load_arrays(r['path'])
        assert len(a)==12 and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items()) and int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        assert a['states'].shape[1]==a['next_states'].shape[1]==120 and np.array_equal(a['states'][1:],a['next_states'][:-1])
    for r in replays:
        old=index[r['original_id']];assert r['complete'] and not r['error'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    for spec in p['candidates']:
        meta=s['models'][spec['id']];assert C.sha(meta['path'])==meta['sha256'] and S.digest(L.read(meta['path'])['policy'])==spec['policy_hash']
    calls=json.loads((OUT/'historical_optimizer_calls.json').read_text());assert len(calls)==s['historical_clone_optimizer_calls']==4
    original=[json.loads(l) for l in Path(p['source_updates']).read_text().splitlines()];rec=json.loads((OUT/'reconstruction.json').read_text());assert len(rec['original_two_rounds'])==2
    for row,old in zip(rec['original_two_rounds'],original):assert row['round']==old['round'] and row['policy_hash']==old['policy_hash'] and row['optimizer_hash']==old['optimizer_hash'] and row['original_updates_exact'] and row['Actor_Adam_rollback_exact']
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(dest,dict(state='PASS',new_optimizer_calls=0,new_physics_steps=0,all_processes_absent=True,all_input_model_payload_SHA_verified=True,
        all16_full120_and12_arrays_verified=True,all16_native_replay_array_and_integration_hashes_exact=True,historical_clone_calls=4,total_commands=32,total_physics=s['physics_steps'],summary_sha256=C.sha(OUT/'summary.json')))
    print('T289 audit PASS zero new native/optimizer')


if __name__=='__main__':main_audit()
