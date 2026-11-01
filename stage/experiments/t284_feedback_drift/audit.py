from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('stage_t284_frozen_core',Path(__file__).with_name('run.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
globals().update({k:v for k,v in vars(core).items() if not k.startswith('_')})


def main_audit():
    dest=ROOT/'stage/docs/long_distance_20261101/T284_AUDIT.json';assert not dest.exists()
    p=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text());assert summary['state']=='COMPLETE'
    w=json.loads((OUT/'workers.json').read_text());launch=json.loads((OUT/'launch.json').read_text())
    for pid in [w['parent'],launch['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('process alive or reused',pid))
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    rows=json.loads((OUT/'cases.json').read_text());assert len(rows)==summary['commands']==4
    assert sum(r['physics_steps'] for r in rows)==summary['physics_steps']<=p['max_physics']
    payloads={}
    for r in rows:
        assert r['complete'] and not r['error'] and C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
        assert len(a)==12 and a['states'].shape[1]==120 and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items())
        ref=p['references']['candidate' if r['id'].startswith('candidate') else 'parent']
        assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
        assert a['native_counts'].astype(np.int64).sum()==r['physics_steps'];payloads[r['id']]=a
    base=json.loads(Path(p['source_protocol']).read_text());parent=policy(base).eval().requires_grad_(False);candidate=policy(base)
    export=summary['candidate_export'];assert C.sha(export['path'])==export['sha256'];saved=L.read(export['path']);candidate.load_state_dict(saved['policy']);candidate.eval().requires_grad_(False)
    assert S.digest(candidate.state_dict())==p['candidate_policy_hash'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    result,a=diagnose(parent,candidate,payloads['parent'],payloads['candidate']);actual=json.loads((OUT/'analysis.json').read_text());meta=actual.pop('payload');assert actual==result and C.sha(meta['path'])==meta['sha256']
    with np.load(meta['path']) as z:assert set(z.files)==set(a) and all(np.array_equal(v,z[k]) for k,v in a.items())
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(dest,dict(state='PASS',physics_added=0,optimizer_calls_added=0,all_processes_absent=True,full12_arrays_and_integration_historical_exact=True,
        decomposition_model_readback_and_arrays_exact=True,source_model_payload_SHA_verified=True,total_commands=4,total_physics=summary['physics_steps'],analysis=result,summary_sha256=C.sha(OUT/'summary.json')))
    print('T284 AUDIT PASS')


if __name__=='__main__':main_audit()
