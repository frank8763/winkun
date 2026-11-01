from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('stage_t285_frozen_core',Path(__file__).with_name('run.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
globals().update({k:v for k,v in vars(core).items() if not k.startswith('_')})


def main_audit():
    dest=ROOT/'stage/docs/long_distance_20261101/T285_AUDIT.json';assert not dest.exists()
    p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE'
    def alarm(signum,frame):raise TimeoutError('T285 audit300s')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['postrun_audit_seconds'])
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('still alive or PID reused',pid))
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    rows=json.loads((OUT/'cases.json').read_text()) if (OUT/'cases.json').exists() else []
    assert len(rows)==s['commands']<=p['max_commands'] and sum(r['physics_steps'] for r in rows)==s['physics_steps']<=p['max_physics']
    payloads={}
    for r in rows:
        assert C.sha(r['path'])==r['sha256'] and r['complete'] and not r['error'];a=read_arrays(r['path'])
        assert set(a)==set(r['saved_array_keys']) and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'];payloads[r['id']]=a
    base=json.loads(Path(p['source_protocol']).read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parent=policy(base);data=prepare_data(parent,p);ax,ay,offset=anchors(base,parent);before=objective(parent,data,ax,ay,offset,base,p['balance'])
    assert before==json.loads((OUT/'parent_objective.json').read_text());states=read_arrays(p['parent_trajectory'])['states']
    for arm,ids in p['arms'].items():
        model,stats=build_candidate(base,p,ids,states);expected=s['arms'][arm]
        assert all(expected[k]==v for k,v in stats.items());assert objective(model,data,ax,ay,offset,base,p['balance'])==expected['measurement']
        export=s['models'][arm];assert C.sha(export['path'])==export['sha256'] and S.digest(L.read(export['path'])['policy'])==S.digest(model.state_dict())
        seen=[r for r in rows if r['arm']==arm];a=[r for r in seen if r['role']=='critical'];b=[r for r in seen if r['role']=='critical_replay']
        if expected['numeric_gate']:
            assert len(a)==len(b)==1 and a[0]['array_hashes']==b[0]['array_hashes'] and a[0]['integration_sha256']==b[0]['integration_sha256']
            assert len(payloads[a[0]['id']])==12 and all(np.array_equal(v,payloads[b[0]['id']][k]) for k,v in payloads[a[0]['id']].items())
        else:assert not seen
        for role,key in [('old_guard','old20'),('diagnostic_six_m','diagnostic_six_m'),('local','local')]:
            rr=[r for r in seen if r['role']==role]+(a if key=='old20' else [])
            assert len(rr)==expected[key+'_executed'] and sum(r['success'] and not r['collision'] for r in rr)==expected[key+'_safe']
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    signal.alarm(0);C.save(dest,dict(state='PASS',physics_added=0,optimizer_calls_added=0,all_processes_absent=True,source_payload_model_SHA_verified=True,
        two_full_projection_and_objective_reconstructions_exact=True,all_saved_arrays_and_native_counts_verified=True,critical_full12_and_integration_replay_exact=True,
        total_commands=s['commands'],total_physics=s['physics_steps'],summary_sha256=C.sha(OUT/'summary.json')))
    print('T285 AUDIT PASS')


if __name__=='__main__':main_audit()
