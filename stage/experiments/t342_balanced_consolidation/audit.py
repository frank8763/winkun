from t342_core import *

def main():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert not (DOC/'T342_AUDIT.json').exists()
    assert s['state']=='COMPLETE' and s['all_workers_stopped'];w=read(OUT/'workers.json');l=read(OUT/'launch.json')
    assert all(not Path('/proc',str(i)).exists() for i in [s['pid'],l['pid'],*w['workers']])
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<start+p['audit_seconds']
    q=Learner(p);data=dataset(p,q.actor,guard);records=[];trace=fit(p,q,data,guard,lambda row:records.append(row))
    f=read(OUT/'fit.json');assert digest(trace)==f['trace_sha256'] and q.hashes()==f['hashes']
    assert metrics(q.actor,data,guard)==f['after'];model=read(OUT/'models.json')['final'];assert K.C.sha(model['path'])==model['sha256']
    v=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu');assert digest(v['state'])==digest(q.state())
    calls=[json.loads(l) for l in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for replica in ['live','shadow']:assert [{k:v for k,v in z.items() if k!='replica'} for z in calls if z['replica']==replica]==records
    rows=[json.loads(l) for l in (OUT/'cases.jsonl').read_text().splitlines()];assert len(rows)+2==s['commands']
    assert sum(r['physics_steps'] for r in rows)==s['physics_steps']<=p['max_physics']
    lookup={(r['role'],r['job_id']):r for r in rows}
    for r in rows:
        guard();assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert set(a)=={'actions','native_counts','end_flags'}
        assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and r['complete'] and not r.get('error')
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        if r['role']=='model_readback27':
            role='final_wide5' if r['job_id'].startswith('wide_dev') else 'final_original2' if r['job_id'].startswith('dev6_') else 'final_old20'
            ref=lookup[(role,r['job_id'])];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    assert s['updates']==dict(live=3000,shadow=3000) and s['RL_updates']==0
    save(DOC/'T342_AUDIT.json',dict(state='PASS',native_added=0,new_training_Adam=0,disposable_historical_Adam=3000,
        full_supervised_history_Actor_Adam_exact=True,model_and_source_SHA_exact=True,all_raw_compact_hashes_and_physics_exact=True,
        model_readback27_native_exact=True,physics_steps=s['physics_steps'],commands=s['commands'],stable_six=s['stable_six'],elapsed_s=time.time()-start,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T342 independent audit PASS,0native/3000discardedhistoricalAdam')

if __name__=='__main__':main()
