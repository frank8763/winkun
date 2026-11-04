from t339_core import *

def main():
    p=read(HERE/'protocol.json');verify(p);dest=DOC/'T339_AUDIT.json';assert not dest.exists()
    start=time.time();deadline=start+p['audit_seconds'];s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and s['all_workers_stopped']
    w=read(OUT/'workers.json')
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*w['workers']]:assert not Path('/proc/'+str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    assert len(cases)+len(reps)+1==s['commands']==40
    physics=sum(r['physics_steps'] for r in [*cases,*reps]);assert physics==s['physics_steps']<=p['max_physics']
    reference=arrays(p['reference']);verified=[]
    for row in cases:
        assert row['complete'] and not row.get('error') and K.C.sha(row['path'])==row['sha256']
        a=arrays(row['path'])
        for name,v in a.items():
            if name in row['array_hashes']:assert digest(v)==row['array_hashes'][name]
        flags=a['end_flags'];assert not flags[:-1].any() and flags[-1].any()
        assert list(map(bool,flags[-1]))==[row['collision'],row['success'],row['timeout']]
        suffix=int(a['native_counts'].astype(np.int64).sum());assert suffix==row.get('suffix_steps',row['physics_steps'])
        if row['role']=='verification':
            index=int(row['job_id'].removeprefix('cached'));cachedrow=p['cached_training_rows'][index]
            assert row['array_hashes']==cachedrow['array_hashes'] and row['integration_sha256']==cachedrow['integration_sha256']
            assert row['cached_replay_not_new_training'];verified.append(cachedrow['id'])
        if not row['full_arrays_saved']:continue
        assert row['prefix_steps']+suffix==row['physics_steps']
        assert np.array_equal(a['states'][0],reference['states'][row['prefix']])
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert K.R.arrays_hash(a['final_native_integration'])==row['integration_sha256']
        for k,x in enumerate(a['states']):
            assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x))<=2e-6
            d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            r=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0.)
            assert abs(r-a['rewards'][k])<1e-12 and r==a['environment_rewards'][k]
        exact=[r for r in reps if r['id']==row['id']];assert len(exact)==1 and exact[0]['physics_steps']==row['physics_steps'] and exact[0]['full_arrays_native_exact']
    assert len(verified)==4 and len(reps)==5
    q=Learner(p,'capg');b=read(OUT/'block.json');assert q.hashes()==b['input_hashes']
    data=cached(p)
    for item in data:
        a=item['arrays'];assert item['row']['policy_hash']==digest(q.actor.state_dict())
        with torch.no_grad():draw=q.actor.draw(torch.tensor(a['states']),torch.tensor(a['innovations']))
        assert all(np.array_equal(v.numpy(),a[name]) for name,v in draw.items())
    trace=optimize(p,q,data,lambda:(_ for _ in ()).throw(TimeoutError('Audit600s')) if time.time()>deadline else None,lambda _r:None)
    assert len(trace)==q.calls==8 and digest(trace)==b['trace_sha256'] and q.hashes()==b['output_hashes']
    model=s['models']['capg'];assert K.C.sha(model['path'])==model['sha256']
    saved=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
    assert q.hashes()==model['hashes']==saved['hashes'] and digest(q.state())==digest(saved['state'])
    for row in [r for r in cases if r['full_arrays_saved']]:
        a=arrays(row['path'])
        with torch.no_grad():draw=q.actor.draw(torch.tensor(a['states']),torch.tensor(a['innovations']),True)
        assert all(np.array_equal(v.numpy(),a[name]) for name,v in draw.items())
    assert s['updates']==dict(live_actor=8,shadow_actor=8)
    events=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for who in ['live','shadow']:assert sum(v['replica']==who and v['stage']=='AFTER_ADAM' for v in events)==8
    save(dest,dict(state='PASS',native_added=0,disposable_historical_Actor_Adam=8,
        all_source_inputs_models_optimizer_and_cached_onpolicy_density_exact=True,all4_cached_native_replays_full_exact=True,
        updated5_native_readback_replays_full_exact=True,all_original_processes_absent=True,
        commands=s['commands'],physics_steps=physics,not_full_original_six_gate=True,
        summary_sha256=K.C.sha(OUT/'summary.json'),elapsed_s=time.time()-start))
    print('T339 independent native evidence/8historicalAdam audit PASS0newnative')

if __name__=='__main__':
    try:main()
    except BaseException:
        save(DOC/'T339_AUDIT_ERROR.json',dict(state='NOT_PASSED',error=traceback.format_exc(),native_added=0));raise
