from recovery_core import *

def main():
    p=read(HERE/'protocol.json');verify(p);dest=DOC/'T338R_AUDIT.json';assert not dest.exists()
    start=time.time();deadline=start+p['audit_seconds'];s=read(OUT/'summary.json')
    assert s['all_workers_stopped']
    w=read(OUT/'workers.json') if (OUT/'workers.json').exists() else dict(workers=[])
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*w['workers']]:assert not Path('/proc/'+str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cases=[json.loads(line) for line in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(line) for line in (OUT/'replays.jsonl').read_text().splitlines()] if (OUT/'replays.jsonl').exists() else []
    blocks=[json.loads(line) for line in (OUT/'blocks.jsonl').read_text().splitlines()] if (OUT/'blocks.jsonl').exists() else []
    physics=sum(r['physics_steps'] for r in [*cases,*reps]);rawcommands=len(cases)+len(reps)+p['preflight_geometry_commands']
    assert s['state']=='COMPLETE','FAILED producer: preserve raw; no full scientific audit PASS'
    assert physics==s['physics_steps']<=p['max_physics'] and rawcommands==s['commands']<=p['max_commands']
    assert s['elapsed_s']<=p['wall_seconds'] and read(OUT/'preflight.json')['synthetic_discarded_Adam_calls']==4
    reference=arrays(p['reference']);byid={r['id']:r for r in [*cases,*p['cached_training_rows']]}
    for row in cases:
        assert K.C.sha(row['path'])==row['sha256'];a=arrays(row['path'])
        if row['full_arrays_saved']:assert {k:digest(v) for k,v in a.items()}==row['array_hashes']
        else:
            for key in ['actions','native_counts','end_flags']:assert digest(a[key])==row['array_hashes'][key]
        flags=a['end_flags'];assert not flags[:-1].any() and flags[-1].any()
        assert bool(flags[-1,0])==row['collision'] and bool(flags[-1,1])==row['success'] and bool(flags[-1,2])==row['timeout']
        assert np.all((a['native_counts']>0)&(a['native_counts']<=100))
        suffix=int(a['native_counts'].astype(np.int64).sum())
        assert suffix==row.get('suffix_steps',row['physics_steps'])
        if row['full_arrays_saved']:
            assert suffix+row['prefix_steps']==row['physics_steps']
            assert np.array_equal(a['states'][0],reference['states'][row['prefix']]) and row['prefix_full12_and_history_exact']
            assert np.array_equal(a['states'][1:],a['next_states'][:-1])
            assert K.R.arrays_hash(a['final_native_integration'])==row['integration_sha256']
            steps=row['prefix_steps']+np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
            assert np.max(np.abs(a['states'][:,112]-(1-steps/40000)))<1e-7
            for k,x in enumerate(a['states']):
                assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x))<=2e-6
                d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
                n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
                r=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0.)
                assert abs(r-a['rewards'][k])<1e-12 and r==a['environment_rewards'][k]
    for r in reps:assert r['id'] in byid and r['physics_steps']==byid[r['id']]['physics_steps'] and r['full_arrays_native_exact']
    def guard():assert time.time()<deadline
    disposable=0
    for arm in p['arms']:
        q=Learner(p,arm)
        for b in [b for b in blocks if b['arm']==arm]:
            guard();assert b['input_hashes']==q.hashes();items=[]
            for ident in b['training_ids']:
                row=byid[ident];a=arrays(row['path']);assert row['policy_hash']==digest(q.actor.state_dict())
                innovations=np.random.default_rng(row['seed']).normal(size=(len(a['states']),2))
                assert np.array_equal(innovations,a['innovations'])
                for k,x in enumerate(a['states']):
                    with torch.no_grad():draw=q.actor.draw(torch.tensor(x[None]),torch.tensor(innovations[k:k+1]))
                    assert all(np.array_equal(v[0].numpy(),a[name][k]) for name,v in draw.items())
                rep=[r for r in [*reps,*p['cached_replays']] if r['id']==ident];assert len(rep)==1
                items.append(dict(row=row,arrays=a))
            trace=optimize(p,q,items,guard,lambda _r:None)
            disposable+=p['epochs']
            assert digest(trace)==b['trace_sha256'] and q.hashes()==b['output_hashes'] and b['retained']
        model=s['models'][arm];assert K.C.sha(model['path'])==model['sha256']
        v=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
        assert q.hashes()==model['hashes']==v['hashes'] and digest(q.state())==digest(v['state'])
    assert p['prior_physics']+physics==s['combined_physics']<=24000000
    assert p['prior_commands']+rawcommands==s['combined_commands']<=900
    assert disposable==s['updates']['live_actor']==s['updates']['shadow_actor']<=p['max_live_Actor_Adam']
    events=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for who in ['live','shadow']:assert sum(e['replica']==who and e['stage']=='AFTER_ADAM' for e in events)==disposable
    save(dest,dict(state='PASS',native_added=0,disposable_historical_Actor_Adam=disposable,commands=rawcommands,physics_steps=physics,
        source_inputs_full_training_arrays_density_reward_clock_history_and_models_exact=True,all_original_processes_absent=True,
        no_new_navigation_training=True,compact_scope='actual actions/native_counts/end_flags and native full-array SHA receipts; not synthetic missing states',
        no_final_old275_six54_fresh16_reload78=True,summary_sha256=K.C.sha(OUT/'summary.json'),elapsed_s=time.time()-start))
    print('T338 independent full evidence/update audit PASS0native')

if __name__=='__main__':
    try:main()
    except BaseException:
        save(DOC/'T338R_AUDIT_ERROR.json',dict(state='NOT_PASSED',error=traceback.format_exc(),no_new_native=True));raise
