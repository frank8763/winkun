from t340_core import *

def main():
    p=read(HERE/'protocol.json');verify(p);dest=DOC/'T340_AUDIT.json';assert not dest.exists()
    start=time.time();deadline=start+p['audit_seconds'];s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and s['all_workers_stopped']
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc/'+str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()] if (OUT/'replays.jsonl').exists() else []
    blocks=[json.loads(v) for v in (OUT/'blocks.jsonl').read_text().splitlines()] if (OUT/'blocks.jsonl').exists() else []
    assert len(cases)+len(reps)+1==s['commands']<=p['max_commands']
    assert sum(v['physics_steps'] for v in [*cases,*reps])==s['physics_steps']<=p['max_physics']
    reference=arrays(p['reference']);byid={v['id']:v for v in cases}
    for row in cases:
        assert row['complete'] and not row.get('error') and K.C.sha(row['path'])==row['sha256'];a=arrays(row['path'])
        for name,v in a.items():
            if name in row['array_hashes']:assert digest(v)==row['array_hashes'][name]
        flags=a['end_flags'];assert not flags[:-1].any() and flags[-1].any()
        assert list(map(bool,flags[-1]))==[row['collision'],row['success'],row['timeout']]
        assert int(a['native_counts'].astype(np.int64).sum())==row.get('suffix_steps',row['physics_steps'])
        if not row['full_arrays_saved']:continue
        assert row['role']=='training' and row['prefix_full12_and_history_exact']
        assert {k:digest(v) for k,v in a.items()}==row['array_hashes']
        assert np.array_equal(a['states'][0],reference['states'][row['prefix']]) and np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert K.R.arrays_hash(a['final_native_integration'])==row['integration_sha256']
        assert row['prefix_steps']+int(a['native_counts'].astype(np.int64).sum())==row['physics_steps']
        for k,x in enumerate(a['states']):
            assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x))<=2e-6
            d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            r=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0.)
            assert abs(r-a['rewards'][k])<1e-12 and r==a['environment_rewards'][k]
        exact=[v for v in reps if v['id']==row['id']];assert len(exact)==1 and exact[0]['physics_steps']==row['physics_steps'] and exact[0]['full_arrays_native_exact']
    def guard():assert time.time()<deadline
    q=Learner(p,'capg');q.opt=torch.optim.Adam(q.actor.trainable(),lr=p['warm_lr']);warm=read(OUT/'warm_block.json')
    assert q.hashes()==warm['input_hashes']
    trace=warmup(p,q.actor,q.opt,demos(p,q.actor),guard,lambda _r:None)
    assert len(trace)==p['warm_steps'] and digest(trace)==warm['trace_sha256'] and q.hashes()==warm['output_hashes']
    def check_model(key):
        record=s['models'][key];assert K.C.sha(record['path'])==record['sha256']
        value=torch.load(io.BytesIO(gzip.decompress(Path(record['path']).read_bytes())),weights_only=False,map_location='cpu')
        assert q.hashes()==record['hashes']==value['hashes'] and digest(q.state())==digest(value['state'])
    check_model('warm');q.opt=torch.optim.Adam(q.actor.trainable(),lr=p['actor_lr']);successes=[];calls=0
    for block in blocks:
        assert q.hashes()==block['input_hashes'];items=[]
        for ident in block['training_ids']:
            row=byid[ident];a=arrays(row['path']);assert row['policy_hash']==digest(q.actor.state_dict())
            innovation=np.random.default_rng(row['seed']).normal(size=(len(a['states']),2));assert np.array_equal(innovation,a['innovations'])
            with torch.no_grad():draw=q.actor.draw(torch.tensor(a['states']),torch.tensor(innovation))
            assert all(np.array_equal(v.numpy(),a[name]) for name,v in draw.items())
            item=dict(row=row,arrays=a);items.append(item)
            if row['success'] and not row['collision']:successes.append(item)
        assert [i['row']['id'] for i in successes]==block['success_ids']
        trace=supported_update(p,q,items,demos(p,q.actor,successes),block['round'],guard,lambda _r:None)
        calls+=p['epochs'];assert digest(trace)==block['trace_sha256'] and q.hashes()==block['output_hashes']
    check_model('final')
    assert s['updates']==dict(live_warm=p['warm_steps'],shadow_warm=p['warm_steps'],live_actor=calls,shadow_actor=calls)
    if s['stable_six']:
        for group,total in [('old275',275),('six54',54),('fresh16',16),('reload78',78)]:
            value=s['result'][group];assert value['success']==total and value['collision']==value['timeout']==0
        assert all(v['independent_model_readback'] for v in cases if v['role']=='reload78')
    save(dest,dict(state='PASS',native_added=0,disposable_historical_warm_Adam=p['warm_steps'],disposable_historical_RL_Adam=calls,
        all_raw_training_arrays_density_reward_history_clock_exact=True,all_model_optimizer_and_exactshadow_reconstructed=True,
        all_native_training_replays_exact=True,source_inputs_unchanged=True,all_original_processes_absent=True,
        commands=s['commands'],physics_steps=s['physics_steps'],stable_six=s['stable_six'],
        summary_sha256=K.C.sha(OUT/'summary.json'),elapsed_s=time.time()-start))
    print('T340 independent raw/warm/RL/model audit PASS0newnative')

if __name__=='__main__':
    try:main()
    except BaseException:
        save(DOC/'T340_AUDIT_ERROR.json',dict(state='NOT_PASSED',error=traceback.format_exc(),native_added=0));raise
