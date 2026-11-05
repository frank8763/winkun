from t358_core import *


def main(outpath):
    out=Path(outpath);p=read(out/'protocol.json');s=read(out/'summary.json');assert s['state']=='COMPLETE' and not (out/'audit.json').exists()
    for pid in [s['pid'],*read(out/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Audit900s')));signal.alarm(900)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time()
    rows=[json.loads(v) for v in (out/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(v) for v in (out/'replays.jsonl').read_text().splitlines()]
    blocks=[json.loads(v) for v in (out/'blocks.jsonl').read_text().splitlines()] if (out/'blocks.jsonl').exists() else []
    calls=[json.loads(v) for v in (out/'optimizer_calls.jsonl').read_text().splitlines()] if (out/'optimizer_calls.jsonl').exists() else []
    policies={};historical=0
    for arm,rho in p['arms'].items():
        obj=NightLearner(p,rho);policies[obj.actor.phash()]=obj.actor.learned();refs=references(p,obj.actor)
        initial=[v for v in rows if v['role']=='initial_'+arm];assert len(initial)==32
        kept_old={v['job_id'] for v in initial[:20] if v['success']};kept_wide={v['job_id'] for v in initial[20:25] if v['success']};kept_narrow={v['job_id'] for v in initial[27:32] if v['success']}
        for b in [v for v in blocks if v['arm']==arm]:
            assert obj.hashes()==b['input_hashes'];before=obj.state();events=[]
            train=[dict(row=v,arrays=arrays(v['path'])) for v in rows if v['role']==f"train_{arm}_r{b['round']}"]
            assert len(train)==4
            trace=optimize(p,obj,train,refs,b['round'],lambda:None,lambda r:events.append(r));historical+=p['epochs']
            assert trace==b['update'] and obj.hashes()==b['proposed_hashes']
            for replica in ['live','shadow']:
                assert [dict(phase=v[3],step=v[4]) for v in calls if v[:3]==[arm,b['round'],replica]]==events
            candidates=[v for v in rows if v['role']==f"candidate_{arm}_r{b['round']}"];assert len(candidates)==32
            so={v['job_id'] for v in candidates[:20] if v['success']};sw={v['job_id'] for v in candidates[20:25] if v['success']};sn={v['job_id'] for v in candidates[27:32] if v['success']}
            keep=kept_old<=so and kept_wide<=sw and kept_narrow<=sn;assert keep==b['kept'] and so==set(b['metrics']['old_success_ids']) and sw==set(b['metrics']['wide_success_ids'])
            assert sn==set(b['metrics']['narrow_success_ids'])
            if keep:kept_old,kept_wide,kept_narrow=so,sw,sn
            else:obj.load(before)
            assert obj.hashes()==b['output_hashes'];policies[obj.actor.phash()]=obj.actor.learned()
        m=s['models'][arm];assert K.C.sha(m['path'])==m['sha256']
        v=torch.load(io.BytesIO(gzip.decompress(Path(m['path']).read_bytes())),weights_only=False,map_location='cpu')
        assert digest(v['state'])==digest(obj.state())
    physics=0;full=0
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:assert {k:digest(z[k]) for k in z.files}==r['saved_array_hashes']
        a=arrays(r['path']);physics+=r['physics_steps'];assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        c,g,t=map(bool,a['end_flags'][-1]);assert c==r['collision'] and (g and not c)==r['success'] and t==r['timeout']
        if not r['full_arrays_saved']:continue
        full+=1;assert {k:digest(v) for k,v in a.items()}==r['full_array_hashes'] and digest(r['full_array_hashes'])==r['all_arrays_digest']
        assert r['circle_shift_x_m']==p['training_shift_x_m'] and r['cap']==40000
        q=NightActor(p,r['rho']).requires_grad_(False);q.load_learned(policies[r['policy_hash']]);q.sample=True;q.rng=np.random.default_rng(r['seed']);h=None
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        for k,x in enumerate(a['states']):
            bh=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(bh,a['hidden_before'][k])
            with torch.no_grad():action,loc,h,_=q.step(torch.tensor(x[None]),h)
            assert np.array_equal(action[0].numpy(),a['actions'][k])
            for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k]),(r['id'],k,name)
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
            reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
            if c or g or t:reward+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
            assert abs(reward-a['rewards'][k])<1e-12
        assert K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256'] and np.array_equal(h[0,0].numpy(),a['last_hidden_after'])
    for r in reps:
        assert r['all_arrays_hidden_history_integration_exact'] and not r['independent_sample']
        assert any(v['job_id']==r['job_id'] and v['policy_hash']==r['policy_hash'] and v['all_arrays_digest']==r['all_arrays_digest'] and v['integration_sha256']==r['integration_sha256'] for v in rows)
        physics+=r['physics_steps']
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==3+len(rows)+len(reps)<=p['max_commands']
    assert historical==s['updates']['live']==s['updates']['shadow'] and full%4==0
    assert s['elapsed_s']<=p['wall_seconds'];save(out/'audit.json',dict(state='PASS',native_added=0,historical_discarded_Adam=historical,
        source_models_raw_SHA_exact=True,full_cases=full,whole_current_sequence_AR_density_reward_hidden_exact=True,
        all_native_model_readbacks_exact=True,all_optimizer_shadow_rollback_reconstructed=True,commands=s['commands'],physics_steps=physics,elapsed_s=time.time()-start))
    print(json.dumps(dict(state='AUDIT_PASS',historical_Adam=historical,native=0)))


if __name__=='__main__':main(sys.argv[1])
