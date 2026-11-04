from t351_core import *
import signal


def main():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T351_AUDIT.json'
    assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T351 audit1200s')));signal.alarm(1200)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time()
    rows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    blocks=[json.loads(v) for v in (OUT/'blocks.jsonl').read_text().splitlines()]
    calls=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()] if (OUT/'optimizer_calls.jsonl').exists() else []
    assert s['commands']==7+len(rows)+len(reps)<=p['max_commands']
    obj=Learner(p);policies={obj.actor.phash():obj.actor.learned()}
    def item(r):return dict(row=r,arrays=arrays(r['path']))
    byid={r['job_id']:r for r in rows if r['role']=='initial_old275'}
    olditems=[item(byid[j['id']]) for j in p['old_jobs']]
    support=[item(r) for r in rows if r['role']=='initial_wide5' and r['full_arrays_saved']]
    refs=reference_data(p,obj.actor,olditems,support,lambda:None);historical=0
    retained_old={r['job_id'] for r in byid.values() if r['success'] and r['job_id'] in {j['id'] for j in p['old_jobs']}}
    retained_wide={r['job_id'] for r in rows if r['role']=='initial_wide5' and r['success']}
    for block in blocks:
        rnd=block['round'];assert obj.hashes()==block['input_hashes'];before=obj.state();events=[]
        train=[item(r) for r in rows if r['role']==f'train_r{rnd}'];assert len(train)==4
        got=update(p,obj,train,refs,rnd,lambda:None,lambda r:events.append(r));historical+=p['epochs']
        assert got==block['update'] and obj.hashes()==block['proposed_hashes'] and block['exact_shadow']
        for replica in ['live','shadow']:
            recorded=[dict(phase=v[2],step=v[3]) for v in calls if v[0]==rnd and v[1]==replica];assert recorded==events
        metrics=block['metrics'];keep=retained_old<=set(metrics['old_success_ids']) and retained_wide<=set(metrics['wide_success_ids']);assert keep==block['retained']
        actual=[r for r in rows if r['role']==f'candidate_r{rnd}'];assert len(actual)==27
        assert set(metrics['old_success_ids'])=={r['job_id'] for r in actual[:20] if r['success']}
        assert set(metrics['wide_success_ids'])=={r['job_id'] for r in actual[20:25] if r['success']}
        if keep:retained_old=set(metrics['old_success_ids']);retained_wide=set(metrics['wide_success_ids'])
        else:obj.load(before)
        assert obj.hashes()==block['output_hashes'];policies[obj.actor.phash()]=obj.actor.learned()
    assert historical==s['updates']['live']==s['updates']['shadow']<=8
    v=torch.load(io.BytesIO(gzip.decompress(Path(s['models']['path']).read_bytes())),weights_only=False,map_location='cpu')
    assert digest(v['state'])==digest(obj.state()) and K.C.sha(s['models']['path'])==s['models']['sha256']
    physics=0;full=0
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:assert {k:digest(z[k]) for k in z.files}==r['saved_array_hashes']
        a=arrays(r['path']);physics+=r['physics_steps'];assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        c,g,t=map(bool,a['end_flags'][-1]);assert c==r['collision'] and (g and not c)==r['success'] and t==r['timeout']
        if not r['full_arrays_saved']:continue
        full+=1;assert {k:digest(v) for k,v in a.items()}==r['full_array_hashes']
        assert digest(r['full_array_hashes'])==r['all_arrays_digest']
        qq=Actor(p).requires_grad_(False);qq.load_learned(policies[r['policy_hash']]);qq.sample=r['stochastic'];qq.rng=np.random.default_rng(r['seed']);h=None
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        for k,x in enumerate(a['states']):
            bh=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(bh,a['hidden_before'][k])
            with torch.no_grad():action,loc,h,_=qq.step(torch.tensor(x[None]),h)
            assert np.array_equal(action[0].numpy(),a['actions'][k])
            for name,value in qq.tape[-1].items():assert np.array_equal(value,a[name][k]),(r['id'],k,name)
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
            reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
            if c or g or t:reward+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
            assert abs(reward-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']
    for r in reps:
        assert r['all_arrays_hidden_history_integration_exact'] and not r['independent_sample']
        assert any(v['job_id']==r['job_id'] and v['policy_hash']==r['policy_hash'] and v['all_arrays_digest']==r['all_arrays_digest'] and v['integration_sha256']==r['integration_sha256'] for v in rows)
        physics+=r['physics_steps']
    assert physics==s['physics_steps']<=p['max_physics'];assert full==22+s['new_training_native_collection']
    assert len([r for r in reps if r['role']=='model_readback27'])==27
    assert s['elapsed_s']<=p['wall_seconds'] and not s['overnight_started'] and not s['entered7or8']
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    save(dest,dict(state='PASS',native_added=0,new_RL_added=0,historical_discarded_Adam=historical,
        actual_live_shadow_ActorAdam_reconstructed=True,all_full120_history_hidden_iid_density_rewards_exact=True,
        full_native_cases=full,all_model_readbacks_exact=True,commands=s['commands'],physics_steps=physics,
        all_native_processes_stopped=True,source_models_raw_SHA_exact=True,stable_six=s['stable_six'],elapsed_s=time.time()-start))
    print(json.dumps(dict(state='AUDIT_PASS',native=0,historical_Adam=historical)))


if __name__=='__main__':main()
