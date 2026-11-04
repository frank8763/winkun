from t348_core import *


def main():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T348_AUDIT.json'
    assert not dest.exists() and s['state']=='COMPLETE'
    launch=read(OUT/'launch.json');workers=read(OUT/'workers.json')
    for pid in [launch['pid'],s['pid'],*workers['workers']]:assert not Path('/proc',str(pid)).exists(),pid
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();deadline=min(start+p['audit_seconds'],launch['all_work_deadline_epoch'])
    def guard():assert time.time()<deadline
    rows=[json.loads(l) for l in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(l) for l in (OUT/'replays.jsonl').read_text().splitlines()]
    blocks=[json.loads(l) for l in (OUT/'blocks.jsonl').read_text().splitlines()]
    calls=[json.loads(l) for l in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    assert len(rows)==59 and len(reps)==54 and len(blocks)==2 and s['commands']==120
    assert s['updates']=={'live':1200,'shadow':1200} and s['new_RL_updates']==0
    data=dataset(p,Actor(p,'reset'),guard);objects={};historical=0
    for block in blocks:
        arm=block['arm'];obj=Learner(p,arm);assert obj.hashes()==block['input_hashes'];events=[]
        trace=fit(p,obj,data,guard,lambda r:events.append(r));historical+=len(trace)
        assert digest(trace)==block['trace_sha256'] and obj.hashes()==block['output_hashes']
        assert read(OUT/('trace_'+arm+'.json'))==trace
        for replica in ['live','shadow']:
            recorded=[{k:v for k,v in row.items() if k not in ['arm','replica']} for row in calls if row['arm']==arm and row['replica']==replica]
            assert recorded==events
        model=s['models'][arm];assert K.C.sha(model['path'])==model['sha256']
        checkpoint=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
        assert digest(checkpoint['state'])==digest(obj.state()) and obj.actor.phash()==model['policy_hash']
        objects[arm]=obj.actor.eval().requires_grad_(False)
    physics=0;full=0
    for row in rows:
        guard();assert K.C.sha(row['path'])==row['sha256'];a=arrays(row['path'])
        assert all(digest(v)==row['array_hashes'][name] for name,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==row['physics_steps'];physics+=row['physics_steps']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        c,g,t=map(bool,a['end_flags'][-1]);assert c==row['collision'] and (g and not c)==row['success'] and t==row['timeout']
        if not row['full_arrays_saved']:continue
        full+=1;q=objects[row['arm']];h=None
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        for k,x in enumerate(a['states']):
            before=np.zeros(16) if h is None else h[0,0].numpy()
            assert np.array_equal(before,a['hidden_before'][k])
            with torch.no_grad():action,loc,new,_=q.step(torch.tensor(x[None]),h)
            assert np.array_equal(action[0].numpy(),a['actions'][k]);h=new if q.memory_on else None
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            count=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**count
            d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
            reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
            if c or g or t:reward+=K.R.GAMMA**(count-1)*(float(g)-float(c)-.5*float(t))
            assert abs(reward-a['rewards'][k])<1e-12
        assert np.array_equal(new[0,0].numpy(),a['last_hidden_after'])
        assert K.R.arrays_hash(a['final_native_integration'])==row['integration_sha256']
        steps=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.abs(a['states'][:,112]-(1-steps/row['cap'])).max()<1e-7
    for row in reps:
        ref=next(r for r in rows if r['arm']==row['arm'] and r['role']=='final27' and r['job_id']==row['job_id'])
        assert row['all_arrays_hidden_history_integration_exact'] and not row['independent_sample']
        assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        assert row['physics_steps']==ref['physics_steps'];physics+=row['physics_steps']
    assert full==4 and historical==1200 and physics==s['physics_steps']<=p['max_physics']
    assert s['elapsed_s']<=p['wall_seconds'] and not s['stable_six'] and not s['entered7or8']
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    save(dest,dict(state='PASS',native_added=0,new_navigation_RL_updates=0,historical_discarded_Adam=historical,
         all_live_shadow_1200_updates_and_final_ActorAdam_reconstructed=True,full_native_payloads=4,
         all_full120_hidden_history_clock_reward_actions_verified=True,all54_model_readbacks_exact=True,
         physics_steps=physics,commands=120,all_native_processes_stopped=True,
         source_models_and_raw_SHA_exact=True,summary_sha256=K.C.sha(OUT/'summary.json'),elapsed_s=time.time()-start))
    print(json.dumps(dict(state='AUDIT_PASS',native=0,historical_Adam=1200,physics_steps=physics)))

if __name__=='__main__':main()
