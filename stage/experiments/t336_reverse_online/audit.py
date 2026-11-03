from t336_core import *

def main():
    p=read(HERE/'protocol.json');verify(p);dest=DOC/'T336_AUDIT.json';assert not dest.exists()
    start=time.time();deadline=min(start+p['audit_seconds'],p['absolute_all_work_deadline_epoch'])
    s=read(OUT/'summary.json');assert s['all_workers_stopped']
    w=read(OUT/'workers.json') if (OUT/'workers.json').exists() else dict(workers=[])
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*w['workers']]:assert not Path('/proc/'+str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cases=[json.loads(line) for line in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(line) for line in (OUT/'replays.jsonl').read_text().splitlines()] if (OUT/'replays.jsonl').exists() else []
    blocks=[json.loads(line) for line in (OUT/'blocks.jsonl').read_text().splitlines()] if (OUT/'blocks.jsonl').exists() else []
    physics=sum(r['physics_steps'] for r in [*cases,*reps]);assert physics==s['physics_steps']<=p['max_physics']
    assert len(cases)+len(reps)+p['preflight_geometry_commands']==s['commands']<=p['max_commands']
    assert s['state']=='COMPLETE','FAILED producer requires a separate explicit partial audit, never promote to PASS'
    reference=arrays(p['reference']);disposable=dict(actor=0,critic=0)
    for row in cases:
        assert K.C.sha(row['path'])==row['sha256'];a=arrays(row['path'])
        if row['full_arrays_saved']:assert {k:digest(v) for k,v in a.items()}==row['array_hashes']
        else:
            for key in ['actions','native_counts','end_flags']:assert digest(a[key])==row['array_hashes'][key]
        flags=a['end_flags'];assert not flags[:-1].any() and flags[-1].any()
        assert bool(flags[-1,0])==row['collision'] and bool(flags[-1,1])==row['success'] and bool(flags[-1,2])==row['timeout']
        assert np.all((a['native_counts']>0)&(a['native_counts']<=100))
        suffix=int(a['native_counts'].astype(np.int64).sum())
        if row['full_arrays_saved']:
            assert suffix==row['suffix_steps'] and suffix+row['prefix_steps']==row['physics_steps']
            assert np.array_equal(a['states'][0],reference['states'][row['prefix']])
            assert np.array_equal(a['states'][1:],a['next_states'][:-1])
            assert row['prefix_full12_and_history_exact']
            assert K.R.arrays_hash(a['final_native_integration'])==row['integration_sha256']
            steps=row['prefix_steps']+np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
            assert np.max(np.abs(a['states'][:,112]-(1-steps/40000)))<1e-7
            for k,x in enumerate(a['states']):
                assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x))<=2e-6
                d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
                n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
                r=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0.)
                assert abs(r-a['rewards'][k])<1e-12 and r==a['environment_rewards'][k]
            rep=[r for r in reps if r['id']==row['id']];assert len(rep)==1 and rep[0]['physics_steps']==row['physics_steps'] and rep[0]['full_arrays_native_exact']
        else:assert suffix==row.get('suffix_steps',row['physics_steps'])
    def guard():assert time.time()<deadline
    for arm in p['arms']:
        q=Learner(p);pd=prior(p,q.actor);old=anchors(p,q.actor);online=[]
        training=[r for r in cases if r['arm']==arm and r['role']=='training']
        by_round={}
        for r in training:
            # Original id begins arm_training_rN_nK.
            roundno=int(r['job_id'].split('_')[0][1:]);by_round.setdefault(roundno,[]).append(r)
        for roundno,rows in sorted(by_round.items()):
            assert len(rows)==4
            for r in rows:
                a=arrays(r['path']);assert r['policy_hash']==digest(q.actor.state_dict())
                for k,x in enumerate(a['states']):
                    with torch.no_grad():
                        mu=q.actor.step(torch.tensor(x[None]))[1][0]
                        act,lp=q.actor.sample(torch.tensor(x[None]),torch.tensor(a['innovations'][k:k+1]))
                    assert np.array_equal(mu.numpy(),a['latent_means'][k]) and np.array_equal(act[0].numpy(),a['actions'][k]) and float(lp[0,0])==a['logp'][k]
                online.append(tensor_data(a,q.actor))
            data=combine(online)
            for b in [v for v in blocks if v['arm']==arm and (v['round']==roundno or (roundno==0 and v['warmup']))]:
                assert q.hashes()==b['input_hashes'];saved=q.actor_state()
                trace=optimize(p,q,pd,data,old,b['seed'],b['steps'],not b['warmup'],guard,lambda _r:None)
                disposable['critic']+=b['steps'];disposable['actor']+=0 if b['warmup'] else b['steps']
                assert digest(trace)==b['trace_sha256'] and q.hashes()==b['output_hashes']
                if not b['warmup']:
                    if not b['accepted']:q.rollback(saved)
                    assert q.hashes()==b['retained_hashes']
        model=s['models'][arm];assert K.C.sha(model['path'])==model['sha256']
        v=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
        assert q.hashes()==model['hashes']==v['hashes'] and digest(q.state())==digest(v['state'])
    assert disposable['actor']==s['updates']['live_actor'] and disposable['critic']==s['updates']['live_critic']
    assert s['updates']['live_actor']==s['updates']['shadow_actor'] and s['updates']['live_critic']==s['updates']['shadow_critic']
    save(dest,dict(state='PASS',native_added=0,disposable_historical_Adam=disposable,
        all_real_training_arrays_clock_history_reward_density_exact=True,all_live_shadow_and_Actor_Adam_rollbacks_exact=True,
        compact_validation_scope='3actualarrays+allfullarraySHA/native receipts; no missing numeric states fabricated',
        commands=s['commands'],physics_steps=physics,all_original_processes_absent=True,
        summary_sha256=K.C.sha(OUT/'summary.json'),elapsed_s=time.time()-start))
    print('T336 full training/update/model/raw independent audit PASS0native')

if __name__=='__main__':
    try:main()
    except BaseException:
        save(DOC/'T336_AUDIT_ERROR.json',dict(state='AUDIT_FAILED_OR_PARTIAL_NOT_PASSED',error=traceback.format_exc(),no_new_native=True))
        raise
