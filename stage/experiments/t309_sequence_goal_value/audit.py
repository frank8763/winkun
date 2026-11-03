from sequence_value import *

def main():
    dest=DOC/'T309_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['inflight']
    w=read(OUT/'workers.json')
    for pid in [w['parent'],*w['workers'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    start=time.time();verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);actor=B.actor(p)
    raw=read(OUT/'cases.json');replays=read(OUT/'replays.json');by={r['id']:r for r in raw};full=[r for r in raw if r['arm']=='center']
    assert len(raw)==len(replays)==128 and len(full)==64 and s['commands']==272
    assert sum(r['physics_steps'] for r in raw+replays)==s['physics_steps']==s['raw_record_physics']<=p['max_physics']
    geometry=read(OUT/'geometry.json');assert len(geometry)==16 and all(r['valid'] and r['physics_steps']==0 for r in geometry)
    for r in raw:
        assert C.sha(r['path'])==r['sha256'] and np.array_equal(r['goal_xy'],r['original_job']['active_goal_xy'])
        if r['arm']=='center':
            a=B.arrays(r['path']);assert len(a)==16 and a['states'].shape[1]==120
            assert {k:S.digest(v) for k,v in a.items()}==r['array_hashes']
            assert np.array_equal(a['states'][1:],a['next_states'][:-1])
            rng=np.random.default_rng(r['seed']);z=rng.normal(size=a['innovations'].shape);assert np.array_equal(z,a['innovations'])
            with torch.no_grad():means=torch.cat([actor.step(torch.tensor(x[None]))[1] for x in a['states']])
            assert np.array_equal(means.numpy(),a['means']) and np.array_equal(T.L.AR.sample(means,z).numpy(),a['actions'])
            assert np.array_equal(D.terms(means,torch.tensor(a['actions']))[1].sum(-1).numpy(),a['logp'])
            points=np.concatenate([a['states'],a['next_states'][-1:]]);poses=np.concatenate([a['poses'],a['next_poses'][-1:]])
            assert len(points)==len(a['history_poses'])
            for x,pose,hp,hg in zip(points,poses,a['history_poses'],a['history_goals']):assert np.abs(H.rebuild(x,pose,hp,r['goal_xy'],hg)-x).max()<=2e-6
            assert np.array_equal(a['aux_costs'],T.endpoint_cost(a,28000)) and np.array_equal(a['rewards'],a['environment_rewards'])
        else:
            a=control_arrays(r);assert set(a)=={'native_counts','end_flags','next_poses','actions','innovations','environment_rewards','first_state','first_pose'}
            for key in ['native_counts','end_flags','next_poses','actions','innovations','environment_rewards']:assert S.digest(a[key])==r['array_hashes'][key]
            rng=np.random.default_rng(r['seed']);z=rng.normal(size=a['innovations'].shape);z[0,1]+=r['original_job']['first_steer_innovation_offset'];assert np.array_equal(z,a['innovations'])
            with torch.no_grad():m=actor.step(torch.tensor(a['first_state'][None]))[1]
            assert np.array_equal(T.L.AR.sample(m,z[:1]).numpy()[0],a['actions'][0])
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and not a['end_flags'][:-1].any()
        poses=np.concatenate([a['poses'][:1] if 'poses' in a else a['first_pose'][None],a['next_poses']])
        for k,n in enumerate(a['native_counts']):
            contact,goal,timeout=map(bool,a['end_flags'][k]);disc=R.GAMMA**int(n);d0=H.fields(poses[k],r['goal_xy'])[1];d1=H.fields(poses[k+1],r['goal_xy'])[1]
            rw=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not(contact or goal or timeout))+(R.GAMMA**(int(n)-1)*(float(goal)-float(contact)-.5*float(timeout)) if contact or goal or timeout else 0.)
            assert abs(rw-a['environment_rewards'][k])<1e-12
    for r in replays:
        ref=by[r['original_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    for family in {r['family'] for r in raw if r['split']=='test'}:
        group=[control_arrays(by[family+'_'+a]) for a in ['minus','center','plus']]
        for a in group:
            n=min(len(a['innovations']),len(group[1]['innovations']));assert np.array_equal(a['first_state'],group[1]['first_state']) and np.array_equal(a['innovations'][1:n],group[1]['innovations'][1:n]) and a['actions'][0,0]==group[1]['actions'][0,0]
    events=[]
    def guard():assert time.time()<start+p['audit_seconds']
    models,opts,losses,initial=fit(p,raw,guard,lambda n,k:events.append(dict(head=n,step=k,invoked=True)),{})
    saved=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()];assert events==saved and len(events)==s['head_optimizer_calls']==6000
    with np.load(OUT/'head_evidence.npz') as z:assert np.array_equal(z['losses'],losses)
    assert evaluate(p,raw,models)==read(OUT/'metrics.json')
    for name,q in models.items():
        r=s['models'][name];assert C.sha(r['path'])==r['sha256'];saved=torch.load(io.BytesIO(gzip.decompress(Path(r['path']).read_bytes())),map_location='cpu',weights_only=False)
        assert S.digest(saved['model'])==S.digest(q.state_dict())==r['model_hash'] and S.digest(saved['optimizer'])==S.digest(opts[name].state_dict())==r['optimizer_hash'] and saved['steps_invoked']==1500 and saved['initial_hash']==initial
    old=read(ROOT/'stage/runs/t308_iid_native_goal_value/cases.json');overlap=[dict(current=r['id'],T308=o['id']) for r in full for o in old if r['array_hashes']==o['array_hashes'] and r['integration_sha256']==o['integration_sha256']]
    verify(p);C.save(dest,dict(state='PASS',native_added=0,Actor_updates=0,disposable_head_Adam_steps=6000,all_original_processes_absent=True,
        native_commands=256,geometry_commands=16,total_commands=272,native_physics=s['physics_steps'],all64_full120_history_density_arrays_exact=True,
        all64_controls_compact_raw_and128_replay_hashes_exact=True,compact_control_full_intermediate_observations_not_saved=True,
        all_original_rewards_and_fixed_goal_control_metrics_exact=True,all_models_Adam_losses_exact=True,T308_exact_array_overlaps=overlap,
        summary_sha256=C.sha(OUT/'summary.json'),elapsed_s=time.time()-start,combined_value_control_gate=s['combined_value_control_gate'],six_metre_navigation_gate_evaluated=False))
    print('T309 independent raw/source/6000Adam/control audit PASS; navigation not evaluated')

if __name__=='__main__':main()
