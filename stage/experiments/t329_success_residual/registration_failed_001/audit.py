from learning import *
import signal
def main():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T329_AUDIT.json';assert not dest.exists()
    assert s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc/'+str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T329 audit1200s')));signal.alarm(p['audit_seconds'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    logs=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    q=model(p);opt=optimizer(q,p,'bc');initial=copy.deepcopy(state(q));empty=copy.deepcopy(opt.state_dict());records=[]
    fit(p,q,opt,lambda:None,lambda r:records.append(dict(r,replica='live')))
    assert records==[r for r in logs if r['kind']=='bc' and r['replica']=='live']
    shadow=model(p);so=optimizer(shadow,p,'bc');sh=[];fit(p,shadow,so,lambda:None,lambda r:sh.append(dict(r,replica='shadow')))
    assert sh==[r for r in logs if r['kind']=='bc' and r['replica']=='shadow']
    patches=read(OUT/'patches.json')
    def verify_patch(name,actor,optimizer_):
        r=patches[name];assert K.C.sha(r['path'])==r['sha256'];v=read_patch(r['path'])
        assert K.S.digest(state(actor))==K.S.digest(v['residual'])
        assert K.S.digest(optimizer_.state_dict())==v['optimizer_hash']==r['optimizer_hash']
        assert K.S.digest(actor.state_dict())==v['policy_hash']==r['policy_hash']
        assert K.S.digest(actor.frozen.state_dict())==p['frozen_policy_hash']==v['frozen_policy_hash']
    verify_patch('bc_proposal',q,opt)
    if not s['blocks']['bc']['accepted']:load(q,initial);opt.load_state_dict(empty)
    verify_patch('selected_after_bc',q,opt)
    rows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];items=[]
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path'])
        for k in ['actions','native_counts','end_flags','environment_rewards']:assert K.S.digest(a[k])==r['array_hashes'][k]
        assert not a['end_flags'][:-1].any() and int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        assert r['complete'] and not r.get('error')
        f=a['end_flags'][-1];assert bool(f[0])==r['collision'] and bool(f[1])==r['success'] and bool(f[2])==r['timeout']
        if r['timeout']:assert r['physics_steps']==r['cap']
        if r['role']!='training':continue
        assert {k:K.S.digest(v) for k,v in a.items()}==r['array_hashes']
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert np.array_equal(a['innovations'],np.random.default_rng(r['job']['seed']).normal(size=a['innovations'].shape))
        steps=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.max(np.abs(a['states'][:,112]-(1-steps/40000)))<1e-7
        with torch.no_grad():
            m=means(q,torch.tensor(a['states']));lp=B.D.terms(m,torch.tensor(a['actions']))[1].sum(-1)
        assert np.array_equal(m.numpy(),a['means']) and np.array_equal(lp.numpy(),a['logp'])
        expected=B.T.L.AR.sample(m,a['innovations']).numpy();assert np.array_equal(expected,a['actions'])
        for k,x in enumerate(a['states']):
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1];n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0)
            assert abs(reward-a['rewards'][k])<1e-12
        items.append(dict(row=r,arrays=a))
    assert len(items)==8
    opt=optimizer(q,p,'pg');before=copy.deepcopy(state(q));empty=copy.deepcopy(opt.state_dict());shadow=copy.deepcopy(q);so=optimizer(shadow,p,'pg')
    rr=[];pg(p,q,opt,items,lambda:None,lambda r:rr.append(dict(r,replica='live')));ss=[];pg(p,shadow,so,items,lambda:None,lambda r:ss.append(dict(r,replica='shadow')))
    assert rr==[r for r in logs if r['kind']=='pg' and r['replica']=='live'] and ss==[r for r in logs if r['kind']=='pg' and r['replica']=='shadow']
    verify_patch('pg_proposal',q,opt)
    if not s['blocks']['pg']['accepted']:load(q,before);opt.load_state_dict(empty)
    verify_patch('final',q,opt)
    reps=read(OUT/'replays.json');assert len(rows)==len(reps)==s['cases']==s['exact_replays']
    by={r['id']:r for r in rows}
    for rep in reps:assert rep['all_arrays_exact'] and rep['physics_steps']==by[rep['id']]['physics_steps'] and rep['integration_sha256']==by[rep['id']]['integration_sha256']
    physics=sum(r['physics_steps'] for r in rows)+sum(r['physics_steps'] for r in reps)
    assert physics==s['physics_steps']<=p['max_physics'] and len(rows)+len(reps)==s['commands']<=p['max_commands']
    assert len(logs)==202 and K.S.digest(q.frozen.state_dict())==p['frozen_policy_hash'];verify(p)
    save(dest,dict(state='PASS',native_added=0,disposable_historical_Adam_calls=202,
        all_actual_BC_PG_Adam_and_rollbacks_reconstructed=True,all_full120_training_history_innovations_density_reward_exact=True,
        compact_evaluation_arrays_verified=True,evaluation_intermediate120_not_stored=True,all_native_replay_receipts_verified=True,
        all_processes_absent=True,commands=s['commands'],physics_steps=physics,final_policy_hash=K.S.digest(q.state_dict()),
        original_275_54_fresh16_reload78_not_executed=True))
    signal.alarm(0);print('T329 independent202historicalAdam/fulldata audit PASS')
if __name__=='__main__':main()
