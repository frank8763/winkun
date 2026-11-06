"""Independent zero-native raw, teacher, likelihood and historical Adam audit."""
from t398_learning import *

def audit_mixed(q,a,p):
    q.sample=False;q.reset_noise();hidden=None;goal=a['task_goal_xy'];prior=N.S.arrays(REFERENCE)
    for i,x in enumerate(a['states']):
        assert np.abs(N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],goal,a['history_goals'][i])-x).max()<=2e-6
        hb=np.zeros(16) if hidden is None else hidden[0,0].numpy().copy()
        with torch.no_grad():mode,_,hidden,_=q.step(torch.tensor(x[None]),hidden)
        assert equal(hb,a['hidden_before'][i])
        for key,v in q.tape[-1].items():assert equal(np.asarray(v),a[key][i]),(i,key)
        if a['ownership'][i]==0:assert equal(mode[0].numpy(),a['actions'][i])
        d0=np.linalg.norm(a['poses'][i,:2]-goal);d1=np.linalg.norm(a['next_poses'][i,:2]-goal);z=N.K.R.GAMMA**int(a['native_counts'][i]);f=a['end_flags'][i]
        reward=-.000002*(1-z)/(1-N.K.R.GAMMA)+.03*d0-z*.03*d1*(not f.any())
        if f.any():reward+=N.K.R.GAMMA**(int(a['native_counts'][i])-1)*(float(f[1])-float(f[0])-.5*float(f[2]))
        assert abs(reward-a['rewards'][i])<1e-12
    assert equal(hidden[0,0].numpy(),a['last_hidden_after'])
    assert equal(a['actions'][:265],prior['actions'][:265])

def teacher(q,p,c):
    folder=OUT/'teachers'/str(c['cycle']);data=N.load_npz(folder/'search.npz');assert digest(data)==c['search_hash'] and q.phash()==c['policy_hash']
    mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['teacher_sd']),(16,2)).copy();best=None;bestid=None;cost=float('inf');nquery=0
    for tr in c['traces']:
        choices=teacher_candidates(mean,sd,best,c['cycle'],tr['generation'],p)
        for idx in tr['ids']:
            theta=choices[idx%12];u=teacher_controls(theta,N.S.arrays(REFERENCE)['actions'],p);n=int(data['lengths'][idx]);assert n>0
            assert equal(theta,data['coefficients'][idx]) and equal(u,data['controls'][idx])
            assert not data['flags'][idx,:n-1].any() and data['flags'][idx,n-1].any()
            actual=objective(u,n,data['flags'][idx,n-1],data['remaining_m'][idx]);assert actual==data['costs'][idx]
            if bestid is None or (actual,idx)<(cost,bestid):best=theta.copy();bestid=idx;cost=actual
            nquery+=1
        elite=sorted(tr['ids'],key=lambda i:(float(data['costs'][i]),i))[:3];assert elite==tr['elite'] and bestid==tr['best_id']
        if not tr['goal_found']:mean=.5*mean+.5*data['coefficients'][elite].mean(0);sd=np.maximum(np.asarray(p['teacher_min_sd']),.5*sd+.5*data['coefficients'][elite].std(0))
        assert equal(mean,np.asarray(tr['mean_after'])) and equal(sd,np.asarray(tr['sd_after']))
    assert nquery==c['queries']<=24 and c['query_exact_repeats']==nquery
    charged=p['teacher_start_hold']*100+2*int(data['counts'].sum());expected_commands=1+2*nquery
    a=N.S.arrays(folder/'teacher_full.npz');audit_mixed(q,a,p)
    if c['full_teacher_safe']:
        b=N.S.arrays(folder/'bridge.npz');audit_mixed(q,b,p)
        first=p['teacher_start_hold'];length=int(data['lengths'][bestid]);assert len(a['states'])==first+length
        assert equal(a['actions'][first:],data['controls'][bestid,:length]) and a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        assert digest(a)==c['full_hash'] and digest(b)==c['bridge_hash']
        assert equal(b['actions'][:first+p['bridge_holds']],a['actions'][:first+p['bridge_holds']])
        assert bool(b['end_flags'][-1,1])==c['short_bridge_safe'] and not b['ownership'][first+p['bridge_holds']:].any()
        charged+=2*int(a['native_counts'][first:].sum()+b['native_counts'][first:].sum());expected_commands+=4
    else:assert len(a['states'])==p['teacher_start_hold'] and not a['end_flags'].any()
    assert c['physics']==charged and c['commands']==expected_commands

def main():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p)
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    commands=[json.loads(v) for v in (OUT/'commands.jsonl').read_text().splitlines()]
    assert len(commands)==s['commands'] and sum(v['physics'] for v in commands)==s['physics'] and not any(v['error'] for v in commands)
    assert s['commands']+read(OUT/'preflight.json')['all_zero_physics_commands']<=p['max_commands'] and s['physics']<=p['max_physics']
    for c in cases:
        assert sha(c['path'])==c['sha256'];a=N.load_npz(c['path']);assert int(a['native_counts'].sum())==c['physics_steps']
        assert bool(a['end_flags'][-1,1])==c['success'] and bool(a['end_flags'][:,0].any())==c['collision']
    for r in reps:assert r['exact'] and r['array_digest'] in {v['array_digest'] for v in cases}
    oldrefs=old_references(q,base,lambda:None)
    contract=read(OUT/'old_reference_contract.json');assert contract['sha256']==digest([(g['x'],g['target']) for g in oldrefs])
    warm=s['result']['warm'];before=state(q,opt)
    actual=warm_fit(q,opt,groups(q,p,[KNOWN_BRIDGE]),oldrefs,p,lambda:None,lambda *_:None)
    assert actual==warm['proposal'] and statehash(q,opt)==warm['proposed_model']['hashes']
    assert warm['accepted']==bool(warm['old8']['valid']==8 and warm['old8']['success']==8)
    if not warm['accepted']:load(q,opt,before)
    assert statehash(q,opt)==warm['retained_hashes']
    opt=torch.optim.Adam(N.trainable(q),lr=p['lr']);assert statehash(q,opt)==warm['post_stage_fresh_Adam_hashes']
    completed=p['warm_steps'];paths=[KNOWN_BRIDGE]
    for i,c in enumerate(s['teachers']):
        teacher(q,p,c)
        if not c['training_labels_allowed']:assert i==len(s['rounds']);break
        paths.append(OUT/'teachers'/str(i)/'bridge.npz');r=s['rounds'][i];before=state(q,opt)
        assert statehash(q,opt)==read(OUT/'blocks'/str(i)/'before_hashes.json')
        batch=[]
        for k in range(4):
            a=N.S.arrays(next(v['path'] for v in cases if v['role']==f'train_r{i}_{k}'));audit_training(q,a,265,[p['seed'],i,k],p);batch.append(a)
        actual=fit(q,opt,batch,265,p,groups(q,p,paths),oldrefs,lambda:None,lambda *_:None);completed+=p['epochs']
        assert actual==r['proposal'] and statehash(q,opt)==r['proposed_model']['hashes']
        oldpass=r['old8'] is not None and r['old8']['valid']==8 and r['old8']['success']==8
        assert r['accepted']==bool(actual['numeric_accept'] and oldpass)
        if not r['accepted']:load(q,opt,before)
        assert statehash(q,opt)==r['retained_hashes']
    assert statehash(q,opt)==s['result']['model']['hashes'] and s['updates']==dict(live=completed,shadow=completed)
    assert s['result']['old275'] is None and s['result']['main54'] is None and not s['result']['stable_new070_80']
    for v in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/v['path'])==v['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,new_optimization=0,discarded_historical_Adam=completed,warm_discarded_historical_Adam=p['warm_steps'],RL_discarded_historical_Adam=completed-p['warm_steps'],all_teacher_queries_and_short_bridge_semantics=True,causal_history_likelihood_rewards=True,Actor_Adam_exact=True,physics=s['physics'],commands=s['commands'],full_six_stable=False))
    N.manifest(OUT);print('T398_INDEPENDENT_AUDIT_PASS')
if __name__=='__main__':main()
