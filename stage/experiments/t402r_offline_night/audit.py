"""Independent zero-native reconstruction; failures preserve original summary."""
from night import *

def main():
    N.offline();p=load_plan();base=verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    for r in cases:
        assert sha(r['path'])==r['sha256'];a=N.load_npz(r['path']);assert int(a['native_counts'].sum())==r['physics']
        assert bool(a['end_flags'][-1,1])==r['success'] and bool(a['end_flags'][:,0].any())==r['collision']
    for row in reps:assert row['exact'] and row['array_digest'] in {r['array_digest'] for r in cases}
    perf=read(OUT/'parallel_probe.json');assert len(perf)==3 and [r['concurrency'] for r in perf]==[1,2,4] and perf[0]['digests']==perf[1]['digests']==perf[2]['digests']
    warm=fixed_actor(base,p,WARM_PARENT).requires_grad_(False)
    for n in [1,2,4]:
        for i in range(4):
            a=N.S.arrays(OUT/'probes'/f'{n}_{i}.npz');audit_full(warm,a,0,None,p);assert int(a['native_counts'].sum())==5000 and not a['end_flags'].any() and digest(a)==perf[[1,2,4].index(n)]['digests'][i]
    parent=fixed_actor(base,p,SAFE_PARENT).requires_grad_(False);oldrows=[r for r in cases if r['role'].startswith('own_old_')];assert len(oldrows)==8 and all(r['success'] for r in oldrows)
    olddata=[N.S.arrays(r['path']) for r in oldrows]
    for a in olddata:audit_full(parent,a,0,None,p)
    old=targets(parent,[(a,0) for a in olddata]);assert digest([(g['x'],g['target']) for g in old])==read(OUT/'old_reference_contract.json')['target_sha256']
    calls=0;selected=None
    for r in s['supervised']:
        q,opt=new_learner(base,p,r['initial_model']);before=state(q,opt)
        if r['kind']=='warm':actual=L.warm_fit(q,opt,L.groups(q,p,[KNOWN_BRIDGE]),old,p,lambda:None,lambda *_:None)
        else:
            newdata=[(N.S.arrays(ROOT/'stage/runs/t399_rejected_warm_diagnostic/episodes/recover53.npz'),265),(N.S.arrays(ROOT/'stage/runs/t399_rejected_warm_diagnostic/episodes/recover54.npz'),270)]
            actual=consolidate_fit(q,opt,old,targets(warm,newdata),p,lambda:None,lambda *_:None)
        calls+=200;assert actual==r['proposal'] and statehash(q,opt)==r['model']['hashes']
        assert r['accepted']==r['checks']['compatible']
        if not r['accepted']:load(q,opt,before)
        else:selected=r['model']['path']
        assert statehash(q,opt)==r['retained_hashes']
    if selected is not None:
        q,opt=new_learner(base,p,selected);refs=L.groups(q,p,[KNOWN_BRIDGE])
        for r in s['rounds']:
            batch=[];before=state(q,opt)
            for k in range(4):
                row=next(v for v in cases if v['role']==f'train_{r["round"]}_{k}');a=N.S.arrays(row['path']);audit_full(q,a,r['prefix_holds'],r['seeds'][k],p);batch.append(a)
            actual=L.fit(q,opt,batch,r['prefix_holds'],p,refs,old,lambda:None,lambda *_:None);calls+=p['epochs']
            assert actual==r['proposal'] and statehash(q,opt)==r['proposed_model']['hashes']
            assert r['accepted']==bool(actual['numeric_accept'] and r['checks'] is not None and r['checks']['old8']['success']==8)
            if not r['accepted']:load(q,opt,before)
            assert statehash(q,opt)==r['retained_hashes']
        assert statehash(q,opt)==s['result']['model']['hashes']
    events=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    assert sum(v[1]=='live' and v[2]=='AFTER' for v in events)==sum(v[1]=='shadow' and v[2]=='AFTER' for v in events)==calls
    assert s['physics']==sum(r['physics'] for r in [*cases,*reps])+60000<=p['max_physics']
    setups=sum(v['concurrency'] for v in perf)+s['result']['parallel_selected'];assert s['commands']==len(cases)+len(reps)+12+setups
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,new_training=0,historical_discarded_Adam=calls,all_live_shadow_and_Adam_exact=True,all_training_and_old8_full_history_hidden_density_rewards_exact=True,probe_all_concurrencies_exact=True,physics=s['physics'],commands=s['commands'],stable_six=False));N.manifest(OUT);print('T402R_INDEPENDENT_AUDIT_PASS',flush=True)
if __name__=='__main__':main()
