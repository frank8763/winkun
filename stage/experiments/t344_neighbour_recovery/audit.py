from t344_core import *

def main():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert not (DOC/'T344_AUDIT.json').exists()
    w=read(OUT/'workers.json');l=read(OUT/'launch.json');assert all(not Path('/proc',str(pid)).exists() for pid in [s['pid'],l['pid'],*w['workers']])
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['updates']==dict(live=16,shadow=16)
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<start+p['audit_seconds']
    rows=[json.loads(l) for l in (OUT/'cases.jsonl').read_text().splitlines()];rep=[json.loads(l) for l in (OUT/'replays.jsonl').read_text().splitlines()];block=[json.loads(l) for l in (OUT/'blocks.jsonl').read_text().splitlines()]
    assert len(rows)+len(rep)+5==s['commands']==92 and sum(r['physics_steps'] for r in [*rows,*rep])==s['physics_steps']<=p['max_physics']
    lookup={r['id']:r for r in rows};loaded={}
    for r in rows:
        guard();assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);loaded[r['id']]=a
        assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and r['complete'] and not r.get('error')
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        if r['full_arrays_saved']:
            assert np.array_equal(a['states'][1:],a['next_states'][:-1])
            for k,x in enumerate(a['states']):
                assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],np.array(r['goal_xy']),a['history_goals'][k])-x))<=2e-6
            for k in range(len(a['rewards'])):
                n=int(a['native_counts'][k]);d0=float(np.linalg.norm(np.asarray(r['goal_xy'])-a['poses'][k,:2]));d1=float(np.linalg.norm(np.asarray(r['goal_xy'])-a['next_poses'][k,:2]));disc=K.R.GAMMA**n;contact,goal,timeout=a['end_flags'][k].astype(bool);terminal=contact or goal or timeout
                rw=-.000002*(1-disc)/(1-K.R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(K.R.GAMMA**(n-1)*(float(goal)-float(contact)-.5*float(timeout)) if terminal else 0.)
                assert abs(rw-float(a['rewards'][k]))<2e-12
    assert len(rep)==9 and all(r['full_arrays_and_integration_exact'] and not r['independent_sample'] for r in rep)
    source=next(r for r in rows if r['role']=='support_regeneration');support=dict(row=source,arrays=loaded[source['id']]);assert read(OUT/'support_contract.json')['integration_exact']
    obj=Learner(p);calls=[json.loads(l) for l in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for b in block:
        guard();assert b['input_hashes']==obj.hashes();batch=[dict(row=lookup[i],arrays=loaded[i]) for i in b['training_ids']]
        for item in batch:
            aa=item['arrays'];x=torch.tensor(aa['states'])
            with torch.no_grad():d=obj.actor.draw(x,torch.tensor(aa['innovations']),False)
            assert all(np.array_equal(v.numpy(),aa[k]) for k,v in d.items())
        successes=[dict(row=lookup[i],arrays=loaded[i]) for i in b['success_ids']];trace=[]
        tr=T.supported_update(p,obj,batch,targets(p,obj.actor,support,successes),b['round'],guard,lambda row:trace.append(row))
        assert digest(tr)==b['trace_sha256'] and obj.hashes()==b['output_hashes']
        # Each round has the same zero-based epoch numbers; slice its16 events.
        for replica in ['live','shadow']:
            records=[{k:v for k,v in r.items() if k!='replica'} for r in calls if r['replica']==replica]
            assert records[b['round']*16:(b['round']+1)*16]==trace
    model=read(OUT/'models.json')['final'];assert K.C.sha(model['path'])==model['sha256']
    v=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu');assert digest(v['state'])==digest(obj.state())
    final={(r['role'],r['job_id']):r for r in rows}
    for r in rows:
        if r['role']=='model_readback27':
            role='final_wide5' if r['job_id'].startswith('wide_dev') else 'final_original2' if r['job_id'].startswith('dev6_') else 'final_old20';ref=final[(role,r['job_id'])]
            assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    save(DOC/'T344_AUDIT.json',dict(state='PASS',native_added=0,new_training_updates=0,disposable_historical_Actor_Adam=16,all_full120_history_draw_density_reward_and_model_Adam_exact=True,
        all9_native_replays_exact=True,model_readback27_exact=True,source_inputs_unchanged=True,commands=s['commands'],physics_steps=s['physics_steps'],elapsed_s=time.time()-start,
        stable_six=False,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T344 audit PASS0native/16historicaldiscardedAdam')

if __name__=='__main__':main()
