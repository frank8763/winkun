from run import *
def main():
    dest=DOC/'T310_AGGREGATE_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');verify(p);start=time.time()
    assert s['state']=='COMPLETE' and s['all_workers_stopped']
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],570156,570159,570160]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    q=B.actor(p);rows=read(ROOT/'stage/runs/t310_initial_history_control/cases.json')+read(OUT/'cases.json');replays=read(OUT/'replays.json');assert len(rows)==20 and len(replays)==19
    assert s['aggregate_commands']==40 and sum(r['physics_steps'] for r in rows+replays)==s['aggregate_physics']<=784008
    lookup={r['id']:r for r in rows};pairs=[]
    for r in rows:
        assert time.time()<start+p['audit_seconds'] and C.sha(r['path'])==r['sha256'];a=B.arrays(r['path'])
        assert len(a)==15 and {k:S.digest(v) for k,v in a.items()}==r['array_hashes']
        assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any() and r['complete'] and not r['error']
        with torch.no_grad():mode=torch.cat([q.step(torch.tensor(x[None]))[0] for x in a['states']]).numpy()
        assert np.array_equal(mode,a['actions']) and not a['means'].any() and not a['logp'].any()
        states=np.concatenate([a['states'],a['next_states'][-1:]]);poses=np.concatenate([a['poses'],a['next_poses'][-1:]])
        for x,pose,hp,hg in zip(states,poses,a['history_poses'],a['history_goals']):assert np.abs(H.rebuild(x,pose,hp,r['goal_xy'],hg)-x).max()<=2e-6
        cap=r['original_job']['case']['step_cap'];assert np.array_equal(a['aux_costs'],T.endpoint_cost(a,cap)) and np.array_equal(a['rewards'],a['environment_rewards'])
        for k,n in enumerate(a['native_counts']):
            contact,goal,timeout=map(bool,a['end_flags'][k]);disc=R.GAMMA**int(n);d0=H.fields(poses[k],r['goal_xy'])[1];d1=H.fields(poses[k+1],r['goal_xy'])[1]
            rw=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not(contact or goal or timeout))+(R.GAMMA**(int(n)-1)*(float(goal)-float(contact)-.5*float(timeout)) if contact or goal or timeout else 0)
            assert abs(rw-a['rewards'][k])<1e-12
        if r['context']=='native':
            old=p['historical_references'][r['case_id']];assert {k:r['array_hashes'][k] for k in B.NATIVE_KEYS}==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
            other=lookup[r['case_id']+'_aligned'];b=B.arrays(other['path'])
            assert r['initial_control']['integration_sha256']==other['initial_control']['integration_sha256']
            assert np.array_equal(a['states'][0,OTHER],b['states'][0,OTHER]) and np.array_equal(a['poses'][0],b['poses'][0])
            assert np.array_equal(a['history_poses'][0],b['history_poses'][0]) and np.array_equal(a['history_times'][0],b['history_times'][0])
            assert np.array_equal(a['states'][0,48:53],b['states'][0,48:53]) and set(other['initial_control']['changed_indices'])<=set(HISTORY)
            assert np.array_equal(b['history_goals'][0],np.tile(np.asarray(other['goal_xy']),(4,1)))
            def outcome(row):return dict(safe=bool(row['success'] and not row['collision']),contact=row['collision'],timeout=row['timeout'],physics_steps=row['physics_steps'],remaining_m=row['remaining_m'])
            pairs.append(dict(id=r['case_id'],distance_m=r['original_job']['case']['distance_m'],side=r['original_job']['case']['side'],native=outcome(r),aligned=outcome(other),native_initial_action=a['actions'][0].tolist(),aligned_initial_action=b['actions'][0].tolist(),only8history_fields_changed=True))
    for r in replays:
        ref=lookup[r['original_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'] and r['initial_control']==ref['initial_control']
    assert S.digest(q.state_dict())==p['policy_hash'];verify(p)
    C.save(dest,dict(state='PASS',native_added=0,optimizer_calls=0,Actor_updates=0,all_original_processes_absent=True,commands=s['aggregate_commands'],physics_steps=s['aggregate_physics'],
        all20full120_history_and_native_arrays=True,all19executed_replays_exact=True,one_replay_not_executed=p['omitted_replay_id'],original_producer_failed=True,engineering_recovery_only=True,all10native_baselines_exact_historical=True,all10paired_initial_integration_and_nonhistory112_exact=True,
        pairs=pairs,elapsed_s=time.time()-start,summary_sha256=C.sha(OUT/'summary.json'),policy_sha256=p['policy_hash'],
        interpretation='Within each fixed reset the frozen policy changes only initial goal-history observations; this is a diagnostic effect, not learning, leakage proof, six stability, reachability proof, or generalization rate. Historical/replay samples not new independent seeds.'))
    print('T310 independent paired-history audit PASS,0native0optimizer')
if __name__=='__main__':main()
