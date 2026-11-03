"""Independent reconstruction of actual live/shadow optimizers; no new native."""
from recover import *
from adapter import event_row

def main():
    dest=DOC/'T312_REMAINING_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');start=time.time()
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['inflight']
    for pid in [s['pid'],*read(OUT/'workers.json')['workers'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    newrows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];newreplays=read(OUT/'replays.json') if (OUT/'replays.json').exists() else [];geometry=read(OUT/'geometry.json');cachedrows=[json.loads(v) for v in (ORIGINAL/'cases.jsonl').read_text().splitlines()];rows=cachedrows+newrows;replays=read(ORIGINAL/'replays.json')+newreplays
    assert not geometry and len(read(ORIGINAL/'geometry.json'))==78 and read(DOC/'T312_FAILED_ENTRY_AUDIT.json')['state']=='PASS_COMPLETED_RAW_AND_V_RECONSTRUCTION_ORIGINAL_PRODUCER_FAILED'
    assert len(newrows)+len(newreplays)==s['commands']<=p['max_commands'] and sum(r['physics_steps'] for r in newrows+newreplays)==s['physics_steps']==s['raw_record_physics']<=p['max_physics']
    assert s['aggregate_commands']<=1172 and s['aggregate_physics']<=32816000
    by={r['id']:r for r in rows};counts=[]
    for r in rows:
        assert C.sha(r['path'])==r['sha256'] and r['complete'] and not r.get('error')
        if r['role']=='training':a=B.arrays(r['path']);assert len(a)==16 and a['states'].shape[1]==120 and {k:S.digest(v) for k,v in a.items()}==r['array_hashes']
        else:
            with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
            assert set(a)=={'actions','native_counts','end_flags','environment_rewards'}
            for k,v in a.items():assert S.digest(v)==r['array_hashes'][k]
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        if r['role']=='training':
            assert np.array_equal(a['states'][1:],a['next_states'][:-1]);rng=np.random.default_rng(r['seed']);assert np.array_equal(rng.normal(size=a['innovations'].shape),a['innovations'])
            assert np.array_equal(r['goal_xy'],r['job']['active_goal_xy']) and np.array_equal(a['rewards'],a['environment_rewards'])
            states=np.concatenate([a['states'],a['next_states'][-1:]]);poses=np.concatenate([a['poses'],a['next_poses'][-1:]])
            for x,pose,hp,hg in zip(states,poses,a['history_poses'],a['history_goals']):assert np.abs(H.rebuild(x,pose,hp,r['goal_xy'],hg)-x).max()<=2e-6
            for k,n in enumerate(a['native_counts']):
                contact,goal,timeout=map(bool,a['end_flags'][k]);disc=R.GAMMA**int(n);d0=H.fields(poses[k],r['goal_xy'])[1];d1=H.fields(poses[k+1],r['goal_xy'])[1]
                rw=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not(contact or goal or timeout))+(R.GAMMA**(int(n)-1)*(float(goal)-float(contact)-.5*float(timeout)) if contact or goal or timeout else 0)
                assert abs(rw-a['rewards'][k])<1e-12
    for r in replays:
        ref=by[r['original_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    for arm in ['mc','gae']:
        for role,want in [('old_regression',275),('six_development',54),('heldout',16),('independent_reload',78)]:
            rr=[r for r in rows if r['arm']==arm and r['role']==role];assert len(rr) in [0,want]
            counts.append(dict(arm=arm,role=role,executed=len(rr),safe=sum(r['success'] and not r['collision'] for r in rr),contact=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),not_executed=want-len(rr),invalid=0,censored=0))
    saved_events=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()];events=[];screens=read(OUT/'screens.json');updates=read(OUT/'round_updates.json') if (OUT/'round_updates.json').exists() else []
    actors={arm:policy(p) for arm in ['mc','gae']};opts={arm:T.L.optimizer(q,p) for arm,q in actors.items()};lastvalues={}
    parent=copy.deepcopy(actors['mc']).eval().requires_grad_(False);ax,ay,offset=T.anchors(p,parent)
    def guard():assert time.time()<start+p['audit_seconds']
    def event(arm,replica,kind,e):events.append(event_row(arm,replica,kind,e))
    rounds=sorted({r['round'] for r in rows if r['role']=='training'});cache_i,cache_v,cache_f,cache_s=cached_first_batch(p)
    for rnd in rounds:
        current={}
        for arm in ['mc','gae']:
            rr=[r for r in rows if r['role']=='training' and r['arm']==arm and r['round']==rnd];assert len(rr)==16;items=load_items(rr)
            for item in items:
                a=item['arrays'];assert item['row']['policy_hash']==S.digest(actors[arm].state_dict())
                with torch.no_grad():m=means(actors[arm],torch.tensor(a['states']))
                assert np.array_equal(m.numpy(),a['means']) and np.array_equal(T.L.AR.sample(m,a['innovations']).numpy(),a['actions'])
            if rnd==0:
                target,metrics=cache_f[arm][1],cache_s[f'{arm}_0']['value']
                assert screens[f'{arm}_0']==cache_s[f'{arm}_0'];vm,vo=cache_v[arm]
            else:
                values,vm,vo,vr=fit_values(p,items,rnd,guard,lambda k,e:event(arm,'live',k,e));vv,sm,so,sr=fit_values(p,items,rnd,guard,lambda k,e:event(arm,'shadow',k,e));assert vr==sr
                target,metrics=build(p,items,values);screen,gram=gradient_screen(p,actors[arm],items,target,guard);screen.update(value=metrics,value_fits=vr);assert screen==screens[f'{arm}_{rnd}']
                with np.load(OUT/'learning'/f'{arm}_{rnd}.npz') as z:assert np.array_equal(z['gram'],gram) and all(np.array_equal(z[f'value{k}'],v) for k,v in enumerate(values))
            current[arm]=(items,target);lastvalues[arm]=(vm,vo)
        if not all(screens[f'{arm}_{rnd}']['value']['heldout_value_gate'] and screens[f'{arm}_{rnd}']['variance_gate'] for arm in ['mc','gae']):continue
        for arm in ['mc','gae']:
            q=actors[arm];opt=opts[arm];before=copy.deepcopy(q.state_dict());before_opt=copy.deepcopy(opt.state_dict());shadow=copy.deepcopy(q);so=T.L.optimizer(shadow,p);so.load_state_dict(before_opt);items,target=current[arm]
            a=actor_steps(p,q,opt,items,target,arm,rnd,ax,ay,offset,guard,lambda k,e:event(arm,'live',k,e));b=actor_steps(p,shadow,so,items,target,arm,rnd,ax,ay,offset,guard,lambda k,e:event(arm,'shadow',k,e));assert a==b
            u=next(u for u in updates if u['arm']==arm and u['round']==rnd);assert a==u['steps'] and S.digest(q.state_dict())==u['candidate_policy_hash'] and S.digest(opt.state_dict())==u['candidate_optimizer_hash']
            if u['native_guard']:
                native=[r for r in rows if r['role']=='old_guard' and r['policy_hash']==u['candidate_policy_hash']];assert len(native)==20
                assert u['native_guard']['all_safe']==all(r['success'] and not r['collision'] for r in native)
            if not u['accepted']:q.load_state_dict(before);opt.load_state_dict(before_opt)
            assert S.digest(q.state_dict())==u['retained_policy_hash'] and S.digest(opt.state_dict())==u['retained_optimizer_hash']
    assert events==saved_events and len(events)==s['optimizer_calls']
    for arm,q in actors.items():
        r=s['models'][arm];assert C.sha(r['path'])==r['sha256'];saved=torch.load(io.BytesIO(gzip.decompress(Path(r['path']).read_bytes())),map_location='cpu',weights_only=False)
        assert S.digest(saved['policy'])==S.digest(q.state_dict())==r['policy_hash'] and S.digest(saved['optimizer'])==S.digest(opts[arm].state_dict())==r['optimizer_hash']
        vm,vo=lastvalues[arm];assert S.digest(saved['value_models'])==S.digest([v.state_dict() for v in vm]) and S.digest(saved['value_optimizers'])==S.digest([v.state_dict() for v in vo])
    assert S.used(ROOT/'stage/runs')<=p['global_bytes'] and S.used(OUT)<=p['final_bytes']
    C.save(dest,dict(state='PASS',native_added=0,disposable_historical_optimizer_calls=len(events),all_actual_recovery_live_shadow_Actor_V_Adam_reconstructed_exact=True,original4000_V_calls_already_independently_reconstructed_not_repeated=True,original_producer_remains_failed=True,
        all_training_full120_history_density_rewards_and_arrays_exact=True,all_training_native_replay_hashes_exact=True,
        compact_evaluation_scope='Actions/counts/endflags/environment_rewards raw plus fullarray/terminalintegrationSHA; intermediate eval120 not saved, not independently reconstructed here.',counts=counts,
        all_original_processes_absent=True,commands=s['commands'],physics_steps=s['physics_steps'],aggregate_commands=s['aggregate_commands'],aggregate_physics=s['aggregate_physics'],source_and_models_SHA_exact=True,elapsed_s=time.time()-start,summary_sha256=C.sha(OUT/'summary.json')))
    print('T312 remaining recovery optimizer audit PASS,0newnative,original FAILED preserved')
if __name__=='__main__':main()
