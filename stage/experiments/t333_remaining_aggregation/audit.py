from t333_learning import *
import signal

def main():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T333_AUDIT.json'
    assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc/'+str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T333 independent audit1800s')));signal.alarm(p['audit_seconds'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    logs=[json.loads(line) for line in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    patches=read(OUT/'patches.json');initial_hash=None;losses={}
    prior=[json.loads(line) for line in (Path(p['previous_run'])/'optimizer_calls.jsonl').read_text().splitlines()]
    logs=prior+logs
    for arm in p['arms']:
        q=model(p);opt=optimizer(q,p,'bc');h=K.S.digest(q.state_dict())
        if initial_hash is None:initial_hash=h
        else:assert h==initial_hash
        d=bc_data(p,q,arm);assert len(d['old'])==28629 and d['policy_prefix_supervised_rows']==0
        assert d['human_correction_rows']==(168 if arm=='recovery' else 0)
        records=[];fit(p,q,opt,lambda:None,lambda row:records.append(dict(row,replica='live')),arm,d)
        assert records==[r for r in logs if r['arm']==arm and r['replica']=='live']
        shadow=model(p);so=optimizer(shadow,p,'bc');other=[]
        fit(p,shadow,so,lambda:None,lambda row:other.append(dict(row,replica='shadow')),arm,d)
        assert other==[r for r in logs if r['arm']==arm and r['replica']=='shadow']
        assert K.S.digest(q.state_dict())==K.S.digest(shadow.state_dict()) and K.S.digest(opt.state_dict())==K.S.digest(so.state_dict())
        row=patches[arm+'_candidate'];assert K.C.sha(row['path'])==row['sha256'];v=read_patch(row['path'])
        assert K.S.digest(state(q))==K.S.digest(v['residual'])
        assert K.S.digest(q.state_dict())==v['policy_hash']==row['policy_hash']
        assert K.S.digest(opt.state_dict())==v['optimizer_hash']==row['optimizer_hash']
        assert K.S.digest(q.frozen.state_dict())==p['frozen_policy_hash']==v['frozen_policy_hash']
        losses[arm]=dict(initial=records[0]['losses'],final=records[-1]['losses'],supervised_rows=len(d['x']),old_rows=len(d['old']))
    rows=[json.loads(line) for line in (OUT/'cases.jsonl').read_text().splitlines()];reps=read(OUT/'replays.json')
    assert len(rows)==len(reps)==s['cases']==s['exact_replays']==54
    by={r['id']:r for r in rows};counts={}
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path'])
        assert set(a)=={'actions','native_counts','end_flags','environment_rewards'}
        for k,v in a.items():assert K.S.digest(v)==r['array_hashes'][k]
        assert r['complete'] and not r.get('error') and not a['end_flags'][:-1].any()
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        f=a['end_flags'][-1];assert bool(f[0])==r['collision'] and bool(f[1])==r['success'] and bool(f[2])==r['timeout']
        assert r['policy_hash']==patches[r['arm']+'_candidate']['policy_hash']
        if r['timeout']:assert r['physics_steps']==r['cap']
        if r['arm'] not in counts:counts[r['arm']]={}
        if r['role'] not in counts[r['arm']]:counts[r['arm']][r['role']]={k:0 for k in ['success','collision','timeout']}
        for k in ['success','collision','timeout']:counts[r['arm']][r['role']][k]+=int(r[k])
    assert counts==s['counts']
    for r in reps:
        assert r['all_arrays_exact'] and r['physics_steps']==by[r['id']]['physics_steps'] and r['integration_sha256']==by[r['id']]['integration_sha256']
    physics=sum(r['physics_steps'] for r in rows)+sum(r['physics_steps'] for r in reps)
    assert physics==s['physics_steps']<=p['max_physics'] and len(rows)+len(reps)==s['commands']<=p['max_commands']
    assert len(logs)==4*p['bc_steps']==1200 and s['real_RL_updates']==0
    verify(p);save(dest,dict(state='PASS',native_added=0,disposable_historical_BC_Adam_calls=1200,
        both_live_shadow_full_Adam_hash_logs_exact=True,frozen_base_unchanged=True,prefix_used_as_human_labels=0,
        paired_same_initialization_normalization_full_old_retention=True,losses=losses,counts=counts,
        compact_raw_arrays_verified=True,native_replay_receipts_verified=True,evaluation_intermediate120_not_stored=True,
        all_processes_absent=True,commands=s['commands'],physics_steps=physics,
        no_original275_six54_fresh16_reload78=True,not_six_stable=True))
    signal.alarm(0);print('T333 independent1200historicalAdam/data/native receipts PASS')

if __name__=='__main__':main()
