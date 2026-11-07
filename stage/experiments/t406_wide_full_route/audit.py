"""Independent raw/update reconstruction after every native process has stopped."""
from t406_core import *
def main():
    N.offline();p=plan();base=verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    for pid in read(OUT/'workers.json')['workers']:assert not Path('/proc',str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p);old,wide=supports(q,base,p,lambda:None);calls=0
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()] if (OUT/'replays.jsonl').exists() else [];cmd=[json.loads(x) for x in (OUT/'commands.jsonl').read_text().splitlines()]
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics_steps'] and bool(a['end_flags'][-1,1])==r['success'] and bool(a['end_flags'][:,0].any())==r['collision']
    initial_old=[r for r in rows if r['role'].startswith('initial_old275_')];ids={j['id'] for j in base['old_jobs']};oldsafe={r['job_id'] for r in initial_old if r['success'] and r['job_id'] in ids};widesafe={j['id'] for j in p['wide5_jobs']}
    for r in s['rounds']:
        before=L.state(q,opt);assert L.statehash(q,opt)==read(OUT/'blocks'/str(r['round'])/'before_hashes.json');batch=[]
        for k in range(4):
            record=next(x for x in rows if x['role']==f'train_{r["round"]}_{k}');a=N.S.arrays(record['path']);STAGE404_NATIVE.audit_array(q,a,p,0,[p['seed'],r['round'],k]);assert not a['ownership'].any();batch.append(a)
        actual=fit(q,opt,batch,p,old,wide,r['round'],lambda:None,lambda *_:None);calls+=p['epochs'];assert actual==r['proposal'] and L.statehash(q,opt)==r['proposed_model']['hashes']
        checks=[x for x in rows if x['role'].startswith(f'proposal_{r["round"]}_old20_')];wc=[x for x in rows if x['role'].startswith(f'proposal_{r["round"]}_wide5_')];nold={x['job_id'] for x in checks if x['success']};nwide={x['job_id'] for x in wc if x['success']};accept=actual['numeric_accept'] and oldsafe<=nold and widesafe<=nwide;assert accept==r['accepted']
        if accept:oldsafe=nold;widesafe=nwide
        else:L.load(q,opt,before)
        assert sorted(oldsafe)==r['old_success_ids'] and sorted(widesafe)==r['wide_success_ids'] and L.statehash(q,opt)==r['retained_hashes']
    assert L.statehash(q,opt)==s['result']['model']['hashes'] and s['updates']==dict(live=calls,shadow=calls)
    model=s['result']['model'];assert sha(model['path'])==model['sha256']
    stored=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
    assert {k:digest(v) for k,v in stored['state'].items()}==L.statehash(q,opt)==stored['hashes']
    assert stored['policy_hash']==q.phash()==model['policy_hash']
    assert len(cmd)==s['commands'] and sum(x['physics'] for x in cmd)==s['physics'] and not any(x['error'] for x in cmd)
    assert sum(x['physics_steps'] for x in [*rows,*reps])==s['physics'] and s['commands']==4+len(rows)+len(reps)
    for r in reps:assert r['exact'] and r['array_digest'] in {x['array_digest'] for x in rows}
    if s['result']['stable_wide_six']:
        for k,n in [('main54',54),('old275',275),('fresh16',16)]:assert s['result'][k]['success']==s['result'][k]['valid']==n
        assert s['result']['reload78']['exact']==78
    for r in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/r['path'])==r['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,new_training=0,historical_discarded_Adam=calls,Actor_Adam_and_rejections_exact=True,final_checkpoint_raw_SHA_and_stored_Actor_Adam_readback_exact=True,training_history_density_rewards_exact=True,raw_physics_commands_exact=True,initial_full_old275_and_wide54_separate=True,physics=s['physics'],commands=s['commands'],stable_wide_six=s['result']['stable_wide_six']));N.manifest(OUT);print('T406_INDEPENDENT_AUDIT_PASS')
if __name__=='__main__':main()
