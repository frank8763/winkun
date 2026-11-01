"""Stopped evidence audit plus disposable exact Actor/Adam update reconstruction."""
from t280_core import *
import copy,json,os,shutil,time
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t280_local_cost';DEST=ROOT/'stage/docs/long_distance_20261101/T280_AUDIT.json'


def rows(name):
    f=OUT/name
    return list(map(json.loads,f.read_text().splitlines())) if f.exists() else []


def main():
    assert not DEST.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    assert s['all_workers_stopped'] and not s['inflight_jobs']
    w=json.loads((OUT/'workers.json').read_text())
    for pid in [w['parent'],*w['workers'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live or reused PID',pid))
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    rr=rows('cases.jsonl');replays=rows('replays.jsonl');optcalls=rows('optimizer_calls.jsonl');committed=rows('round_updates.jsonl')
    assert len(rr)+len(replays)==s['commands']<=p['max_commands']
    assert sum(r['physics_steps'] for r in rr+replays)==s['physics_steps']<=p['max_physics']
    arrays={};full={}
    for r in rr:
        assert C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
        assert set(a)==set(r['saved_array_keys']) and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and np.all((a['native_counts']>0)&(a['native_counts']<=100))
        assert np.array_equal(a['rewards'],a['environment_rewards']-r['aux_eta']*a['aux_costs'])
        assert 0<=discounted_cost(a['aux_costs'],a['native_counts'])<=1+1e-12
        if r['role']=='training':
            assert len(a)==12 and a['states'].shape[1]==a['next_states'].shape[1]==120
            assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and S.digest(a['states'][0])==r['initial120_sha256']
            assert np.array_equal(endpoint_cost(a,28000),a['aux_costs']);full[r['id']]=dict(row=r,arrays=a)
        arrays[r['id']]=r
    for r in replays:
        ref=arrays[r['original_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    assert sum(r['phase']=='live' for r in optcalls)==s['optimizer_calls_live'] and sum(r['phase']=='shadow' for r in optcalls)==s['optimizer_calls_shadow']
    cache=json.loads((OUT/'native_guard_cache.json').read_text())['cache']
    for ph,g in cache.items():
        native=[r for r in rr if r['role']=='old20_guard' and r['policy_hash']==ph]
        safe=[r['complete'] and r['success'] and not r['collision'] for r in native]
        assert len(native)==20 and g['safe']==sum(safe) and g['all_safe']==all(safe)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);clone_calls=0;reconstructed=[]
    def count(opt):
        original=opt.step
        def invoke(*args,**kw):
            nonlocal clone_calls
            clone_calls+=1;return original(*args,**kw)
        opt.step=invoke
    for arm in p['arms']:
        q=policy(p);opt=L.optimizer(q,p);count(opt);frozen=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,offset=anchors(p,frozen)
        for record in [r for r in committed if r['arm']==arm]:
            rnd=record['round'];items=[full[r['id']] for r in rr if r['arm']==arm and r['role']=='training' and r['id'].startswith(f'{arm}_r{rnd}_')]
            assert len(items)==16 and all(i['row']['complete'] for i in items)
            d=L.training_batch(items,q);shadow=copy.deepcopy(q);so=L.optimizer(shadow,p);so.load_state_dict(copy.deepcopy(opt.state_dict()));count(so)
            live,audit=paired_update(q,opt,shadow,so,d,ax,ay,offset,p,rnd,lambda:None,lambda actor:cache[S.digest(actor.state_dict())],lambda phase,event:None)
            assert live==audit==record['updates'] and S.digest(q.state_dict())==record['policy_hash'] and S.digest(opt.state_dict())==record['optimizer_hash']
            reconstructed.append(dict(arm=arm,round=rnd,accepted=sum(r['accepted'] for r in live)))
        export=s['arms'][arm]['export'];assert export and C.sha(export['path'])==export['sha256']
        saved=L.read(export['path']);assert S.digest(saved['policy'])==export['policy_hash']==S.digest(q.state_dict())
        assert S.digest(saved['optimizer'])==export['optimizer_hash']==S.digest(opt.state_dict())
    assert clone_calls<=p['postrun_clone_optimizer_max_calls']
    assert C.sha(p['checkpoint'])==p['checkpoint_sha256'] and S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(DEST,dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_added=0,navigation_Actor_updates_added=0,
        disposable_optimizer_reconstruction_calls=clone_calls,all_processes_absent=True,all_original_sources_models_and_payload_SHA_verified=True,
        total_physics_from_raw=s['physics_steps'],total_commands_from_raw=s['commands'],full120_training_payloads=len(full),
        complete_exact_replays=len(replays),retained_Actor_Adam_rounds_reconstructed=reconstructed,
        all_final_Actor_Adam_state_hashes_reconstructed_exact=True,safe_T255_unchanged=True,
        arms=s['arms'],own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),summary_sha256=C.sha(OUT/'summary.json'),audit_epoch=time.time(),
        attribution='Reward-only within this matched pilot; not matched against T267 and not a novelty/multiseed/other-map claim.'))
    print(json.dumps(dict(state='PASS',clone_reconstruction_calls=clone_calls,arms=s['arms'])),flush=True)


if __name__=='__main__':main()
