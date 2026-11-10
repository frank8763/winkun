"""Read-only morning inspection; zero native, gradients, or optimizer steps.

This is deliberately not the unexecuted T449 full historical-Adam audit.
Original failed source, protocol, sealed raw files and summary stay unchanged.
"""
from pathlib import Path
import collections,hashlib,importlib.util,json,sys,time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t449_offline_joint_recovery';DOC=ROOT/'stage/docs/overnight_20261109'
def read(f):return json.loads(f.read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    start=time.time();p=read(OUT/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='FAILED'
    ids=[read(OUT/'launch.json')['pid'],read(OUT/'driver_started.json')['pid']]+read(OUT/'SEALED.json')['scientific_pids']
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    assert all(sha(f)==h for f,h in p['inputs'].items())
    local=read(DOC/'LOCAL_MANIFEST.json')
    for r in local['files']:assert (ROOT/r['path']).stat().st_size==r['bytes'] and sha(ROOT/r['path'])==r['sha256']
    spec=importlib.util.spec_from_file_location('stage449_readonly_inspection',ROOT/'stage/experiments/t449_offline_joint_recovery/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.E.N.offline();m.R.C.ban_learning();rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()));replays=list(map(json.loads,(OUT/'replays.jsonl').read_text().splitlines()));ev=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines()))
    assert s['physics']==sum(r['physics'] for r in rows+replays)<=p['max_physics']
    assert s['commands']==p['preflight_commands']+16*len(read(OUT/'worker_generations.json'))+len(rows)+len(replays)<=p['max_commands']
    expert=m.E.safe_initial(m.E.N.science(p),p).eval().requires_grad_(False);full=0;compact=0
    for r in rows:
        assert time.time()-start<600
        assert sha(r['path'])==r['sha256'];a=m.E.N.S.arrays(r['path'])
        assert int(a['native_counts'].sum())==r['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert bool(a['end_flags'][-1,0])==r['collision'] and bool(a['end_flags'][-1,1])==r['success'] and bool(a['end_flags'][-1,2])==r['timeout']
        if r['full']:
            assert m.digest(a)==r['array_digest'] and all(m.digest(a[k])==v for k,v in r['array_hashes'].items());m.audit_full_history(p,r,a);full+=1
            if r['teacher']:
                with m.torch.no_grad():lab=expert.base_actions(m.torch.tensor(a['states'])).float().numpy()
                assert m.np.array_equal(lab,a['teacher_actions'])
        else:compact+=1
    bypath={r['path']:r for r in rows}
    for r in replays:
        previous=bypath[r['reference_path']];assert r['array_digest']==previous['array_digest'] and r['array_hashes']==previous['array_hashes'] and r['integration_sha256']==previous['integration_sha256']
    counts=collections.Counter((v[0],v[1]) for v in ev if v[2]=='AFTER')
    for phase,n in [('tail',1024),('recovery',1500),('ppo0',4)]:
        for which in ['live','shadow']:
            assert [v[2:] for v in ev if v[:2]==[phase,which]]==[[where,k] for k in range(n) for where in ['BEFORE','AFTER']]
    parent=set(p['night_initial_result']['success_ids']);allids={j['id'] for j in p['main_jobs']+p['old275_jobs']};results=[]
    for e in read(OUT/'phases.json'):
        rr=[r for r in rows if r['role']=='FINAL_'+e['phase']];assert len(rr)==329
        ss={r['job_id'] for r in rr if r['success'] and not r['collision']};candidate=e['candidate'];assert ss==set(candidate['success_ids']) and sha(candidate['model']['path'])==candidate['model']['sha256']
        ck=m.P.checkpoint(candidate['model']['path']);assert m.digest(ck['joint_state'])==candidate['model']['joint_hash'];assert ck['policy_hash']==candidate['model']['policy_hash']
        for key,scope in [('wide54',rr[:54]),('old275',rr[54:])]:assert m.R.M.tally(scope)==candidate[key]
        assert e['accepted']==(parent<=ss)
        results.append(dict(phase=e['phase'],wide54=candidate['wide54'],old275=candidate['old275'],failed=sorted(allids-ss),repaired_vs_parent=sorted(ss-parent),lost_vs_parent=sorted(parent-ss),accepted=e['accepted'],model=candidate['model']))
    corrections=read(OUT/'corrections.json');assert len(corrections)==7 and all(r['success'] and not r['collision'] for r in corrections)
    result=dict(state='PASS_RAW_NAVIGATION_AND_FULL_HISTORY_SCOPE_ONLY',new_native=0,new_gradients=0,optimizer_steps=0,historical_optimizer_reconstruction=0,
        original_main_state='FAILED',original_full_audit='NOT_EXECUTED',all_recorded_processes_stopped=True,all_original_local_manifest_members_and_locked_inputs_SHA_verified=True,
        physics=s['physics'],commands=s['commands'],elapsed_main_seconds=s['elapsed_s'],raw_full_trajectories_rebuilt=full,compact_records_hashed_and_terminal_counts_verified=compact,exact_native_replays=len(replays),
        compact_final658_full120_NOT_reconstructed=True,phase_models_Actor_V_and_mature_Adam_bytes_hash_verified_NOT_training_reconstructed=True,
        live_BC_Actor_Adam=2524,shadow_BC_Actor_Adam=2524,live_PPO_Actor_Adam=4,live_PPO_V_Adam=4,shadow_PPO_Actor_Adam=4,shadow_PPO_V_Adam=4,
        PPO_proposal_not_serialized_or_panel_evaluated=True,preflight_Adam_counts_separate=read(OUT/'preflight.json'),results=results,
        true_unconsumed_fresh16=None,independent_reload78=None,six_gate=False,eight_started=False,elapsed_inspection_seconds=time.time()-start)
    (DOC/'POSTCHECK_20261110.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['state','physics','commands','raw_full_trajectories_rebuilt','elapsed_inspection_seconds']},ensure_ascii=False))
if __name__=='__main__':main()
