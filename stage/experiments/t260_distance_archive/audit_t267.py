"""Stopped T267 accounting, including censored/partial batches; zero navigation."""
from pathlib import Path
import json
import hashlib
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t267_causal_credit_course'
DOC=ROOT/'stage/docs/long_distance_20261101'
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S


def main():
    destination=DOC/'T267_AUDIT.json'
    assert not destination.exists(), 'Do not repeat or overwrite a stopped-run audit'
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['all_workers_stopped']
    workers=json.loads((OUT/'workers.json').read_text())
    for pid in [workers['parent'],*workers['workers']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError('Process exists or PID reused: '+str(pid))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def rows(name):
        p=OUT/name
        return list(map(json.loads,p.read_text().splitlines())) if p.exists() else []
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    names=['baseline.jsonl','training.jsonl','replays.jsonl','candidate_guards.jsonl',
        'round_centers.jsonl','course_centers.jsonl','old_regression.jsonl','development.jsonl',
        'heldout.jsonl','verification.jsonl','next_distance.jsonl']
    raw={n:rows(n) for n in names};geometry=rows('geometry.jsonl')
    physics=sum(r['physics_steps'] for rr in raw.values() for r in rr)
    commands=len(geometry)+sum(map(len,raw.values()))
    assert physics==summary['physics_steps'] and commands==summary['commands']
    train=raw['training.jsonl'];replays=raw['replays.jsonl'];byid={r['id']:r for r in train}
    assert len(byid)==len(train)==summary['unique_training_episodes']
    assert len(replays)==summary['exact_replays']<=len(train)
    for b in replays:
        a=byid[b['id']]
        assert a['array_hashes']==b['array_hashes'] and a['integration_sha256']==b['integration_sha256']
    assert sum(r['complete'] for r in train)==summary['completed_training_episodes']
    assert sum(not r['complete'] for r in train)==summary['censored_training_episodes']
    payloads={}
    for rr in raw.values():
        for r in rr:
            if r.get('path'):
                assert sha(r['path'])==r['sha256'];payloads[r['path']]=r['sha256']
    for r in train:
        with L.np.load(r['path']) as z:
            assert z['states'].shape==z['next_states'].shape and z['states'].shape[1]==120
            assert set(z.files)==set(r['array_hashes'])
            assert all(S.digest(z[k])==h for k,h in r['array_hashes'].items())
    updates=rows('updates.jsonl');events=rows('proposal_events.jsonl')
    assert len(updates)==summary['rounds'] and all(r['shadow_exact'] for r in updates)
    accepted=sum(u['accepted'] for r in updates for u in r['updates'])
    assert accepted==summary['accepted_RL_updates']==summary['shadow_accepted_updates']
    assert summary['retained_round_training_episodes']==16*len(updates)
    results={}
    for name,nkey,skey in [('old_regression.jsonl','old_regression_n','old_regression_safe'),
        ('development.jsonl','new_development_n','new_development_safe'),
        ('heldout.jsonl','heldout_n','heldout_safe')]:
        rr=raw[name];assert len(rr)==summary[nkey] and sum(map(safe,rr))==summary[skey]
        results[name]=dict(recorded=len(rr),complete=sum(r['complete'] for r in rr),
            safe=sum(map(safe,rr)),contact=sum(r['collision'] for r in rr),
            timeout=sum(r['timeout'] for r in rr),censored=sum(not r['complete'] for r in rr))
    plan=json.loads((OUT/'protocol.json').read_text())
    assert (OUT/'protocol.json').read_bytes()==(ROOT/'stage/experiments/t267_causal_credit_course/protocol.json').read_bytes()
    for path,h in plan['input_hashes'].items():assert sha(path)==h,path
    export=summary['export']
    if export:
        assert sha(export['path'])==export['sha256']
        data=L.read(export['path']);assert S.digest(data['policy'])==export['policy_hash']
        assert S.digest(data['optimizer'])==export['optimizer_hash']
    signatures={(json.dumps(r['array_hashes'],sort_keys=True),r['integration_sha256'])
        for f in plan['historical_training_logs'] for r in map(json.loads,Path(f).read_text().splitlines())}
    overlap=sum((json.dumps(r['array_hashes'],sort_keys=True),r['integration_sha256']) in signatures for r in train)
    assert overlap==summary['exact_historical_training_overlap']
    own=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    global_used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert own<plan['final_bytes'] and global_used<plan['global_bytes']
    result=dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_steps_added=0,optimizer_calls_added=0,
        all_processes_absent=True,original_summary_sha256=sha(OUT/'summary.json'),
        protocol_sha256=sha(OUT/'protocol.json'),physics_from_raw=physics,commands_from_raw=commands,
        geometry_commands=len(geometry),inflight_physics_upper_bound=summary['inflight_physics_upper_bound'],
        training_records=len(train),exact_replays=len(replays),historical_overlap=overlap,
        retained_updates=accepted,rounds=len(updates),
        full120_payloads_and_all_array_hashes_verified=len(train),all_saved_payload_hashes_verified=len(payloads),
        results=results,unexecuted_expected={k:max(0,n-len(raw[k])) for k,n in
            [('old_regression.jsonl',275),('development.jsonl',54),('heldout.jsonl',16),('verification.jsonl',78)]},
        rolled_back_partial_rounds=sum(e['phase']=='rollback' for e in events),export=export,
        six_m_gate=summary['new_distance_gate'],seven_m_tests=len(raw['next_distance.jsonl']),
        own_bytes=own,global_newruns_bytes=global_used,audit_epoch=time.time(),
        interpretation='Missing/censored items separated; incomplete regression does not imply a measured failure rate.')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
