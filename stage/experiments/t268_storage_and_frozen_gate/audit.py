"""Independent stopped-run evidence audit for frozen T268; zero navigation."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t268_storage_and_frozen_gate'
DEST=ROOT/'stage/docs/long_distance_20261101/T268_AUDIT.json'
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,policy


def main():
    assert not DEST.exists(),'Do not overwrite a recorded audit'
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['all_workers_stopped'] and s['optimizer_calls']==s['new_navigation_training_episodes']==0
    w=json.loads((OUT/'workers.json').read_text())
    for pid in [w['parent'],*w['workers']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError('Process exists or PID reused: '+str(pid))
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name,h in p['input_hashes'].items():assert sha(name)==h,name
    assert (OUT/'protocol.json').read_bytes()==(ROOT/'stage/experiments/t268_storage_and_frozen_gate/protocol.json').read_bytes()
    names=['old_missing.jsonl','development.jsonl','heldout.jsonl','verification.jsonl','next_distance.jsonl']
    rows=lambda f:list(map(json.loads,(OUT/f).read_text().splitlines())) if (OUT/f).exists() else []
    raw={n:rows(n) for n in names};geometry=rows('geometry.jsonl')
    physics=sum(r['physics_steps'] for rr in raw.values() for r in rr)
    commands=len(geometry)+sum(map(len,raw.values()))
    assert physics==s['physics_steps'] and commands==s['commands']
    count=0
    for rr in raw.values():
        for r in rr:
            assert r['policy_hash']==p['policy_hash'] and sha(r['path'])==r['sha256']
            with L.np.load(r['path']) as z:
                assert set(z.files)==set(r['array_hashes'])-{'states','next_states'}
                assert all(S.digest(z[k])==r['array_hashes'][k] for k in z.files)
            count+=1
    old=list(map(json.loads,gzip.decompress(Path(p['reused_old_rows']).read_bytes()).splitlines()))
    assert len(old)==224
    for r in old:assert r['policy_hash']==p['policy_hash'] and sha(r['path'])==r['sha256']
    assert len({r['source_reset_id'] for r in old+raw['old_missing.jsonl']})==len(old)+len(raw['old_missing.jsonl'])
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    assert s['old_combined_safe']==sum(map(safe,old+raw['old_missing.jsonl']))
    for key,name in [('old_missing','old_missing.jsonl'),('six_development','development.jsonl'),('heldout','heldout.jsonl'),('reload','verification.jsonl')]:
        rr=raw[name];assert s[key]['n']==len(rr) and s[key]['safe']==sum(map(safe,rr))
    payload=L.read(p['checkpoint']);assert S.digest(payload['policy'])==p['policy_hash'] and sha(p['checkpoint'])==s['checkpoint_sha256']==p['checkpoint_sha256']
    own=S.used(OUT);used=S.used(ROOT/'stage/runs');assert own<p['final_bytes'] and used<p['global_bytes']
    assert physics<=p['max_physics'] and commands<=p['max_commands'] and s['elapsed_s']<p['wall_seconds']+20
    result=dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_steps_added=0,optimizer_calls_added=0,
        all_parent_workers_absent=True,physics_from_raw=physics,commands_from_raw=commands,
        new_saved_episode_payloads_and_arrays_verified=count,reused_old224_not_new_physical_repeats=True,
        old_combined_n=s['old_combined_n'],old_combined_safe=s['old_combined_safe'],six_development=s['six_development'],
        six_by_side=s['six_by_side'],heldout=s['heldout'],reload=s['reload'],six_m_gate=s['six_m_gate'],
        actual_seven_frozen_tests=len(s['next_distance']),checkpoint_sha256=s['checkpoint_sha256'],
        policy_hash=s['policy_hash'],original_summary_sha256=sha(OUT/'summary.json'),
        protocol_sha256=sha(OUT/'protocol.json'),own_bytes=own,global_bytes=used,audit_epoch=time.time(),
        model_unchanged_no_optimizer=True,unexecuted={name:max(0,n-len(raw[name])) for name,n in
            [('old_missing.jsonl',51),('development.jsonl',54),('heldout.jsonl',16),('verification.jsonl',78)]})
    DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
