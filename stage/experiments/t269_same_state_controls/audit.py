"""Independent audit of ended fixed-state controls and one-hold interventions."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t269_same_state_controls'
DEST=ROOT/'stage/docs/long_distance_20261101/T269_AUDIT.json'
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,C,policy,np


def main():
    assert not DEST.exists(),'Do not overwrite audit'
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['all_workers_stopped'] and s['all_native_envs_closed']
    w=json.loads((OUT/'workers.json').read_text())
    try:os.kill(w['parent'],0)
    except ProcessLookupError:pass
    else:raise AssertionError('Parent still exists or PID reused')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name,h in p['input_hashes'].items():assert sha(name)==h,name
    assert (OUT/'protocol.json').read_bytes()==(ROOT/'stage/experiments/t269_same_state_controls/protocol.json').read_bytes()
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()))
    assert sum(r['physics_steps'] for r in rows)==s['physics_steps'] and len(rows)==s['commands']==s['cases']
    arrays={}
    for r in rows:
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:
            assert set(z.files)==set(r['array_hashes'])
            assert all(S.digest(z[k])==h for k,h in r['array_hashes'].items())
            a={k:z[k].copy() for k in z.files}
        assert a['states'].shape==a['next_states'].shape and a['states'].shape[1]==120
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        arrays[r['id']]=a
    source=rows[0];assert source['array_hashes']==p['reference']['array_hashes'] and source['integration_sha256']==p['reference']['integration_sha256']
    assert rows[1]['array_hashes']==source['array_hashes'] and rows[1]['integration_sha256']==source['integration_sha256']
    for begin in p['snapshot_steps']:
        name=f'control_{begin}';r=next(r for r in rows if r['id']==name)
        assert r['integration_sha256']==source['integration_sha256']
        assert all(np.array_equal(v,arrays[source['id']][k][begin//100:]) for k,v in arrays[name].items())
        context=json.loads((OUT/f'context_{begin}.json').read_text())
        assert context['packet120_sha256']==S.digest(arrays[source['id']]['states'][begin//100])
    assert json.loads((OUT/'restoration_gate.json').read_text())['no_intervention_simulated_before_gate']
    for r in rows[4:]:
        intervention=r['intervention'];begin=r['start_step'];base=arrays[f'control_{begin}'];a=arrays[r['id']]
        assert np.array_equal(a['states'][0],base['states'][0])
        original=np.asarray(intervention['original_action'],dtype=np.float32)
        assert np.array_equal(original,base['actions'][0])
        expected=original.copy();expected[1]=np.clip(expected[1]+np.float32(intervention['steering_delta']),-1,1)
        assert np.array_equal(a['actions'][0],expected) and a['actions'][0,0]==base['actions'][0,0]
        assert r['absolute_end_step']<=p['job']['case']['step_cap'] and intervention['maximum_hold_steps']==100
    payload=L.read(p['checkpoint']);assert S.digest(payload['policy'])==p['policy_hash'] and sha(p['checkpoint'])==p['checkpoint_sha256']
    assert s['state']=='COMPLETE' and len(rows)==8 and s['optimizer_calls']==0
    assert s['physics_steps']<=p['max_physics'] and s['elapsed_s']<p['wall_seconds']+10
    own=S.used(OUT);used=S.used(ROOT/'stage/runs');assert own<p['final_bytes'] and used<p['global_bytes']
    result=dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_steps_added=0,optimizer_calls_added=0,
        all_processes_absent=True,total_physics_from_raw=s['physics_steps'],commands=len(rows),
        all8_full120_payloads_and_all10arrays_sha_verified=True,source_and_repeat_exact_T268=True,
        both_restored_control_suffixes_all_arrays_and_integration_exact=True,both120_history_contexts_exact=True,
        exactly4_one_hold_interventions_same_initial_state_verified=True,original_model_unchanged=True,
        original_summary_sha256=sha(OUT/'summary.json'),protocol_sha256=sha(OUT/'protocol.json'),
        branches=[dict(id=r['id'],success=r['success'],collision=r['collision'],timeout=r['timeout'],
            remaining_m=r['remaining_m'],absolute_end_step=r['absolute_end_step'],contact_witness=r['contact_witness']) for r in rows[4:]],
        own_bytes=own,global_bytes=used,audit_epoch=time.time(),not_new_training_or_deployment_success=True)
    DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
