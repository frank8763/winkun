"""Recover already completed T598 evidence without resuming the failed run.

Uses its previously authorized, not yet consumed128 historical audit updates.
No new navigation, no scientific learning, no change to T598 originals.
"""
from pathlib import Path
import sys,time,json,os,hashlib,datetime
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t598_independent_recurrent_recovery'))
import t598_runtime as R
L=R.L;torch=L.torch
read,sha,digest=L.read,L.sha,L.digest
OLD=ROOT/'stage/runs/t598_independent_recurrent_recovery'
OUT=ROOT/'stage/runs/t599_partial_evidence_recovery'
DOC=ROOT/'stage/docs/day_20261116/t599'


def save(path,obj):
    text=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    assert len(text.encode())<768*1024
    with path.open('x') as f:f.write(text)


def register():
    assert not OUT.exists() and read(OLD/'summary.json')['state']=='FAILED'
    assert not (OLD/'audit.json').exists()
    assert read(ROOT/'stage/docs/day_20261116/t598/AUTHORIZATION.json')['audit_historical_Adam']==128
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T598.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    end=datetime.datetime.fromisoformat('2026-11-16T20:00:00+08:00').timestamp()
    assert time.time()+720<end
    assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())+768*1024<2**30
    raw=read(ROOT/'stage/docs/long_distance_20261101/source_manifest_T598.json')['files']
    locked={str(ROOT/x['path']):x['sha256'] for x in raw}
    locked.update({str(f):sha(f) for f in [Path(__file__),ROOT/'stage/docs/day_20261116/t598/AUTHORIZATION.json']})
    p=dict(id='T599_T598_PARTIAL_EVIDENCE_ONLY',seconds=600,all_stop=end,own_bytes=768*1024,
        new_native=0,new_learning=0,historical_audit_Adam=128,new_seeds=0,CPU=[14,15],GPU=False,
        authorization='T598 already approved128 historical audit updates. The original driver stopped before invoking its audit; this consumes those once, not a fresh learning allocation.',
        inputs=locked,scope='Preserve original FAILED summary; verify52 completed cases/5exact replays and original final ActorAdam. Two key replays remain unexecuted. No original audit PASS claim.')
    OUT.mkdir();DOC.mkdir(parents=True,exist_ok=True);save(OUT/'protocol.json',p);save(DOC/'PLAN.json',p)
    print('REGISTERED_PARTIAL_EVIDENCE_NO_NEW_TRAINING',flush=True)


def main():
    p=read(OUT/'protocol.json');assert not (OUT/'launch.json').exists();start=time.time()
    save(OUT/'launch.json',dict(epoch=start,pid=os.getpid(),protocol_sha256=sha(OUT/'protocol.json')))
    def guard():assert time.time()-start<p['seconds'] and time.time()<p['all_stop']
    for f,h in p['inputs'].items():assert sha(f)==h,f
    wp=read(OLD/'worker_generations.json');pids=[619006,619008,619014]+[i for x in wp for i in x['workers']]
    assert not any(Path('/proc',str(i)).exists() for i in pids)
    op=read(OLD/'protocol.json');s=read(OLD/'summary.json');assert s['state']=='FAILED' and not (OLD/'audit.json').exists()
    for f,h in op['inputs'].items():assert sha(f)==h,f
    rows=R.G.rows(OLD);reps=R.G.rows(OLD,'replays.jsonl');assert len(rows)==52 and len(reps)==5
    assert [r['job_id'] for r in rows[:3]]==[j['id'] for j in op['handoff_jobs']]
    assert [r['job_id'] for r in rows[3:]]==[j['id'] for j in op['evaluation_jobs']]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);R.P.frozen=L.frozen
    for row in rows:
        guard();assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        assert int(a['native_counts'].sum())==row['physics']
        assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        if row['full']:R.inspect(op,[(a,row)])
    by={r['path']:r for r in rows}
    for r in reps:
        ref=by[r['reference_path']]
        assert r['exact'] and r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rows+reps)<=op['max_physics']
    # Runtime already starts with2commands, and main erroneously adds another2
    # before its only2-worker pool. Both the charged count and discrepancy stay.
    assert s['commands']==len(rows)+len(reps)+6==63
    check=R.assess(op,rows[3:]);hh=rows[:3]
    missing=sorted(set(op['key_replay_ids'])-{r['job_id'] for r in reps if r['role']=='EXACT_KEY4_COMPACT'})
    assert len(missing)==2
    saved=L.P.checkpoint(OLD/'proposal.pt.gz');model=L.frozen(op,OLD/'proposal.pt.gz')
    assert model.phash()==saved['policy_hash'] and L.fixed(model)==saved['fixed_sha']
    obj=L.make(op);events=[]
    u=L.fit(obj,L.dataset(op),op,guard,lambda t,k:events.append(['live',t,k]))
    assert u['steps']==128 and digest(L.state(obj))==saved['joint_hash']
    calls=R.G.rows(OLD,'optimizer_calls.jsonl')
    assert [e for e in calls if e[0]=='live']==events and [['live',t,k] for w,t,k in calls if w=='shadow']==events
    assert L.fixed(obj[0])==saved['fixed_sha'] and sha(op['base_model']['path'])==op['base_model']['sha256']
    for f,h in p['inputs'].items():assert sha(f)==h,f
    result=dict(state='PARTIAL_EVIDENCE_VERIFIED_NOT_COMPLETE_ORIGINAL_AUDIT',
        original_state='FAILED',original_summary_unchanged=True,new_native=0,new_learning=0,
        historical_audit_Adam=128,all52_saved_payload_SHA=True,all3_full_handoff_histories_and_extra_memories=True,
        other49_compact_NOT_full_pose_reconstruction=True,all5_completed_replays_exact=True,
        missing_key_replays=missing,missing_key_replays_not_counted_as_navigation_failures=True,
        proposal_full_ActorAdam_reconstruction_exact=True,base_T550_unchanged=True,
        no_recorded_postrollback_joint_hash_NOT_independently_verified=True,
        actual_charged_commands=63,payload_commands=57,pool_lifecycle_actual=4,
        duplicated_pool_start_charge=2,original_locked_complete_case_counter_required=65,
        physics=s['physics'],original_elapsed_s=s['elapsed_s'],handoff=[dict(id=r['job_id'],success=r['success'],collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m']) for r in hh],
        original49=check,accepted=False,full407=None,fresh16=None,reload78=None,
        update_reconstructed=u,elapsed_s=time.time()-start,all_original_PIDs_gone=True)
    save(OUT/'result.json',result);print('T599_PARTIAL_EVIDENCE_VERIFIED',result['elapsed_s'],flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
