"""Posthumous partial T633 audit; never invent the overwritten compact file."""
from pathlib import Path
import gzip
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t633_conditional_recovery_pg'))
import t633_runtime as Old
M=Old.M


def run(p,out):
    root=ROOT/'stage/runs/t633_conditional_recovery_pg'
    read=Old.read;prior=read(root/'protocol.json')
    cfg=read(prior['actor_configuration']);cfg.update(prior);prior=cfg
    # An independently authorized audit window, not a mutation or extension of
    # the original protocol. No native commands, no proposed policy retained.
    prior=dict(prior,absolute_all_stop=p['absolute_all_stop'],absolute_learning_stop=p['absolute_all_stop'])
    Old.setup(prior)
    def records(name):
        with gzip.open(root/(name+'.gz'),'rt') as f:return [json.loads(line) for line in f if line.strip()]
    rows=records('cases.jsonl');reps=records('replays.jsonl');missing=[];verified=[]
    for row in rows:
        if Old.sha(row['path'])!=row['sha256']:
            missing.append(row);continue
        a=Old.R.B.R.E.N.S.arrays(row['path'])
        if row['full']:
            assert all(M.digest(v)==row['array_hashes'][key] for key,v in a.items())
            Old.N.inspect(prior,[(a,row)])
        else:
            assert all(M.digest(a[key])==row['array_hashes'][key] for key in ['actions','native_counts','end_flags'])
        verified.append(row)
    assert len(missing)==1 and missing[0]['request']['job'].get('student_takeover_first')==20
    # Use the *original* matching metadata row, never a dict keyed only by the
    # repeated path, to corroborate the replay. Metadata is not lost raw bytes.
    for replay in reps:
        candidates=[r for r in rows if r['path']==replay['reference_path'] and r['array_hashes']==replay['array_hashes']]
        assert len(candidates)==1 and replay['exact']
        assert replay['integration_sha256']==candidates[0]['integration_sha256']
    s=read(root/'summary.json');r=s['result']
    assert sum(v['physics'] for v in rows+reps)==s['physics'] and len(rows)+len(reps)+6==s['commands']
    assert Old.R.metrics(prior,r['evaluation_rows'])==r['final49']
    assert not r['accepted'] and not r['handoffs'][0]['success']
    Old.R.gpu_guard(prior);M.configure_gpu(prior);obj=M.make(prior,prior['raw_sigma'])
    before=M.state(obj);events=[]
    def event(kind,k):
        events.append(['live',kind,k])
        with (out/'historical_optimizer_calls.jsonl').open('a') as f:
            f.write(json.dumps([kind,k])+'\n');f.flush()
    u=Old.L.fit(obj,prior,[Old.load(v,prior) for v in r['training_rows']],Old.load(r['anchor'],prior,True),
                Old.load(prior['nominal_eight_row'],prior,True),lambda:Old.R.gpu_guard(prior),
                event)
    assert u==r['update'] and u['attempted_optimizer']==8
    obj[0].cpu();assert M.digest(M.state(obj))==r['proposal']['joint_hash']
    assert obj[0].phash()==r['proposal']['policy_hash'] and Old.sha(r['proposal']['path'])==r['proposal']['sha256']
    calls=records('optimizer_calls.jsonl')
    assert [v for v in calls if v[0]=='live']==events
    assert [['live',*v[1:]] for v in calls if v[0]=='shadow']==events
    M.load(obj,before);assert M.digest(M.state(obj))==r['retained_state_sha']
    report=dict(state='PARTIAL_AUDIT_ORIGINAL_MISSING_COMPACT',
         independent_historical_SGD=8,new_native=0,new_live_policy_updates=0,
         verified_existing_case_files=len(verified),total_case_occurrences=len(rows),
         missing_original_compact_records=missing,all_training_full_history_and_model_updates_reconstructed=True,
         independent_original_handoff_raw_validation=False,
         original_failed_audit_preserved=True,no_raw_file_recreated=True,
         all_original_files_and_metadata_unmodified=True,epoch=time.time())
    (out/'T633_PARTIAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
