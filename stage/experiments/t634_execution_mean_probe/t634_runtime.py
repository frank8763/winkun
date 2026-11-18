"""Zero-learning action-readout contrast, then conditional complete49."""
from pathlib import Path
import fcntl
import os
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t634_execution_mean_probe'
DOC=ROOT/'stage/docs/day_20261118/t634'
sys.path.insert(0,str(ROOT/'stage/experiments/t631_explored_recovery'))
import t631_runtime as Base
from unique_runtime import Runtime
import t634_native as N
M=N.M;R=Base.R;read,save,sha=M.read,M.save,M.sha


def plan():
    p=read(HERE/'protocol.json');cfg=read(p['actor_configuration']);cfg.update(p);return cfg


def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=p['max_bytes']
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072<2**30
    assert R.shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f


def setup(p):
    Base.HERE=HERE;Base.OUT=OUT;Base.DOC=DOC;Base.verify=verify;Base.setup(p)
    R.B.N.worker=N.worker;R.B.N.N.inspect=N.inspect


def model_for(p,mode):
    return dict(p['student_initial'],policy_hash=N.output_hash(p['student_initial']['policy_hash'],mode,p['raw_sigma']))


def main():
    p=plan();verify(p);setup(p)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=Runtime(p);error=None;result=dict(modes={},final49=None,accepted=False,no_global_promotion=True)
    try:
        rt.pool=R.B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for mode in ['mode','expected']:
            values=[];model=model_for(p,mode)
            for job in p['sentinel_jobs']:
                j=dict(job,execution_mean_mode=mode)
                vv=rt.request([j],model,'FRESH_TRAIN4');row=vv[0][1]
                rt.request([j],model,'EXACT_SENTINEL',references=[row]);values.append(row)
            result['modes'][mode]=values;save(OUT/'progress.json',result)
        if all(R.safe(r) for r in result['modes']['expected']):
            model=model_for(p,'expected');jobs=[dict(j,execution_mean_mode='expected') for j in p['evaluation_jobs']]
            vv=rt.request(jobs,model,'FINAL49');rows=[r for a,r in vv]
            keys=[r for r in rows if r['job_id'] in p['key_replay_ids']]
            rt.request([r['request']['job'] for r in keys],model,'EXACT_KEY4',references=keys)
            metrics=R.metrics(p,rows);lost=sorted(set(p['student_success_ids'])-set(metrics['success_ids']))
            result.update(evaluation_rows=rows,final49=metrics,lost_student_successes=lost,
                          accepted=R.accepted(metrics) and not lost)
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),physics=sum(rt.counts),
             commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result,
             Actor_updates_started=dict(live=0,shadow=0),Actor_updates_completed=dict(live=0,shadow=0),
             no_global_promotion=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=R.B.read_rows('cases.jsonl');reps=R.B.read_rows('replays.jsonl')
    assert len({r['path'] for r in rows})==len(rows)
    by={r['path']:r for r in rows}
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.B.R.E.N.S.arrays(row['path'])
        if row['full']:
            assert all(M.digest(v)==row['array_hashes'][k] for k,v in a.items());N.inspect(p,[(a,row)])
        else:assert all(M.digest(a[k])==row['array_hashes'][k] for k in ['actions','native_counts','end_flags'])
    for replay in reps:
        original=by[replay['reference_path']]
        assert replay['exact'] and replay['array_hashes']==original['array_hashes']
        assert replay['integration_sha256']==original['integration_sha256']
    r=s['result'];assert len(r['modes']['mode'])==len(r['modes']['expected'])==4
    for row,old in zip(r['modes']['mode'],p['control_reference_rows']):
        # Newly counted diagnostic control, compared with retained old hashes;
        # this does not recreate the T633 overwritten compact file.
        assert row['array_hashes']==dict(old['array_hashes'],execution_mean_mode=M.digest(N.np.array(0,N.np.uint8)))
        assert row['integration_sha256']==old['integration_sha256']
    if r['final49'] is not None:
        assert all(R.safe(x) for x in r['modes']['expected'])
        metrics=R.metrics(p,r['evaluation_rows']);assert metrics==r['final49']
        lost=sorted(set(p['student_success_ids'])-set(metrics['success_ids']))
        assert lost==r['lost_student_successes'] and r['accepted']==(R.accepted(metrics) and not lost)
    assert len(rows)+len(reps)+6==s['commands']<=p['max_commands']
    assert sum(x['physics'] for x in rows+reps)==s['physics']<=p['max_physics']
    save(OUT/'audit.json',dict(state='PASS',new_native=0,new_live_Actor=0,
        every_original_full_readout_history_rebuilt=True,all_saved_bytes_SHA=True,
        unique_evaluation_paths=True,all_exact_replays_counted=True,original_control_history_hashes_matched=True))


def previous_audit():
    p=plan();verify(p)
    from previous_audit import run
    run(p,OUT)


if __name__=='__main__':{'main':main,'audit':audit,'previous_audit':previous_audit}[sys.argv[1]]()
