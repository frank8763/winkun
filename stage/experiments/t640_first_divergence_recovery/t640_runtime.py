"""One causally selected early recovery, not a takeover-time grid."""
from pathlib import Path
import fcntl,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t640_first_divergence_recovery';DOC=ROOT/'stage/docs/day_20261118/t640'
sys.path.insert(0,str(ROOT/'stage/experiments/t639_neural_recovery'))
import t639_runtime as B
B.HERE=HERE;B.OUT=OUT;B.DOC=DOC
read,save,sha,digest=B.read,B.save,B.sha,B.digest


def main():
    p=B.plan();B.verify(p);B.setup(p)
    assert read(ROOT/'stage/runs/t639_neural_recovery/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/runs/t639_neural_recovery/SEALED.json')['all_scientific_processes_stopped']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=B.os.getpid(),epoch=time.time()))
    rt=B.Runtime(p);error=None;result=dict(recovery=None,qualified=False,learning=None,full407=None)
    try:
        rt.pool=B.R.B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        a,row=rt.request(p['handoff_job'],'early_neural_recovery')
        rt.request(p['handoff_job'],'exact_early_recovery',row)
        result.update(recovery=row,qualified=B.R.safe(row))
        save(OUT/'RECOVERY_ADMISSION.json',dict(passed=result['qualified'],teacher=p['reference_model'],recovery=row,
            source_T501=p['source_row'],first_divergence=p['first_divergence'],not_original_start_student_success=True))
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=B.os.getpid(),error=error,
             physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,result=result,new_optimizer=0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=B.plan();B.verify(p);B.setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(v)).exists() for v in [g['parent'],*g['workers']])
    rows=B.R.B.read_rows('cases.jsonl');assert len(rows)==2 and len({r['path'] for r in rows})==2
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=B.R.B.R.E.N.S.arrays(row['path'])
        assert all(digest(v)==row['array_hashes'][k] for k,v in a.items());B.N.inspect(p,[(a,row)])
    assert rows[0]['array_hashes']==rows[1]['array_hashes'] and rows[0]['integration_sha256']==rows[1]['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics'] and s['commands']==8
    assert s['result']['qualified']==B.R.safe(rows[0])
    assert sha(p['reference_model']['path'])==p['reference_model']['sha256']
    save(OUT/'audit.json',dict(state='PASS',new_optimizer=0,new_native=0,
         actual_T501_prefix_and_teacher_own64plus32plus32_history_rebuilt=True,
         two_full_unique_arrays_and_exact_replay=True,qualified=s['result']['qualified']))


if __name__=='__main__':globals()[sys.argv[1]]()
