"""One recovery and exact replay at a measured actual candidate divergence."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t643_old_policy_recovery';DOC=ROOT/'stage/docs/day_20261118/t643'
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_runtime as R
import t643_native as Native
R.HERE=HERE;R.OUT=OUT;R.DOC=DOC
read,save,sha,digest=R.read,R.save,R.sha,R.digest


def setup(p):
    R.setup(p);R.N.worker=Native.worker;R.N.inspect=Native.inspect


def main():
    p=R.plan();R.verify(p);setup(p);R.E.N.offline();R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=R.Runtime(p);error=None;result=dict(qualified=False,recovery=None,new_learning=None,full407=None)
    try:
        rt.pool=R.N.Pool(dict(p,out=str(OUT),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request([p['handoff_job']],p['initial'],'old_policy_recovery',True);a,row=vv[0]
        rt.request([p['handoff_job']],p['initial'],'exact_recovery',True,[row])
        result.update(qualified=R.safe(row),recovery=row)
        save(OUT/'RECOVERY_ADMISSION.json',dict(passed=result['qualified'],row=row,first_divergence=p['first_divergence'],
            training_only_not_student_success=True,no_runtime_router=True))
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,
           physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,new_optimizer=0,result=result))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=R.plan();R.verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(v)).exists() for v in [g['parent'],*g['workers']])
    rows=[json.loads(z) for z in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert len(rows)==2 and len({r['path'] for r in rows})==2
    for row in rows:
        assert sha(row['path'])==row['sha256'];Native.inspect(p,R.E.N.S.arrays(row['path']),row)
    assert rows[0]['array_hashes']==rows[1]['array_hashes'] and rows[0]['integration_sha256']==rows[1]['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics'] and s['commands']==8
    assert s['result']['qualified']==R.safe(rows[0]) and sha(p['initial']['path'])==p['initial']['sha256']
    save(OUT/'audit.json',dict(state='PASS',new_optimizer=0,new_native=0,
        actual_candidate_prefix_and_boundary_exact=True,T501_teacher_own_full_history_and_carry_rebuilt=True,
        both_full_raw_unique_and_exact=True,qualified=s['result']['qualified']))


if __name__=='__main__':globals()[sys.argv[1]]()
