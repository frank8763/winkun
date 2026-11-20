"""T501's conditional recovery on four actual student histories, not deployment."""
from pathlib import Path
import fcntl,gzip,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t668_observed_departure_recovery';DOC=ROOT/'stage/docs/weekend_20261120_23/t668'
import t668_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t665_remaining_verification'));import t665_runtime as T
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'));import t654_native as N
read,sha,digest,np=T.read,T.sha,T.digest,T.np
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait


def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f


T.verify=verify


def setup(p):
    T.setup(p);T.R.N.worker=N.worker;T.R.N.inspect=T.O.N.inspect=N.inspect
    T.N.inspect=N.inspect;N.P.frozen=N.L.frozen


def collect():
    p=read(OUT/'protocol.json');verify(p);setup(p);T.R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=T.Runtime(p);error=None
    try:
        T.pool(rt)
        for job in p['recovery_jobs']:
            vv=rt.request([job],p['initial'],job['id']+'_recovery',True)
            rt.request([job],p['initial'],job['id']+'_exact',True,[vv[0][1]])
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        original=[r for r in rt.rows if not r.get('exact')]
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,rows=rt.rows,results=[dict(id=r['job_id'],safe=T.Prior.safe(r),collision=r['collision'],remaining_m=r['remaining_m'],recovery_first=r['request']['job']['recovery_first']) for r in original],
            qualified_recovery_ids=[r['job_id'] for r in original if T.Prior.safe(r)],new_Actor=0,new_Value=0,new_optimizer=0,
            conditional_recovery_not_original_start_single_actor=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=read(OUT/'protocol.json');verify(p);setup(p);s=read(OUT/'summary.json')
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    import json
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert rows==s['rows'] and len(rows)==8 and not list(OUT.rglob('optimizer_calls.jsonl'))
    by={r['path']:r for r in rows};assert len(by)==8
    for row in rows:
        assert sha(row['path'])==row['sha256'] and row['full'] and row['seed'] is None
        N.inspect(p,T.R.E.N.S.arrays(row['path']),row)
        assert row['loaded_model_sha256']==p['initial']['sha256']
        assert row['request']['job']==next(j for j in p['recovery_jobs'] if j['id']==row['job_id'])
        if row.get('exact'):
            prev=by[row['reference_path']];assert row['array_hashes']==prev['array_hashes'] and row['integration_sha256']==prev['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    assert s['commands']==14<=p['max_commands']
    save(OUT/'audit.json',dict(state='PASS',all8_full_prefix_and_own_carry_and_exact=True,new_native=0,new_optimizer=0,
        qualified_recovery_ids=s['qualified_recovery_ids'],conditional_recovery_not_original_start_single_actor=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
