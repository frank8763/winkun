"""Four fixed forgetting counterexamples, full causal histories, no learning."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t666_failure_full_history';DOC=ROOT/'stage/docs/weekend_20261120_23/t666'
import t666_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t665_remaining_verification'))
import t665_runtime as T
import native as Native
read,sha,digest=T.read,T.sha,T.digest
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait


def plan():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json')
    return p


def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json')
    assert T.R.used(OUT)<=p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>=40*2**30
    Wait.resources()
    h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f


T.verify=verify


def setup(p):
    T.setup(p);T.R.N.worker=Native.worker
    T.R.N.inspect=T.O.N.inspect=Native.inspect
    T.N.inspect=Native.inspect
    Native.P.frozen=Native.frozen


def expected_row(pair,which):
    return pair['reference_row'] if which=='reference' else pair['candidate_row']


def collect():
    p=plan();verify(p);setup(p);T.R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=T.Runtime(p);error=None;checks=[]
    try:
        T.pool(rt)
        for pair in p['pairs']:
            for which in ['reference','candidate']:
                model=pair[which];prior=expected_row(pair,which)
                vv=rt.request([pair['job']],model,pair['id']+'_'+which,True)
                row=vv[0][1]
                # These are deterministic reproductions of measured cases,
                # not additional independent demonstrations or a new seed.
                assert row['array_hashes']==prior['array_hashes']
                assert row['integration_sha256']==prior['integration_sha256']
                assert row['success']==prior['success'] and row['collision']==prior['collision']
                rt.request([pair['job']],model,pair['id']+'_'+which+'_exact',True,[row])
                checks.append(dict(id=pair['id'],which=which,source_path=prior['path'],row_path=row['path'],exact_prior_arrays=True))
                save(OUT/'PROGRESS.json',dict(complete_originals=len(checks),planned_originals=8,physics=sum(rt.counts),commands=rt.commands))
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,
            physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
            verified=checks,rows=rt.rows,new_Actor=0,new_Value=0,new_optimizer=0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json')
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert rows==s['rows'] and len(rows)==16 and not list(OUT.rglob('optimizer_calls.jsonl'))
    by={r['path']:r for r in rows};assert len(by)==16
    report=[]
    for row in rows:
        assert row['full'] and row['seed'] is None and sha(row['path'])==row['sha256']
        a=T.R.E.N.S.arrays(row['path']);T.N.inspect(p,a,row)
        if row.get('exact'):
            prev=by[row['reference_path']]
            assert row['array_hashes']==prev['array_hashes'] and row['integration_sha256']==prev['integration_sha256']
        else:
            pair=next(x for x in p['pairs'] if x['id']==row['job_id'])
            which='reference' if row['role'].endswith('_reference') else 'candidate'
            prior=expected_row(pair,which)
            assert row['array_hashes']==prior['array_hashes'] and row['integration_sha256']==prior['integration_sha256']
            assert row['request']['job']==pair['job'] and row['loaded_model_sha256']==pair[which]['sha256']
            report.append(dict(id=row['job_id'],which=which,success=T.Prior.safe(row),collision=row['collision'],path=row['path']))
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    assert s['commands']==22<=p['max_commands']
    save(OUT/'audit.json',dict(state='PASS',all16_full_history_and_exact_arrays=True,
        prior_compact_arrays_reproduced=True,new_optimizer=0,new_native=0,results=report))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
