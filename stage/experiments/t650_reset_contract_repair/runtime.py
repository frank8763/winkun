"""Repair validation only, no learning; six canonical originals and six exact."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t650_reset_contract_repair';DOC=ROOT/'stage/docs/day_20261119/t650'
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_runtime as R
import t650_native as N
import t645_runtime as O
read,save,sha,digest,np=R.read,R.save,R.sha,R.digest,R.np


def plan():return read(HERE/'protocol.json')
def verify(p):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=p['max_bytes'] and R.used(ROOT/'stage/runs')<=10*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f


def main():
    p=plan();verify(p);R.OUT=OUT;O.OUT=OUT;R.DOC=DOC;R.verify=verify;R.setup(p);R.N.worker=N.worker
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=O.Runtime(p);error=None;fresh=[]
    try:
        rt.pool=R.N.Pool(dict(p,out=str(OUT),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['jobs'],p['initial'],'CANONICAL_FRESH',True);fresh=[r for a,r in vv]
        for (a,row),old in zip(vv,p['references']):
            assert row['policy_hash']==old['policy_hash'] and row['array_hashes']==old['array_hashes'] and row['integration_sha256']==old['integration_sha256'],'Canonical full-trajectory mismatch'
        rt.request(p['jobs'],p['initial'],'CANONICAL_EXACT',True,fresh)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
            successes=sum(r['success'] and not r['collision'] for r in fresh),fresh=fresh,new_Actor=0,new_Critic=0,new_optimizer=0,not_new_independent_navigation_training=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);R.OUT=OUT;R.setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):assert all(not Path('/proc',str(v)).exists() for v in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()];assert len(rows)==12 and len({r['path'] for r in rows})==12
    by={r['path']:r for r in rows}
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        assert all(digest(v)==row['array_hashes'][k] for k,v in a.items());N.F.M.audit_full_history(p,row,a)
        ref=next(v for v in p['references'] if v['job_id']==row['job_id']);assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        if row.get('exact'):assert row['array_hashes']==by[row['reference_path']]['array_hashes']
    assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics'] and s['commands']==18
    assert sha(p['initial']['path'])==p['initial']['sha256']
    save(OUT/'audit.json',dict(state='PASS',all_six_canonical_full_arrays_exact=True,all_six_replays_exact=True,all_initial_scene_observation_contracts_before_any_step=True,new_native=0,new_optimizer=0))

if __name__=='__main__':globals()[sys.argv[1]]()
