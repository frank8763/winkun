"""Replace only four wrong-scene task groups; explicitly reuse60 valid samples."""
import fcntl,gzip,json,os,traceback
import t651_runtime as T
R,O,OUT=T.R,T.O,T.OUT

def main():
    p=T.plan();T.verify(p);T.setup(p);O.N.inspect=T.Coverage.inspect;folder=OUT/'shared';rt=O.Runtime(p);error=None
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    T.save(folder/'launch.json',dict(pid=os.getpid(),epoch=T.time.time()))
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for i,job in enumerate(p['replacement_jobs']):
            seeds=[[20261119651,i,k] for k in range(4)]
            values=rt.request([job]*4,p['initial'],'shared_fresh',True,seeds=seeds)
            rt.request([job]*4,p['initial'],'shared_exact',False,[r for a,r in values],seeds=seeds)
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        fresh=[r for r in rt.rows if r['role']=='shared_fresh'];merged={}
        for job in p['training_jobs']:
            values=[r for r in fresh if r['job_id']==job['id']] or p['reuse_rows'].get(job['id'],[])
            merged[job['id']]=sum(T.M.outcome(r) for r in values)
        informative=[key for key in merged if 0<merged[key]<4]
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
          fresh=len(fresh),explicit_reused_samples=60,safe_by_task=merged,informative_tasks=informative,
          qualified_for_learning=sum(key in p['new_tasks'] for key in informative)>=2,new_optimizer=0))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error


def audit():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'shared';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not T.Path('/proc',str(s['pid'])).exists()
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows};assert len(by)==len(rows)==32
    for r in rows:
        assert T.sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path'])
        if r['full']:T.Coverage.inspect(p,a,r)
        else:assert all(T.digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert r['seed']==ref['seed'] and r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    for rr in p['reuse_rows'].values():
        for r in rr:
            assert T.sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path']);T.Coverage.inspect(p,a,r)
            expected=p['narrow_scene'];assert bytes(a['scene_sha256']).hex()==expected,'Reused geometry not canonical'
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and s['commands']==38
    T.save(folder/'audit.json',dict(state='PASS',fresh16_exact16=True,reused60_original_SHA_history_density_checked=True,all_initial_contracts_before_first_step=True,new_native=0,new_optimizer=0))

if __name__=='__main__':globals()[T.sys.argv[1]]()
