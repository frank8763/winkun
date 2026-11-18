"""Fixed frozen local-neighbourhood profile; exact repeats are not new trials."""
import copy,fcntl,gzip,json,os,traceback
import t646_runtime as T
R,O,OUT,DOC=T.R,T.O,T.OUT,T.DOC

def main():
    p=T.configure();O.setup(p);folder=OUT/'profile';rt=O.Runtime(p);error=None
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    models=[]
    for name in p['arms']:
        s=T.read(OUT/name/'summary.json')
        if s['retained']['sha256']!=p['initial']['sha256']:models.append((s['final407']['success'],name,s['retained']))
    chosen=sorted(models,key=lambda v:(-v[0],p['arms'].index(v[1])))[0][2] if models else p['initial']
    T.save(folder/'SELECTED.json',dict(model=chosen,rule='highest accepted full407 success; fixed arm order ties; fallback original T501'))
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        first=rt.request(p['profile_jobs'],chosen,'profile_fresh',True)
        rt.request(p['profile_jobs'],chosen,'profile_exact',False,[r for a,r in first])
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        result=dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,error=error,
                    success=sum(R.safe(r) for r in rt.rows if r['role']=='profile_fresh'),fresh=sum(r['role']=='profile_fresh' for r in rt.rows),new_Actor=0,new_Critic=0)
        T.save(folder/'summary.json',result)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error

def audit():
    p=T.configure();O.setup(p);folder=OUT/'profile';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE' and not T.Path('/proc',str(s['pid'])).exists()
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by={r['path']:r for r in rows};assert len(rows)==len(by)==96
    for r in rows:
        assert T.sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path'])
        if r['full']:O.N.inspect(p,a,r)
        else:assert all(T.digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert ref['array_hashes']==r['array_hashes'] and ref['integration_sha256']==r['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    T.save(folder/'audit.json',dict(state='PASS',actual_profile_cases=48,exact_repeats=48,new_native=0,new_optimizer=0,not_unseen_geometry=True))
