"""One shared fresh T501 dataset; exact repeats excluded from learning samples."""
import fcntl,gzip,json,os,traceback
import t649_runtime as T
R,O,OUT=T.R,T.O,T.OUT

def main():
    p=T.plan();T.verify(p);T.setup(p);O.N.inspect=T.Coverage.inspect;folder=OUT/'shared';rt=O.Runtime(p);error=None
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (folder/'launch.json').exists();T.save(folder/'launch.json',dict(pid=os.getpid(),epoch=T.time.time()))
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for i,job in enumerate(p['training_jobs']):
            seeds=[[20261119649,i,k] for k in range(4)]
            values=rt.request([job]*4,p['initial'],'shared_fresh',True,seeds=seeds)
            rt.request([job]*4,p['initial'],'shared_exact',False,[r for a,r in values],seeds=seeds)
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        fresh=[r for r in rt.rows if r['role']=='shared_fresh'];groups={j['id']:[r for r in fresh if r['job_id']==j['id']] for j in p['training_jobs']}
        safe={k:sum(R.safe(v) for v in g) for k,g in groups.items()}
        informative=[key for key,v in safe.items() if 0<v<4]
        new_informative=[key for key in informative if key in p['new_tasks']]
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,fresh=len(fresh),safe_by_task=safe,informative_tasks=informative,
           new_informative_tasks=new_informative,qualified_for_learning=len(new_informative)>=2,optimizer=0,all_samples_used_in_PG=True,exact_not_independent=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error


def audit():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'shared';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not T.Path('/proc',str(s['pid'])).exists()
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows}
    assert len(rows)==len(by)==152
    for r in rows:
        assert T.sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path'])
        if r['full']:T.Coverage.inspect(p,a,r)
        else:assert all(T.digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert r['loaded_model_sha256']==p['initial']['sha256']
        if r.get('exact'):
            ref=by[r['reference_path']];assert ref['seed']==r['seed'] and ref['array_hashes']==r['array_hashes'] and ref['integration_sha256']==r['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    fresh=[r for r in rows if r['role']=='shared_fresh'];safe={j['id']:sum(R.safe(r) for r in fresh if r['job_id']==j['id']) for j in p['training_jobs']}
    assert safe==s['safe_by_task'];informative=[k for k in safe if 0<safe[k]<4]
    assert informative==s['informative_tasks'] and s['qualified_for_learning']==(sum(k in p['new_tasks'] for k in informative)>=2)
    T.save(folder/'audit.json',dict(state='PASS',all76_fresh_full_history_hidden_density_reward_checked=True,all76_exact=True,new_native=0,new_optimizer=0))

if __name__=='__main__':globals()[T.sys.argv[1]]()
