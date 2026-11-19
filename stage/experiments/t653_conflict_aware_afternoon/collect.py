"""Fixed two missing old-task groups,16 actual T501 episodes each, no teacher."""
import fcntl,gzip,json,os,traceback
import t653_runtime as T
R,O,OUT=T.R,T.O,T.OUT

def main():
    p=T.plan();T.verify(p);T.setup(p);O.N.inspect=T.Coverage.inspect;folder=OUT/'shared';rt=O.Runtime(p);error=None
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    T.save(folder/'launch.json',dict(pid=os.getpid(),epoch=T.time.time()))
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for i,job in enumerate(p['extra_jobs']):
            for offset in range(0,16,4):
                seeds=[[20261119653,i,k] for k in range(offset,offset+4)]
                vv=rt.request([job]*4,p['initial'],'shared_fresh',True,seeds=seeds)
                rt.request([job]*4,p['initial'],'shared_exact',False,[r for a,r in vv],seeds=seeds)
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3;fresh=[r for r in rt.rows if r['role']=='shared_fresh'];safe={j['id']:sum(T.safe(r) for r in fresh if r['job_id']==j['id']) for j in p['extra_jobs']}
        qualified=len(fresh)==32 and all(0<v<16 for v in safe.values())
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            fresh=len(fresh),new32_not_replay32=True,explicit_old76_reuse=True,safe_by_task=safe,qualified_for_learning=qualified,
            learning_gate='both missing old tasks must have success AND failure in16actual samples',new_optimizer=0))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error


def audit():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'shared';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not T.Path('/proc',str(s['pid'])).exists()
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows};assert len(by)==len(rows)==64
    for row in rows:
        assert T.sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:T.Coverage.inspect(p,a,row)
        else:assert all(T.digest(v)==row['array_hashes'][key] for key,v in a.items())
        assert row['loaded_model_sha256']==p['initial']['sha256']
        if row.get('exact'):
            ref=by[row['reference_path']];assert ref['seed']==row['seed'] and row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
    safe={j['id']:sum(T.safe(r) for r in rows if r['role']=='shared_fresh' and r['job_id']==j['id']) for j in p['extra_jobs']}
    assert safe==s['safe_by_task'] and s['qualified_for_learning']==all(0<v<16 for v in safe.values())
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and s['commands']==70
    T.save(folder/'audit.json',dict(state='PASS',new32_history_hidden_density_reward_checked=True,new32_exact=True,all_initial_canonical_contracts=True,new_native=0,new_optimizer=0))

if __name__=='__main__':globals()[T.sys.argv[1]]()
