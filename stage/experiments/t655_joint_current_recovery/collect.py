"""Finite shared current-policy collection: equal old/new task coverage."""
import fcntl,gzip,json,os,traceback
import t655_runtime as T
R,O,OUT=T.R,T.O,T.OUT

def main():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'shared';rt=O.Runtime(p);error=None
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    T.save(folder/'launch.json',dict(pid=os.getpid(),epoch=T.time.time()))
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for method in p['methods']:
            jobs=p['original_jobs'] if method=='original' else p['recovery_jobs']
            for k in range(8):
                seeds=[[20261119655,i,k] for i in range(3)]
                vv=rt.request(jobs,p['initial'],'old_'+method+'_fresh',True,seeds=seeds)
                rt.request(jobs,p['initial'],'old_'+method+'_exact',False,[r for a,r in vv],seeds=seeds)
        for k in range(4):
            seeds=[[20261119655,100+i,k] for i in range(8)]
            vv=rt.request(p['field_train_jobs'],p['initial'],'field_fresh',True,seeds=seeds)
            rt.request(p['field_train_jobs'],p['initial'],'field_exact',False,[r for a,r in vv],seeds=seeds)
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3;safe={method:{j['id']:sum(T.safe(r) for r in rt.rows if r['role']=='old_'+method+'_fresh' and r['job_id']==j['id']) for j in p['original_jobs']} for method in p['methods']}
        qualified=all(all(n>0 for n in v.values()) and any(n<8 for n in v.values()) for v in safe.values())
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            fresh=80,qualified_for_learning=qualified,safe_by_task=safe,gate='Both supports require at least one GOAL in every old group and a mixed old group per support',
            current_policy=p['initial'],no_new_teacher=True,new_optimizer=0))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error

def audit():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'shared';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not T.Path('/proc',str(s['pid'])).exists()
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=T.shared_rows();by={r['path']:r for r in rows};assert len(rows)==len(by)==160
    for row in rows:
        assert T.sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:T.Native.inspect(p,a,row)
        else:assert all(T.digest(v)==row['array_hashes'][key] for key,v in a.items())
        assert row['loaded_model_sha256']==p['initial']['sha256']
        if row.get('exact'):
            ref=by[row['reference_path']];assert ref['seed']==row['seed'] and ref['array_hashes']==row['array_hashes'] and ref['integration_sha256']==row['integration_sha256']
    safe={method:{j['id']:sum(T.safe(r) for r in rows if r['role']=='old_'+method+'_fresh' and r['job_id']==j['id']) for j in p['original_jobs']} for method in p['methods']}
    assert safe==s['safe_by_task'] and s['qualified_for_learning']==all(all(n>0 for n in v.values()) and any(n<8 for n in v.values()) for v in safe.values())
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and s['commands']==166
    T.save(folder/'audit.json',dict(state='PASS',all80_actual_history_and_density_rebuilt=True,all80_exact=True,masked_prefixes_physically_identical=True,new_native=0,new_optimizer=0))
if __name__=='__main__':globals()[T.sys.argv[1]]()
