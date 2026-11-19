"""Three fixed teacher suffix qualifications and five prior field reconstructions."""
import fcntl,gzip,json,os,traceback
import t654_runtime as T
R,O,OUT=T.R,T.O,T.OUT

def main():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'shared';rt=O.Runtime(p);error=None
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    T.save(folder/'launch.json',dict(pid=os.getpid(),epoch=T.time.time()))
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['recovery_jobs'],p['teacher_model'],'teacher_recovery',True)
        rt.request(p['recovery_jobs'],p['teacher_model'],'exact_recovery',False,[r for a,r in vv])
        vv=rt.request(p['field_reconstruct_jobs'],p['initial'],'field_reconstruct',True,p['field_reconstruct_references'])
        rt.request(p['field_reconstruct_jobs'],p['initial'],'exact_field_reconstruct',False,[r for a,r in vv])
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3;fresh=[r for r in rt.rows if r['role']=='teacher_recovery'];safe={j['id']:any(T.safe(r) for r in fresh if r['job_id']==j['id']) for j in p['recovery_jobs']}
        qualified=len(fresh)==3 and all(safe.values())
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            qualified_for_learning=qualified,safe_by_task=safe,new_teacher_suffixes=3,reconstructed_old_field_outcomes=5,
            no_reset_of_episode_clock_or_hidden=True,new_optimizer=0))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error

def audit():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'shared';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not T.Path('/proc',str(s['pid'])).exists()
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows};assert len(rows)==len(by)==16
    old={r['path']:r for r in p['field_reconstruct_references']}
    for row in rows:
        assert T.sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:T.Native.inspect(p,a,row)
        else:assert all(T.digest(v)==row['array_hashes'][key] for key,v in a.items())
        model=p['teacher_model'] if 'recovery' in row['role'] else p['initial'];assert row['loaded_model_sha256']==model['sha256']
        if row.get('exact'):
            ref=(by|old)[row['reference_path']];assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
    safe={j['id']:any(T.safe(r) for r in rows if r['role']=='teacher_recovery' and r['job_id']==j['id']) for j in p['recovery_jobs']}
    assert safe==s['safe_by_task'] and s['qualified_for_learning']==all(safe.values())
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and s['commands']==22
    T.save(folder/'audit.json',dict(state='PASS',all_three_actual_prefixes_and_teacher_carry_rebuilt=True,all_exact_verified=True,
        five_field_outcomes_identical_prior_not_new_independent=True,new_native=0,new_optimizer=0))
if __name__=='__main__':globals()[T.sys.argv[1]]()
