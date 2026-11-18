"""Bounded stratified original-start stochastic coverage; no optimizer."""
import fcntl,gzip,json,os,time,traceback
import t646_runtime as T
O,R,OUT=T.O,T.R,T.OUT

def inspect(p,a,row):
    assert row['seed'] is not None and not a['ownership'].any() and a['stochastic'].all()
    assert T.np.array_equal(a['actions'],T.np.clip(a['raw_actions'],-1,1).astype(T.np.float32))
    assert T.np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    O.N.F.M.audit_full_history(p,row,a)
    assert all(T.digest(v)==row['array_hashes'][k] for k,v in a.items())
    assert int(a['native_counts'].sum())==row['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1

def main():
    p=T.configure();O.setup(p);O.N.inspect=inspect;folder=OUT/'coverage'
    rt=O.Runtime(p);error=None;reason='ALL_FIXED_COVERAGE_COMPLETE'
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=p['initial']  # fixed original baseline, never selected post hoc
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        # Round-robin tasks before increasing replicate count; no seed search.
        queue=[(j,[20261118646,99,i,k]) for k in range(32) for i,j in enumerate(p['coverage_jobs'])]
        for off in range(0,len(queue),3):
            if time.time()+650>=rt.deadline:reason='FINITE_CLOCK_RESERVE_FOR_EXACT_AND_AUDIT';break
            batch=queue[off:off+3];jobs=[j for j,seed in batch];seeds=[seed for j,seed in batch]
            first=rt.request(jobs,model,'coverage_fresh',True,seeds=seeds)
            rt.request(jobs,model,'coverage_exact',False,[r for a,r in first],seeds=seeds)
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        first=[r for r in rt.rows if r['role']=='coverage_fresh']
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,stop_reason=reason,
            physics=sum(rt.counts),commands=rt.commands,fresh=len(first),success=sum(R.safe(r) for r in first),new_Actor=0,new_Critic=0,
            tasks={j['id']:dict(attempts=sum(r['job_id']==j['id'] for r in first),safe_success=sum(r['job_id']==j['id'] and R.safe(r) for r in first)) for j in p['coverage_jobs']},
            new_success_rows=[r for r in first if R.safe(r)],not_deterministic_acceptance=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error

def audit():
    p=T.configure();O.setup(p);folder=OUT/'coverage';s=T.read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not T.Path('/proc',str(s['pid'])).exists()
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by={r['path']:r for r in rows};assert len(by)==len(rows)==2*s['fresh']
    for r in rows:
        assert T.sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path'])
        if r['full']:inspect(p,a,r)
        else:assert all(T.digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            previous=by[r['reference_path']];assert previous['seed']==r['seed'] and previous['array_hashes']==r['array_hashes'] and previous['integration_sha256']==r['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    T.save(folder/'audit.json',dict(state='PASS',fresh=s['fresh'],actual_success=s['success'],all_full_history_hidden_density_reward_rebuilt=True,
        full_exact_checks=True,new_native=0,new_optimizer=0,not_deterministic_navigation_promotion=True))
