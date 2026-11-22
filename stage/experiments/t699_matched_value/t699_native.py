"""Uncensored canonical native comparisons, shared CPU arithmetic and audits."""
import copy, fcntl, gzip, os, sys, time
from t699_common import *
import t699_actor as A
import t698_runtime as S
N,T,torch,np,digest=S.N,S.T,A.torch,A.np,A.digest
BASE_INSPECT=N.ORIGINAL_INSPECT

def worker(pipe,index,p,counts):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    assert not torch.cuda.is_initialized()
    N.C.ACTIVE=p;real=N.G._real_affinity
    def pin(pid,mask):
        return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;N.G.T.Session=N.C.Session
    N.G.B.native_setup=lambda folder,pc:N.C.C.consistent_reset.setup(N.F,folder,pc,N.C.C.CANONICAL_SETUP)
    N.P.frozen=A.frozen
    N.G.T.worker(pipe,index,p,counts)

def verify(p,*unused,force=False):
    root=read(OUT/'protocol.json');immutable(root,full=force)
    assert time.time()<p['absolute_all_stop'] and used(Path(p['out']))<p['child_bytes']
    guard=read(DOC/'AFFINITY_GUARD.json')
    assert guard['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(guard['pid'])).exists()

def setup(p):
    folder=Path(p['out']);S.OUT=folder;S.ROOTOUT=OUT;S.HERE=HERE;S.ROOTDOC=DOC
    T.OUT=OUT;T.DOC=DOC;T.HERE=HERE;T.verify=verify;S.verify=verify
    # Runtime output redirection is process-local. No historical source is modified.
    T.setup(p)
    N.worker=worker;N.inspect=BASE_INSPECT
    T.R.N.worker=worker;T.R.N.inspect=T.O.N.inspect=T.N.inspect=BASE_INSPECT
    T.R.N.Pool=N.Pool;N.P.frozen=A.frozen
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def model(p,name):
    if name=='historical_A':return p['existing_A']
    if name=='historical_parent':return p['initial']
    return read(OUT/'fit/summary.json')['finals'][name]['model']

def plan(child):
    p=read(OUT/'protocol.json');c=read(OUT/child/'protocol.json')
    assert c['_root_sha256']==sha(OUT/'protocol.json')
    return dict(p,**c)

def evaluate(child):
    p=plan(child);folder=Path(p['out']);verify(p,force=True);setup(p)
    T.R.E.L.install_stop_handlers();m=model(p,p['model_name'])
    assert sha(m['path'])==m['sha256'] and read(OUT/'fit/audit.json')['state']=='PASS'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rt=S.NativeRuntime(p);save(folder/'science_started.json',dict(pid=os.getpid(),epoch=time.time()))
    error=None
    try:
        T.pool(rt)
        if p['suite']=='known':
            first=p['screen_jobs'];all_jobs=p['evaluation_jobs']
        else:
            first=p['new_jobs'];all_jobs=first
        returned=rt.request(first,m,'fresh',True);refs=[r for a,r in returned]
        exact=p['screen_jobs'] if p['suite']=='known' else p['new_exact_jobs']
        by={r['job_id']:r for r in refs}
        rt.request(exact,m,'exact',False,[by[j['id']] for j in exact])
        rest=[j for j in all_jobs if j['id'] not in by]
        # No navigation gate removes a comparator, even if it loses all old IDs.
        for off in range(0,len(rest),42):
            value=rt.request(rest[off:off+42],m,'rest',False)
            by.update({r['job_id']:r for a,r in value})
        assert len(by)==len(all_jobs)
        ids=sorted(k for k,r in by.items() if T.Prior.safe(r))
        core=set(p['core407_ids']);old=set(p['required403_ids']);source=set(p['source_ids'])
        summary=dict(state='COMPLETE',pid=os.getpid(),model=m,suite=p['suite'],physics=sum(rt.counts),commands=rt.commands+7,
            success_ids=ids,success=len(ids),total=len(all_jobs),core_success=len(set(ids)&core),
            lost403=sorted(old-set(ids)) if p['suite']=='known' else None,
            source_success_ids=sorted(source&set(ids)),supplementary_success_ids=sorted(set(ids)-core) if p['suite']=='known' else [],
            collision_ids=sorted(k for k,r in by.items() if r['collision']),
            timeout_ids=sorted(k for k,r in by.items() if not r['success'] and not r['collision']),
            boundary_ids=sorted(k for k,r in by.items() if 'fullfield_start_pose' in r['request']['job'] and r.get('field_boundary')),
            elapsed_seconds=time.time()-rt.start)
    except BaseException:
        import traceback
        error=traceback.format_exc()
        summary=dict(state='FAILED',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands+7,
            completed_rows=len(rt.rows),elapsed_seconds=time.time()-rt.start,model=m,suite=p['suite'])
    finally:
        T.close(rt);verify(p,force=True);save(folder/'summary.json',summary)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if error:raise RuntimeError(error)

def audit(child):
    p=plan(child);folder=Path(p['out']);verify(p,force=True);setup(p)
    s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for generation in read(folder/'worker_generations.json'):
        assert all(not Path('/proc',str(pid)).exists() for pid in [generation['parent'],*generation['workers']])
    rows=[json.loads(line) for line in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by_path={r['path']:r for r in rows};fresh=[r for r in rows if not r.get('exact')]
    assert len(by_path)==len(rows)
    for row in rows:
        verify(p);assert sha(row['path'])==row['sha256']
        assert row['policy_hash']==s['model']['policy_hash'] and row['loaded_model_sha256']==s['model']['sha256']
        arr=T.R.E.N.S.arrays(row['path'])
        if row['full']:BASE_INSPECT(p,arr,row)
        else:assert all(digest(v)==row['array_hashes'][k] for k,v in arr.items())
        if row.get('exact'):
            previous=by_path[row['reference_path']]
            assert row['array_hashes']==previous['array_hashes'] and row['integration_sha256']==previous['integration_sha256']
    jobs=p['evaluation_jobs'] if p['suite']=='known' else p['new_jobs']
    assert len(fresh)==len(jobs) and {r['job_id'] for r in fresh}=={j['id'] for j in jobs}
    ids=sorted(r['job_id'] for r in fresh if T.Prior.safe(r));assert ids==s['success_ids']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    assert len(rows)+14==s['commands']<=p['max_commands']
    if p['suite']=='known':assert sorted(set(p['required403_ids'])-set(ids))==s['lost403']
    # All independent models share the same prospective, policy-independent reset hashes.
    if p['suite']=='new':
        contracts={r['id']:r for r in p['new_initial_contracts']}
        for row in fresh:
            arr=T.R.E.N.S.arrays(row['path']);c=contracts[row['job_id']]
            assert digest(arr['states'][0])==c['packet_hash']
            assert digest(arr['initial_native_integration'])==c['native_hash']
    verify(p,force=True)
    save(folder/'audit.json',dict(state='PASS',all_arrays_hashes_owncarry_density_reward_native_integral_and_exact_rows=True,
        actual_science_rows=len(rows),actual_physics=s['physics'],all_source_dependencies_sha_verified=True,
        new_native=0,comparator_not_screened_for_forgetting=True))

if __name__=='__main__':
    {'evaluate':evaluate,'audit':audit}[sys.argv[1]](sys.argv[2])
