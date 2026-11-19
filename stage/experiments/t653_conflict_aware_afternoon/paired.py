"""Frozen paired local-neighbourhood outcomes, all predeclared models/cases."""
import fcntl,gzip,json,os,traceback
import t653_runtime as T
R,O,OUT=T.R,T.O,T.OUT

def main():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'paired';rt=O.Runtime(p);error=None;models=[('T501',p['initial'])]
    for name in p['arms']:
        f=OUT/name/'summary.json'
        if f.exists():
            s=T.read(f);assert s['state']=='COMPLETE' and T.read(OUT/name/'audit.json')['state']=='PASS'
            models.append((name,s['proposal']))
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for name,model in models:
            vv=rt.request(p['paired_jobs'],model,'paired_'+name,True)
            rt.request(p['paired_jobs'],model,'exact_'+name,False,[r for a,r in vv])
    except BaseException:error=traceback.format_exc()
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        bases=[*p['protected_tasks'],'t502_field_sector5_offset1']
        result={name:dict(safe=sum(T.safe(r) for r in rt.rows if r['role']=='paired_'+name),n=sum(r['role']=='paired_'+name for r in rt.rows),model=model,
            by_task={base:dict(safe=sum(T.safe(r) for r in rt.rows if r['role']=='paired_'+name and r['job_id'].startswith('pair_'+base+'_')),n=8) for base in bases}) for name,model in models}
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,models=result,new_optimizer=0,not_unseen_geometry_or_promotion=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();assert error is None,error


def audit():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'paired';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not T.Path('/proc',str(s['pid'])).exists()
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows}
    assert len(rows)==len(by)==48*len(s['models'])
    for row in rows:
        assert T.sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:O.N.inspect(p,a,row)
        else:assert all(T.digest(v)==row['array_hashes'][key] for key,v in a.items())
        if row.get('exact'):
            ref=by[row['reference_path']];assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
    for name,v in s['models'].items():
        actual=[r for r in rows if r['role']=='paired_'+name]
        assert len(actual)==24 and v['safe']==sum(T.safe(r) for r in actual)
        for base,stats in v['by_task'].items():
            selected=[r for r in actual if r['job_id'].startswith('pair_'+base+'_')]
            assert len(selected)==stats['n']==8 and stats['safe']==sum(T.safe(r) for r in selected)
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and s['commands']==len(rows)+6<=p['max_commands']
    T.save(folder/'audit.json',dict(state='PASS',all_frozen_model_full_history_and_exact=True,models=len(s['models']),new_native=0,new_optimizer=0))

if __name__=='__main__':globals()[T.sys.argv[1]]()
