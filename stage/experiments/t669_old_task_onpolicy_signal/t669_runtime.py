"""Real old-task current-policy outcome support; no teacher or learning."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t669_old_task_onpolicy_signal';DOC=ROOT/'stage/docs/weekend_20261120_23/t669'
import t669_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t665_remaining_verification'));import t665_runtime as T
read,sha,digest,np=T.read,T.sha,T.digest,T.np
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait


def verify(p,*args):
    root=read(OUT/'protocol.json');assert root['authorization_approved'] and time.time()<root['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in root['inputs'].items():assert sha(f)==h,f


T.verify=verify


def panel(p,rows):
    out=[]
    for job in p['old_signal_jobs']:
        vals=[r for r in rows if r['job_id']==job['id'] and not r.get('exact')]
        assert len(vals)==8
        safe=sum(T.Prior.safe(r) for r in vals)
        outcomes=[T.M.outcome(r,'collision_cost') for r in vals]
        out.append(dict(id=job['id'],n=8,safe=safe,collision=sum(r['collision'] for r in vals),timeout=sum(r['timeout'] for r in vals),
            binary_mixed=0<safe<8,collision_outcome_mixed=len(set(outcomes))>1,outcomes=outcomes,
            mean_collision_objective=sum(outcomes)/8))
    return out


def collect():
    p=read(OUT/'protocol.json');verify(p);p['initial']=T.M.initial_variant(p,OUT);save(OUT/'INITIAL_MODEL.json',p['initial'])
    T.setup(p);T.R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=T.Runtime(p);error=None;results=None
    try:
        T.pool(rt);jobs=[j for j in p['old_signal_jobs'] for _ in range(8)]
        seeds=[[p['signal_seed'],i] for i in range(len(jobs))]
        values=rt.request(jobs,p['initial'],'old_original_start',True,seeds=seeds)
        rt.request(jobs,p['initial'],'exact_old_original_start',True,[r for a,r in values],seeds=seeds)
        results=panel(p,rt.rows)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,rows=rt.rows,panel=results,model=p['initial'],new_Actor=0,new_Value=0,new_optimizer=0,
            sampled_success_not_deterministic_acceptance=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=read(OUT/'protocol.json');verify(p);p['initial']=read(OUT/'INITIAL_MODEL.json');T.setup(p);s=read(OUT/'summary.json')
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert rows==s['rows'] and len(rows)==64 and not list(OUT.rglob('optimizer_calls.jsonl'))
    by={r['path']:r for r in rows};assert len(by)==64
    for row in rows:
        assert sha(row['path'])==row['sha256'] and row['full'] and row['seed'] is not None
        T.N.inspect(p,T.R.E.N.S.arrays(row['path']),row)
        assert row['loaded_model_sha256']==p['initial']['sha256'] and row['policy_hash']==p['initial']['policy_hash']
        assert row['request']['job']==next(j for j in p['old_signal_jobs'] if j['id']==row['job_id'])
        if row.get('exact'):
            prev=by[row['reference_path']];assert row['array_hashes']==prev['array_hashes'] and row['integration_sha256']==prev['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and s['commands']==70<=p['max_commands']
    assert panel(p,rows)==s['panel']
    save(OUT/'audit.json',dict(state='PASS',all64_full_history_density_and_exact=True,new_native=0,new_optimizer=0,
        mixed_binary_tasks=sum(x['binary_mixed'] for x in s['panel']),mixed_collision_tasks=sum(x['collision_outcome_mixed'] for x in s['panel']),
        sampled_success_not_deterministic_acceptance=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
