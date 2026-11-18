"""Eight bounded recovery samples and exact replays; never updates weights."""
from pathlib import Path
import copy,fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t644_sampled_recovery';DOC=ROOT/'stage/docs/day_20261118/t644'
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_runtime as R
import t644_native as Native
R.HERE=HERE;R.OUT=OUT;R.DOC=DOC
read,save,sha,digest=R.read,R.save,R.sha,R.digest


def setup(p):
    R.setup(p);R.N.worker=Native.worker;R.N.inspect=Native.inspect


class Runtime(R.Runtime):
    def request(self,jobs,model,role,seeds,references=None):
        results=[]
        for off in range(0,len(jobs),3):
            self.guard();js=jobs[off:off+3]
            assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics']
            assert self.commands+len(js)+3<=self.p['max_commands'];self.commands+=len(js)
            req=[]
            for i,j in enumerate(js):
                key=f'{len(self.rows)+i:04d}_{role}'
                req.append(dict(model=model['path'],job=copy.deepcopy(j),seed=seeds[off+i],query=False,record_key=key,
                    full_path=str(OUT/'episodes'/(key+'.npz')) if references is None else None))
            for i,((a,row),request) in enumerate(zip(self.pool.request(req),req)):
                full=references is None
                path=Path(request['full_path']) if full else OUT/'replays'/(request['record_key']+'.npz')
                if not full:
                    path.parent.mkdir(exist_ok=True);R.E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},self.p['compact_bytes'])
                row.update(job_id=request['job']['id'],request=request,role=role,seed=request['seed'],full=full,path=str(path),sha256=sha(path),
                    loaded_model_sha256=model['sha256'],field_boundary=bool((R.np.abs(a['poses'][:,:2])>7).any()))
                if references is not None:row.update(reference_path=references[off+i]['path'],exact=True)
                self.rows.append(row)
                with gzip.open(OUT/'cases.jsonl.gz','ab') as f:f.write((json.dumps(row)+'\n').encode())
                assert row['policy_hash']==model['policy_hash']
                try:
                    Native.inspect(self.p,a,row)
                    if references is not None:
                        previous=references[off+i]
                        assert row['array_hashes']==previous['array_hashes'] and row['integration_sha256']==previous['integration_sha256']
                except BaseException:
                    if not full:R.E.N.write_npz(OUT/'exact_failure.npz',R.E.N.S.packed(a),self.p['full_bytes'])
                    raise
                results.append((a,row))
        return results


def main():
    p=R.plan();R.verify(p);setup(p);R.E.N.offline();R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=Runtime(p);error=None;fresh=[]
    try:
        rt.pool=R.N.Pool(dict(p,out=str(OUT),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for off in range(0,8,3):
            seeds=p['exploration_seeds'][off:off+3];jobs=[p['handoff_job'] for _ in seeds]
            vv=rt.request(jobs,p['initial'],'sample',seeds);rr=[r for a,r in vv]
            rt.request(jobs,p['initial'],'exact_sample',seeds,rr);fresh+=rr
            save(OUT/'PROGRESS.json',dict(fresh=len(fresh),safe_success=sum(R.safe(r) for r in fresh),physics=sum(rt.counts),commands=rt.commands))
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3;safe=[r for r in fresh if R.safe(r)]
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,new_optimizer=0,result=dict(samples=len(fresh),safe_success=len(safe),qualified_rows=safe,
                first_qualified=safe[0] if safe else None,new_learning=None,full407=None,not_deterministic_student_success=True)))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=R.plan();R.verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(v)).exists() for v in [g['parent'],*g['workers']])
    rows=[json.loads(z) for z in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert len(rows)==16 and len({r['path'] for r in rows})==16
    by={r['path']:r for r in rows};fresh=[r for r in rows if r['role']=='sample'];assert len(fresh)==8
    assert [r['seed'] for r in fresh]==p['exploration_seeds']
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:Native.inspect(p,a,row)
        else:
            assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
            source=by[row['reference_path']]
            assert row['seed']==source['seed'] and row['array_hashes']==source['array_hashes'] and row['integration_sha256']==source['integration_sha256']
    safe=[r for r in fresh if R.safe(r)]
    assert s['result']['safe_success']==len(safe) and s['result']['qualified_rows']==safe
    assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics'] and s['commands']==22
    assert sha(p['initial']['path'])==p['initial']['sha256']
    save(OUT/'audit.json',dict(state='PASS',new_optimizer=0,new_native=0,all8_fresh_full_history_rng_density_rewards_rebuilt=True,
        all8_exact_hashes_and_compact_commands_verified=True,physical_prefix_and_own_carry_checked=True,
        safe_success=len(safe),unique8_seeds_not8_independent_training_runs=True))


if __name__=='__main__':globals()[sys.argv[1]]()
