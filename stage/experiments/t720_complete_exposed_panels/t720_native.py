"""Complete exposed panels, reuse samepolicy audited rows without reexecuting."""
from pathlib import Path
import importlib.util,sys,os,time,fcntl,json,gzip,traceback
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t720_native_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
N=C.M.N;INSPECT=N.BASE_INSPECT

def worker(pipe,index,p,counts):
    return C.worker(pipe,index,p,counts)

def plan(name):
    p=N.plan(name);selection=C.read(C.DOC/'FIXED_SELECTION.json');p.update(selection['panels'][p['panel']]);return p

def arrays(row):assert C.sha(row['path'])==row['sha256'];return N.T.R.E.N.S.arrays(row['path'])
def initial_match(a,b):
    for key in ['states','poses','history_poses','history_goals','history_times']:assert N.np.array_equal(a[key][0],b[key][0]),('initial',key)
    for key in ['initial_native_integration','task_goal_xy','scene_sha256']:assert N.np.array_equal(a[key],b[key]),('initial',key)

def collect(name):
    p=plan(name);folder=Path(p['out']);N.verify(p,force=True);N.setup(p);N.T.R.E.L.install_stop_handlers();model=p['candidate_model'];assert C.sha(model['path'])==model['sha256'];lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=N.S.NativeRuntime(p);error=None;by={r['job_id']:r for r in p['reused_rows']}
    C.save(folder/'REUSED.json',dict(rows=p['reused_rows'],exact=p['reused_exact_rows'],source_audit=str(C.ROOT/'stage/runs/t719_outcome_weighted_replay/validation/audit.json'),new_native=0))
    try:
        N.T.pool(rt);values=rt.request(p['fresh_jobs'],model,'fresh',True);by.update({r['job_id']:r for a,r in values});refs=[by[j['id']] for j in p['exact_jobs']];rt.request(p['exact_jobs'],model,'exact',True,refs)
        assert set(by)=={j['id'] for j in p['all_jobs']}
        safe=sorted(k for k,r in by.items() if N.T.Prior.safe(r));base={r['job_id']:r for r in p['baseline_rows']};old={k for k,r in base.items() if N.T.Prior.safe(r)}
    except BaseException:error=traceback.format_exc();C.save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True));safe=[];old=set()
    finally:
        N.T.close(rt);N.verify(p,force=True);C.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,model=model,total=len(p['all_jobs']),new_fresh=len(p['fresh_jobs']),reused=len(p['reused_rows']),success=len(safe),success_ids=safe,baseline_success=len(old),gained=sorted(set(safe)-old),lost=sorted(old-set(safe)),collision_ids=sorted(k for k,r in by.items() if r['collision']),timeout_ids=sorted(k for k,r in by.items() if not r['success'] and not r['collision']),boundary_ids=sorted(k for k,r in by.items() if 'fullfield_start_pose' in r['request']['job'] and r.get('field_boundary')),physics=sum(rt.counts),commands=rt.commands+7,elapsed_seconds=time.time()-rt.start,new_learning=0));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit(name):
    p=plan(name);folder=Path(p['out']);N.verify(p,force=True);N.setup(p);sm=C.read(folder/'summary.json');assert sm['state']=='COMPLETE' and not Path('/proc',str(sm['pid'])).exists()
    for g in C.read(folder/'worker_generations.json'):assert all(not Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rr=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];references={r['path']:r for r in rr+p['reused_rows']};by={r['job_id']:r for r in p['reused_rows']};base={r['job_id']:r for r in p['baseline_rows']}
    for row in rr+p['reused_rows']+p['reused_exact_rows']:
        N.verify(p);assert row['full'] and row['seed'] is None and row['loaded_model_sha256']==sm['model']['sha256'] and row['policy_hash']==sm['model']['policy_hash'];a=arrays(row);INSPECT(p,a,row);assert int(a['native_counts'].sum())==row['physics'];assert all(N.digest(v)==row['array_hashes'][k] for k,v in a.items());assert not a['ownership'].any() and not a['stochastic'].any();initial_match(a,arrays(base[row['job_id']]))
        if row.get('exact'):
            previous=references[row['reference_path']];assert row['array_hashes']==previous['array_hashes'] and row['integration_sha256']==previous['integration_sha256']
        else:by[row['job_id']]=row
    assert set(by)=={j['id'] for j in p['all_jobs']};assert len(rr)==len(p['fresh_jobs'])+len(p['exact_jobs']);assert sum(row['physics'] for row in rr)==sm['physics']<=p['max_physics'];assert len(rr)+14==sm['commands']<=p['max_commands'];assert sorted(k for k,r in by.items() if N.T.Prior.safe(r))==sm['success_ids']
    N.verify(p,force=True);C.save(folder/'audit.json',dict(state='PASS',complete_original_starts=len(by),new_scientific_rows=len(rr),reused_original=len(p['reused_rows']),reused_exact=len(p['reused_exact_rows']),all_initial_native_scene_goal_packet_paired=True,all_full_payload_owncarry_density_native_and_exact_SHA=True,no_reexecution_of_consumed_T719=True,new_native=0,new_learning=0))

def install():N.evaluate=collect;N.audit=audit;N.BASE_INSPECT=INSPECT;C.M.worker=worker
