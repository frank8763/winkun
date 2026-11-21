"""Fixed antithetic episode-parameter sampling, no Actor optimizer or promotion."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t679_episode_parameter_exploration';DOC=ROOT/'stage/docs/weekend_20261120_23/t679'
import t679_wait as Wait
from status_io import save
import t679_calibration as C
T,M,torch,np=C.T,C.M,C.torch,C.np
read,sha,digest=C.read,C.sha,C.digest
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait

def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop'];assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes'] and T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f

T.verify=verify

def collect():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);T.R.E.L.install_stop_handlers();torch.set_num_threads(1)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p);error=None;cal=None;models=[];fresh=[];last=[0.]
    def guard():
        if time.time()-last[0]>5:rt.guard();last[0]=time.time()
        assert time.time()<min(rt.start+p['calibration_seconds'],p['absolute_native_stop'])-10
    try:
        q=M.L.frozen(p,p['initial']['path']);base=q.phash();cal,theta=C.calibration(p,q,guard);save(OUT/'CALIBRATION.json',cal)
        if cal['state']=='PASS':
            for i,seed in enumerate(p['evaluation_direction_seeds']):
                epsilon=theta.new_tensor(C.direction(seed))
                for sign in [-1,1]:
                    guard();name=f'direction{i:02d}_{"minus" if sign<0 else "plus"}'
                    try:C.set_parameters(q,theta+sign*cal['scale']*epsilon);model=C.export_variant(p,q,OUT/'models',name)
                    finally:C.set_parameters(q,theta)
                    assert q.phash()==base;models.append(dict(name=name,seed=seed,sign=sign,model=model))
            del q,theta
            T.pool(rt)
            for item in models:
                vv=rt.request(p['collection_jobs'],item['model'],item['name']+'_fresh',True,seeds=None);rr=[r for a,r in vv];fresh.extend(rr)
                rt.request(p['collection_jobs'],item['model'],item['name']+'_exact',False,rr,seeds=None)
            save(OUT/'MODELS.json',models)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        groups={k:[r for r in fresh if r['job_id']==k] for k in p['collection_ids']};support={};offsets={}
        center=M.L.frozen(p,p['initial']['path'])
        for r in fresh:offsets[r['path']]=C.offset_stats(center,r)
        historical=C.B.dataset(p)
        for key,rr in groups.items():
            y=[M.outcome(r,'collision_cost') for r in rr];support[key]=dict(n=len(rr),outcomes=y,mixed=len(set(y))>1,safe=sum(T.Prior.safe(r) for r in rr),collision=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),
                field_boundary=sum(bool(r.get('field_boundary')) for r in rr) if key in p['field_collection_ids'] else None,
                antithetic_return_differences=[y[2*i+1]-y[2*i] for i in range(len(y)//2)],
                historical_IID_descriptive=dict(n=len(historical[key]),outcomes=[M.outcome(r,'collision_cost') for r in historical[key]],not_new_independent_or_budget_matched=True))
        checks=dict(calibration_pass=cal is not None and cal['state']=='PASS',all64fresh=len(fresh)==64,at_least2_old_mixed=sum(support[k]['mixed'] for k in p['old_collection_ids'])>=2,
                    at_least2_field_mixed=sum(support[k]['mixed'] for k in p['field_collection_ids'])>=2)
        gate='PARAMETER_EXPLORATION_SUPPORT_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED'
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
            gate=gate,checks=checks,calibration=cal,models=models,rows=fresh,support=support,conditional_offset_statistics=offsets,new_Actor=0,new_Value=0,new_optimizer=0,
            perturbation_success_not_center_policy_improvement=True,paired_antithetic_not_independent64samples=True))
        fcntl.flock(lock,fcntl.LOCK_UN)
    assert error is None,error

def audit():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    q=M.L.frozen(p,p['initial']['path']);base=q.phash();cal,theta=C.calibration(p,q,guard);assert cal==s['calibration']
    for item in s['models']:
        model=item['model'];assert sha(model['path'])==model['sha256']
        try:
            C.set_parameters(q,theta+item['sign']*cal['scale']*theta.new_tensor(C.direction(item['seed'])))
            loaded=M.L.frozen(p,model['path']);assert loaded.phash()==q.phash()==model['policy_hash'] and digest(loaded.state_dict())==digest(q.state_dict())
        finally:C.set_parameters(q,theta)
    allrows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (OUT/'cases.jsonl.gz').exists() else [];by={r['path']:r for r in allrows}
    for r in allrows:
        assert sha(r['path'])==r['sha256'];a=T.R.E.N.S.arrays(r['path']);assert r['seed'] is None and not r.get('teacher')
        if r['full']:T.N.inspect(p,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    assert sum(r['physics'] for r in allrows)==s['physics']<=p['max_physics'] and (len(allrows)+6 if allrows else 3)==s['commands']<=p['max_commands']
    assert {r['path']:C.offset_stats(q,r) for r in s['rows']}==s['conditional_offset_statistics']
    assert q.phash()==base and not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'audit.json',dict(state='PASS',calibration_variants_all_arrays_exact=True,new_native=0,new_optimizer=0,gate=s['gate'],center_model_unchanged=True));fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
