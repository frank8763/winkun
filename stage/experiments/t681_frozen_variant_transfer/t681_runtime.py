"""Complete the two fixed source-positive models' field maps, no optimization."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t681_frozen_variant_transfer';DOC=ROOT/'stage/docs/weekend_20261120_23/t681'
import t681_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t679_episode_parameter_exploration'))
import t679_calibration as C
T,M,torch,np=C.T,C.M,C.torch,C.np
read,sha,digest=C.read,C.sha,C.digest
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait

def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')-T.R.used(OUT)+p['max_bytes']<p['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f

T.verify=verify

def source_rows(p):
    rows=read(p['source_summary'])['rows']
    return {item['name']:[r for r in rows if r['policy_hash']==item['model']['policy_hash']] for item in p['transfer_models']}

def baseline_rows(p):
    rows=[json.loads(x) for x in Path(p['baseline_field_cases']).read_text().splitlines()]
    jobs={j['id']:j for j in p['evaluation_jobs']};result={}
    for r in rows:
        assert sha(r['path'])==r['sha256'] and r['seed'] is None
        job=dict(jobs[r['job_id']]);job.pop('initial_contract',None)
        assert job==r['request']['job'] and r['loaded_model_sha256']==p['safe_retained']['sha256']
        a=T.R.E.N.S.arrays(r['path']);boundary=bool((np.abs(a['poses'][:,:2])>p['field_limit_m']).any())
        result[r['job_id']]=dict(safe=T.Prior.safe(dict(r,field_boundary=boundary)),collision=r['collision'],timeout=r['timeout'],
            boundary=boundary,complete=r['complete'],censored=r['administrative_censor'],source_path=r['path'],sha256=r['sha256'])
    assert len(result)==24 and {k for k,v in result.items() if v['safe']}==set(p['field_ids'])&set(p['before_success_ids'])
    return result

def field_report(p,fresh):
    previous=source_rows(p);baseline=baseline_rows(p);reports={};protected=set(p['before_success_ids'])
    for item in p['transfer_models']:
        name=item['name'];hist=previous[name];rr=[r for r in hist if r['job_id'] in p['discovery_field_ids']]
        new=[r for r in fresh if r['policy_hash']==item['model']['policy_hash']];rr+=new
        by={r['job_id']:r for r in rr}
        assert len(rr)==len(by)==24 and set(by)==set(p['field_ids']) and len(new)==20
        safe={k for k,r in by.items() if T.Prior.safe(r)}
        outside=sorted(safe-set(p['discovery_field_ids'])-protected)
        reports[name]=dict(model=item['model'],field_success=len(safe),success_ids=sorted(safe),
            discovery_success_ids=sorted(safe&set(p['discovery_field_ids'])),
            transfer_success_ids=sorted(safe-set(p['discovery_field_ids'])),new_to_T501_transfer_success_ids=outside,
            retained_T501_field_ids=sorted(safe&protected),lost_T501_field_ids=sorted((set(p['field_ids'])&protected)-safe),
            known_old_results={r['job_id']:dict(safe=T.Prior.safe(r),collision=r['collision'],timeout=r['timeout'],source_path=r['path']) for r in hist if r['job_id'] in p['known_old_ids']},
            field_rows=[dict(id=k,safe=T.Prior.safe(by[k]),collision=by[k]['collision'],timeout=by[k]['timeout'],boundary=by[k]['field_boundary'],complete=by[k]['complete'],censored=by[k]['administrative_censor'],T501=baseline[k],
                source='discovery_reused' if k in p['discovery_field_ids'] else 'new_transfer',path=by[k]['path'],sha256=by[k]['sha256']) for k in p['field_ids']],
            full407=None,accepted=False,known_old_failure_rules_out_retention=True)
    gate='FROZEN_VARIANT_TRANSFER_OBSERVED' if all(v['new_to_T501_transfer_success_ids'] for v in reports.values()) else 'NO_TWO_VARIANT_TRANSFER_SUPPORT'
    return reports,gate

def collect():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);T.R.E.L.install_stop_handlers();torch.set_num_threads(1)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p);error=None;fresh=[];reports=None;gate=None
    try:
        save(OUT/'REUSED_SOURCE_ROWS.json',dict(rows=source_rows(p),new_native=0,not_new_samples=True))
        T.pool(rt)
        for item in p['transfer_models']:
            vv=rt.request(p['transfer_jobs'],item['model'],item['name']+'_transfer',True,seeds=None)
            rr=[r for a,r in vv];fresh.extend(rr)
            rt.request(p['transfer_jobs'],item['model'],item['name']+'_exact',False,rr,seeds=None)
        reports,gate=field_report(p,fresh)
        save(OUT/'FIELD_TRANSFER.json',dict(models=reports,gate=gate,no_center_update=True,full407=None))
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,gate=gate,models=reports,rows=fresh,new_Actor=0,new_Value=0,new_optimizer=0,
            center_unchanged=True,full407=None,retained=p['safe_retained'],postselection_development_transfer_not_blinded_generalization=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);torch.set_num_threads(1);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    raw=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by={r['path']:r for r in raw};assert len(by)==len(raw)==80
    models={x['model']['policy_hash']:x['model'] for x in p['transfer_models']}
    policies={h:M.L.frozen(p,m['path']) for h,m in models.items()}
    reused=source_rows(p);assert reused==read(OUT/'REUSED_SOURCE_ROWS.json')['rows']
    for r in raw+[v for rows in reused.values() for v in rows]:
        guard();assert sha(r['path'])==r['sha256'] and r['seed'] is None and not r.get('teacher')
        model=models[r['policy_hash']];assert r['loaded_model_sha256']==model['sha256'] and sha(model['path'])==model['sha256']
        a=T.R.E.N.S.arrays(r['path'])
        if r['full']:
            T.N.inspect(p,a,r)
            with torch.no_grad():loc=policies[r['policy_hash']].sequence(torch.tensor(a['states'])).numpy()
            assert np.max(np.abs(loc-a['locations']))<1e-10 and not a['stochastic'].any()
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    fresh=[r for r in raw if not r.get('exact')];assert fresh==s['rows'] and len(fresh)==40
    assert sum(r['physics'] for r in raw)==s['physics']<=p['max_physics'] and len(raw)+6==s['commands']<=p['max_commands']
    reports,gate=field_report(p,fresh);assert reports==s['models'] and gate==s['gate']
    assert read(OUT/'FIELD_TRANSFER.json')==dict(models=reports,gate=gate,no_center_update=True,full407=None)
    assert all(any(not r['safe'] for r in v['known_old_results'].values()) for v in reports.values())
    assert all(q.phash()==h for h,q in policies.items()) and not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'audit.json',dict(state='PASS',all40fresh40exact_and16source_rows_verified=True,all_arrays_native_integrals_and_owncarry=True,
        independent_per_model_field24_report_exact=True,new_native=0,new_optimizer=0,no_model_promoted=True,gate=gate))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
