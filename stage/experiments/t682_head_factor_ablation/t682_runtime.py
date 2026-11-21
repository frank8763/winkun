"""Static head-factor intervention, four actors, no learning or runtime routing."""
from pathlib import Path
import copy,fcntl,gzip,io,json,lzma,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t682_head_factor_ablation';DOC=ROOT/'stage/docs/weekend_20261120_23/t682'
import t682_wait as Wait
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

def parent_rows(p):
    rows=[r for f in p['source_summaries'] for r in read(f)['rows']]
    result={item['name']:[r for r in rows if r['policy_hash']==item['model']['policy_hash'] and r['job_id'] in p['panel_ids']] for item in p['factor_parents']}
    assert all(len(v)==len({r['job_id'] for r in v})==10 for v in result.values())
    return result

def baseline_rows(p):
    old=[json.loads(x) for x in lzma.decompress(Path(p['baseline_old_cases']).read_bytes()).decode().splitlines()]
    fields=[json.loads(x) for x in Path(p['baseline_field_cases']).read_text().splitlines()]
    selected=[r for r in old+fields if r['job_id'] in p['panel_ids']]
    assert len(selected)==len({r['job_id'] for r in selected})==10
    result={};original=M.L.frozen(dict(p,raw_sigma=[.2,.2]),p['safe_retained']['path'])
    for r in selected:
        j=copy.deepcopy(next(j for j in p['factor_jobs'] if j['id']==r['job_id']));j.pop('initial_contract',None)
        assert j==r['request']['job'] and r['loaded_model_sha256']==p['safe_retained']['sha256'] and r['seed'] is None
        if r['job_id'] in p['old_panel_ids']:
            ref=read(r['path']+'.t629-remote.json');assert ref['sha256']==r['sha256'] and not ref['offline_available']
            assert T.Prior.safe(r) and r['job_id'] in p['before_success_ids']
        else:
            assert sha(r['path'])==r['sha256'];a=T.R.E.N.S.arrays(r['path'])
            assert all(digest(v)==r['array_hashes'][k] for k,v in a.items()) and int(a['native_counts'].sum())==r['physics']
            with torch.no_grad():location=original.sequence(torch.tensor(a['states'])).numpy()
            assert np.max(np.abs(location-a['locations']))<1e-10
            r=dict(r,field_boundary=bool((np.abs(a['poses'][:,:2])>p['field_limit_m']).any()))
        result[r['job_id']]=r
    return result

def hybrid(p,parent,head):
    q=M.L.frozen(p,p['initial']['path']);source=M.L.frozen(p,parent['model']['path'])
    assert M.L.fixed_hash(q)==M.L.fixed_hash(source)
    module=getattr(q,head+'_adapter');module.load_state_dict(getattr(source,head+'_adapter').state_dict())
    base=M.L.frozen(p,p['initial']['path'])
    for k,v in q.state_dict().items():
        expected=source.state_dict()[k] if k.startswith(head+'_adapter.') else base.state_dict()[k]
        assert torch.equal(v,expected)
    assert sum(v.numel() for v in module.parameters())==5953
    return q

def export(p,q,name,parent,head):
    packet=copy.deepcopy(M.P.checkpoint(p['initial']['path']))
    packet['joint_state']['actor']=copy.deepcopy(q.state_dict());packet['policy_hash']=q.phash();packet['joint_hash']=digest(packet['joint_state'])
    packet['static_head_factor_intervention']=dict(parent=parent['model'],changed_head=head,no_optimizer=True,no_runtime_selector=True)
    buf=io.BytesIO();torch.save(packet,buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<p['model_bytes']
    path=OUT/'models'/(name+'.pt.gz')
    with path.open('xb') as f:f.write(blob)
    loaded=M.L.frozen(p,path);assert loaded.phash()==q.phash() and digest(loaded.state_dict())==digest(q.state_dict())
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),bytes=len(blob),joint_hash=packet['joint_hash'])

def outcome(r):return int(T.Prior.safe(r))-int(r['collision'])

def result(p,models,fresh):
    original=baseline_rows(p);parents=parent_rows(p);by={(r['policy_hash'],r['job_id']):r for r in fresh}
    assert len(fresh)==len(by)==40
    reports={};feasible={};newsets={}
    for parent in p['factor_parents']:
        name=parent['name'];both={r['job_id']:r for r in parents[name]};table=[];headrows={}
        for head in ['drive','steer']:
            model=next(m for m in models if m['parent']==name and m['head']==head)
            rows={k:by[(model['model']['policy_hash'],k)] for k in p['panel_ids']};headrows[head]=rows
            safe={k for k,r in rows.items() if T.Prior.safe(r)}
            feasible[(name,head)]=set(p['old_panel_ids'])<=safe
            newsets[(name,head)]=safe&set(p['field_panel_ids'])
        for key in p['panel_ids']:
            rr={'initial00':original[key],'drive10':headrows['drive'][key],'steer01':headrows['steer'][key],'both11':both[key]}
            y={k:outcome(r) for k,r in rr.items()}
            table.append(dict(id=key,Y=y,interaction=y['both11']-y['drive10']-y['steer01']+y['initial00'],
                outcomes={k:dict(safe=T.Prior.safe(r),collision=r['collision'],timeout=r['timeout'],boundary=r.get('field_boundary',False),
                    complete=r['complete'],censored=r['administrative_censor'],path=r['path'],sha256=r['sha256']) for k,r in rr.items()},
                initial_old_raw_not_reread_this_round=key in p['old_panel_ids']))
        reports[name]=dict(table=table,heads={h:dict(old_safe=sum(T.Prior.safe(r) for k,r in headrows[h].items() if k in p['old_panel_ids']),
            field_success_ids=sorted(newsets[(name,h)]),safe_ids=sorted(k for k,r in headrows[h].items() if T.Prior.safe(r))) for h in ['drive','steer']})
    checks={}
    for head in ['drive','steer']:
        common=set.intersection(*(newsets[(item['name'],head)] for item in p['factor_parents']))
        checks[head]=dict(both_parents_old4_retained=all(feasible[(x['name'],head)] for x in p['factor_parents']),
            common_new_field_ids=sorted(common),qualified=all(feasible[(x['name'],head)] for x in p['factor_parents']) and len(common)>=2)
    qualified=[h for h,v in checks.items() if v['qualified']]
    gate='HEAD_FACTOR_SEPARATION_PASS' if len(qualified)==1 else ('HEAD_FACTOR_AMBIGUOUS' if qualified else 'SCIENTIFIC_GATE_FAILED')
    return dict(gate=gate,checks=checks,qualified_head=qualified[0] if len(qualified)==1 else None,parents=reports,
        full407=None,accepted=False,new_optimizer=0,no_automatic_learning=True,old4_not_original384_retention=True,
        historical_initial_old_control_metadata_reused_not_full_raw_reaudited=True)

def collect():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);T.R.E.L.install_stop_handlers();torch.set_num_threads(1)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p);error=None;fresh=[];models=[];report=None
    try:
        for parent in p['factor_parents']:
            for head in ['drive','steer']:
                rt.guard();q=hybrid(p,parent,head);name=parent['name']+'_'+head+'_only'
                model=export(p,q,name,parent,head);models.append(dict(name=name,parent=parent['name'],head=head,model=model))
        save(OUT/'MODELS.json',models);T.pool(rt)
        for item in models:
            vv=rt.request(p['factor_jobs'],item['model'],item['name']+'_fresh',True,seeds=None);rr=[r for a,r in vv];fresh.extend(rr)
            rt.request(p['factor_jobs'],item['model'],item['name']+'_exact',False,rr,seeds=None)
        report=result(p,models,fresh);save(OUT/'FACTOR_RESULTS.json',report)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,gate=report['gate'] if report else None,result=report,models=models,rows=fresh,new_Actor=0,new_Value=0,new_optimizer=0,
            center_unchanged=True,full407=None,retained=p['safe_retained']))
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
    models=read(OUT/'MODELS.json');assert models==s['models'];policies={};parentmap={x['name']:x for x in p['factor_parents']}
    for item in models:
        guard();q=hybrid(p,parentmap[item['parent']],item['head']);model=item['model'];assert sha(model['path'])==model['sha256']
        loaded=M.L.frozen(p,model['path']);assert loaded.phash()==q.phash()==model['policy_hash'] and digest(loaded.state_dict())==digest(q.state_dict())
        policies[q.phash()]=(q,model)
    raw=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in raw}
    assert len(raw)==len(by)==80
    for r in raw:
        guard();assert sha(r['path'])==r['sha256'] and r['seed'] is None and not r.get('teacher')
        q,model=policies[r['policy_hash']];assert r['loaded_model_sha256']==model['sha256']
        a=T.R.E.N.S.arrays(r['path'])
        if r['full']:
            T.N.inspect(p,a,r)
            with torch.no_grad():loc=q.sequence(torch.tensor(a['states'])).numpy()
            assert np.max(np.abs(loc-a['locations']))<1e-10
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    for parent in p['factor_parents']:
        q=M.L.frozen(p,parent['model']['path'])
        for r in parent_rows(p)[parent['name']]:
            guard();assert sha(r['path'])==r['sha256'];a=T.R.E.N.S.arrays(r['path']);T.N.inspect(p,a,r)
            with torch.no_grad():loc=q.sequence(torch.tensor(a['states'])).numpy()
            assert np.max(np.abs(loc-a['locations']))<1e-10
    fresh=[r for r in raw if not r.get('exact')];assert fresh==s['rows'] and len(fresh)==40
    assert sum(r['physics'] for r in raw)==s['physics']<=p['max_physics'] and len(raw)+6==s['commands']<=p['max_commands']
    again=result(p,models,fresh);assert again==s['result']==read(OUT/'FACTOR_RESULTS.json') and not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'audit.json',dict(state='PASS',four_static_factor_models_rebuilt_exactly=True,all40full40exact_arrays_integrals_and_owncarry=True,
        all20parent_full_reverified=True,historical_initial_old4_metadata_only_reuse=True,independent_factor_results_exact=True,
        new_native=0,new_optimizer=0,no_model_promoted=True,gate=again['gate']))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
