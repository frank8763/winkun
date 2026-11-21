"""Real frozen loaders, source provenance and unused task partition preflight."""
from pathlib import Path
import ast,copy,importlib.util,json,sys,time
import t681_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,M,torch,np=S.T,S.M,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    preceding_preflight=read(HERE/'protocol.json')['preflight_seconds'] if (HERE/'protocol.json').exists() else 0
    src=ROOT/'stage/runs/t679_episode_parameter_exploration';last=ROOT/'stage/runs/t680_parameter_estimator_gate'
    for folder in [src,last]:
        assert read(folder/'FINAL_REPORT.json')['state']=='COMPLETE' and read(folder/'audit.json')['state']=='PASS'
        seal=read(folder/'SEALED.json');assert seal['all_scientific_processes_stopped']
        assert all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids']+[read(folder/'DRIVER_STARTED.json')['pid']])
    assert read(last/'RESULTS.json')['state']=='SCIENTIFIC_GATE_FAILED'
    prior=read(src/'protocol.json');p=copy.deepcopy(prior);p.pop('registration_epoch',None)
    source=read(src/'summary.json');allmodels=read(src/'MODELS.json')
    discovery=prior['field_collection_ids'];protected=set(prior['before_success_ids']);assert len(protected)==384
    selected=[m for m in allmodels if any(r['policy_hash']==m['model']['policy_hash'] and r['job_id'] in discovery and r['job_id'] not in protected and T.Prior.safe(r) for r in source['rows'])]
    assert [m['name'] for m in selected]==['direction00_minus','direction03_plus']
    jobs=[j for j in prior['evaluation_jobs'] if j['id'].startswith('t502_field_')];assert len(jobs)==24
    new=[j for j in jobs if j['id'] not in discovery];assert len(new)==20 and not set(j['id'] for j in new)&set(discovery)
    a=read(DOC.parent/'AUTHORIZATION.json');assert a['approved'];now=time.time()
    p.update(id='T681_FIXED_FROZEN_VARIANT_TRANSFER',authorization_approved=False,out=str(OUT),arm='frozen_transfer',
        source_summary=str(src/'summary.json'),transfer_models=selected,discovery_field_ids=discovery,known_old_ids=prior['old_collection_ids'],
        baseline_field_cases=str(ROOT/'stage/runs/t502_fullfield_probe/probe/cases.jsonl'),
        field_ids=[j['id'] for j in jobs],transfer_jobs=new,max_physics=4*sum(j['case']['step_cap'] for j in new),max_commands=86,
        max_bytes=256*2**20,child_bytes=256*2**20,global_bytes=10*2**30,main_seconds=2400,audit_seconds=300,
        absolute_learning_stop=min(now+2400,a['absolute_learning_stop']),absolute_native_stop=min(now+2400,a['absolute_native_stop']),
        absolute_all_stop=min(now+3000,a['absolute_all_stop']),new_Actor=0,new_Value=0,new_optimizer=0,finalization_seconds=60)
    assert p['max_physics']==4973312 and len(selected)*len(new)*2+6==p['max_commands']
    assert 40*p['full_bytes']+40*p['compact_bytes']+8*2**20<p['max_bytes']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);T.setup(p)
    original=M.L.frozen(dict(p,raw_sigma=[.2,.2]),p['safe_retained']['path']);initial=M.L.frozen(p,p['initial']['path'])
    assert digest(original.state_dict())==digest(initial.state_dict())
    baseline=S.baseline_rows(p);assert len(baseline)==24
    models={m['model']['policy_hash']:M.L.frozen(p,m['model']['path']) for m in selected};rows=S.source_rows(p)
    sources=[]
    for item in selected:
        model=item['model'];q=models[model['policy_hash']];assert sha(model['path'])==model['sha256'] and q.phash()==model['policy_hash']
        rr=rows[item['name']];assert len(rr)==8 and {r['job_id'] for r in rr}==set(prior['collection_ids'])
        assert any(not T.Prior.safe(r) for r in rr if r['job_id'] in p['known_old_ids'])
        assert not {r['job_id'] for r in rr}&{j['id'] for j in new}
        before=digest(q.state_dict())
        for r in rr:
            assert sha(r['path'])==r['sha256'] and r['loaded_model_sha256']==model['sha256'] and r['seed'] is None
            aa=T.R.E.N.S.arrays(r['path']);T.N.inspect(p,aa,r)
            with torch.no_grad():loc=q.sequence(torch.tensor(aa['states'])).numpy()
            assert np.max(np.abs(loc-aa['locations']))<1e-10
            assert np.array_equal(np.clip(aa['raw_actions'],-1,1).astype(np.float32),aa['actions'])
            sources.append(r['path'])
        assert digest(q.state_dict())==before and all(not x.requires_grad for x in q.parameters())
    # Include both original fresh and exact provenance; do not execute native.
    source_cases=[json.loads(v) for v in S.gzip.decompress((src/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    selected_hashes=set(models);cases=[r for r in source_cases if r['policy_hash'] in selected_hashes]
    assert len(cases)==32
    by={r['path']:r for r in cases}
    for r in cases:
        assert sha(r['path'])==r['sha256']
        if r.get('exact'):
            ref=by[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    T.N.P.frozen=T.N.L.frozen;assert T.N.P.frozen is T.N.L.frozen
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    spec=importlib.util.spec_from_file_location('_t681_helper_preflight',HERE/'affinity_guard.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.OUT==OUT and helper.CG.name=='stage-t681.service'
    resources=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',both_actual_frozen_loaders_and_causal_location_identity=True,
        all16source_full32fresh_exact_provenance=True,exhaustive_two_positive_source_models=True,discovery4_transfer20_disjoint=True,
        no_consumed_task_rerun=True,no_parameter_change=True,canonical_worker_loader=True,source_known_old_failure=True,
        full407=None,resource_helper_configuration=True,new_native=0,new_optimizer=0,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T681_PRIMARY_REVIEW.md'}
    for folder in [src,last]:deps|={folder/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json']}
    deps|={src/'summary.json',src/'MODELS.json',src/'cases.jsonl.gz',last/'RESULTS.json'}
    baseline_folder=ROOT/'stage/runs/t502_fullfield_probe/probe'
    deps|={baseline_folder/f for f in ['cases.jsonl','summary.json','audit.json']}
    deps|={Path(v['source_path']) for v in baseline.values()}
    deps|={Path(m['model']['path']) for m in selected}|{Path(r['path']) for r in cases}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start+preceding_preflight+3
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_native_stop','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()
