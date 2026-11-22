"""Legacy-source audit plus fixed GPU geometry, no fit or physical action."""
from pathlib import Path
import copy,gzip,json,os,resource,sys,time,traceback
import t698_actor as A
import t698_resources as Resources
ROOT=A.ROOT;DOC=ROOT/'stage/docs/weekend_20261120_23/t698';RESULT=DOC/'cpu_allocation_recovery'
INPUTDOC=DOC.parent/'t697'

def main():
    started=time.time();assert not (RESULT/'PREPARATION.json').exists() and not (RESULT/'PREPARATION_REJECTION.json').exists()
    os.sched_setaffinity(0,{14});Resources.resources();Resources.gpu()
    (RESULT/'PREPARATION_STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),epoch=time.time(),absolute_stop=started+900))+'\n')
    def forbidden(*args,**kwargs):raise AssertionError('No native,optimizer,coefficient commit in preparation')
    import mujoco
    mujoco.mj_step=forbidden;A.torch.optim.Adam.step=forbidden;A.fit=forbidden
    A.torch.set_num_threads(1);A.torch.use_deterministic_algorithms(True)
    last=[0.]
    def guard():
        assert time.time()<A.read(DOC/'PREPARATION_STARTED.json')['absolute_stop']-10
        if time.time()-last[0]>5:Resources.resources();Resources.gpu();last[0]=time.time()
    prior=A.read(ROOT/'stage/runs/t696_existing_success_compatibility/protocol.json');p=copy.deepcopy(prior)
    retained=A.read(DOC.parent/'RETAINED_MODEL.json');assert A.sha(retained['model']['path'])==retained['model']['sha256']
    p.update(initial=retained['model'],original_T501=prior['original_T501'],protected410_rows=copy.deepcopy(prior['protected403_rows']))
    with gzip.open(ROOT/'stage/runs/t696_existing_success_compatibility/evaluation/cases.jsonl.gz','rt') as inp:
        by={r['job_id']:r for r in map(json.loads,inp) if r['full'] and not r.get('exact')}
    previous={r['id'] for r in p['protected410_rows']}
    for key in sorted(set(retained['full431_success_ids'])-previous):
        r=by[key];p['protected410_rows'].append(dict(id=key,path=r['path'],sha256=r['sha256'],original384=False))
    assert len(p['protected410_rows'])==410
    restored=A.read(INPUTDOC/'RESTORE_REPORT.json');selection=A.read(INPUTDOC/'SOURCE_INVENTORY.json')
    sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'));import t608_native as Legacy
    source_root=A.read(ROOT/'stage/runs/t626_multitask_closed_loop/protocol.json')
    sourcep=A.read(source_root['actor_configuration']);sourcep.update(source_root)
    original=A.Parent.Parent.ORIGINAL(p,p['original_T501']['path']);parent=A.Parent.frozen(p,p['initial']['path'])
    before=A.digest(parent.state_dict());source_checks=[];novel=[]
    for chosen,restored_row in zip(selection['selected'],restored['rows']):
        guard();row=chosen['row'];assert row['job_id']==restored_row['id']
        path=restored_row['path'];assert A.sha(path)==row['sha256'];a=A.L.T.R.E.N.S.arrays(path)
        assert all(A.digest(v)==row['array_hashes'][k] for k,v in a.items())
        assert not a['ownership'].any() and a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
        assert bool((abs(a['poses'][:,:2])<=7).all())
        assert chosen['row']['request']['job']==next(j for j in prior['evaluation_jobs'] if j['id']==chosen['id'])
        source=Legacy.L.frozen(sourcep,row['request']['model'])
        body_equal=all(A.torch.equal(source.state_dict().get(key,A.torch.empty(0)),value) for key,value in original.state_dict().items())
        prior_loader=Legacy.P.frozen
        try:
            Legacy.P.frozen=Legacy.L.frozen
            Legacy.inspect(sourcep,[(a,row)])
        finally:Legacy.P.frozen=prior_loader
        A.source_trajectory(parent,a,chosen['id'],None,guard)
        source_checks.append(dict(id=chosen['id'],sha256=row['sha256'],source_model_sha256=A.sha(row['request']['model']),
                                  source_body_equals_current=body_equal,original_three_carry_audit=True,candidate_own64_reconstructed_without_sourcehidden=True,decisions=len(a['states'])))
        novel.append(dict(id=chosen['id'],path=path,sha256=row['sha256'],source_model=row['request']['model'],source_audit=chosen['source_audit']))
    p.update(novel_rows=novel,maximum_centers=1024,support_fraction=.49,minimum_feature_separation=1e-10,model_bytes=8*2**20)
    print(json.dumps(dict(stage='source_qualification_PASS',checks=source_checks)),flush=True)
    protected,sources,manifest=A.data(p,guard)
    print(json.dumps(dict(stage='GPU_geometry',protected=len(protected),source=sum(len(x['features']) for x in sources))),flush=True)
    # Old source inspection sets offline CUDA visibility; restore the
    # preregistered geometry device before its first CUDA operation.
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    geometry,info=A.construction(p,protected,sources,guard)
    assert A.digest(parent.state_dict())==before
    result=dict(state='GPU_GEOMETRY_AND_LEGACY_SOURCE410_PROTECTION_PASS_NO_COEFFICIENT_COMMIT',geometry=info,
        source_checks=source_checks,protected410_rows=p['protected410_rows'],novel_rows=novel,manifest=manifest,
        GPU0_actual=True,CPU_deploy_kernel_basis_all410_zero=True,new_native=0,new_Actor=0,new_Value=0,new_optimizer=0,
        elapsed_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (RESULT/'PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['manifest','protected410_rows','novel_rows']}),flush=True)

if __name__=='__main__':
    started=time.time()
    try:main()
    except BaseException:
        (RESULT/'PREPARATION_REJECTION.json').write_text(json.dumps(dict(state='PREPARATION_REJECTED_NO_COEFFICIENT_COMMIT',error=traceback.format_exc(),new_native=0,new_Actor=0,new_optimizer=0,elapsed_seconds=time.time()-started,no_source_or_radius_rescue=True),indent=2)+'\n');raise
    finally:
        f=DOC.parent/'LEDGER.json';l=A.read(f);job=next(j for j in l['jobs'] if j['id']=='T698_CPU_ALLOCATION_RECOVERY');assert job['state']=='PREPARATION_RESERVED_NO_RUN_OR_COEFFICIENT_COMMIT';elapsed=time.time()-started;job.update(state='PREPARATION_PASS' if (RESULT/'PREPARATION.json').exists() else 'PREPARATION_REJECTED',science_seconds=elapsed);l['science_seconds']+=elapsed;f.write_text(json.dumps(l,indent=2)+'\n')
