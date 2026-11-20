"""Provenance, LOO and active-set fixtures, zero optimizer calls."""
from pathlib import Path
import ast,json,os,sys,time
import t670_analysis as A
from status_io import save
ROOT,HERE,OUT,DOC=A.ROOT,A.HERE,A.OUT,A.DOC
read,sha=A.read,A.sha


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    src=ROOT/'stage/runs/t669_old_task_onpolicy_signal';source=read(src/'protocol.json');s=read(src/'summary.json')
    assert read(src/'audit.json')['state']=='PASS' and read(src/'audit.json')['mixed_collision_tasks']==4
    assert all(not Path('/proc',str(pid)).exists() for pid in read(src/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(src/'DRIVER_STARTED.json')['pid'])).exists()
    fresh=ROOT/'stage/runs/t663_daytime_factorial_relay/ppo_binary_s005';record=read(fresh/'ROUNDS.json')[0]
    model=read(src/'INITIAL_MODEL.json');assert model['policy_hash']==record['training_rows'][0]['policy_hash']
    casep=fresh/'cases.jsonl.gz'
    import gzip
    cases=[json.loads(x) for x in gzip.decompress(casep.read_bytes()).decode().splitlines()]
    exact={r['reference_path']:r for r in cases if r.get('exact')}
    assert len(record['training_rows'])==64
    for row in record['training_rows']:
        assert row['path'] in exact and row['array_hashes']==exact[row['path']]['array_hashes']
        assert row['integration_sha256']==exact[row['path']]['integration_sha256']
        assert sha(row['path'])==row['sha256'] and row['policy_hash']==model['policy_hash']
    auth=read(DOC.parent/'AUTHORIZATION.json');assert auth['approved']
    now=time.time();p=dict(id='T670_OLD_OUTCOME_GRADIENT_FEASIBILITY',authorization_approved=False,out=str(OUT),
        source_protocol=str(src/'protocol.json'),old_summary=str(src/'summary.json'),fresh_rounds=str(fresh/'ROUNDS.json'),
        initial=model,old_ids=[x['id'] for x in s['panel']],sealed_sources=[str(src),str(ROOT/'stage/runs/t665_remaining_verification')],
        main_seconds=600,audit_seconds=600,absolute_all_stop=min(now+1800,auth['absolute_all_stop']),
        max_bytes=16*2**20,global_bytes=10*2**30,Adam_eps=1e-8,finite_parameter_epsilon=1e-6,
        numeric_atol=1e-5,numeric_rtol=1e-3,max_physics=0,max_commands=0,new_Actor=0,new_Value=0,new_optimizer=0)
    os.sched_setaffinity(0,{14});A.torch.set_num_threads(1);res=A.T.Wait.Raw.resources()
    # Genuine active-set fixture: raw update violates two independent old
    # constraints, while a third unconstrained coordinate retains ascent.
    arr=A.np.array([[1.,0.,0.],[0.,1.,0.]])
    d,qp=A.project(A.np.array([-1.,-2.,3.]),arr)
    assert A.np.array_equal(d,A.np.array([0.,0.,3.])) and qp['active']==[0,1]
    good=A.np.array([1.,2.,3.]);d2,_=A.project(good,arr);assert A.np.array_equal(d2,good)
    bad,qp2=A.project(A.np.array([-1.,2.]),A.np.array([[1.,0.],[-1.,0.]]));assert abs(bad[0])<1e-12 and bad[1]==2.
    ys=A.np.array([-1.,1.,0.,1.]);ad=ys-(ys.sum()-ys)/(len(ys)-1);assert abs(ad.mean())<1e-15
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',four_old_mixed_tasks=True,same_policy_sigma_prior64_and_exact_verified=True,
        active_set_KKT_and_feasible_zero_fixture=True,LOO_finite_fixture=True,new_native=0,new_optimizer=0,resources=res)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in source['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'AUTHORIZATION.json',src/'summary.json',src/'audit.json',src/'protocol.json',src/'INITIAL_MODEL.json',Path(model['path']),fresh/'ROUNDS.json',casep}
    for row in record['training_rows']:deps.add(Path(row['path']))
    for row in s['rows']:deps.add(Path(row['path']))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)}
    save(HERE/'protocol.json',p);save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'})
    print(json.dumps(pre))


if __name__=='__main__':main()
