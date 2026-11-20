"""Lock full407 data provenance and actual diagnostic entry; zero native/optimizer."""
from pathlib import Path
import ast,json,os,sys,time
import t673_analysis as B
from status_io import save
ROOT,HERE,OUT,DOC=B.ROOT,B.HERE,B.OUT,B.DOC

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    src=ROOT/'stage/runs/t672_full_suite_onpolicy';source=B.read(src/'protocol.json');report=B.read(src/'FINAL_REPORT.json')
    assert report['state']=='COMPLETE' and report['completed_batches']==8
    seal=B.read(src/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
    driver=B.read(src/'DRIVER_STARTED.json')['pid'];assert not Path('/proc',str(driver)).exists()
    summaries=[str(src/c['name']/'summary.json') for c in source['batch_configurations']]
    for f in summaries:assert B.read(Path(f).with_name('audit.json'))['state']=='PASS'
    auth=B.read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p=dict(id='T673_ALL407_SIGNAL_AND_DISPLACEMENT_DIAGNOSTIC',authorization_approved=False,out=str(OUT),source_run=str(src),source_driver_pid=driver,
        science_protocol=str(src/'protocol.json'),summaries=summaries,initial=source['initial'],protected384=source['before_success_ids'],
        main_seconds=1800,audit_seconds=1800,absolute_all_stop=min(now+4500,auth['absolute_all_stop']),max_bytes=64*2**20,global_bytes=10*2**30,
        numeric_ids=['dev_task6_13','dev_task6_25','dev_task7_17','dev_task6_21','t502_field_sector0_offset1','t502_field_sector7_offset2'],
        finite_parameter_epsilon=1e-6,numeric_atol=1e-8,numeric_rtol=1e-4,max_physics=0,max_commands=0,new_Actor=0,new_Value=0,new_optimizer=0)
    assert len(p['protected384'])==384 and len(set(p['protected384']))==384
    groups=B.dataset(p);science=dict(source,out=str(OUT),arm='t673');B.T.setup(science)
    B.torch.set_num_threads(1);B.torch.use_deterministic_algorithms(True);q=B.M.L.frozen(science,p['initial']['path'])
    assert q.phash()==p['initial']['policy_hash'] and source['raw_sigma']==[.05,.05]
    # Exercise the actual group loader and full sequence/carry/density path.
    group,y=B.A.group(q,groups['dev_task6_13'],science);assert y==[-1.,-1.,-1.,1.]
    scores=B.np.array([[1.,2.],[-3.,4.],[5.,-6.],[7.,8.]])
    grad=B.combination(scores,y);adv=B.np.array(y)-(sum(y)-B.np.array(y))/3
    assert B.np.allclose(grad,(scores*adv[:,None]).mean(0))
    assert B.np.array_equal(B.combination(scores,[1,1,1,1]),B.np.zeros(2))
    sigma=B.torch.tensor([.05,.1],dtype=B.torch.float64);delta=B.torch.tensor([.05,.1],dtype=B.torch.float64)
    assert float(B.conditional_KL(delta,B.torch.zeros_like(delta),sigma.tolist()))==1.
    assert float(.5*((delta/.2)**2).sum())!=1.
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    resources=B.T.Wait.Raw.resources()
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_all407_dataset_shape=True,actual_mixed_task_loader_own_carry_and_density=True,
             analytic_LOO_and_all_same_zero_fixture=True,heterogeneous_actual_sigma_KL_fixture=True,resources=resources,new_native=0,new_optimizer=0,elapsed_seconds=time.time()-start)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in source['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'AUTHORIZATION.json',DOC.parent/'T673_PRIMARY_REVIEW.md',src/'protocol.json',src/'SEALED.json',src/'DRIVER_STARTED.json',src/'FINAL_REPORT.json'}
    for f in summaries:
        deps|={Path(f),Path(f).with_name('audit.json')}
    for rows in groups.values():
        for row in rows:assert B.sha(row['path'])==row['sha256'];deps.add(Path(row['path']))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):B.sha(f) for f in sorted(deps)}
    p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'});print(json.dumps(pre))

if __name__=='__main__':main()
