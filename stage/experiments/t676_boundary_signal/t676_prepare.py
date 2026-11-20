"""Real cost/group/likelihood seam and boundary applicability fixtures."""
from pathlib import Path
import ast,copy,sys,time,json
import t676_analysis as C
from status_io import save
ROOT,HERE,OUT,DOC=C.ROOT,C.HERE,C.OUT,C.DOC

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    src=ROOT/'stage/runs/t675_progress_measurement';prior=C.read(src/'protocol.json')
    assert C.read(src/'FINAL_REPORT.json')['state']=='COMPLETE' and C.read(src/'audit.json')['state']=='PASS'
    folders=[str(src),*prior['sealed_sources']]
    for f in folders:
        seal=C.read(Path(f)/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
        assert not Path('/proc',str(C.read(Path(f)/'DRIVER_STARTED.json')['pid'])).exists()
    old=ROOT/'stage/runs/t673_all_task_signal_gate';olds=C.read(old/'protocol.json');auth=C.read(DOC.parent/'AUTHORIZATION.json');assert auth['approved'];now=time.time()
    p=dict(id='T676_RECORDED_BOUNDARY_SEVERITY_SIGNAL',authorization_approved=False,out=str(OUT),summaries=prior['summaries'],initial=prior['initial'],sealed_sources=folders,
        science_protocol=olds['science_protocol'],protected384=olds['protected384'],outcome_result=str(old/'RESULTS.json'),outcome_gradients=str(old/'gradients.npz'),
        exact_cases=[str(Path(f).with_name('cases.jsonl.gz')) for f in prior['summaries'] if Path(f).parent.name.startswith('field_')],
        numeric_ids=['t502_field_sector0_offset2','t502_field_sector1_offset1','t502_field_sector7_offset1'],variation_floor=1e-12,
        finite_parameter_epsilon=1e-6,numeric_atol=1e-8,numeric_rtol=1e-4,main_seconds=900,audit_seconds=900,
        absolute_all_stop=min(now+2400,auth['absolute_all_stop']),max_bytes=16*2**20,global_bytes=10*2**30,
        max_physics=0,max_commands=0,new_Actor=0,new_Value=0,new_optimizer=0)
    row=dict(field_boundary=True,request={'job':{'fullfield_start_pose':[6.8,0.,0.]}},sha256='fixture')
    a=dict(poses=C.np.array([[6.8,0.,0.],[7.7,0.,0.]]),next_poses=C.np.array([[7.7,0.,0.],[7.9,0.,0.]]))
    item=C.boundary(row,a);assert abs(item['cost']-.1)<1e-14 and item['first_recorded_crossing']==1
    exempt=copy.deepcopy(row);exempt['request']['job']={};assert C.boundary(exempt,a)['cost']==0
    terminal=copy.deepcopy(row);terminal['field_boundary']=False
    b=dict(poses=C.np.array([[6.8,0.,0.]]),next_poses=C.np.array([[7.2,0.,0.]]));assert C.boundary(terminal,b)['terminal_extra_crossing']
    groups=C.B.dataset(p);science=C.read(p['science_protocol']);science.update(out=str(OUT),arm='t676');C.T.setup(science)
    C.torch.set_num_threads(1);C.torch.use_deterministic_algorithms(True);q=C.M.L.frozen(science,p['initial']['path']);assert not q.core.batch_first
    g,costs,values,ys=C.cost_group(q,groups[p['numeric_ids'][0]],science);assert len(g)==8 and C.np.isfinite(costs).all()
    assert all(abs(x['adv']-float(-costs[i]+(costs.sum()-costs[i])/7))<1e-15 for i,x in enumerate(g))
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',field_applies_nonfield_exempt_fixture=True,recorded_vs_final_nextpose_distinction=True,
             actual_full_history_cost_density_and_LOO=True,resources=C.T.Wait.Raw.resources(),new_native=0,new_optimizer=0);save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T676_PRIMARY_REVIEW.md',src/'protocol.json',src/'RESULTS.json',src/'audit.json',src/'FINAL_REPORT.json',src/'SEALED.json',src/'DRIVER_STARTED.json',old/'RESULTS.json',old/'gradients.npz',ROOT/'stage/experiments/t645_iterative_recovery_relay/t645_runtime.py'}|{Path(f) for f in p['exact_cases']}
    import gzip
    for filename in p['exact_cases']:
        for row in [json.loads(line) for line in gzip.decompress(Path(filename).read_bytes()).decode().splitlines()]:
            if row.get('exact'):assert C.sha(row['path'])==row['sha256'];deps.add(Path(row['path']))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):C.sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21;save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'});print(json.dumps(pre))

if __name__=='__main__':main()
