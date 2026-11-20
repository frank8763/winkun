"""Registered only after projection fixtures, provenance and real loader pass."""
from pathlib import Path
import ast,sys,time,json
import t674_analysis as C
from status_io import save
ROOT,HERE,OUT,DOC=C.ROOT,C.HERE,C.OUT,C.DOC

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    src=ROOT/'stage/runs/t673_all_task_signal_gate';prior=C.read(src/'protocol.json');s=C.read(src/'RESULTS.json')
    assert C.read(src/'FINAL_REPORT.json')['state']=='COMPLETE' and C.read(src/'audit.json')['state']=='PASS'
    for folder in [src,Path(prior['source_run'])]:
        seal=C.read(folder/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
        assert not Path('/proc',str(C.read(folder/'DRIVER_STARTED.json')['pid'])).exists()
    auth=C.read(DOC.parent/'AUTHORIZATION.json');now=time.time();assert auth['approved']
    p=dict(id='T674_44_OUTCOME_DISPLACEMENT_PROJECTION_GATE',authorization_approved=False,out=str(OUT),science_protocol=prior['science_protocol'],summaries=prior['summaries'],
        initial=prior['initial'],protected384=prior['protected384'],prior_result=str(src/'RESULTS.json'),prior_gradients=str(src/'gradients.npz'),
        sealed_sources=[str(src),prior['source_run']],main_seconds=600,audit_seconds=600,absolute_all_stop=min(now+1800,auth['absolute_all_stop']),
        max_bytes=16*2**20,global_bytes=10*2**30,nonlinear_tolerance=1e-12,max_physics=0,max_commands=0,new_Actor=0,new_Value=0,new_optimizer=0)
    assert s['direction_summaries']['all407']['old_negative_ids'] and len(s['direction_summaries']['all407']['old_negative_ids'])==10
    arr=C.np.array([[1.,0.,0.],[0.,1.,0.]]);d,kkt=C.projection(C.np.array([-1.,-2.,3.]),arr)
    assert C.np.allclose(d,[0,0,3],atol=1e-12) and kkt['active']==2
    d,_=C.projection(C.np.array([1.,2.,3.]),arr);assert C.np.array_equal(d,[1.,2.,3.])
    d,_=C.projection(C.np.array([-1.,2.]),C.np.array([[1.,0.],[-1.,0.]]));assert C.np.allclose(d,[0.,2.],atol=1e-12)
    rng=C.np.random.default_rng(674)
    for _ in range(8):
        rows=rng.normal(size=(4,9));target=rng.normal(size=9)
        got,_=C.projection(target,rows);reference,_=C.B.A.project(target,rows)
        assert C.np.max(C.np.abs(got-reference))<1e-9
    groups=C.B.dataset(p);assert len(groups)==407
    science=C.read(p['science_protocol']);science.update(out=str(OUT),arm='t674');C.T.setup(science);C.torch.set_num_threads(1)
    q=C.M.L.frozen(science,p['initial']['path']);C.B.A.group(q,groups['dev_task6_13'],science)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',dual_NNLS_active_inactive_rank_deficient_fixtures=True,real_full_history_loader=True,
             exhaustive_small4_constraint_crosscheck8=True,full407_actual_prior_conflict=True,resources=C.T.Wait.Raw.resources(),new_native=0,new_optimizer=0)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T674_PRIMARY_REVIEW.md',src/'protocol.json',src/'RESULTS.json',src/'gradients.npz',src/'audit.json',src/'FINAL_REPORT.json',src/'SEALED.json',src/'DRIVER_STARTED.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):C.sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'});print(json.dumps(pre))

if __name__=='__main__':main()
