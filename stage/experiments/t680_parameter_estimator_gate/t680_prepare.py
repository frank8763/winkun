"""Raw antipair estimator and denominator fixtures before source/data lock."""
from pathlib import Path
import ast,sys,time,json
import t680_analysis as A
from status_io import save
ROOT,HERE,OUT,DOC=A.ROOT,A.HERE,A.OUT,A.DOC

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True);src=ROOT/'stage/runs/t679_episode_parameter_exploration'
    prior=A.read(src/'protocol.json');s=A.read(src/'summary.json');seal=A.read(src/'SEALED.json')
    assert A.read(src/'FINAL_REPORT.json')['state']=='COMPLETE' and A.read(src/'audit.json')['state']=='PASS' and s['gate']=='PARAMETER_EXPLORATION_SUPPORT_PASS'
    assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
    assert not Path('/proc',str(A.read(src/'DRIVER_STARTED.json')['pid'])).exists()
    auth=A.read(DOC.parent/'AUTHORIZATION.json');assert auth['approved'];now=time.time()
    p=dict(id='T680_FOUR_PAIR_ESTIMATOR_STABILITY',authorization_approved=False,out=str(OUT),source_run=str(src),source_protocol=str(src/'protocol.json'),summary=str(src/'summary.json'),calibration=str(src/'CALIBRATION.json'),
           main_seconds=600,audit_seconds=600,absolute_all_stop=min(now+1800,auth['absolute_all_stop']),max_bytes=16*2**20,global_bytes=10*2**30,
           new_Actor=0,new_Value=0,new_optimizer=0,max_physics=0,max_commands=0)
    eps=A.np.array([[1.,0.],[0.,1.],[1.,1.],[1.,-1.]]);diff=A.np.array([1.,-1.,0.,2.]);g=A.estimate(diff,eps,.5)
    assert A.np.array_equal(g,A.np.array([.75,-.75])) and A.cosine(g,g)>1-1e-14 and A.cosine(g,A.np.zeros(2)) is None
    mask=A.np.array([False,True,True,True]);assert A.np.allclose(A.estimate(diff[mask],eps[mask],.5),(diff[mask,None]*eps[mask]).sum(0)/3)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',raw_antithetic_formula_and_deleted_denominator_fixture=True,zero_cosine_defined_missing=True,
             source64_and4_antipairs=True,resources=A.C.T.Wait.Raw.resources(),new_native=0,new_optimizer=0);save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T680_PRIMARY_REVIEW.md',src/'protocol.json',src/'summary.json',src/'CALIBRATION.json',src/'audit.json',src/'FINAL_REPORT.json',src/'SEALED.json',src/'DRIVER_STARTED.json'}
    for item in s['models']:deps.add(Path(item['model']['path']))
    for row in s['rows']:deps.add(Path(row['path']))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):A.sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21;save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'});print(json.dumps(pre))

if __name__=='__main__':main()
