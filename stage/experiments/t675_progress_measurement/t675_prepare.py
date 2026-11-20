"""Actual measurement seam and potential-telescoping fixtures before registration."""
from pathlib import Path
import ast,sys,time,json
import t675_analysis as C
from status_io import save
ROOT,HERE,OUT,DOC=C.ROOT,C.HERE,C.OUT,C.DOC

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    src=ROOT/'stage/runs/t674_projection_feasibility';prior=C.read(src/'protocol.json');res=C.read(src/'RESULTS.json')
    assert C.read(src/'FINAL_REPORT.json')['state']=='COMPLETE' and C.read(src/'audit.json')['state']=='PASS' and res['state']=='SCIENTIFIC_GATE_FAILED'
    assert not res['checks']['field_linear_ascent'] and not res['checks']['all_old_nonlinear_nonnegative']
    folders=[str(src),*prior['sealed_sources']]
    for folder in folders:
        seal=C.read(Path(folder)/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
        assert not Path('/proc',str(C.read(Path(folder)/'DRIVER_STARTED.json')['pid'])).exists()
    auth=C.read(DOC.parent/'AUTHORIZATION.json');assert auth['approved'];now=time.time()
    p=dict(id='T675_RECORDED_RADIAL_PROGRESS_VALIDATION',authorization_approved=False,out=str(OUT),summaries=prior['summaries'],initial=prior['initial'],sealed_sources=folders,
        minimum_progress_range=.01,main_seconds=600,audit_seconds=600,absolute_all_stop=min(now+1800,auth['absolute_all_stop']),max_bytes=16*2**20,global_bytes=10*2**30,
        max_physics=0,max_commands=0,new_Actor=0,new_Value=0,new_optimizer=0)
    row=dict(job_id='fixture',physics=1,remaining_m=2.5,success=False,collision=False,timeout=True,complete=True,administrative_censor=False,request={'job':{}},sha256='fixture')
    a=dict(poses=C.np.array([[0.,0.,0.]]),next_poses=C.np.array([[1.5,2.,0.]]),task_goal_xy=C.np.array([3.,4.]),native_counts=C.np.array([1]),end_flags=C.np.array([[0,0,1]]))
    item=C.measure(row,a);assert item['initial_m']==5. and item['normalized_terminal_progress']==.5
    for mid in [-4.,-2.]:
        phi=C.np.array([-5.,mid,0.]);assert C.np.diff(phi).sum()==5.
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_measurement_function_fixture=True,terminal_zero_potential_telescopes_constant_fixture=True,
             no_new_reward_or_Actor=True,resources=C.B.T.Wait.Raw.resources(),new_native=0,new_optimizer=0);save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T675_PRIMARY_REVIEW.md',src/'protocol.json',src/'RESULTS.json',src/'FINAL_REPORT.json',src/'audit.json',src/'SEALED.json',src/'DRIVER_STARTED.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):C.sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21;save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'});print(json.dumps(pre))

if __name__=='__main__':main()
