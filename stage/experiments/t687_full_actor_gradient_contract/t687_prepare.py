"""Real complete-history BPTT entry preflight; zero optimizer/native calls."""
from pathlib import Path
import ast,json,os,resource,sys,time
import t687_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
read,sha=S.read,S.sha
from status_io import save

def main():
    start=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists();DOC.mkdir(exist_ok=True)
    last=ROOT/'stage/runs/t686_direct_stochastic_validation';rep=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert rep['state']=='COMPLETE' and rep['audit']['state']=='PASS'
    ids=seal['scientific_pids']+[rep['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    science_path=ROOT/'stage/runs/t678_reference_target_pilot/protocol.json';science=read(science_path)
    auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p=dict(id='T687_FULL_ORIGINAL_ACTOR_BPTT_CONTRACT',authorization_approved=False,out=str(OUT),
        weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),science_protocol=str(science_path),initial=science['initial'],
        KL_panel_ids=science['gradient_panel_ids'],numeric_epsilon=1e-6,numeric_atol=1e-8,numeric_rtol=1e-4,
        new_Actor=0,new_Value=0,new_optimizer=0,max_physics=0,max_commands=0,
        main_seconds=2400,audit_seconds=2400,max_bytes=64*2**20,child_bytes=64*2**20,global_bytes=10*2**30,
        absolute_learning_stop=min(now+2400,auth['absolute_learning_stop']),absolute_native_stop=min(now+2400,auth['absolute_native_stop']),
        absolute_all_stop=min(now+6000,auth['absolute_all_stop']),finalization_seconds=60)
    assert len(p['KL_panel_ids'])==68
    resources=S.Wait.Raw.resources()
    assert S.T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    assert sha(p['initial']['path'])==p['initial']['sha256']
    # Any unexpected optimizer or native call is a preflight failure, not a
    # discarded learning update. These entry points are never needed here.
    def forbidden(*args,**kwargs):raise AssertionError('T687 forbids optimizer/native calls')
    S.torch.optim.Adam.step=forbidden
    if 'mujoco' in sys.modules:sys.modules['mujoco'].mj_step=forbidden
    phase=[]
    def guard(name,key):
        assert time.time()<start+240
        S.Wait.Raw.resources();phase.append(name)
    result,_=S.compute(p,guard,short=True)
    elapsed=time.time()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<auth['memory_bytes_cap'] and elapsed<240
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',complete4_history_plus2_origin_reference_BPTT=True,
        numeric=result['numeric'],real_full64772_loader=True,adapter_dimension=11906,
        sequence_step_identity_and_exact_state_restore=True,original68_ID_KL_panel_unchanged=True,
        old_T686_all_processes_gone=True,new_native=0,new_optimizer=0,
        elapsed_seconds=elapsed,peak_rss_bytes=peak,resources=resources,
        full1724_gradient_and_adapter_slice_identity_pending_registered_main=True)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in science['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T687_PRIMARY_REVIEW.md',science_path}
    deps|={last/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json','summary.json','DIRECT_RESULTS.json']}
    deps|={Path(science[k]) for k in ['outcome_gradients','prior_target_gradients','reference_contract']}
    deps|=set(map(Path,science['summaries']))
    deps|={Path(r['path']) for rr in S.B.dataset(science).values() for r in rr}
    deps|={Path(r['source_path']) for r in read(science['reference_contract'])['reference_rows']}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre),flush=True)

if __name__=='__main__':main()
