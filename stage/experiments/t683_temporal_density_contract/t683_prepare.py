"""Small real-Actor prefix, no native worker and no optimizer construction."""
from pathlib import Path
import ast,copy,json,sys,time
import t683_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
read,sha=S.read,S.sha
from status_io import save

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    last=ROOT/'stage/runs/t682_head_factor_ablation';report=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert report['state']=='COMPLETE' and report['audit']['state']=='PASS' and report['gate']=='SCIENTIFIC_GATE_FAILED'
    ids=seal['scientific_pids']+[report['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']]
    assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in ids)
    science=read(last/'protocol.json');hist=read(ROOT/'stage/runs/t679_episode_parameter_exploration/protocol.json')['calibration_rows']
    assert len(hist)==8
    auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p=dict(id='T683_TEMPORAL_LATENT_DENSITY_CONTRACT',authorization_approved=False,out=str(OUT),weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),
        science_protocol=str(last/'protocol.json'),initial=science['initial'],history_rows=hist,new_Actor=0,new_Value=0,new_optimizer=0,
        max_physics=0,max_commands=0,max_bytes=32*2**20,child_bytes=32*2**20,global_bytes=10*2**30,
        main_seconds=600,audit_seconds=600,absolute_learning_stop=min(now+600,auth['absolute_learning_stop']),
        absolute_native_stop=min(now+600,auth['absolute_native_stop']),absolute_all_stop=min(now+1800,auth['absolute_all_stop']),
        finalization_seconds=60,numeric_epsilon=1e-6,numeric_atol=1e-8,numeric_rtol=1e-4)
    S.torch.set_num_threads(1);result,arrays=S.compute(p,True)
    assert result['state']=='TEMPORAL_LATENT_CONTRACT_PASS',result
    resources=S.Wait.Raw.resources();assert S.T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_Actor32step_prefix_IID_identity=True,
        temporal_sampler_reset_causality_and_raw_joint=True,numeric=result['numeric'],fixtures=result['fixtures'],
        no_native_or_optimizer=True,full_history_test_pending_registered_main=True,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in science['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T683_PRIMARY_REVIEW.md'}
    deps|={last/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json','FACTOR_RESULTS.json']}
    deps|={Path(r['path']) for r in hist}|{Path(p['initial']['path']),ROOT/'stage/runs/t679_episode_parameter_exploration/protocol.json'}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start+12
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()
