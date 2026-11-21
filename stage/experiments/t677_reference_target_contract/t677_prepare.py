"""Frozen-target zero point and real references before finite registration."""
from pathlib import Path
import ast,sys,time,json
import t677_analysis as C
from status_io import save
ROOT,HERE,OUT,DOC=C.ROOT,C.HERE,C.OUT,C.DOC

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    src=ROOT/'stage/runs/t676_boundary_signal';prior=C.read(src/'protocol.json');result=C.read(src/'RESULTS.json')
    assert C.read(src/'FINAL_REPORT.json')['state']=='COMPLETE' and C.read(src/'audit.json')['state']=='PASS' and result['state']=='SCIENTIFIC_GATE_FAILED'
    folders=[str(src),*prior['sealed_sources']]
    for f in folders:
        s=C.read(Path(f)/'SEALED.json');assert s['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])
        assert not Path('/proc',str(C.read(Path(f)/'DRIVER_STARTED.json')['pid'])).exists()
    auth=C.read(DOC.parent/'AUTHORIZATION.json');assert auth['approved'];now=time.time()
    p=dict(id='T677_REFERENCE_TARGET_ZERO_POINT_AND_DIRECTION',authorization_approved=False,out=str(OUT),summaries=prior['summaries'],initial=prior['initial'],sealed_sources=folders,
        science_protocol=prior['science_protocol'],protected384=prior['protected384'],outcome_result=prior['outcome_result'],outcome_gradients=prior['outcome_gradients'],
        finite_parameter_epsilon=1e-6,numeric_atol=1e-9,numeric_rtol=1e-4,main_seconds=600,audit_seconds=600,
        absolute_all_stop=min(now+1800,auth['absolute_all_stop']),max_bytes=32*2**20,global_bytes=10*2**30,
        max_physics=0,max_commands=0,new_Actor=0,new_Value=0,new_optimizer=0)
    science=C.read(p['science_protocol']);science.update(out=str(OUT),arm='t677');C.T.setup(science)
    C.torch.set_num_threads(1);C.torch.use_deterministic_algorithms(True)
    q=C.M.L.frozen(science,p['initial']['path']);q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    refs=C.M.references(science,q,lambda:None);own=C.own_targets(q,refs);variables=[v for v in q.parameters() if v.requires_grad]
    zero,grad=C.loss_gradient(q,variables,own);assert zero<1e-20 and C.np.linalg.norm(grad)<1e-12
    contract=C.read(science['reference_contract']);origins=[r['origin'] for r in contract['reference_rows']]
    assert len(refs)==387 and origins.count('T501_original_success')==384 and origins.count('T653_mean_success')==3
    assert all(not g['target'].requires_grad for g in own)
    sample=C.torch.tensor([[-1.2,-.4],[.3,1.2]],dtype=C.torch.float64,requires_grad=True)
    target=sample.detach().clamp(-1,1);term=C.M.P.action_loss(sample,target).sum();gg=C.torch.autograd.grad(term,sample)[0]
    assert float(term)==0 and not gg.any()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all387_real_reference_self_target_zero=True,loss=zero,gradient_norm=float(C.np.linalg.norm(grad)),
        same_action_loss_hinge_and_detached_target=True,reference_provenance384_plus3=True,resources=C.T.Wait.Raw.resources(),new_native=0,new_optimizer=0)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T677_PRIMARY_REVIEW.md',src/'protocol.json',src/'RESULTS.json',src/'audit.json',src/'FINAL_REPORT.json',src/'SEALED.json',src/'DRIVER_STARTED.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):C.sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21;save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'});print(json.dumps(pre))

if __name__=='__main__':main()
