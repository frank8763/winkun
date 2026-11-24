"""Bounded zero-physics witness of both real worker bindings and manifest joins."""
from pathlib import Path
import importlib.util,time,unittest.mock as mock
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t715_admission_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
def main():
    start=time.time();C.old_stopped();p=C.read(C.ROOT/'stage/runs/t712_retention_tradeoff/protocol.json');p['stage_cpus']=list(range(8,16));checks=[]
    original=C.os.sched_setaffinity
    def witness(pipe,index,pc,counts):
        assert C.M.N.N.P.frozen is C.A.frozen
        assert C.M.N.N.G.T.Session is (C.Geo.Session if pc.get('layout_test') else C.M.N.N.C.Session)
        checks.append(dict(layout=bool(pc.get('layout_test')),worker=index))
    try:
        with mock.patch.object(C.M.N.N.G.T,'worker',witness):
            for layout in [False,True]:C.worker(None,0,dict(p,layout_test=layout),None)
    finally:C.os.sched_setaffinity=original
    test=C.read(C.OUT/'prepared_test.json');contracts={r['id']:r for r in test['new_initial_contracts']}
    assert len(test['new_jobs'])==72 and len(test['new_exact_jobs'])==24
    for job in test['new_jobs']+test['new_exact_jobs']:assert job['layout_initial_contract']==contracts[job['id']]
    source=C.read(C.ROOT/'stage/docs/day_20261124/t714/SOURCE_SELECTION.json');acq=C.read(C.DOC/'ACQUISITION.json')
    assert {r['job_id'] for r in source['rows']}=={j['id'] for j in acq['jobs']} and len(acq['exact_jobs'])==18
    fs=importlib.util.spec_from_file_location('t715_loss_check',HERE/'fit.py');F=importlib.util.module_from_spec(fs);fs.loader.exec_module(F);t=F.torch
    x=t.tensor([-2.,-.5,.5,2.],dtype=t.float64,requires_grad=True)
    loss=F.interval_error(x,t.ones(4,dtype=t.float64));loss.sum().backward();assert t.equal(loss,t.tensor([9.,2.25,.25,0.],dtype=t.float64)) and t.equal(x.grad,t.tensor([-6.,-3.,-1.,0.],dtype=t.float64))
    x=t.tensor([-2.,-.5,.5,2.],dtype=t.float64,requires_grad=True)
    loss=F.interval_error(x,-t.ones(4,dtype=t.float64));loss.sum().backward();assert t.equal(loss,t.tensor([0.,.25,2.25,9.],dtype=t.float64)) and t.equal(x.grad,t.tensor([0.,1.,3.,6.],dtype=t.float64))
    x=t.tensor([-2.],dtype=t.float64,requires_grad=True);((x.clamp(-1,1)-1)**2).sum().backward();assert x.grad.item()==0,'Literal clipping red witness changed'
    assert F.interval_error(t.tensor([2.]),t.tensor([1.])).item()==0
    x=t.tensor([.3],dtype=t.float64,requires_grad=True);F.interval_error(x,t.tensor([.1],dtype=t.float64)).sum().backward();assert abs(x.grad.item()-.4)<1e-12
    x=t.tensor([1-2**-25],dtype=t.float64);assert x.float().item()==1 and F.interval_error(x,t.tensor([1.],dtype=t.float64)).item()>0
    C.save(C.DOC/'LOSS_PREFLIGHT.json',dict(state='PASS',literal_clamp_wrong_side_gradient_zero_RED=True,interval_wrong_side_gradient_nonzero_GREEN=True,saturated_correct_side_loss_gradient_zero=True,unsaturated_MSE_gradient=True,float32_inside_boundary_conservative_surrogate_witness=True,new_model_updates=0,new_native=0))
    C.save(C.DOC/'ADMISSION_CHECK.json',dict(state='PASS',worker_bindings=checks,test_contracts=72,exact=24,acquisition_jobs=18,learning=0,physics=0,elapsed_seconds=time.time()-start))
    print('T715_ADMISSION_PASS',flush=True)
if __name__=='__main__':main()
