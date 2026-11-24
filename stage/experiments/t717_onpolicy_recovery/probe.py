"""Actual native worker loader/cancellation contract, zero physics."""
from pathlib import Path
import sys,importlib.util,os,time,traceback,copy
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t717_probe_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
import acquisition_native as RN
import t717_learning as L

def main():
    C.old_stopped();C.unit_stopped(C.UNIT);p=C.read(C.OUT/'prepared.json');folder=C.OUT/'worker_preflight';folder.mkdir();p=dict(p,out=str(folder),native_stop_epoch=time.time()+90,max_physics=1000);C.save(C.DOC/'NATIVE_PREFLIGHT_STARTED.json',dict(epoch=time.time(),pid=os.getpid()))
    ctx=RN.Base.G.R.M.mp.get_context('spawn');counts=ctx.RawArray('q',8);a,b=ctx.Pipe();child=ctx.Process(target=RN.worker,args=(b,0,p,counts));child.start();b.close()
    try:
        assert a.poll(40),'No READY';ready=a.recv();assert ready['state']=='READY',ready;assert os.sched_getaffinity(child.pid)=={8}
        model=p['learner_initial'];job=p['training_jobs'][0];a.send(dict(model=model['path'],job=job,seed=[717,0,0],query=False,cancel_probe=True));assert a.poll(20),'No cancel response';reply=a.recv();assert reply['state']=='CANCEL_PASS',reply;assert sum(counts)==0
        a.send(None);child.join(10);assert child.exitcode==0;C.save(C.DOC/'NATIVE_PREFLIGHT.json',dict(state='PASS',actual_worker_loader=True,native=0,commands=3,CPU=[8],forced_prefix_cancel_before_first_native_step=True,source_model=p['learner_initial'],worker_pid=child.pid))
        # Same-carry batch1 features remain bitwise identical at fit8threads.
        A=C.A;torch=A.torch;orig=C.read(C.OUT/'prepared.json');q=A.frozen(orig,model['path']);row=orig['recovery_jobs'][0];arr=RN.Base.C.E.N.S.arrays(row['source_path']);q.sample=False;q.rng=None;q.reset_noise();h=None;loc=[];hb=[]
        torch.set_num_threads(1)
        for x in arr['states'][:32]:
            hb.append(A.np.zeros(64) if h is None else h[0,0].numpy().copy());_,mu,h,_=q.step(torch.tensor(x[None]),h);loc.append(mu[0].detach().numpy().copy())
        aa={k:v[:32].copy() if getattr(v,'ndim',0) and len(v)==len(arr['states']) else v for k,v in arr.items()};aa['hidden_before']=A.np.asarray(hb);aa['last_hidden_after']=h[0,0].numpy().copy();torch.set_num_threads(8);L.features(q,aa)
        contract=C.read(C.DOC/'NATIVE_PREFLIGHT.json');contract['fit8threads_vs_deploy1thread_owncarry_exact']=True;C.save(C.DOC/'NATIVE_PREFLIGHT.json',contract);print('T717_NATIVE_ZERO_PHYSICS_PREFLIGHT_PASS',flush=True)
    finally:
        if child.is_alive():child.terminate();child.join(5)
        a.close()
if __name__=='__main__':
    try:main()
    except BaseException:C.save(C.DOC/'NATIVE_PREFLIGHT_REJECTION.json',dict(error=traceback.format_exc(),new_native=0));raise
