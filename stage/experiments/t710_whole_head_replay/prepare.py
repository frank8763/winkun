"""No scientific fit/integration: real loader/helper/inspector and synthetic full-head optimizer."""
from pathlib import Path
import importlib.util,json,os,sys,time,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('t710_preflight_controller',HERE/'run.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A=C.A;D=C.DOC;O=C.OUT

def main():
    start=time.time();C.old_stopped();assert not O.exists();O.mkdir();(O/'fit').mkdir();A.torch.set_num_threads(1);os.sched_setaffinity(0,{15})
    import mujoco
    def forbidden(*a,**kw):raise AssertionError('T710 preparation forbids physics/optimizer')
    original_lbfgs=A.torch.optim.LBFGS.step
    mujoco.mj_step=forbidden;A.torch.optim.Adam.step=forbidden;A.torch.optim.LBFGS.step=forbidden
    p=C.read(ROOT/'stage/runs/t704_frozen_new_starts/protocol.json');ref=C.read(ROOT/'stage/runs/t701_compact_retention/fit/summary.json')['finals']['replay_refined108']['model']
    q=A.frozen(p,ref['path']);r=A.frozen(p,ref['path']);assert C.sha(ref['path'])==ref['sha256'] and q.phash()==ref['policy_hash']
    arrays=A.Old.Existing.L.T.R.E.N.S.arrays(p['protected403_rows'][0]['path']);h=hh=None;q.sample=r.sample=False;q.rng=r.rng=None;q.reset_noise();r.reset_noise()
    with A.torch.no_grad():
        for row in arrays['states'][:8]:
            x=A.torch.tensor(row[None]);cmd,_,h,_=q.step(x,h);other,_,hh,_=r.step(x,hh)
            assert A.torch.equal(h,hh) and A.Old.Existing.same_float32(cmd.numpy(),other.numpy())
    import unittest.mock as mock
    seen=[]
    def witness(pipe,index,pc,counts):assert C.M.N.N.P.frozen is A.frozen;seen.append(index)
    with mock.patch.object(C.M.N.N.G.T,'worker',witness):C.worker(None,0,dict(p,stage_cpus=list(range(8,16))),None)
    assert seen==[0]
    row=dict(job_id=p['protected403_rows'][0]['id'],request=dict(model=p['initial']['path']),seed=None,teacher=False,physics=int(arrays['native_counts'].sum()),array_hashes={k:A.digest(v) for k,v in arrays.items()})
    C.M.N.BASE_INSPECT(p,arrays,row)
    try:C.M.N.BASE_INSPECT(p,arrays,dict(row,array_hashes=dict(row['array_hashes'],actions='0'*64)))
    except AssertionError:pass
    else:raise AssertionError('Native inspector accepted corrupt action hash')
    # Exercise full-head optimizer, gradients, fixed chunks and rollback on toy data.
    torch=A.torch;torch.manual_seed(20261123710)
    head=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Linear(4,2)).double()
    x=torch.tensor([[.1,.3,-.5],[.2,-.4,.7]],dtype=torch.float64);y=torch.zeros((2,2),dtype=torch.float64)
    opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,max_eval=25,history_size=20,line_search_fn='strong_wolfe')
    calls=[];before={k:v.clone() for k,v in head.state_dict().items()}
    def closure():
        opt.zero_grad(set_to_none=True);loss=(head(x)-y).square().mean();loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in head.parameters());calls.append(float(loss));return loss
    for _ in range(3):original_lbfgs(opt,closure)
    assert calls[-1]<=calls[0] and any(not torch.equal(head.state_dict()[k],v) for k,v in before.items())
    saved={k:v.clone() for k,v in head.state_dict().items()};op=__import__('copy').deepcopy(opt.state_dict())
    class Budget(Exception):pass
    try:
        def interrupted():
            with torch.no_grad():next(head.parameters()).add_(.1)
            raise Budget()
        original_lbfgs(opt,interrupted)
    except Budget:head.load_state_dict(saved);opt.load_state_dict(op)
    assert all(torch.equal(head.state_dict()[k],v) for k,v in saved.items())
    assert not torch.isfinite(head(torch.full((1,3),float('nan'),dtype=torch.float64))).all()
    test=O/'helper_preflight';test.mkdir();(D/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None))
    command=['systemd-run','--user','--unit=stage-t710-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight']
    v=subprocess.run(command,capture_output=True,text=True);assert v.returncode==0,v.stderr;assert C.read(D/'helper_preflight/AFFINITY_GUARD.json')['state']=='ENDED'
    worst=2*((224+69)*p['fresh_bytes']+(431+28)*p['compact_bytes']+64*2**20)+64*2**20
    assert worst<3*2**30 and C.used(ROOT/'stage/runs')+3*2**30<8*2**30
    assert time.time()-start<=180 and time.time()<C.read(D/'PREPARATION_BOUND.json')['absolute_stop']
    C.save(D/'PREFLIGHT.json',dict(state='PASS',actual_reference_loader_owncarry_batch1_exact=True,worker_loader_binding=True,full_inspector_corruption_RED=True,actual_system_python_helper_cgroup=True,
        synthetic_fullhead_LBFGS_gradient_loss_and_interrupt_rollback_PASS=True,synthetic_nonfinite_RED=True,synthetic_optimizer_attempts=4,synthetic_gradient_evaluations=len(calls),no_actual_dataset_or_model_optimization=True,new_Actor=0,new_optimizer=0,new_native=0,
        storage_worst_bytes=worst,elapsed_seconds=time.time()-start,resource=C.C.resources()))
    print('T710_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        C.save(D/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),new_native=0,new_Actor=0));raise
