"""No scientific fit/integration: real loader/helper/inspector and toy QP."""
from pathlib import Path
import importlib.util,json,os,sys,time,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('t705_preflight_controller',HERE/'run.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A=C.A;D=C.DOC;O=C.OUT

def main():
    start=time.time();C.old_stopped();assert not O.exists();O.mkdir();(O/'fit').mkdir();A.torch.set_num_threads(1);os.sched_setaffinity(0,{15})
    import mujoco
    def forbidden(*a,**kw):raise AssertionError('T705 preparation forbids physics/optimizer')
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
    fs=importlib.util.spec_from_file_location('t705_toy_qp',HERE/'fit.py');F=importlib.util.module_from_spec(fs);fs.loader.exec_module(F)
    Z=A.torch.ones((410,1),dtype=A.torch.float64);Y=A.torch.zeros((410,2),dtype=A.torch.float64);Y[403:]=1;groups=A.torch.arange(410);records=[dict(id='synthetic_'+str(i),length=1) for i in range(410)];W0=A.torch.full((1,2),.5,dtype=A.torch.float64)
    results,info=F.optimize(Z,Y,groups,records,W0,lambda:None,lambda *a:None)
    assert info['constrained_final']['source_loss']<=info['source_bound']+1e-12 and info['solves']<=34
    assert (results['constrained']-.5).abs().max()<1e-6
    try:F.solve([A.torch.tensor([[-2.]],dtype=A.torch.float64)]*2,[W0,W0],W0,1e-10,1.)
    except A.torch.linalg.LinAlgError:pass
    else:raise AssertionError('Indefinite synthetic QP not rejected')
    test=O/'helper_preflight';test.mkdir();(D/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None))
    command=['systemd-run','--user','--unit=stage-t705-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight']
    v=subprocess.run(command,capture_output=True,text=True);assert v.returncode==0,v.stderr;assert C.read(D/'helper_preflight/AFFINITY_GUARD.json')['state']=='ENDED'
    worst=2*((224+69)*p['fresh_bytes']+(431+28)*p['compact_bytes']+64*2**20)+64*2**20
    assert worst<3*2**30 and C.used(ROOT/'stage/runs')+3*2**30<8*2**30
    assert time.time()-start<=180 and time.time()<C.read(D/'PREPARATION_BOUND.json')['absolute_stop']
    C.save(D/'PREFLIGHT.json',dict(state='PASS',actual_reference_loader_owncarry_batch1_exact=True,worker_loader_binding=True,full_inspector_corruption_RED=True,actual_system_python_helper_cgroup=True,
        synthetic_convex_feasibility_stationarity_and_indefinite_RED=True,synthetic_solves=info['solves'],no_actual_dataset_or_model_optimization=True,new_Actor=0,new_optimizer=0,new_native=0,
        storage_worst_bytes=worst,elapsed_seconds=time.time()-start,resource=C.C.resources()))
    print('T705_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        C.save(D/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),new_native=0,new_Actor=0));raise
