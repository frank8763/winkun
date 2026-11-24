"""Bounded geometry qualification and real loader/worker/audit preflight."""
from pathlib import Path
import copy, importlib.util, json, os, subprocess, sys, time, traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
s=importlib.util.spec_from_file_location('t712_prepare_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
A,torch,O,D=C.A,C.A.torch,C.OUT,C.DOC

def main():
    start=time.time();end=C.read(D/'PREPARATION_BOUND.json')['absolute_stop'];C.old_stopped();C.C.resources();assert not O.exists();O.mkdir();(O/'fit').mkdir()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,{15})
    import mujoco
    def forbidden(*args,**kw):raise AssertionError('Preparation forbids native physics and actual optimization')
    original=torch.optim.LBFGS.step;mujoco.mj_step=forbidden;torch.optim.Adam.step=forbidden;torch.optim.LBFGS.step=forbidden
    def guard():assert time.time()<end-10;C.C.resources()
    p=C.read(ROOT/'stage/runs/t710_whole_head_replay/protocol.json')
    ref=C.read(ROOT/'stage/runs/t703_projected_local/fit/summary.json')['finals']['proj8']['model'];assert C.sha(ref['path'])==ref['sha256']
    q=A.frozen(p,ref['path']);again=A.frozen(p,ref['path']);assert q.phash()==ref['policy_hash'] and A.digest(q.state_dict())==A.digest(again.state_dict())
    fs=importlib.util.spec_from_file_location('t712_geometry_module',HERE/'fit.py');F=importlib.util.module_from_spec(fs);fs.loader.exec_module(F)
    radii=F.plain_radii(q,guard);changed=radii!=q.projected_radii;effective=changed&(q.projected_weights!=0).any(-1)
    assert (radii>=q.projected_radii).all()
    C.save(D/'GEOMETRY_CHECK.json',dict(state='NONVACUOUS' if effective.any() else 'VACUOUS_NO_DUPLICATE_NATIVE',reference=ref,centers=1737,stored_radius_changes=int(changed.sum()),effective_radius_changes=int(effective.sum()),plain_radii_hash=A.digest(radii),protected_radii_hash=A.digest(q.projected_radii),all_other_state_unchanged=True,new_native=0,new_actual_model_optimization=0,elapsed_seconds=time.time()-start))
    for model in [q,A.frozen(p,p['repair_initial']['path'])]:
        model.sample=False;model.rng=None;model.reset_noise();h=None
        arr=A.Old.Existing.L.T.R.E.N.S.arrays(p['protected403_rows'][0]['path'])
        with torch.no_grad():
            for row in arr['states'][:8]:
                guard();_,_,h,_=model.step(torch.tensor(row[None]),h)
                assert torch.isfinite(h).all() and h.shape[-1]==64
    import unittest.mock as mock
    witness=[]
    def check(pipe,index,pc,counts):assert C.M.N.N.P.frozen is A.frozen;witness.append(index)
    with mock.patch.object(C.M.N.N.G.T,'worker',check):C.worker(None,0,dict(p,stage_cpus=list(range(8,16))),None)
    assert witness==[0]
    row=dict(job_id=p['protected403_rows'][0]['id'],request=dict(model=p['initial']['path']),seed=None,teacher=False,physics=int(arr['native_counts'].sum()),array_hashes={k:A.digest(v) for k,v in arr.items()})
    C.M.N.BASE_INSPECT(p,arr,row)
    try:C.M.N.BASE_INSPECT(p,arr,dict(row,array_hashes=dict(row['array_hashes'],actions='0'*64)))
    except AssertionError:pass
    else:raise AssertionError('Corrupt native actions accepted')
    # Synthetic optimizer only: test deterministic full-head gradients and
    # rollback of an interrupted line-search trial. No Stage data are trained.
    torch.manual_seed(202611240712);head=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Linear(4,2)).double()
    x=torch.tensor([[.1,.3,-.5],[.2,-.4,.7]],dtype=torch.float64);opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,line_search_fn='strong_wolfe');calls=[]
    def closure():
        opt.zero_grad(set_to_none=True);loss=head(x).square().mean();loss.backward();calls.append(float(loss));return loss
    for _ in range(3):original(opt,closure)
    assert calls[-1]<calls[0];saved=copy.deepcopy(head.state_dict());osaved=copy.deepcopy(opt.state_dict())
    class Interrupted(Exception):pass
    def interrupted():
        with torch.no_grad():next(head.parameters()).add_(.1)
        raise Interrupted()
    try:original(opt,interrupted)
    except Interrupted:head.load_state_dict(saved);opt.load_state_dict(osaved)
    assert A.digest(head.state_dict())==A.digest(saved) and A.digest(opt.state_dict())==A.digest(osaved)
    test=O/'helper_preflight';test.mkdir();(D/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None))
    command=['systemd-run','--user','--unit=stage-t712-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight']
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
    assert C.read(D/'helper_preflight/AFFINITY_GUARD.json')['state']=='ENDED'
    assert time.time()<end and time.time()-start<300
    C.save(D/'PREFLIGHT.json',dict(state='PASS',geometry_check=C.read(D/'GEOMETRY_CHECK.json')['state'],frozen_loaders=True,actual_worker_loader_binding=True,actual_corrupt_payload_rejected=True,synthetic_fullhead_optimizer_and_deadline_rollback=True,synthetic_gradient_evaluations=len(calls),synthetic_optimizer_attempts=4,actual_helper_cgroup=True,new_Actor=0,new_native=0,new_actual_optimizer=0,elapsed_seconds=time.time()-start,resources=C.C.resources()))
    print('T712_PREFLIGHT_PASS',C.read(D/'GEOMETRY_CHECK.json')['stored_radius_changes'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:C.save(D/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),all_raw_preserved=True,new_native=0,new_actual_optimization=0));raise
