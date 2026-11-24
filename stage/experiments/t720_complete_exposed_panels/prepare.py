"""Actual zero-physics frozen loader, reused initial pairing, worker/helper."""
from pathlib import Path
import sys,importlib.util,os,time,copy,json,subprocess,traceback
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t720_prepare_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
import t720_native as RN
A,np,torch=C.A,C.A.np,C.A.torch

def main():
    started=time.time();C.old_stopped();C.unit_stopped(C.UNIT);assert not (C.OUT/'prepared.json').exists() and not (C.OUT/'launch.json').exists();C.OUT.mkdir(exist_ok=True);(C.OUT/'fit').mkdir(exist_ok=True);os.sched_setaffinity(0,{15});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import mujoco
    def forbidden(*a,**k):raise AssertionError('T720 preparation native/optimizer forbidden')
    mujoco.mj_step=forbidden;torch.optim.Adam.step=forbidden;torch.optim.LBFGS.step=forbidden
    p=C.read(C.ROOT/'stage/runs/t719_outcome_weighted_replay/protocol.json');selection=C.read(C.DOC/'FIXED_SELECTION.json');model=selection['model'];RN.N.N.P.frozen=A.frozen;assert C.sha(model['path'])==model['sha256'];q=A.frozen(p,model['path']);other=A.frozen(p,model['path']);assert q.phash()==other.phash()==model['policy_hash'] and A.digest(q.state_dict())==A.digest(other.state_dict());checks=[]
    assert C.read(C.ROOT/'stage/runs/t719_outcome_weighted_replay/validation/audit.json')['state']=='PASS'
    for name,panel in selection['panels'].items():
        assert C.read(panel['baseline_audit'])['state']=='PASS';base={r['job_id']:r for r in panel['baseline_rows']};assert len(base)==len(panel['all_jobs'])
        for r in panel['baseline_rows']:assert C.sha(r['path'])==r['sha256']
        for row in panel['reused_rows']+panel['reused_exact_rows']:
            assert row['role'] in ['original','exact_original'] and row['seed'] is None and row['loaded_model_sha256']==model['sha256'];a=RN.arrays(row);RN.INSPECT(p,a,row);assert all(A.digest(v)==row['array_hashes'][k] for k,v in a.items());assert not a['ownership'].any() and not a['stochastic'].any();RN.initial_match(a,RN.arrays(base[row['job_id']]))
        checks.append(dict(panel=name,total=len(panel['all_jobs']),new_fresh=len(panel['fresh_jobs']),new_exact=len(panel['exact_jobs']),reused_original=len(panel['reused_rows']),reused_exact=len(panel['reused_exact_rows'])))
    fixture=selection['panels']['panel78']['reused_rows'][0];a=RN.arrays(fixture);RN.INSPECT(p,a,fixture);bad=copy.deepcopy(a);bad['actions'][0,0]=np.float32(.123456)
    try:RN.INSPECT(p,bad,fixture)
    except AssertionError:pass
    else:raise AssertionError('Corrupt full payload accepted')
    folder=C.OUT/'worker_preflight';folder.mkdir(exist_ok=True);wp=dict(p,out=str(folder),native_stop_epoch=time.time()+90,max_physics=1000);ctx=RN.N.N.G.R.M.mp.get_context('spawn');counts=ctx.RawArray('q',8);a,b=ctx.Pipe();child=ctx.Process(target=RN.worker,args=(b,0,wp,counts));child.start();b.close()
    try:
        assert a.poll(40);ready=a.recv();assert ready['state']=='READY',ready;assert os.sched_getaffinity(child.pid)=={8}
        a.send(dict(model=model['path'],job=selection['panels']['panel78']['fresh_jobs'][0],seed=None,query=False,cancel_probe=True));assert a.poll(20);reply=a.recv();assert reply['state']=='CANCEL_PASS',reply;assert sum(counts)==0;a.send(None);child.join(10);assert child.exitcode==0
    finally:
        if child.is_alive():child.terminate();child.join(5)
        a.close()
    test=C.OUT/'helper_preflight';test.mkdir();(C.DOC/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None));command=['systemd-run','--user','--unit=stage-t720-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={C.ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight'];v=subprocess.run(command,capture_output=True,text=True);assert v.returncode==0,v.stderr
    for k in ['inputs','input_stats','_input_manifest','_replay_manifest']:p.pop(k,None)
    C.save(C.OUT/'prepared.json',p);C.save(C.DOC/'PREFLIGHT.json',dict(state='PASS',model=model,checks=checks,total174_reused18=True,source_array_SHAs=True,reused_full_original_owncarry_initial_pairing=True,corrupt_payload_red=True,actual_model_loader_worker_helper=True,new_native=0,new_optimizer=0,commands=3,elapsed_seconds=time.time()-started,resources=C.C.resources()));print('T720_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:C.save(C.DOC/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),new_native=0,new_optimizer=0,raw_preserved=True));raise
