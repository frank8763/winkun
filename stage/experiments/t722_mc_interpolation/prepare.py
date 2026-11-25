"""Zero native and zero optimizer; one fixed midpoint with exact reconstruction."""
from pathlib import Path
import importlib.util,sys,os,copy,io,gzip,time,json,subprocess
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('T722_preparation_controller',HERE/'run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
import t722_learning as L
import acquisition_native as RN

def main():
    started=time.time();C.old_stopped();C.unit_stopped(C.UNIT);assert not (C.OUT/'prepared.json').exists();p=C.read(C.ROOT/'stage/runs/t721_credit_assignment/protocol.json');ref=p['_study_data'];assert C.sha(ref['path'])==ref['sha256'];p.update(C.read(ref['path']));L.torch.set_num_threads(1);RN.Base.G._real_affinity(0,{15})
    import mujoco
    def forbidden(*a,**kw):raise AssertionError('No T722 preparation optimizer or native')
    mujoco.mj_step=forbidden;L.torch.optim.Adam.step=forbidden;L.torch.optim.LBFGS.step=forbidden
    b0=p['learner_initial'];mc=C.read(C.ROOT/'stage/runs/t721_credit_assignment/mc1_train2/summary.json')['model'];assert C.read(C.ROOT/'stage/runs/t721_credit_assignment/mc1_train2/audit.json')['state']=='PASS'
    for m in [b0,mc]:assert C.sha(m['path'])==m['sha256']
    before=L.checkpoint(b0['path'])['joint_state']['actor'];after=L.checkpoint(mc['path'])['joint_state']['actor'];assert set(before)==set(after)
    state={k:(before[k]+.5*(after[k]-before[k]) if k.startswith('replay_head.') else before[k].clone()) for k in before}
    assert sum(v.numel() for k,v in state.items() if k.startswith('replay_head.'))==20198
    assert all(L.torch.equal(before[k],after[k]) for k in before if not k.startswith('replay_head.'))
    for who in ['shadow','independent']:
        independently={k:before[k].clone() for k in before}
        for k in before:
            if k.startswith('replay_head.'):independently[k]=before[k]+(after[k]-before[k])/2
        assert L.digest(state)==L.digest(independently)
    q=C.A.frozen(p,b0['path']);q.load_state_dict(state);joint=dict(actor=copy.deepcopy(state));buffer=io.BytesIO();L.torch.save(dict(joint_state=joint,architecture=L.B.ARCH,policy_hash=q.phash(),joint_hash=L.digest(joint),new_RL=False,optimizer=0,parameter_interpolation=.5,runtime_teacher=False),buffer);blob=gzip.compress(buffer.getvalue(),mtime=0);assert len(blob)<3*2**20
    file=C.OUT/'midpoint.pt.gz';file.write_bytes(blob);model=dict(path=str(file),sha256=C.sha(file),bytes=len(blob),policy_hash=q.phash(),joint_hash=L.digest(joint));assert L.digest(C.A.frozen(p,file).state_dict())==L.digest(state)
    C.save(C.DOC/'MODEL_CONSTRUCTION.json',dict(state='PASS',baseline=b0,MC=mc,model=model,alpha=.5,formula='B0+0.5*(MC-B0) for replay_head only',live_shadow_independent_exact=True,unchanged_all_body_tensors=True,optimizer=0,native=0,added_parameters=0))
    # Full real new-schema payload fixture and corrupt-array rejection, no native.
    sourcefolder=C.ROOT/'stage/runs/t721_credit_assignment/mc1_recovery18';rows=[json.loads(v) for v in gzip.decompress((sourcefolder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];row=rows[0];a=RN.Base.C.E.N.S.arrays(row['path']);RN.inspect(p,a,row);bad=copy.deepcopy(a);bad['actions'][0,0]=L.np.float32(.123456)
    try:RN.inspect(p,bad,row)
    except AssertionError:pass
    else:raise AssertionError('Corrupt payload accepted')
    # NativeRuntime worker loader/cancel seam, new source and no first physics step.
    folder=C.OUT/'worker_preflight';folder.mkdir();wp=dict(p,out=str(folder),native_stop_epoch=time.time()+90,max_physics=1000);ctx=RN.Base.G.R.M.mp.get_context('spawn');counts=ctx.RawArray('q',8);a,b=ctx.Pipe();child=ctx.Process(target=RN.worker,args=(b,0,wp,counts));child.start();b.close()
    try:
        assert a.poll(40);ready=a.recv();assert ready['state']=='READY',ready;assert os.sched_getaffinity(child.pid)=={8}
        a.send(dict(model=model['path'],job=p['panels']['old403']['jobs'][0],seed=None,query=False,cancel_probe=True));assert a.poll(20);reply=a.recv();assert reply['state']=='CANCEL_PASS' and sum(counts)==0,reply;a.send(None);child.join(10);assert child.exitcode==0
    finally:
        if child.is_alive():child.terminate();child.join(5)
        a.close()
    for key in ['inputs','input_stats','_input_manifest','_study_data','schedule']:p.pop(key,None)
    p.update(C.read(C.DOC/'AUTHORIZATION.json'));p.update(candidate_model=model,max_bytes=2**30,child_bytes=640*2**20,global_bytes=10*2**30,admission_bytes=8*2**30,fresh_bytes=1210052,full_bytes=1210052)
    maximum=692*1210052+64*2**20;assert maximum<p['max_bytes'];assert C.used(C.ROOT/'stage/runs')+p['max_bytes']+128*2**20<8*2**30
    # New resource helper tested in its actual registered temporary cgroup.
    test=C.OUT/'helper_preflight';test.mkdir();(C.DOC/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None));cmd=['systemd-run','--user','--unit=stage-t722-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={C.ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight'];v=subprocess.run(cmd,capture_output=True,text=True);assert v.returncode==0,v.stderr
    C.save(C.OUT/'prepared.json',p);C.save(C.DOC/'PREFLIGHT.json',dict(state='PASS',zero_native=True,zero_optimizer=True,real_worker_loader_cancel=True,actual_helper=True,corrupt_payload_red=True,midpoint_live_shadow_independent_exact=True,same_tensor_capacity=True,full_payload_storage_upper=maximum,extra_128MiB_margin=True,source_model_SHA=True,elapsed_seconds=time.time()-started,resources=C.C.resources()))
    print('T722_PREPARATION_PASS',model['sha256'],flush=True)
if __name__=='__main__':main()
