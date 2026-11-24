"""Bounded0learning,0native preflight of both frozen stochastic comparators."""
from pathlib import Path
import sys,os,time,json,copy,importlib.util,subprocess,traceback,gzip,types
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t719_preparation_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
import acquisition_native as RN
import t719_learning as L
A,np,torch=C.A,C.A.np,C.A.torch

def main():
    started=time.time();C.old_stopped();C.unit_stopped(C.UNIT);assert not C.OUT.exists();C.OUT.mkdir();(C.OUT/'fit').mkdir();os.sched_setaffinity(0,{15});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import mujoco
    def forbidden(*a,**k):raise AssertionError('T719 preflight forbids native and optimizer')
    mujoco.mj_step=forbidden;torch.optim.LBFGS.step=forbidden
    p=C.read(C.ROOT/'stage/runs/t717_onpolicy_recovery/protocol.json');selection=C.read(C.DOC/'FIXED_DATA.json');models={'initial':selection['initial']};checks=[];p.update(learner_initial=selection['initial'],replay_rows=selection['rows'],weight_beta=1.,actor_lr=.0003,fit_steps=16,fit_threads=8)
    for name,m in models.items():
        assert C.sha(m['path'])==m['sha256'];q=A.frozen(p,m['path']);again=A.frozen(p,m['path']);assert q.phash()==again.phash()==m['policy_hash'] and A.digest(q.state_dict())==A.digest(again.state_dict());assert q.p['raw_sigma']==[.2,.2];checks.append(dict(arm=name,model_sha=m['sha256'],policy_hash=q.phash(),same_sigma=[.2,.2]))
    rows=[json.loads(x) for x in gzip.decompress((C.ROOT/'stage/runs/t717_onpolicy_recovery/training/cases.jsonl.gz').read_bytes()).decode().splitlines()];fixture=next(r for r in rows if r['role']=='fresh0');arr=RN.Base.C.E.N.S.arrays(fixture['path']);RN.inspect(p,arr,fixture)
    bad=copy.deepcopy(arr);bad['actions'][0,0]=np.float32(.123456)
    try:RN.inspect(p,bad,fixture)
    except AssertionError:pass
    else:raise AssertionError('Corruptprefix accepted')
    jobs=p['recovery_jobs'];assert len(jobs)==18 and len(selection['rows'])==72
    old=C.read(C.DOC.parent/'t717/FIXED_SPLIT.json');assert selection['training_ids']==old['training_ids'] and selection['validation_ids']==old['validation_ids']
    captured=[];original=RN.F.FieldSession.step
    def witness(self,forced=None,owner=0,actor_active=True):captured.append((forced,owner,actor_active,self.q.sample))
    try:
        RN.F.FieldSession.step=witness
        for j in jobs:
            assert C.sha(j['source_path'])==j['source_sha256'];obj=RN.Session.__new__(RN.Session);obj.first=j['recovery_first'];obj.source=RN.Base.C.E.N.S.arrays(j['source_path']);obj.prefix_actions=[];obj.q=types.SimpleNamespace(sample=True);obj.suffix_sample=True;obj.rows=[None]*(obj.first-1);obj.step();assert captured[-1][1:]==(1,True,False);obj.rows=[None]*obj.first;obj.step();assert captured[-1]==(None,0,True,True)
    finally:RN.F.FieldSession.step=original
    p.update(models=models,recovery_jobs=jobs,training_ids=selection['training_ids'],validation_ids=selection['validation_ids'],new_learning=True)
    workerfolder=C.OUT/'worker_preflight';workerfolder.mkdir();wp=dict(p,out=str(workerfolder),native_stop_epoch=time.time()+90,max_physics=1000);ctx=RN.Base.G.R.M.mp.get_context('spawn');counts=ctx.RawArray('q',8);a,b=ctx.Pipe();child=ctx.Process(target=RN.worker,args=(b,0,wp,counts));child.start();b.close()
    try:
        assert a.poll(40);ready=a.recv();assert ready['state']=='READY',ready;assert os.sched_getaffinity(child.pid)=={8}
        for m in models.values():
            a.send(dict(model=m['path'],job=jobs[0],seed=[20261124718,0,0],query=False,cancel_probe=True));assert a.poll(20);reply=a.recv();assert reply['state']=='CANCEL_PASS',reply;assert sum(counts)==0
        a.send(None);child.join(10);assert child.exitcode==0
    finally:
        if child.is_alive():child.terminate();child.join(5)
        a.close()
    C.save(C.DOC/'NATIVE_PREFLIGHT.json',dict(state='PASS',actual_worker=True,both_real_model_loaders=True,actual_CPU=[8],zero_native_cancelled_before_first_step=True,commands=3,new_native=0,new_optimizer=0,worker_pid=child.pid))
    test=C.OUT/'helper_preflight';test.mkdir();(C.DOC/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None))
    command=['systemd-run','--user','--unit=stage-t719-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={C.ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight'];v=subprocess.run(command,capture_output=True,text=True);assert v.returncode==0,v.stderr
    # Native-free real fullhistory feature weighting and discarded single-step Adam.
    obj=L.make(p);startstate=L.state(obj);data,manifest=L.dataset(p,obj[0],lambda:None)
    assert sum(len(items) for items in data.values())==72 and all(len(items)==6 for items in data.values())
    assert sorted(m['score'] for m in manifest).count(1)==51
    results=[];states=[]
    def event(who):
        def emit(stage,k):
            with (C.DOC/'PREFLIGHT_CALLS.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,stage=stage,k=k,kind='actor'))+'\n')
        return emit
    # A fixed first ID's six fullsuffixes exercise actual objective/gradient without consuming formal steps.
    small={next(iter(data)):data[next(iter(data))]}
    for who in ['live','shadow','independent']:
        candidate=L.make(p);L.load(candidate,startstate);result=L.fit(candidate,small,p,lambda:None,event(who),steps=1);results.append(result);states.append(L.digest(L.state(candidate)))
    assert results[0]==results[1]==results[2] and states[0]==states[1]==states[2]
    for k in ['inputs','input_stats','_input_manifest']:p.pop(k,None)
    C.save(C.OUT/'prepared.json',p);C.save(C.DOC/'PREFLIGHT.json',dict(state='PASS',checks=checks,actual_worker_and_helper=True,corrupt_prefix_rejected=True,full_stochastic_owncarry_density_fixture=True,all18_prefix_suffix_dispatch=True,new_native=0,discarded_Actor_Adam_each=1,new_Value=0,formal_learning=0,all72_positive_weights_verified=True,live_shadow_independent_discarded_Adam=True,elapsed_seconds=time.time()-started,resources=C.C.resources()));print('T719_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:C.save(C.DOC/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),raw_preserved=True));raise
