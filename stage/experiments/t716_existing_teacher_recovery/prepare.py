"""One bounded zero-native, zero-learning recovery-runtime qualification."""
from pathlib import Path
import importlib.util,json,copy,time,os,sys,types,traceback,subprocess
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t716_prepare_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
import recovery_native as RN
A,N=C.A,C.M.N

def main():
    started=time.time();b=C.read(C.DOC/'ACTUAL_PREPARATION_BOUND.json');assert started<b['absolute_stop'];C.old_stopped();assert not C.OUT.exists();C.OUT.mkdir();(C.OUT/'fit').mkdir()
    A.torch.set_num_threads(1);A.torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,{15})
    import mujoco
    def forbidden(*args,**kwargs):raise AssertionError('No native/learning in T716preparation')
    mujoco.mj_step=forbidden;A.torch.optim.Adam.step=forbidden;A.torch.optim.LBFGS.step=forbidden
    def guard():assert time.time()<b['absolute_stop']-10
    p=C.read(C.ROOT/'stage/runs/t715_command_interval/protocol.json');assignment=C.read(C.DOC/'FIXED_ASSIGNMENT.json')['rows'];groups={'compact':[],'replay50':[]};finals={};checks=[]
    for row in assignment:
        guard();teacher=row['teacher'];baseline=row['baseline'];modelpath=teacher['request']['model'];assert C.sha(modelpath)==teacher['loaded_model_sha256']
        q=A.frozen(p,modelpath);assert q.phash()==teacher['policy_hash'];again=A.frozen(p,modelpath);assert A.digest(q.state_dict())==A.digest(again.state_dict());del again
        for r in [teacher,baseline]:
            assert C.sha(r['path'])==r['sha256'];arr=RN.Base.C.E.N.S.arrays(r['path']);assert not arr['ownership'].any() and not arr['stochastic'].any()
            assert {k:A.digest(v) for k,v in arr.items()}==r['array_hashes']
        arm=row['assigned_teacher'];qmodel=dict(path=modelpath,sha256=C.sha(modelpath),policy_hash=q.phash(),bytes=Path(modelpath).stat().st_size)
        if arm in finals:assert finals[arm]['model']==qmodel
        finals[arm]=dict(model=qmodel)
        job=copy.deepcopy(baseline['request']['job']);job.update(source_path=baseline['path'],source_sha256=baseline['sha256'],recovery_first=row['recovery_first']);groups[arm].append(job)
        arr=RN.Base.C.E.N.S.arrays(baseline['path']);assert 0<job['recovery_first']<len(arr['states']) and not arr['end_flags'][:job['recovery_first']].any()
        # Existing teacher trajectory is a positive full inspector fixture.
        original=RN.Base.C.E.N.S.arrays(teacher['path']);fixture=dict(original);fixture['ownership']=original['ownership'].copy();fixture['ownership'][:1]=1;fixture['recovery_first']=A.np.asarray(1,A.np.int64);fixture['source_prefix_actions']=A.np.zeros_like(original['actions']);fixture['source_prefix_actions'][:1]=original['actions'][:1]
        fixturejob=copy.deepcopy(teacher['request']['job']);fixturejob.update(source_path=teacher['path'],source_sha256=teacher['sha256'],recovery_first=1)
        fixture_row=dict(teacher,request=dict(teacher['request'],job=fixturejob),array_hashes={k:A.digest(v) for k,v in fixture.items()});RN.inspect(p,fixture,fixture_row)
        corrupted=dict(fixture);corrupted['actions']=fixture['actions'].copy();corrupted['actions'][0,0]*=-1;corrupted['actions'][0,0]+=.01
        try:RN.inspect(p,corrupted,fixture_row)
        except AssertionError:pass
        else:raise AssertionError('Corrupt native prefix accepted')
        checks.append(dict(id=row['id'],arm=arm,recovery_first=job['recovery_first'],loaded_teacher_policy=q.phash(),positive_inspector=True,corrupt_prefix_rejected=True));del q
    assert len(groups['compact'])==14 and len(groups['replay50'])==4
    captured=[];original_step=RN.F.FieldSession.step
    def step(self,forced=None,owner=0,actor_active=True):captured.append((forced,owner,actor_active))
    try:
        RN.F.FieldSession.step=step
        for jobs in groups.values():
            for job in jobs:
                obj=RN.Session.__new__(RN.Session);obj.first=job['recovery_first'];obj.source=RN.Base.C.E.N.S.arrays(job['source_path']);obj.prefix_actions=[];obj.q=types.SimpleNamespace(sample=True);obj.rows=[None]*(obj.first-1);obj.step();assert captured[-1][1:]==(1,True) and A.np.array_equal(captured[-1][0],obj.source['actions'][obj.first-1]);obj.rows=[None]*obj.first;obj.step();assert captured[-1]==(None,0,True) and not obj.q.sample
    finally:RN.F.FieldSession.step=original_step
    from unittest.mock import patch
    witnesses=[]
    def witness(pipe,index,pc,counts):assert RN.Base.P.frozen is A.frozen and RN.Base.G.T.Session is RN.Session and RN.Base.C.E.payload is RN.payload;witnesses.append(index)
    with patch.object(RN.Base.G.T,'worker',witness):C.worker(None,0,p,None)
    assert witnesses==[0]
    test=C.OUT/'helper_preflight';test.mkdir();(C.DOC/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None))
    cmd=['systemd-run','--user','--unit=stage-t716-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={C.ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight']
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    C.save(C.OUT/'fit/summary.json',dict(state='REUSED_FROZEN_TEACHERS_NO_FIT',finals=finals));C.save(C.OUT/'fit/audit.json',dict(state='PASS',finals={k:dict(state='PASS',audited_historical_teacher_reused=True) for k in finals}))
    p.update(assignment=assignment,recovery_groups=groups,new_learning=0)
    for key in ['inputs','input_stats','_input_manifest']:p.pop(key,None)
    C.save(C.OUT/'prepared.json',p);C.save(C.DOC/'PREFLIGHT.json',dict(state='PASS',source_ids=18,compact=14,replay50=4,models=list(finals),checks=checks,actual_helper=True,actual_worker_binding=True,forced_prefix_then_teacher_owncarry_dispatch=True,full_original_teacher_payload_inspector=True,corrupt_prefix_red=True,new_native=0,new_learning=0,elapsed_seconds=time.time()-started,resources=C.C.resources()))
    print('T716_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:C.save(C.DOC/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),new_native=0,new_learning=0));raise
