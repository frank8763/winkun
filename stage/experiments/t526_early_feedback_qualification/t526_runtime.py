"""Native student-state recovery qualification followed by conditional rehearsal."""
from pathlib import Path
import copy,fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t525_student_state_recovery'));import t525_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t526_early_feedback_qualification'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
BASE_PAYLOAD=E.payload
# Reuse only the already verified strict-mapping feedback equations.
sys.path.insert(0,str(ROOT/'stage/experiments/t514_strict_teacher_recovery'))
import t514_runtime as Feedback
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
class RecoverySession(F.FieldSession):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs);self.teacher_labels=[];self.holds=int(job['teacher_holds'])
        from training.e2e_phase10_lidar_v7_curriculum.phase10_strict_mujoco_env import Phase10StrictMuJoCoEnv
        from training.e2e_phase10_lidar_v7_curriculum.stage9_runtime_fullplant_fix_v5_2 import WHEELBASE_M
        assert type(self.env)._policy_to_commands is Phase10StrictMuJoCoEnv._policy_to_commands and float(WHEELBASE_M)==self.q.p['physical_wheelbase_m']
        self.teacher_parameters=np.asarray([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer],np.float64)
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        command=Feedback.command(self.q.p,self.env._pose()[:3],self.goal,self.teacher_parameters)
        Feedback.roundtrip(self.q.p,self.env,self.env._pose()[:3],self.goal,command)
        self.teacher_labels.append(command.copy());k=len(self.rows);first=self.q.p['teacher_first'];use=k>=first and (self.holds<0 or k<first+self.holds)
        return super().step(command if use else None,owner=int(use),actor_active=True)
def payload(session,tape=None):
    a=BASE_PAYLOAD(session,tape);a['teacher_actions']=np.asarray(session.teacher_labels[:len(a['states'])],np.float32);a['teacher_holds']=np.asarray(session.holds,np.int64);a['teacher_vehicle_parameters']=session.teacher_parameters.copy();return a

def worker(pipe,index,p,counts):
    G.T.Session=RecoverySession;G.B.native_setup=F.field_setup;P.frozen=F.L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';old=read(f) if f.exists() else [];old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,old)
        except BaseException:self.close();raise
    def receive(self,i):
        v=super().receive(i)
        if isinstance(v,tuple):v[1].update(teacher=True,teacher_first=self.p['teacher_first'],feedback_teacher_contract=self.p['controller_contract'],training_mixture_NOT_autonomous=True)
        return v

def check_prefix(p,a):
    b=E.N.S.arrays(p['baseline_row']['path']);k=p['teacher_first'];assert len(a['states'])>k
    for key in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:
        assert np.array_equal(a[key][:k],b[key][:k]),key
    for key in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],b[key][k]),key
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration']) and a['native_counts'][:k].sum()==p['prefix_physics_steps']
    owner=np.zeros(len(a['states']),np.uint8);holds=int(a['teacher_holds']);owner[k:None if holds<0 else k+holds]=1
    assert np.array_equal(owner,a['ownership']) and not a['stochastic'].any()

def audit_recovery(p,row,a):
    check_prefix(p,a);q=F.L.frozen(p,p['initial']['path']);q.sample=False;h=None;own=[]
    with torch.no_grad():
        for i,x in enumerate(a['states']):
            u,_,h,_=q.step(torch.tensor(x[None]),h);own.append(u[0].numpy().copy())
            expected=Feedback.command(p,a['poses'][i],a['task_goal_xy'],a['teacher_vehicle_parameters']);assert np.array_equal(expected,a['teacher_actions'][i])
            assert np.array_equal(a['actions'][i],a['teacher_actions'][i] if a['ownership'][i] else own[-1])
    b=dict(a,actions=np.asarray(own),ownership=np.zeros(len(own),np.uint8));P.frozen=F.L.frozen;M.audit_full_history(p,dict(row,teacher=False),b)
    return F.detail(p,row,a)

def main(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);P.frozen=F.L.frozen;E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        job=dict(p['probe_job'],teacher_holds=-1);vv=rt.request([job],p['initial'],'EARLY_STRICT_FEEDBACK_RECOVERY',full=True);a,row=vv[0];check_prefix(p,a);detail=F.detail(p,row,a)
        rt.request([job],p['initial'],'EXACT_EARLY_FEEDBACK',full=True,references=[row]);teacher_ok=detail['field_qualified_success'];handoff=None
        if teacher_ok:
            job=dict(p['probe_job'],teacher_holds=20);values=rt.request([job],p['initial'],'FOUR_SECOND_FEEDBACK_THEN_STUDENT',full=True);b,rr=values[0];check_prefix(p,b);rt.request([job],p['initial'],'EXACT_FOUR_SECOND_HANDOFF',full=True,references=[rr]);handoff=dict(tally=G.R.M.tally([rr]),autonomous_decisions_after_return=max(0,len(b['states'])-p['teacher_first']-20),detail=F.detail(p,rr,b))
        result=dict(recovery_teacher_qualified=teacher_ok,teacher_detail=detail,handoff=handoff,eligible_for_future_learning=bool(teacher_ok and handoff and handoff['detail']['field_qualified_success'] and handoff['autonomous_decisions_after_return']>0),Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,student_fullfield24=None,old8=None,accepted=False,new_autonomous_fullfield_success_NOT_established=True,fresh16=None,reload78=None)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder);rows=G.rows(folder);rep=G.rows(folder,'replays.jsonl');result=s['result']
    for r in rows:
        assert r['full'] and sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());audit_recovery(p,r,a)
    by={r['path']:r for r in rows}
    for r in rep:assert r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256'] and r['exact']
    a=E.N.S.arrays(rows[0]['path']);detail=F.detail(p,rows[0],a);assert detail==result['teacher_detail'];teacher_ok=detail['field_qualified_success'];assert teacher_ok==result['recovery_teacher_qualified']
    if teacher_ok:
        b=E.N.S.arrays(rows[1]['path']);assert F.detail(p,rows[1],b)==result['handoff']['detail'] and G.R.M.tally([rows[1]])==result['handoff']['tally']
        assert result['handoff']['autonomous_decisions_after_return']==max(0,len(b['states'])-p['teacher_first']-20)
    assert result['eligible_for_future_learning']==bool(teacher_ok and result['handoff']['detail']['field_qualified_success'] and result['handoff']['autonomous_decisions_after_return']>0)
    assert len(rows)==len(rep)==(2 if teacher_ok else 1);assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==len(rows)+len(rep)+4
    assert not (folder/'optimizer_calls.jsonl').exists() and all(sha(f)==h for f,h in p['protected_files'].items())
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,all_student_causal_history_feedback_mapping_rewards_and_arrays_rebuilt=True,prefix_clock_goal_actuator_hidden_preserved=True,all_full_arrays_and_integrations_match_native_replays=True,teacher_and_handoff_success_NOT_new_student_model=True,physics=s['physics'],commands=s['commands']));print('T526_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
