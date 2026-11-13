"""One training-only terminal feedback controller; frozen student, zero fit."""
from pathlib import Path
import fcntl,math,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=F.G,F.P,F.E,F.M,F.torch,F.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t513_goal_feedback_teacher'
BASE_PAYLOAD=E.payload
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def command(p,pose,goal,parameters):
    """Point-goal curvature feedback; no path, obstacle map or future states."""
    speed_gain,max_wheel,wheelbase,yaw_gain,max_steer=map(float,parameters)
    assert speed_gain>0 and max_wheel>0 and wheelbase>0 and yaw_gain>0 and 0<max_steer<math.pi/2
    dx,dy=np.asarray(goal)-np.asarray(pose)[:2];d=math.hypot(dx,dy)
    error=math.atan2(math.sin(math.atan2(dy,dx)-pose[2]),math.cos(math.atan2(dy,dx)-pose[2]))
    curvature=2*math.sin(error)/max(d,p['curvature_distance_floor_m'])
    steer=float(np.clip(math.atan(wheelbase*curvature/yaw_gain),-max_steer,max_steer))
    vmax=speed_gain*max_wheel
    desired=min(p['nominal_speed_m_s'],vmax)*min(1.,d/p['slow_distance_m'])*max(p['minimum_heading_speed_fraction'],math.cos(error))
    # Exact inverse of native _policy_to_commands, including its .05 floor.
    drive=float(np.clip(2*desired/vmax-1,-1,1))
    rotation=math.tan(steer)*max(desired,.05)/wheelbase
    turn=float(np.clip(rotation/1.2,-1,1))
    return np.asarray([drive,turn],np.float32)

class TeacherSession(F.FieldSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.teacher_parameters=np.asarray([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer],np.float64)
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        if len(self.rows)<self.q.p['teacher_first']:return super().step(owner=0)
        u=command(self.q.p,self.env._pose()[:3],self.goal,self.teacher_parameters)
        return super().step(u,owner=1,actor_active=True)

def payload(session,tape=None):
    a=BASE_PAYLOAD(session,tape)
    a['teacher_actions']=a['actions'].copy();a['teacher_vehicle_parameters']=session.teacher_parameters.copy()
    return a

def worker(pipe,index,p,counts):
    G.T.Session=TeacherSession;G.B.native_setup=F.field_setup;P.frozen=F.L.frozen;E.payload=payload
    G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';old=read(f) if f.exists() else []
            old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False));save(f,old)
        except BaseException:self.close();raise
    def receive(self,i):
        v=super().receive(i)
        if isinstance(v,tuple):v[1].update(teacher=True,teacher_first=self.p['teacher_first'],not_student_navigation=True,teacher_behavior_hash=digest(dict(model=v[1]['policy_hash'],first=self.p['teacher_first'],controller=self.p['controller_contract'])))
        return v

def check_teacher(p,a,baseline):
    k=p['teacher_first'];assert len(a['states'])>k and not a['ownership'][:k].any() and a['ownership'][k:].all() and not a['stochastic'].any()
    for key in ['states','actions','native_counts','next_states','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:
        assert np.array_equal(a[key][:k],baseline[key][:k]),key
    for key in ['states','poses','hidden_before','history_poses','history_goals','history_times']:
        assert np.array_equal(a[key][k],baseline[key][k]),key
    assert int(a['native_counts'][:k].sum())==p['prefix_physics_steps']
    assert np.array_equal(a['teacher_actions'],a['actions'])
    for i in range(k,len(a['states'])):
        assert np.array_equal(command(p,a['poses'][i],a['task_goal_xy'],a['teacher_vehicle_parameters']),a['actions'][i]),i

def main(phase):
    assert phase=='probe';folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True)
    E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vals=rt.request([p['probe_job']],p['initial'],'ONE_FIXED_TRAINING_ONLY_FEEDBACK_RECOVERY',full=True)
        a,r=vals[0];check_teacher(p,a,E.N.S.arrays(p['baseline_row']['path']))
        rt.request([p['probe_job']],p['initial'],'EXACT_SAME_FEEDBACK_RECOVERY',full=True,references=[r])
        detail=F.detail(p,r,a);save(folder/'diagnosis.json',detail)
        result=dict(teacher_tally=G.R.M.tally([r]),qualified_teacher_successes=int(detail['field_qualified_success']),
                    learning_data_eligible=bool(detail['field_qualified_success']),teacher_rows=[r],student_autonomous_success_NOT_tested=True,
                    Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,model=p['initial'],fullfield24=None,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder);P.frozen=F.L.frozen
    rows,reps=G.rows(folder),G.rows(folder,'replays.jsonl');assert len(rows)==len(reps)==1
    r=rows[0];a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and digest(a)==r['array_digest']
    assert all(digest(v)==r['array_hashes'][key] for key,v in a.items())
    M.audit_full_history(p,r,a);check_teacher(p,a,E.N.S.arrays(p['baseline_row']['path']))
    assert r['teacher'] and not r['administrative_censor'] and r['loaded_model_sha256']==p['initial']['sha256']
    detail=F.detail(p,r,a);assert detail==read(folder/'diagnosis.json')
    assert int(detail['field_qualified_success'])==s['result']['qualified_teacher_successes']
    assert reps[0]['exact'] and reps[0]['array_hashes']==r['array_hashes'] and reps[0]['integration_sha256']==r['integration_sha256']
    assert s['physics']==r['physics']+reps[0]['physics']<=p['max_physics'] and s['commands']==6
    assert sha(p['initial']['path'])==p['initial']['sha256'] and not (folder/'optimizer_calls.jsonl').exists()
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,teacher_action_feedback_and_full_history_density_reward_arrays_exact=True,
         exact_self_prefix_clock_hidden_history_not_reset=True,teacher_NOT_autonomous_student=True,model_unchanged=True,physics=s['physics'],commands=s['commands']))
    print('T513_FEEDBACK_TEACHER_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
