"""Existing GAP teacher only during one qualified training suffix."""
from pathlib import Path
import ast,gzip,json,math,os,sys,time
import t637_data as D
B,L,P,E,torch,np=D.B,D.L,D.P,D.E,D.torch,D.np
ROOT=D.ROOT
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'))
import t502_runtime as F
sys.path.insert(0,str(ROOT/'stage/experiments/t630_early_departure_recovery'))
import reset_contract as Reset
G=L.G
BASE_PAYLOAD=E.payload
TEACHER_SOURCE=ROOT/'stage/experiments/t550_offline_obstacle_relay/t550_teacher.py'
# Execute only the unchanged pure law, avoiding unrelated legacy runtime
# imports with module-global patches. Its complete source file is SHA locked.
tree=ast.parse(TEACHER_SOURCE.read_text())
function=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='control')
namespace=dict(np=np,math=math,ANGLES=np.linspace(-.75*math.pi,.75*math.pi,48,dtype=np.float32).astype(np.float64))
exec(compile(ast.Module(body=[function],type_ignores=[]),str(TEACHER_SOURCE),'exec'),namespace)
control=namespace['control']


class Session(F.FieldSession):
    def __init__(self,job,*args,**kw):
        super().__init__(job,*args,**kw)
        self.first=int(job.get('gap_teacher_first',-1))
        self.source=E.N.S.arrays(job['source_path']) if self.first>=0 else None
        self.scans=[];self.controls=[]
        self.parameters=np.array([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer])

    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        if self.first<0:return super().step()
        k=len(self.rows);before=E.N.K.R.integration(self.env).copy();scan=self.env._lidar48().copy()
        u=control('GAP',self.env._pose()[:3],self.goal,scan,self.parameters)
        assert np.array_equal(before,E.N.K.R.integration(self.env))
        self.scans.append(scan);self.controls.append(u)
        if k<self.first:
            # Actual T501 evolution, not a replay of an expert prefix.
            super().step()
            assert np.array_equal(self.rows[-1][0],self.source['states'][k])
            assert np.array_equal(self.rows[-1][1],self.source['actions'][k])
            return
        return super().step(u,owner=1,actor_active=True)


def payload(s,tape=None):
    a=BASE_PAYLOAD(s,tape)
    if s.first>=0:
        n=len(a['states'])
        a.update(teacher_scans=np.asarray(s.scans[:n],np.float32),teacher_actions=np.asarray(s.controls[:n],np.float32),
            teacher_vehicle_parameters=s.parameters.copy(),teacher_first_declared=np.array(s.first,np.int64))
    return a


def worker(pipe,index,p,counts):
    real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:Reset.field_setup(F,folder,pc)
    P.frozen=L.frozen;E.payload=payload
    G.T.worker(pipe,index,p,counts)


class Pool(B.Pool):
    def __init__(self,p,counts,guard):
        self.p,self.guard=p,guard;self.children=[];self.pipes=[]
        ctx=G.R.M.mp.get_context('spawn');out=Path(p['out'])
        try:
            for i in range(3):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close()
                self.children.append(c);self.pipes.append(a)
            record=dict(parent=os.getpid(),workers=[c.pid for c in self.children])
            D.save(out/'worker_generations.json',[record]);self.ready=[self.receive(i) for i in range(3)]
            actual=[sorted(os.sched_getaffinity(c.pid)) for c in self.children]
            assert actual==[[8],[10],[12]] and all(v['state']=='READY' for v in self.ready)
            D.save(out/'worker_generations.json',[dict(record,actual_affinity=actual)])
        except BaseException:self.close();raise

    def request(self,items):
        assert 1<=len(items)<=3
        for i,r in enumerate(items):self.pipes[i].send(r)
        result=[]
        for i,r in enumerate(items):
            a,row=self.receive(i)
            with gzip.open(Path(self.p['out'])/('native_'+r['record_key']+'.json.gz'),'xb') as f:
                f.write(json.dumps(dict(request=r,native_result=row)).encode())
            result.append((a,row))
        return result


def inspect(p,a,r):
    if 'teacher_first_declared' not in a:
        # Historical failures are valid measurements, too; general audit has
        # no hardcoded success terminal assertion from the six-case repro.
        F.M.audit_full_history(p,r,a)
    else:
        k=int(a['teacher_first_declared']);source=E.N.S.arrays(r['request']['job']['source_path'])
        assert k>0 and len(a['states'])>k
        for key in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before']:
            assert np.array_equal(a[key][:k],source[key][:k]),key
        for key in ['states','poses','hidden_before','history_poses','history_goals','history_times']:
            assert np.array_equal(a[key][k],source[key][k]),key
        assert np.array_equal(a['ownership'],np.arange(len(a['states']))>=k)
        for i,(pose,scan) in enumerate(zip(a['poses'],a['teacher_scans'])):
            u=control('GAP',pose,a['task_goal_xy'],scan,a['teacher_vehicle_parameters'])
            assert np.array_equal(u,a['teacher_actions'][i])
            if i>=k:assert np.array_equal(u,a['actions'][i])
        proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32)
        F.M.audit_full_history(p,dict(r,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))
    assert int(a['native_counts'].sum())==r['physics']
    assert not a['stochastic'].any() and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
    assert all(D.digest(v)==r['array_hashes'][key] for key,v in a.items())
