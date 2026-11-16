"""Fixed actual stochastic prefix training-teacher recovery qualification, zero learning."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t582_fresh_multitask_pair'))
import t582_learning as L
OUT=ROOT/'stage/runs/t583_qualified_gap_recovery';DOC=ROOT/'stage/docs/day_20261116/t583'
sys.path.insert(0,str(ROOT/'stage/experiments/t577_offline_recovery_night'))
import t577_runtime as R
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np
Base,F,G,E,P=R.Base,R.F,R.G,R.E,R.P


def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def verify(p,full=False):
    root=read(OUT/'protocol.json');assert root['authorization_approved'] is True;assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes']
    assert used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15}
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f


def setup():
    R.HERE=HERE;R.verify=verify;R.L.frozen=L.frozen;R.setup(OUT);Base.verify=verify;P.frozen=L.frozen


def worker(pipe,index,p,counts):
    R.L.frozen=L.frozen;R.worker(pipe,index,p,counts)


class Pool(R.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
            save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise





sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'))
import t550_teacher as Teacher
NORMAL=R.NORMAL
PREFIX_KEYS=['states','next_states','actions','native_counts','poses','next_poses','hidden_before',
             'history_poses','history_goals','history_times','locations','innovations','raw_actions','atom','logp','stochastic']


class Session(F.FieldSession):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        self.first=int(job.get('gap_recovery_first',-1));self.teacher_actions=[];self.teacher_scans=[]
        self.teacher_parameters=np.asarray([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer],np.float64)

    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        if self.first<0:return super().step()
        before=E.N.K.R.integration(self.env).copy();scan=self.env._lidar48().copy()
        u=Teacher.control('GAP',self.env._pose()[:3],self.goal,scan,self.teacher_parameters)
        assert np.array_equal(before,E.N.K.R.integration(self.env))
        self.teacher_actions.append(u.copy());self.teacher_scans.append(scan)
        use=len(self.rows)>=self.first
        if use:self.q.sample=False
        return super().step(u if use else None,owner=int(use),actor_active=True)


def payload(s,tape=None):
    a=NORMAL(s,tape)
    if s.first>=0:
        n=len(a['states']);a.update(teacher_actions=np.asarray(s.teacher_actions[:n],np.float32),
            teacher_scans=np.asarray(s.teacher_scans[:n],np.float32),teacher_vehicle_parameters=s.teacher_parameters.copy(),
            teacher_first_declared=np.asarray(s.first,np.int64))
    return a


# Override the inherited worker before any process is created. Actor inference
# and physics remain the same; only training-owned actions after k differ.
def worker(pipe,index,p,counts):
    G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload
    G.cpu_worker(pipe,index,p,counts)


original_receive=Pool.receive

def receive(self,i):
    value=original_receive(self,i)
    if isinstance(value,tuple):
        a,r=value;r['teacher']='teacher_first_declared' in a
        if r['teacher']:r.update(teacher_first=int(a['teacher_first_declared']),teacher_kind='FIXED_GAP_TRAINING_ONLY',training_mixture_NOT_autonomous=True)
    return value

Pool.receive=receive


def prefix_assert(a,b,k):
    assert k>0 and len(a['states'])>k and len(b['states'])>k
    assert all(np.array_equal(a[name][:k],b[name][:k]) for name in PREFIX_KEYS)
    assert all(np.array_equal(a[name][k],b[name][k]) for name in ['states','poses','hidden_before','history_poses','history_goals','history_times'])
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])


def audit_mixed(p,row,a,baseline):
    k=int(a['teacher_first_declared']);prefix_assert(a,baseline,k)
    owner=np.arange(len(a['states']))>=k
    assert np.array_equal(a['ownership'],owner) and a['stochastic'][:k].all() and not a['stochastic'][k:].any()
    assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    for i,(pose,scan) in enumerate(zip(a['poses'],a['teacher_scans'])):
        u=Teacher.control('GAP',pose,a['task_goal_xy'],scan,a['teacher_vehicle_parameters'])
        assert np.array_equal(u,a['teacher_actions'][i])
        if owner[i]:assert np.array_equal(u,a['actions'][i])
    # The historical checker assumes entirely deterministic teacher episodes.
    # Check its unmodified per-step model/history/density/reward loop on the
    # recorded Actor proposals, then separately verify actual owned commands.
    proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32)
    assert np.array_equal(proposals[:k],a['actions'][:k])
    actor_view=dict(a,actions=proposals,ownership=np.zeros(len(owner),np.uint8))
    F.M.audit_full_history(p,dict(row,teacher=False),actor_view)


def main():
    p=read(OUT/'protocol.json');setup();verify(p,True);E.N.offline();E.L.install_stop_handlers()
    rt=Base.Runtime(p);state='FAILED';error=None;result=None
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        basevals=rt.request(p['jobs'],p['initial'],'HISTORICAL_NOISY_PREFIX_SOURCE_REBUILD',seeds=p['seeds'],full=True)
        for (a,row),old in zip(basevals,p['source_rows']):
            F.M.audit_full_history(p,row,a)
            assert row['array_hashes']==old['array_hashes'] and row['integration_sha256']==old['integration_sha256']
        jobs=[dict(j,gap_recovery_first=k) for j,k in zip(p['jobs'],p['teacher_firsts'])]
        vals=rt.request(jobs,p['initial'],'ACTUAL_FIXED_GAP_RECOVERY',seeds=p['seeds'],full=True)
        outcomes=[]
        for index,((a,row),(baseline,base_row)) in enumerate(zip(vals,basevals)):
            audit_mixed(p,row,a,baseline)
            outcomes.append(dict(job_id=row['job_id'],teacher_first=p['teacher_firsts'][index],
                prefix_seconds=float(baseline['native_counts'][:p['teacher_firsts'][index]].sum()*.002),
                safe_goal=bool(row['success'] and not row['collision'] and row['field_detail']['field_qualified_success']),
                collision=row['collision'],timeout=row['timeout'],remaining_m=row['remaining_m'],row=row,
                autonomous_student_success=False))
        rt.request(jobs,p['initial'],'EXACT_FIXED_GAP_RECOVERY',seeds=p['seeds'],full=True,reference=[r for a,r in vals])
        result=dict(outcomes=outcomes,qualified_success_data=[x['row'] for x in outcomes if x['safe_goal']],
            new_Actor=0,new_Adam=0,new_RL=0,model_unchanged=True,new_autonomous_student_evaluation=False,
            old275=None,six54=None,eight54=None,full407=None,fresh16=None,reload78=None,
            teacher_goal_NOT_student_navigation_success=True,globally_promoted=False)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');setup();verify(p,True);G.stopped(OUT)
    assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==8 and len(reps)==4
    arrays=[]
    for row in rows:
        a=E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256'] and int(a['native_counts'].sum())==row['physics']
        assert all(digest(v)==row['array_hashes'][key] for key,v in a.items());arrays.append(a)
    for i in range(4):
        F.M.audit_full_history(p,rows[i],arrays[i])
        assert rows[i]['array_hashes']==p['source_rows'][i]['array_hashes']
        assert rows[i]['integration_sha256']==p['source_rows'][i]['integration_sha256']
        audit_mixed(p,rows[4+i],arrays[4+i],arrays[i])
        item=s['result']['outcomes'][i]
        assert item['safe_goal']==bool(rows[4+i]['success'] and not rows[4+i]['collision'] and rows[4+i]['field_detail']['field_qualified_success'])
    by={r['path']:r for r in rows}
    for row in reps:
        assert row['exact'] and row['array_hashes']==by[row['reference_path']]['array_hashes']
        assert row['integration_sha256']==by[row['reference_path']]['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
    assert s['commands']==16<=p['max_commands'] and not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,all8_full_history_density_reward_and_teacher_ownership_verified=True,
        actual_noisy_prefix_and_boundary_exact=True,all4_teacher_native_replays_exact=True,
        baseline4_historical_repeats_NOT_independent=True,all_models_unchanged=True,physics=s['physics'],commands=s['commands']))
    print('T583_AUDIT_PASS',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
