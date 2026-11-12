"""Frozen full-field transfer probe with genuine reset/history contracts.

Eight new start sectors, three destination sectors each. Static geometry is
used only for reset qualification; deployment still receives no map/planner.
"""
from pathlib import Path
import copy,fcntl,math,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'));import speed_runtime as R
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t502_fullfield_probe'
BASE_SETUP=G.B.BASE_SETUP;BASE_TASK=E.N.K.R.task_env;BASE_SESSION=G.T.Session

def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
R.OUT=OUT;R.HERE=HERE;R.verify=verify

def field_setup(folder,p):
    module,native,create,w,reward=BASE_SETUP(folder,E.N.science(p))
    def task_env(create,job):
        factory_job=copy.deepcopy(job)
        if 'fullfield_start_pose' in job:factory_job['case']['distance_m']=6.
        env,obs,actual=BASE_TASK(create,factory_job)
        if 'fullfield_start_pose' not in job:return env,obs,actual
        import mujoco
        pose=np.asarray(job['fullfield_start_pose'],float);goal=np.asarray(job['fullfield_goal_xy'],float)
        assert np.isfinite(pose).all() and np.isfinite(goal).all() and np.abs(pose[:2]).max()<7 and np.abs(goal).max()<7
        # New episode reset, never a mid-trajectory teleport or teacher route.
        env.data.qpos[:2]=pose[:2];env.data.qpos[3:7]=[math.cos(pose[2]/2),0.,0.,math.sin(pose[2]/2)]
        env.data.qvel[:]=0.;env.data.ctrl[:]=0.;env.data.time=0.;env.step_count=0;env.prev_action=np.zeros(2,np.float32)
        mujoco.mj_forward(env.model,env.data);env.history.clear();obs=np.asarray(env._observation(update_history=False),np.float32)
        before=E.N.K.R.integration(env).copy();obs=w.alternate(env,goal,w.get(env));assert np.array_equal(before,E.N.K.R.integration(env))
        hp,hg,ht=w.get(env);assert np.max(np.abs(hp-pose))<1e-10 and np.all(hg==goal) and np.max(abs(ht))<1e-10
        assert not env._collision();env.max_steps=job['case']['step_cap'];return env,obs,actual
    E.N.K.R.task_env=task_env
    return module,native,create,w,reward

class FieldSession(BASE_SESSION):
    def __init__(self,job,native,create,witness,guard,physics,q=None):
        self.env,self.obs,_=E.N.K.R.task_env(create,job);self.obs=E.N.K.C.perturb_anchor(self.env,self.obs,job)
        self.w,self.native,self.q,self.guard,self.physics=witness,native,q,guard,physics
        self.steps=0;self.previous=np.zeros(2,np.float32);self.pn=0;self.hidden=torch.zeros((1,1,64),dtype=torch.float64)
        self.rows=[];self.hist=[];self.hb=[];self.ha=[];self.proposal=[];self.owner=[];self.digests=[]
        self.term=self.trunc=False;self.info={};self.cap=job['case']['step_cap'];self.goal=self.env.goal_xy.copy();self.scene=E.N.K.C.model_hash(self.env);self.initial=E.N.K.R.integration(self.env).copy()
        assert not self.env._collision() and 0<self.cap<=90000 and len(self.initial)==203
        if q is not None:q.reset_noise()

def worker(pipe,index,p,counts):
    G.T.Session=FieldSession;G.B.native_setup=field_setup;P.frozen=L.frozen
    G.cpu_worker(pipe,index,p,counts)

class Pool(R.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';old=read(f) if f.exists() else [];old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False));save(f,old)
        except BaseException:self.close();raise

def detail(p,row,a):
    dt=a['native_counts']*.002;t=np.r_[0.,np.cumsum(dt)];poses=np.vstack([a['poses'],a['next_poses'][-1]]);d=np.linalg.norm(poses[:,:2]-a['task_goal_xy'],axis=1)
    speed=np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1)/dt;radial=(d[:-1]-d[1:])/dt
    angle=np.arctan2(a['task_goal_xy'][1]-a['poses'][:,1],a['task_goal_xy'][0]-a['poses'][:,0]);heading=np.abs(np.arctan2(np.sin(angle-a['poses'][:,2]),np.cos(angle-a['poses'][:,2])))*180/np.pi
    boundary=bool((np.abs(poses[:,:2])>p['field_limit_m']).any());n=min(100,len(dt))
    return dict(id=row['job_id'],native_success=row['success'],collision=row['collision'],timeout=row['timeout'],seconds=float(t[-1]),remaining_m=row['remaining_m'],min_decision_distance_m=float(d.min()),closest_time_s=float(t[d.argmin()]),mean_speed_m_s=float(np.average(speed,weights=dt)),last20_heading_deg=float(np.average(heading[-n:],weights=dt[-n:])),moving_away_seconds=float(dt[radial<-.001].sum()),field_boundary_crossed=boundary,field_qualified_success=bool(row['success'] and not row['collision'] and not boundary),terminal_contacts=row['terminal_contacts'])

def main(phase):
    assert phase=='probe';folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['field_jobs'],p['initial'],'FROZEN_FULLFIELD24',full=True);rows=[r for a,r in vv]
        metrics=[detail(p,r,a) for a,r in vv];save(folder/'diagnosis.json',metrics)
        rt.request(p['field_jobs'],p['initial'],'EXACT_FULLFIELD24',full=True,references=rows)
        result=dict(model=p['initial'],native_tally=G.R.M.tally(rows),field_qualified_goals=sum(m['field_qualified_success'] for m in metrics),boundary_crossed=sum(m['field_boundary_crossed'] for m in metrics),all24_field_pass=all(m['field_qualified_success'] for m in metrics),distance_groups={str(i):G.R.M.tally([r for r in rows if r['request']['job']['sector_offset']==i]) for i in [1,2,3]},Actor_Adam=0,Value_Adam=0,prior_old275=p['locked_best']['old275'],prior_wide6=p['locked_best']['wide6'],prior_wide8=p['locked_best']['wide8'],fullfield_all_possible_tasks_NOT_proven=True,fresh16=None)
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder);P.frozen=L.frozen
    rows,rep=G.rows(folder),G.rows(folder,'replays.jsonl');assert len(rows)==len(rep)==24;metrics=[]
    for r in rows:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
        job=r['request']['job'];assert np.max(abs(a['poses'][0]-job['fullfield_start_pose']))<1e-10 and np.array_equal(a['task_goal_xy'],job['fullfield_goal_xy']) and np.array_equal(a['initial_native_integration'],np.asarray(p['geometry_contract'][r['job_id']]['initial_integration']))
        assert not a['hidden_before'][0].any();metrics.append(detail(p,r,a))
    assert metrics==read(folder/'diagnosis.json');by={r['path']:r for r in rows}
    for r in rep:
        ref=by[r['reference_path']];assert r['exact'] and r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rows+rep) and s['commands']==52 and sha(p['initial']['path'])==p['initial']['sha256']
    assert digest(P.checkpoint(p['initial']['path'])['joint_state'])==p['initial']['joint_hash']
    assert s['result']['native_tally']==G.R.M.tally(rows) and s['result']['field_qualified_goals']==sum(m['field_qualified_success'] for m in metrics)
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,full24_reset_history_hidden_density_rewards_and24exact=True,model_and_joint_unchanged=True,physics=s['physics'],commands=s['commands']))
    print('T502_FULLFIELD_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
