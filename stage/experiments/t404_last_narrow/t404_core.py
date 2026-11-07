"""Last authorized narrow-scene attempt; teacher only in training."""
from pathlib import Path
import sys,os,json,time,copy,hashlib,io,gzip,math,shutil,signal,subprocess,traceback,fcntl,contextlib,resource,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t398_warm_recovery_rl'))
import t398_learning as L
from t398_learning import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t404_last_narrow';DOC=ROOT/'stage/docs/review_20261107'
PARENT=ROOT/'stage/runs/t402r_offline_night/candidate.pt.gz'
PRIOR=ROOT/'stage/runs/t402r_offline_night'

def plan():return read(OUT/'protocol.json')
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes']);return N.science(p)
def actor(base,p,model=PARENT):
    q=L.Actor(base,p).eval();q.load_learned(N.S.read_model(model));q.sample=False;q.reset_noise();return q
def optimizer(q,p):return torch.optim.Adam(N.trainable(q),lr=p['lr'])
def learner(base,p,model=PARENT):
    q=actor(base,p,model);return q,optimizer(q,p)
def payload(s):
    a=s.arrays();a.update(hold_sha256=np.asarray(s.digests,np.uint8),ownership=np.asarray(s.owner,np.uint8),task_goal_xy=s.goal.copy(),scene_sha256=np.frombuffer(bytes.fromhex(s.scene),np.uint8).copy());return a
def old_groups(q,p):
    gs=[]
    with torch.no_grad():
        for f in p['old_history_files']:
            a=N.S.arrays(f);x=torch.tensor(a['states']);b=q.base_actions(x)
            gs.append(dict(x=x,b=b,target=q.sequence(x,b).detach().clone(),first=0,weight=1.))
    return gs
def correction_groups(q,cohorts,p):
    gs=L.datasets(q)
    for g in gs:g.update(weight=.25,stop=len(g['x']))
    for c in cohorts:
        a=N.S.arrays(c['full_path']);first=c['first'];assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        assert np.all(a['ownership'][:first]==(1 if c['context']=='goal' else 0)) and np.all(a['ownership'][first:]==1)
        x=torch.tensor(a['states']);gs.append(dict(x=x,b=q.base_actions(x),y=torch.tensor(a['actions'],dtype=torch.float64),first=first,stop=len(x),weight=.5/len(cohorts)))
    return gs
def targets(q,cohorts):
    gs=[]
    with torch.no_grad():
        for c in cohorts:
            a=N.S.arrays(c['full_path']);x=torch.tensor(a['states']);b=q.base_actions(x);gs.append(dict(x=x,b=b,target=q.sequence(x,b).detach().clone(),first=c['first'],weight=1.))
    return gs
def consolidate(q,opt,old,new,p,guard,event):
    trace=[];params=N.trainable(q)
    for k in range(p['consolidation_steps']):
        guard();opt.zero_grad(set_to_none=True);losses=[]
        for g in [old[k%len(old)],new[k%len(new)]]:
            pred=q.sequence(g['x'],g['b']);f=g['first'];loss=(pred[f:]-g['target'][f:]).square().mean();loss.backward();losses.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([*losses,norm])
    return dict(trace=trace,steps=p['consolidation_steps'],new_RL=0)
def select_first(a,reference,p,context='origin'):
    if context=='goal':return p['goal_prefix_hold']
    n=min(len(a['poses']),len(reference['poses']));distance=np.linalg.norm(a['poses'][:n,:2]-reference['poses'][:n,:2],axis=1)
    yaw=np.abs((a['poses'][:n,2]-reference['poses'][:n,2]+np.pi)%(2*np.pi)-np.pi)
    eligible=np.flatnonzero((distance>p['deviation_m'])|(yaw>math.radians(p['deviation_deg'])))
    # A descriptive, predeclared query-location heuristic, not a safety label.
    end=max(1,len(a['poses'])-p['before_contact_holds']);return min(int(eligible[0]) if len(eligible) else end,end)
def teacher_controls(theta,reference,first,p):return L.ORIGINAL.teacher_controls(theta,reference,dict(p,teacher_start_hold=first))
def teacher_candidates(mean,sd,best,cycle,generation,p):return L.ORIGINAL.teacher_candidates(mean,sd,best,cycle,generation,p)
def objective(u,n,flags,remaining):return L.ORIGINAL.objective(u,n,flags,remaining)
def search_buffer(first):
    return L.ORIGINAL.teacher_buffer(dict(teacher_start_hold=first))
def audit_array(q,a,p,first=0,seed=None):
    q.reset_noise();q.rng=None if seed is None else np.random.default_rng(np.random.SeedSequence(seed));h=None
    with torch.no_grad():
        for i,x in enumerate(a['states']):
            q.sample=bool(a['stochastic'][i]);assert not q.sample or seed is not None
            hb=np.zeros(16) if h is None else h[0,0].numpy().copy();cmd,_,h,_=q.step(torch.tensor(x[None]),h)
            assert equal(hb,a['hidden_before'][i])
            for k,v in q.tape[-1].items():assert equal(np.asarray(v),a[k][i]),(i,k)
            if not a['ownership'][i]:assert equal(cmd[0].numpy(),a['actions'][i])
            assert np.abs(N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
            n=int(a['native_counts'][i]);z=N.K.R.GAMMA**n;f=a['end_flags'][i];d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy'])
            rw=-.000002*(1-z)/(1-N.K.R.GAMMA)+.03*d0-z*.03*d1*(not f.any())
            if f.any():rw+=N.K.R.GAMMA**(n-1)*(float(f[1])-float(f[0])-.5*float(f[2]))
            assert abs(rw-a['rewards'][i])<1e-12
    assert equal(h[0,0].numpy(),a['last_hidden_after']);q.sample=False
    if seed is not None:assert not a['stochastic'][:first].any() and a['stochastic'][first:].all() and not a['ownership'].any()

def native_worker(pipe,index,p,counts):
    N.offline();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True);N.configure(OUT);base=N.science(p)
    q=actor(base,p).requires_grad_(False);s=w=None;folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('w');stdout,stderr=sys.stdout,sys.stderr
    def prohibited(*a,**k):raise AssertionError('Native worker cannot optimize')
    torch.optim.Adam.step=prohibited;torch.Tensor.backward=prohibited
    def guard(physical=False):L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            m,native,create,w,_=N.S.P.C.B.setup(folder/'native',base);original=copy.deepcopy(m.ACTIVE);L.install_stop_handlers();pipe.send(dict(state='READY',pid=os.getpid(),cpu=index,native=0))
            while True:
                req=pipe.recv()
                if req is None:break
                q.load_learned(N.S.read_model(req['model']));q.reset_noise();q.sample=False;q.rng=None if req['seed'] is None else np.random.default_rng(np.random.SeedSequence(req['seed']))
                j=req['job'];m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m']);before=counts[index+1]
                s=L.Session(j,native,create,w,guard,charge,q);L.install_stop_handlers()
                if req.get('cancel_probe'):
                    old_guard=s.guard
                    def cancel(physical=False):
                        if physical:raise L.StageStop('zero physical worker cancel')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Cancellation missing')
                    except L.StageStop:pass
                    assert not s.rows and s.steps==0;s.close();s=None;pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                while not s.done():
                    q.sample=req['seed'] is not None and len(s.rows)>=req['first'];s.step(owner=0)
                a=payload(s);r=s.result();r.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),cpu=index,pid=os.getpid())
                s.close();s=None;pipe.send((a,r))
    except BaseException:
        if s is not None and s.rows:
            try:N.write_npz(folder/'partial.npz',N.S.packed(payload(s)),p['full_bytes'])
            except BaseException:save(folder/'partial_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if s:s.close()
        if w:w.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()

class Pool:
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();child=ctx.Process(target=native_worker,args=(b,i,p,counts));child.start();b.close();self.children.append(child);self.pipes.append(a)
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]))
        self.ready=[self.receive(i) for i in range(4)];assert all(v['state']=='READY' for v in self.ready)
    def receive(self,i):
        start=time.time()
        while not self.pipes[i].poll(.05):self.guard();assert self.children[i].is_alive() and time.time()-start<300,'Worker died or stalled'
        x=self.pipes[i].recv()
        if isinstance(x,dict) and x.get('state')=='FAILED':raise RuntimeError(x['error'])
        return x
    def request(self,items):
        assert 1<=len(items)<=4
        for i,x in enumerate(items):self.pipes[i].send(x)
        return [self.receive(i) for i in range(len(items))]
    def close(self):
        for c,p in zip(self.children,self.pipes):
            if c.is_alive():
                try:p.send(None)
                except (EOFError,BrokenPipeError):pass
        for c in self.children:
            c.join(5)
            if c.is_alive():c.terminate();c.join(1)
            if c.is_alive():c.kill();c.join()
        for p in self.pipes:p.close()
        assert all(not c.is_alive() for c in self.children)
