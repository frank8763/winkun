"""Predeclared offline Stage night, shared-policy isolated native workers."""
from pathlib import Path
import sys,multiprocessing as mp,contextlib,resource,fcntl
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t398_warm_recovery_rl'))
import t398_learning as L
from t398_learning import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t402r_offline_night';DOC=ROOT/'stage/docs/overnight_20261106'
SAFE_PARENT=ROOT/'stage/runs/t393_safe_demo_pg/candidate.pt.gz'
WARM_PARENT=ROOT/'stage/runs/t398_warm_recovery_rl/warm/proposed.pt.gz'

def load_plan():return read(OUT/'protocol.json')
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes']);return N.science(p)
def fixed_actor(base,p,model):
    q=Actor(base,p).eval();q.load_learned(N.S.read_model(model));q.sample=False;q.reset_noise();return q
def new_learner(base,p,model):
    q=fixed_actor(base,p,model);params=N.trainable(q);assert sum(x.numel() for x in params)==6864
    return q,torch.optim.Adam(params,lr=p['lr'])
def payload(s):
    a=s.arrays();a.update(hold_sha256=np.asarray(s.digests,np.uint8),ownership=np.asarray(s.owner,np.uint8),task_goal_xy=s.goal.copy(),scene_sha256=np.frombuffer(bytes.fromhex(s.scene),np.uint8).copy());return a
def audit_full(q,a,first,seed,p):
    h=None;q.reset_noise();q.rng=None if seed is None else np.random.default_rng(np.random.SeedSequence(seed));ref=N.S.arrays(REFERENCE)
    assert equal(a['actions'][:first],ref['actions'][:first])
    expected=np.zeros(len(a['states']),np.uint8);expected[:first]=1;assert equal(expected,a['ownership'])
    with torch.no_grad():
        for i,x in enumerate(a['states']):
            q.sample=seed is not None and i>=first
            hb=np.zeros(16) if h is None else h[0,0].numpy().copy()
            cmd,_,h,_=q.step(torch.tensor(x[None]),h)
            assert equal(hb,a['hidden_before'][i])
            for k,v in q.tape[-1].items():assert equal(np.asarray(v),a[k][i]),(i,k)
            if i>=first:assert equal(cmd[0].numpy(),a['actions'][i])
            assert np.abs(N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
            n=int(a['native_counts'][i]);z=N.K.R.GAMMA**n;flags=a['end_flags'][i]
            d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy'])
            r=-.000002*(1-z)/(1-N.K.R.GAMMA)+.03*d0-z*.03*d1*(not flags.any())
            if flags.any():r+=N.K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
            assert abs(r-a['rewards'][i])<1e-12
    assert equal(h[0,0].numpy(),a['last_hidden_after']);q.sample=False

def worker(pipe,index,p,counts):
    """Every process owns its own MjData/history/hidden; no internet or learner."""
    N.offline();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    N.configure(OUT);base=N.science(p);q=fixed_actor(base,p,SAFE_PARENT).requires_grad_(False);prior=N.S.arrays(REFERENCE);s=w=None
    # Initialize Torch internals before prohibiting optimizer/backward hooks.
    def prohibited(*a,**k):raise AssertionError('Native worker cannot optimize')
    torch.optim.Adam.step=prohibited;torch.Tensor.backward=prohibited
    folder=OUT/p['worker_tag']/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('w');oldout,olderr=sys.stdout,sys.stderr
    def guard(physical=False):
        check_stop();assert time.time()<p['native_stop_epoch']-3 and sum(counts)<=p['max_physics']
    def charge():counts[index]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            m,native,create,w,_=N.S.P.C.B.setup(folder/'native',base)
            original=copy.deepcopy(m.ACTIVE);install_stop_handlers()
            pipe.send(dict(state='READY',pid=os.getpid(),cpu=index,native=0))
            while True:
                request=pipe.recv()
                if request is None:break
                if request.get('cancel_probe'):
                    def deny(physical=False):
                        if physical:raise StageStop('zero-native subprocess cancellation')
                    m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=p['centre_left']['circle_shift_x_m']
                    s=Session(p['centre_left'],native,create,w,deny,charge,q);install_stop_handlers();initial=s.initial.copy()
                    try:s.step();raise AssertionError('Subprocess cancellation missing')
                    except StageStop:pass
                    assert s.steps==0 and not s.rows and equal(initial,N.K.R.integration(s.env));s.close();s=None
                    pipe.send(dict(state='CANCEL_PROBE_PASS',physics=counts[index]));continue
                guard();before=counts[index];begin=time.perf_counter();q.load_learned(N.S.read_model(request['model']));q.reset_noise();q.sample=False
                q.rng=None if request['seed'] is None else np.random.default_rng(np.random.SeedSequence(request['seed']))
                j=request['job'];m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
                s=Session(j,native,create,w,guard,charge,q);install_stop_handlers()
                for i in range(request['first']):
                    row,_=s.step(prior['actions'][i],owner=1);assert not s.done()
                    for k,name in enumerate(N.R.NAMES):assert equal(np.asarray(row[k]),prior[name][i]),(i,name)
                q.sample=request['seed'] is not None
                if request.get('hold_limit'):
                    for i in range(request['hold_limit']):s.step(owner=0);assert not s.done()
                else:
                    while not s.done():s.step(owner=0)
                a=payload(s);r=s.result();r.update(policy_hash=q.phash(),physics=counts[index]-before,array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),elapsed_s=time.perf_counter()-begin,pid=os.getpid(),cpu=index,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                s.close();s=None;pipe.send((a,r))
    except BaseException:
        error=traceback.format_exc()
        if s is not None and s.rows:
            try:N.write_npz(folder/'partial.npz',N.S.packed(payload(s)),p['full_bytes'])
            except BaseException:save(folder/'partial_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=error,physics=counts[index]))
        except BaseException:pass
        raise
    finally:
        if s:s.close()
        if w:w.close()
        sys.stdout,sys.stderr=oldout,olderr
        if not log.closed:log.close()
        pipe.close()

class Pool:
    def __init__(self,p,n,guard,tag):
        p=dict(p,worker_tag=tag);self.p=p;self.guard=guard;self.n=n;ctx=mp.get_context('spawn');self.counts=ctx.RawArray('q',n);self.children=[];self.pipes=[];self.ready=[]
        for i in range(n):
            a,b=ctx.Pipe();child=ctx.Process(target=worker,args=(b,i,p,self.counts));child.start();b.close();self.children.append(child);self.pipes.append(a)
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]))
        for i in range(n):self.ready.append(self.receive(i))
        assert all(v['state']=='READY' for v in self.ready)
    def receive(self,index):
        begin=time.time()
        while not self.pipes[index].poll(.05):
            self.guard();assert self.children[index].is_alive(),'Worker died';assert time.time()-begin<180,'Worker response timed out'
        v=self.pipes[index].recv()
        if isinstance(v,dict) and v.get('state')=='FAILED':raise RuntimeError(v['error'])
        return v
    def request(self,items):
        assert len(items)<=self.n
        for i,item in enumerate(items):self.pipes[i].send(item)
        return [self.receive(i) for i in range(len(items))]
    def close(self):
        for child,pipe in zip(self.children,self.pipes):
            if child.is_alive():
                try:pipe.send(None)
                except (BrokenPipeError,EOFError):pass
        for child in self.children:
            child.join(5)
            if child.is_alive():child.terminate();child.join(1)
            if child.is_alive():child.kill();child.join()
        for pipe in self.pipes:pipe.close()
        assert all(not child.is_alive() for child in self.children)

def targets(q,items):
    groups=[]
    with torch.no_grad():
        for a,first in items:
            x=torch.tensor(a['states']);b=q.base_actions(x);groups.append(dict(x=x,b=b,target=q.sequence(x,b).detach().clone(),first=first,weight=1.))
    return groups
def consolidate_fit(q,opt,old,new,p,guard,event,steps=200):
    params=N.trainable(q);trace=[];immutable=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(steps):
        guard();opt.zero_grad(set_to_none=True);losses=[]
        for group in [old[k%len(old)],new[k%len(new)]]:
            pred=q.sequence(group['x'],group['b']);loss=(pred[group['first']:]-group['target'][group['first']:]).square().mean();loss.backward();losses.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
        assert digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})==immutable
        trace.append([*losses,norm])
    return dict(trace=trace,steps=steps,new_RL=0)
