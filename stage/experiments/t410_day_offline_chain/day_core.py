"""Sealed wide-route pilot: one Actor, frozen parent, zero-function new capacity.

The extra logit residual is part of the sole deployed Actor. Historical policies
only supply fixed learner targets; there is no runtime teacher or task routing.
This module has no import-time filesystem writes or simulator execution.
"""
from pathlib import Path
import sys,os,json,time,copy,shutil,signal,traceback,contextlib,multiprocessing as mp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t408_joint_wide'))
from t408_core import *
import t408_core as STAGE408
import t404_core as STAGE404

HERE=Path(__file__).resolve().parent
ROOT_OUT=ROOT/'stage/runs/t410_day_offline_chain'
OUT=ROOT_OUT
DOC=ROOT/'stage/docs/day_20261107'
EXTRA_SEED=20261107410

def set_out(out):
    global OUT
    OUT=Path(out)
    N.configure(OUT)

configure=set_out

class Actor(L.Actor):
    """Parent recurrent representation plus a trainable causal residual."""
    def __init__(self,base,p):
        super().__init__(base,p)
        # The parent can be the original T404 or an admitted extra-aware model.
        # Load original keys before constructing the wider residual; then load
        # any saved extra keys after construction. No stale adapter is merged.
        initial=N.S.read_model(p['initial_model'])
        original={k:v for k,v in initial.items() if not k.startswith('extra.')}
        self.load_learned(original)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(EXTRA_SEED)
            self.extra=torch.nn.Sequential(torch.nn.Linear(136,32),torch.nn.ReLU(),torch.nn.Linear(32,2)).double()
            torch.nn.init.zeros_(self.extra[-1].weight)
            torch.nn.init.zeros_(self.extra[-1].bias)
        if any(k.startswith('extra.') for k in initial):
            self.load_learned(initial)
        self.requires_grad_(False)
        self.extra.requires_grad_(True)
        self.sample=False
        self.reset_noise()
    def location_with_observation(self,b,h,s):
        joined=torch.cat([h,s],1)
        return 1.25*torch.tanh(torch.atanh(b/1.25)+self.head(h)+self.adapter(joined)+self.extra(joined))

def params(q):
    """Do not call historical N.trainable, which unfreezes the parent."""
    q.requires_grad_(False)
    q.extra.requires_grad_(True)
    result=list(q.extra.parameters())
    assert sum(v.numel() for v in result)==4450
    assert all(not v.requires_grad for k,v in q.named_parameters() if not k.startswith('extra.'))
    return result

def original_hash(q):
    return digest({k:v for k,v in q.state_dict().items() if not k.startswith('extra.')})

def learner(base,p,model=None):
    q=Actor(base,p).eval()
    if model is not None:
        state=N.S.read_model(model)
        # An original parent model leaves exactly the registered zero residual.
        if not any(k.startswith('extra.') for k in state):
            with torch.no_grad():
                q.extra[-1].weight.zero_();q.extra[-1].bias.zero_()
        q.load_learned(state)
    return q,torch.optim.Adam(params(q),lr=p['lr'])

def export(q,opt,path):
    buffer=io.BytesIO()
    torch.save(dict(state=L.state(q,opt),hashes=L.statehash(q,opt),policy_hash=q.phash(),memory_on=True,
                    architecture='frozen_T404_plus_136x32x2_zero_initialized_logit_residual',
                    deployed_actor_count=1,extra_seed=EXTRA_SEED,trainable_parameters=4450),buffer)
    data=gzip.compress(buffer.getvalue(),mtime=0)
    assert len(data)<=512*1024
    Path(path).write_bytes(data)
    return dict(path=str(path),sha256=sha(path),hashes=L.statehash(q,opt),policy_hash=q.phash())

def groups(q,old_arrays,p,broad):
    old=[];left=[]
    assert len(old_arrays)==8 and len(broad)==211 and {g['task'] for g in broad}==set(range(8))
    for a in old_arrays:
        x=torch.tensor(a['states'])
        old.append(dict(x=x,b=q.base_actions(x),target=torch.tensor(a['locations'])))
    for file in p['left_history_files']:
        a=N.S.arrays(file);x=torch.tensor(a['states'])
        assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        left.append(dict(x=x,b=q.base_actions(x),target=torch.tensor(a['locations'])))
    a=N.S.arrays(p['right_history_file']);x=torch.tensor(a['states'])
    assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
    right=dict(x=x,b=q.base_actions(x),target=torch.tensor(a['locations']))
    assert len(left)==5
    return dict(old=old,left=left,right=right,broad=broad)

def support_losses(q,g,p,k):
    """Equal total LEFT/RIGHT weights; task-balanced old broad references."""
    old=g['old'][k%8]
    options=[item for item in g['broad'] if item['task']==k%8]
    assert options
    broad=options[(k//8)%len(options)]
    left=g['left'][k%5];right=g['right']
    losses=[]
    for item,weight,target in [(old,p.get('old_weight',1.),old['target']),
                               (broad,p.get('broad_old_weight',1.),broad['b']),
                               (left,p.get('left_weight',.5),left['target']),
                               (right,p.get('right_weight',.5),right['target'])]:
        losses.append((q.sequence(item['x'],item['b'])-target).square().mean()*weight)
    assert p.get('left_weight',.5)==p.get('right_weight',.5)
    return losses

def gradient(loss,variables):
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(loss,variables)])

def protect(vector,reference):
    """One fixed half-space projection; not a nonlinear safety guarantee."""
    norm2=torch.dot(reference,reference)
    dot=torch.dot(vector,reference)
    if float(norm2)>1e-30 and float(dot)<0:
        return vector-dot/norm2*reference
    return vector

def install_gradient(variables,vector):
    offset=0
    for variable in variables:
        count=variable.numel();variable.grad=vector[offset:offset+count].reshape_as(variable).clone();offset+=count
    assert offset==vector.numel()

def auxiliary(q,g,p,k,projected=False,variables=None):
    variables=params(q) if variables is None else variables
    losses=support_losses(q,g,p,k)
    values=[float(loss.detach()) for loss in losses]
    # Separate graphs make all gradients exact for the same current parameters.
    old=gradient(losses[0]+losses[1],variables)
    left=gradient(losses[2],variables);right=gradient(losses[3],variables)
    pl=protect(left,old) if projected else left
    pr=protect(right,old) if projected else right
    correction=old.new_zeros(old.shape)
    corrections=g.get('corrections',[])
    correction_losses=[]
    for item in corrections:
        first=int(item['first']);assert 0<=first<len(item['x'])
        target=item['target'] if 'target' in item else item['actions']
        if not isinstance(target,torch.Tensor):target=torch.tensor(target,dtype=torch.float64)
        prediction=q.sequence(item['x'],item['b'])
        loss=(prediction[first:]-target[first:]).square().mean()*p.get('correction_weight',.5)/len(corrections)
        current=gradient(loss,variables)
        correction+=protect(current,old) if projected else current
        correction_losses.append(float(loss.detach()))
    detail=dict(old_norm=float(old.norm()),left_norm=float(left.norm()),right_norm=float(right.norm()),
                old_left_dot=float(torch.dot(old,left)),old_right_dot=float(torch.dot(old,right)),
                protected_old_left_dot=float(torch.dot(old,pl)),protected_old_right_dot=float(torch.dot(old,pr)),
                left_right_dot=float(torch.dot(left,right)),old_task=k%8,left_history_index=k%5,
                projection_enabled=bool(projected),correction_norm=float(correction.norm()),
                correction_losses=correction_losses,correction_teacher_prefix_masked=True)
    return old+pl+pr+correction,values,detail,old

def fit(q,opt,g,p,guard,event,steps=None,projected=False):
    count=p.get('consolidation_steps',p.get('fit_steps',600)) if steps is None else steps
    assert 0<count<=600
    variables=params(q);fixed=original_hash(q);trace=[];components=[]
    for k in range(count):
        guard();opt.zero_grad(set_to_none=True)
        vector,losses,detail,_=auxiliary(q,g,p,k,projected,variables)
        install_gradient(variables,vector)
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append([*losses,norm]);components.append(detail)
        assert all(v.grad is None for name,v in q.named_parameters() if not name.startswith('extra.'))
        if k%64==0:assert original_hash(q)==fixed
    assert original_hash(q)==fixed
    return dict(trace=trace,components=components,steps=count,new_RL=0,projected=bool(projected),
                trainable_parameters=4450,original_frozen_hash=fixed,
                left_weight=p.get('left_weight',.5),right_weight=p.get('right_weight',.5))

def fit_A(q,opt,g,p,guard,event,steps=None):
    return fit(q,opt,g,p,guard,event,steps,False)

def fit_B(q,opt,g,p,guard,event,steps=None):
    return fit(q,opt,g,p,guard,event,steps,True)

def ppo_fit(q,opt,items,g,p,rnd,guard,event,projected=False):
    batch,returns,errors=L.blocks(q,items,0,p)
    variables=params(q);fixed=original_hash(q);trace=[];components=[]
    for k in range(p.get('epochs',8)):
        guard();opt.zero_grad(set_to_none=True);pg=0.;pg_vector=torch.zeros(sum(v.numel() for v in variables),dtype=torch.float64)
        for x,b,raw,atom,oldlog,adv in batch:
            ratio=(L.logp(q.sequence(x,b),raw,atom,p)-oldlog).exp()
            coefficient=adv/(4*400)
            loss=-torch.minimum(ratio*coefficient,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*coefficient).sum()
            pg+=float(loss.detach());pg_vector+=gradient(loss,variables)
        aux,losses,detail,old=auxiliary(q,g,p,rnd*p.get('epochs',8)+k,projected,variables)
        adjusted=protect(pg_vector,old) if projected else pg_vector
        detail.update(pg_norm=float(pg_vector.norm()),projected_pg_norm=float(adjusted.norm()),
                      old_pg_dot=float(torch.dot(old,pg_vector)),protected_old_pg_dot=float(torch.dot(old,adjusted)))
        install_gradient(variables,adjusted+aux)
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append([pg,*losses,norm]);components.append(detail)
        assert original_hash(q)==fixed
    with torch.no_grad():
        maximum=max(float((L.logp(q.sequence(x,b),raw,atom,p)-oldlog).abs().max()) for x,b,raw,atom,oldlog,_ in batch)
    return dict(trace=trace,gradient_components=components,returns=returns.tolist(),density_errors=errors,
                post_max_abs_logratio=maximum,numeric_accept=bool(np.isfinite(maximum) and maximum<=p['max_logratio']),
                original_frozen_hash=fixed,trainable_parameters=4450,projected=bool(projected))

def native_worker(pipe,index,p,counts):
    """Same validated Stage plant/session; only the constructed Actor differs."""
    N.offline();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    set_out(p['out']);base=N.science(p);q=Actor(base,p).eval().requires_grad_(False)
    session=witness=None;folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('w');stdout,stderr=sys.stdout,sys.stderr
    def prohibited(*a,**k):raise AssertionError('Native worker cannot optimize')
    torch.optim.Adam.step=prohibited;torch.Tensor.backward=prohibited
    def guard(physical=False):
        L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():
        counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,witness,_=N.S.P.C.B.setup(folder/'native',base)
            original=copy.deepcopy(module.ACTIVE);L.install_stop_handlers()
            pipe.send(dict(state='READY',pid=os.getpid(),cpu=index,native=0))
            while True:
                request=pipe.recv()
                if request is None:break
                q.load_learned(N.S.read_model(request['model']));q.reset_noise();q.sample=False
                q.rng=None if request['seed'] is None else np.random.default_rng(np.random.SeedSequence(request['seed']))
                job=request['job'];module.ACTIVE=copy.deepcopy(original)
                module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m'])
                before=counts[index+1]
                session=L.Session(job,native,create,witness,guard,charge,q);L.install_stop_handlers()
                if request.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise L.StageStop('zero physical worker cancel')
                    session.guard=cancel
                    try:session.step();raise AssertionError('Cancellation missing')
                    except L.StageStop:pass
                    assert not session.rows and session.steps==0
                    session.close();session=None;pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                controls=request.get('controls')
                force_first=int(request.get('force_first',0))
                if controls is not None:
                    controls=np.asarray(controls,np.float32)
                    assert controls.ndim==2 and controls.shape[1]==2 and len(controls)>=400
                    assert np.isfinite(controls).all() and np.abs(controls).max()<=1 and 0<=force_first<400
                    assert request['seed'] is None,'Forced teacher data must not be called on-policy'
                while not session.done():
                    index_now=len(session.rows)
                    q.sample=request['seed'] is not None and index_now>=request.get('first',0)
                    if controls is not None and index_now>=force_first:
                        session.step(controls[index_now],owner=1,actor_active=True)
                    else:session.step(owner=0)
                arrays=STAGE404.payload(session);row=session.result()
                row.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(arrays),
                           integration_sha256=digest(arrays['final_native_integration']),cpu=index,pid=os.getpid())
                session.close();session=None;pipe.send((arrays,row))
    except BaseException:
        if session is not None and session.rows:
            try:N.write_npz(folder/'partial.npz',N.S.packed(STAGE404.payload(session)),p['full_bytes'])
            except BaseException:save(folder/'partial_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if session:session.close()
        if witness:witness.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()

class Pool(STAGE404.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];context=mp.get_context('spawn')
        set_out(p['out'])
        try:
            for index in range(4):
                parent,child=context.Pipe();process=context.Process(target=native_worker,args=(child,index,p,counts))
                process.start();child.close();self.children.append(process);self.pipes.append(parent)
            save(OUT/'workers.json',dict(parent=os.getpid(),workers=[process.pid for process in self.children]))
            self.ready=[self.receive(index) for index in range(4)]
            assert all(row['state']=='READY' for row in self.ready)
        except BaseException:self.close();raise

audit_array=STAGE404.audit_array

def audit_mixed(q,a,p,first):
    assert 0<=first<len(a['states'])
    assert not a['stochastic'].any()
    assert not a['ownership'][:first].any() and np.all(a['ownership'][first:]==1)
    STAGE404.audit_array(q,a,p,0,None)
