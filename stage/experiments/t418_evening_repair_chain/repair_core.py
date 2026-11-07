"""Safe-function student, expert-query likelihood BC and separately sampled PPO.

Fixed phase repair trains extra only; protect/rl train encoder/core/extra. Base/head/adapter always frozen. A frozen LEFT policy may
generate training labels, but autonomous deployment executes one student Actor.
"""
from pathlib import Path
import sys,os,json,time,copy,contextlib,traceback,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t410_day_offline_chain'))
from day_core import *
import day_core as DAY_CORE
import t406_core as LEFT_EXPERT
import t404_core as NATIVE_AUDIT
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t418_evening_repair_chain'
DOC=ROOT/'stage/docs/day_20261107'
TEACHER_KEYS=['teacher_actions','teacher_locations','teacher_hidden_before','teacher_hidden_after']

def capacity(out,limit):
    assert N.K.S.used(out)<=limit and N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30

N.capacity=capacity

def set_out(out):
    global OUT
    OUT=Path(out);N.configure(OUT)

configure=set_out

class Actor(DAY_CORE.Actor):
    def __init__(self,base,p):
        super().__init__(base,p)
        # Build the common safe-function topology, then load the explicitly
        # registered research candidate. The safe deployment model is untouched.
        with torch.no_grad():
            self.head.weight.zero_();self.head.bias.zero_()
            self.adapter[-1].weight.zero_();self.adapter[-1].bias.zero_()
            self.extra[-1].weight.zero_();self.extra[-1].bias.zero_()
        self.sample=False;self.reset_noise()
        if p.get('student_start_model'):self.load_learned(N.S.read_model(p['student_start_model']))
        self.requires_grad_(False)
        for name,v in self.named_parameters():v.requires_grad_(name.startswith(trained_prefixes(self)))

TRAINED_PREFIXES=('encoder.','core.','extra.')
def trained_prefixes(q):return ('extra.',) if q.contract['phase']=='repair' else TRAINED_PREFIXES
def params(q):
    q.requires_grad_(False)
    for name,v in q.named_parameters():v.requires_grad_(name.startswith(trained_prefixes(q)))
    result=[v for v in q.parameters() if v.requires_grad]
    assert result and all(v.requires_grad==name.startswith(trained_prefixes(q)) for name,v in q.named_parameters())
    return result
qparams=params
def original_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith(trained_prefixes(q))})

def safe_initial(base,p):return Actor(base,p).eval()
actor=safe_initial

def learner(base,p,model=None):
    q=safe_initial(base,p)
    if model is not None:q.load_learned(N.S.read_model(model))
    return q,torch.optim.Adam(params(q),lr=p['lr'])

def export(q,opt,path):
    buffer=io.BytesIO()
    torch.save(dict(state=L.state(q,opt),hashes=L.statehash(q,opt),policy_hash=q.phash(),memory_on=True,
        architecture='safe_T255_function_plus_adaptable_student_encoder_core_and_extra',
        deployed_actor_count=1,trainable_parameters=sum(v.numel() for v in q.parameters() if v.requires_grad),trainable_prefixes=trained_prefixes(q),
        teacher_at_deployment=False),buffer)
    data=gzip.compress(buffer.getvalue(),mtime=0);assert len(data)<=512*1024
    Path(path).write_bytes(data)
    return dict(path=str(path),sha256=sha(path),hashes=L.statehash(q,opt),policy_hash=q.phash())

def atoms(actions):
    actions=torch.as_tensor(actions)
    assert bool(torch.isfinite(actions).all()) and float(actions.abs().max())<=1.
    return torch.where(actions>=1,1,torch.where(actions<=-1,-1,0)).to(torch.int8)

def labelled(q,x,raw,actions):
    x=torch.as_tensor(x);raw=torch.as_tensor(raw,dtype=torch.float64)
    action=torch.as_tensor(actions);assert raw.shape==action.shape==(len(x),2)
    atom=atoms(action)
    assert torch.equal(atom.abs()==1,action.abs()==1)
    # Interior raw labels retain the actual teacher location before float32
    # conversion. Boundary labels use the existing exact tail probability.
    return dict(x=x,b=q.base_actions(x),raw=raw.detach().clone(),atom=atom.detach().clone())

def query_group(q,a):
    assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
    assert all(key in a for key in TEACHER_KEYS)
    result=labelled(q,a['states'],a['teacher_locations'],a['teacher_actions'])
    result.update(query_all_actual_history=True,first=0,ownership_does_not_mask_expert_query=True)
    return result

def build_groups(q,initial_arrays,p,broad=None):
    assert len(initial_arrays)==14
    if broad is None:broad=N.old_data(q,N.science(p),lambda:None)
    assert len(broad)==211 and {item['task'] for item in broad}==set(range(8))
    anchors=[];old=[]
    for a,row in initial_arrays:
        if not row['success']:continue
        item=labelled(q,a['states'],a['locations'],a['actions']);item.update(id=row['job_id'],own_initial_success=True)
        anchors.append(item)
        if row['job_id'] in {j['id'] for j in p['old8_jobs']}:old.append(item)
    assert anchors
    old_broad=[]
    for item in broad:
        row=labelled(q,item['x'],item['b'],item['b'].clamp(-1,1).float());row.update(task=item['task'],id=item['id']);old_broad.append(row)
    left=[]
    for file in p['left_history_files']:
        a=N.S.arrays(file);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        left.append(labelled(q,a['states'],a['locations'],a['actions']))
    a=N.S.arrays(p['right_history_file']);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
    right=labelled(q,a['states'],a['locations'],a['actions'])
    return dict(old=old,anchors=anchors,broad=old_broad,left=left,right=right,corrections=[])

groups=build_groups

def nll(q,item,p):
    prediction=q.sequence(item['x'],item['b'])
    loss=-L.logp(prediction,item['raw'],item['atom'],p).mean()
    assert bool(torch.isfinite(loss));return loss

def old_kl(q,item,p):
    # KL between equal-variance latent Gaussians at this fixed history. It is
    # exactly zero at the safe parent and does not push saturated old labels.
    prediction=q.sequence(item['x'],item['b']);sd=torch.tensor(p['raw_sigma'],dtype=torch.float64)
    return ((prediction-item['raw']).square()/(2*sd.square())).mean()

def support_losses(q,g,p,k):
    # Every current old-eight history contributes at every optimizer step.
    # Broad historical support remains task balanced, but never substitutes
    # for those actual current safe-policy trajectories.
    old=sum(old_kl(q,item,p) for item in g['anchors'])/len(g['anchors'])*p.get('old_weight',1.)
    options=[item for item in g['broad'] if item['task']==k%8];assert options
    broad=old_kl(q,options[(k//8)%len(options)],p)*p.get('broad_old_weight',1.)
    left=nll(q,g['left'][k%5],p)*p.get('left_weight',.25)
    right=nll(q,g['right'],p)*p.get('right_weight',.25)
    queries=g.get('corrections',[])
    current=(sum(nll(q,item,p) for item in queries)/len(queries)*p.get('query_weight',1.)) if queries else old.new_zeros(())
    return [old,broad,left,right,current]

def protect_vector(vector,reference):
    norm2=torch.dot(reference,reference);dot=torch.dot(vector,reference)
    return vector-dot/norm2*reference if float(norm2)>1e-30 and float(dot)<0 else vector

def vector_gradient(loss,variables):
    return torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,variables)])

def install_vector(variables,vector):
    offset=0
    for v in variables:
        n=v.numel();v.grad=vector[offset:offset+n].reshape_as(v).clone();offset+=n
    assert offset==vector.numel()

def backward_auxiliary(q,losses,variables,pg=None):
    if q.contract['phase']=='protect':
        old=vector_gradient(losses[0]+losses[1],variables)
        projected=[protect_vector(vector_gradient(loss,variables),old) for loss in losses[2:] if loss.requires_grad]
        total=old+sum(projected,torch.zeros_like(old))
        if pg is not None:total+=protect_vector(pg,old)
        install_vector(variables,total)
    else:
        for loss in losses:
            if loss.requires_grad:loss.backward()


def fit_bc(q,opt,g,p,guard,event,steps=None):
    count=p.get('bc_steps',200) if steps is None else steps;assert 0<count<=200
    assert g.get('corrections'),'BC requires actual complete verified expert-query data'
    variables=params(q);fixed=original_hash(q);trace=[]
    for k in range(count):
        guard();opt.zero_grad(set_to_none=True);losses=support_losses(q,g,p,k)
        values=[float(loss.detach()) for loss in losses]
        backward_auxiliary(q,losses,variables)
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append([*values,norm])
        assert all(v.grad is None for name,v in q.named_parameters() if not name.startswith(trained_prefixes(q)))
        if k%32==0:assert original_hash(q)==fixed
    assert original_hash(q)==fixed
    return dict(trace=trace,steps=count,new_RL=0,trainable_parameters=sum(v.numel() for v in variables),
        original_frozen_hash=fixed,own_initial_success_histories_each_step=True,expert_query_all_steps=True,
        query_groups=len(g['corrections']),likelihood='existing_float32_boundary_tail_and_continuous_raw_Gaussian')

def fit_ppo(q,opt,items,g,p,rnd,guard,event):
    batch,returns,errors=L.blocks(q,items,0,p);variables=params(q);fixed=original_hash(q)
    trace=[];components=[]
    for k in range(p.get('epochs',8)):
        guard();opt.zero_grad(set_to_none=True);pg=0.
        for x,b,raw,atom,oldlog,adv in batch:
            ratio=(L.logp(q.sequence(x,b),raw,atom,p)-oldlog).exp();coefficient=adv/(4*400)
            loss=-torch.minimum(ratio*coefficient,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*coefficient).sum()
            assert bool(torch.isfinite(loss));loss.backward();pg+=float(loss.detach())
        before=[v.grad.detach().clone() for v in variables]
        losses=support_losses(q,g,p,rnd*p.get('epochs',8)+k)
        if q.contract['phase']=='protect':
            pgflat=torch.cat([a.reshape(-1) for a in before]);backward_auxiliary(q,losses,variables,pgflat)
        else:backward_auxiliary(q,losses,variables)
        aux=[v.grad.detach()-a for v,a in zip(variables,before)]
        pg_norm=float(sum(v.square().sum() for v in before).sqrt());aux_norm=float(sum(v.square().sum() for v in aux).sqrt())
        dot=float(sum((a*b).sum() for a,b in zip(before,aux)))
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append([pg,*[float(loss.detach()) for loss in losses],norm])
        components.append(dict(pg_norm=pg_norm,aux_norm=aux_norm,pg_aux_cosine=None if not pg_norm*aux_norm else dot/(pg_norm*aux_norm)))
        assert original_hash(q)==fixed
    with torch.no_grad():maximum=max(float((L.logp(q.sequence(x,b),raw,atom,p)-oldlog).abs().max()) for x,b,raw,atom,oldlog,_ in batch)
    return dict(trace=trace,gradient_components=components,returns=returns.tolist(),density_errors=errors,
        post_max_abs_logratio=maximum,numeric_accept=bool(np.isfinite(maximum) and maximum<=p['max_logratio']),
        original_frozen_hash=fixed,trainable_parameters=sum(v.numel() for v in variables),teacher_data_only_auxiliary=True,own_initial_success_histories_each_step=True)

def expert(base,p):
    teacher=LEFT_EXPERT.embedded(base,p).eval().requires_grad_(False)
    teacher.load_learned(N.S.read_model(p['teacher_model']));teacher.sample=False;teacher.reset_noise()
    assert teacher.phash()==p['teacher_policy_hash']
    assert not torch.count_nonzero(teacher.adapter[-1].weight) and not torch.count_nonzero(teacher.adapter[-1].bias)
    return teacher

def payload(session,tape=None):
    result=NATIVE_AUDIT.payload(session)
    if tape is not None:
        assert len(tape)==len(result['states']);result.update({key:np.asarray([row[key] for row in tape]) for key in TEACHER_KEYS})
    return result

def native_worker(pipe,index,p,counts):
    N.offline();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    set_out(p['out']);base=N.science(p);student=safe_initial(base,p).requires_grad_(False);teacher=expert(base,p)
    session=witness=None;tape=None;folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('w');stdout,stderr=sys.stdout,sys.stderr
    def prohibited(*a,**k):raise AssertionError('Native worker cannot optimize')
    torch.optim.Adam.step=prohibited;torch.Tensor.backward=prohibited
    def guard(physical=False):L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,witness,_=N.S.P.C.B.setup(folder/'native',base);original=copy.deepcopy(module.ACTIVE);L.install_stop_handlers()
            pipe.send(dict(state='READY',pid=os.getpid(),cpu=index,native=0))
            while True:
                request=pipe.recv()
                if request is None:break
                student.load_learned(N.S.read_model(request['model']));student.reset_noise();student.sample=False
                student.rng=None if request['seed'] is None else np.random.default_rng(np.random.SeedSequence(request['seed']))
                teacher.reset_noise();teacher.sample=False;teacher_hidden=None
                query=bool(request.get('query',request.get('teacher',False)));case=request.get('case',{})
                first=int(case.get('first',request.get('first',p.get('teacher_first',20))))
                if query:assert request['seed'] is None and first in [0,p.get('teacher_first',20)]
                teacher_kind=case.get('teacher_kind','left_carry');assert teacher_kind in ['safe_reset','left_carry']
                job=request['job'];module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m'])
                before=counts[index+1];session=L.Session(job,native,create,witness,guard,charge,student);L.install_stop_handlers();tape=[] if query else None
                if request.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise L.StageStop('zero physical worker cancel')
                    session.guard=cancel
                    try:session.step();raise AssertionError('Cancellation missing')
                    except L.StageStop:pass
                    assert not session.rows and session.steps==0;session.close();session=None;pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                while not session.done():
                    hold=len(session.rows);student.sample=request['seed'] is not None and hold>=request.get('first',0)
                    if query:
                        x=session.packet().copy()
                        if teacher_kind=='left_carry':
                            hb=np.zeros(16) if teacher_hidden is None else teacher_hidden[0,0].numpy().copy()
                            with torch.no_grad():command,location,teacher_hidden,_=teacher.step(torch.tensor(x[None]),teacher_hidden)
                            ha=teacher_hidden[0,0].numpy().copy()
                        else:
                            # T255 base uses its original hidden-reset function per current120 observation.
                            # Empty arrays explicitly mean there is no carried teacher memory here.
                            hb=ha=np.empty(0,dtype=np.float64)
                            with torch.no_grad():location=student.base_actions(torch.tensor(x[None]));command=location.clamp(-1,1).float()
                        tape.append(dict(teacher_actions=command[0].numpy().copy(),teacher_locations=location[0].numpy().copy(),teacher_hidden_before=hb,teacher_hidden_after=ha))
                        if hold>=first:session.step(command[0].numpy(),owner=1,actor_active=True)
                        else:session.step(owner=0)
                        assert equal(session.rows[-1][0],x)
                    else:session.step(owner=0)
                arrays=payload(session,tape);row=session.result();row.update(policy_hash=student.phash(),physics=counts[index+1]-before,
                    array_digest=digest(arrays),integration_sha256=digest(arrays['final_native_integration']),cpu=index,pid=os.getpid(),
                    teacher=query,teacher_policy_hash=(teacher.phash() if teacher_kind=='left_carry' else 'T255_base_hidden_reset') if query else None,first=first if query else 0)
                if query:
                    row.update(student_policy_hash=student.phash(),case_id=case.get('id',job['id']),
                        no_native_reset_at_handoff=True,teacher_hidden_reconstructed_from_actual_episode_history=teacher_kind=='left_carry',teacher_kind=teacher_kind,
                        teacher_has_carried_hidden=teacher_kind=='left_carry',
                        source_control=bool(case.get('source_control',False)))
                session.close();session=None;pipe.send((arrays,row))
    except BaseException:
        if session is not None and session.rows:
            try:N.write_npz(folder/'partial.npz',N.S.packed(payload(session,None if tape is None else tape[:len(session.rows)])),p['query_full_bytes'] if tape is not None else p['full_bytes'])
            except BaseException:save(folder/'partial_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if session:session.close()
        if witness:witness.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()

class Pool(NATIVE_AUDIT.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];context=mp.get_context('spawn');set_out(p['out'])
        try:
            for index in range(4):
                parent,child=context.Pipe();process=context.Process(target=native_worker,args=(child,index,p,counts));process.start();child.close();self.children.append(process);self.pipes.append(parent)
            save(OUT/'workers.json',dict(parent=os.getpid(),workers=[process.pid for process in self.children]));self.ready=[self.receive(index) for index in range(4)];assert all(row['state']=='READY' for row in self.ready)
        except BaseException:self.close();raise

audit_array=NATIVE_AUDIT.audit_array

def audit_query(student,a,case,p):
    first=case['first'];kind=case.get('teacher_kind','left_carry')
    flags={name:v.requires_grad for name,v in student.named_parameters()};student.requires_grad_(False)
    try:
        # Actual student prefix/hidden/density/reward is replayed, including the
        # subsequent student hidden updates under externally executed actions.
        audit_array(student,a,p,first,None)
        teacher=expert(N.science(p),p);teacher.reset_noise();teacher.sample=False;hidden=None
        with torch.no_grad():
            for index,x in enumerate(a['states']):
                if kind=='left_carry':
                    before=np.zeros(16) if hidden is None else hidden[0,0].numpy().copy()
                    command,loc,hidden,_=teacher.step(torch.tensor(x[None]),hidden)
                    assert equal(before,a['teacher_hidden_before'][index]);assert equal(hidden[0,0].numpy(),a['teacher_hidden_after'][index])
                else:
                    assert kind=='safe_reset' and a['teacher_hidden_before'].shape==(len(a['states']),0) and a['teacher_hidden_after'].shape==(len(a['states']),0)
                    loc=student.base_actions(torch.tensor(x[None]));command=loc.clamp(-1,1).float()
                assert equal(command[0].numpy(),a['teacher_actions'][index]);assert equal(loc[0].numpy(),a['teacher_locations'][index])
                if index>=first:assert equal(command[0].numpy(),a['actions'][index])
        assert np.all(a['ownership'][:first]==0) and np.all(a['ownership'][first:]==1)
        return dict(state='EXACT_ACTUAL_HISTORY_TEACHER_QUERY',first=first,teacher_kind=kind,all_steps_labelled=True,not_onpolicy=True)
    finally:
        student.reset_noise();student.sample=False
        for name,v in student.named_parameters():v.requires_grad_(flags[name])


def checked_query(q,entry,p):
    row,case=entry['row'],entry['case']
    assert row['success'] and not row['collision'] and not row['timeout'] and sha(row['path'])==row['sha256']
    assert sha(entry['source_student_model'])==entry['source_student_sha256']
    source=safe_initial(N.science(p),p);source.load_learned(N.S.read_model(entry['source_student_model']))
    assert source.phash()==row['policy_hash'];a=N.S.arrays(row['path']);audit_query(source,a,case,p)
    return query_group(q,a)
