"""Replace all compact-memory corrections with one smooth fixed-capacity head.

Trainable encoder/causal GRU/readout; frozen original command adapters, no deployment teacher,
center bank, nearest-neighbor lookup or task/router input. Distillation only.
"""
from pathlib import Path
import copy,gzip,io,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'));import t552_actor as Prior
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as Base
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
ARCH='T562_single120_64carry_fullhistory_encoder_core_sine93186'

class Sine(torch.nn.Module):
    def forward(self,x):return torch.sin(30.*x)

class Actor(Base.Actor):
    def __init__(self,p):
        super().__init__(p)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['student_seed'])
            self.student=torch.nn.Sequential(torch.nn.Linear(184,128),Sine(),torch.nn.Linear(128,128),Sine(),torch.nn.Linear(128,2)).double()
            with torch.no_grad():
                self.student[0].weight.uniform_(-1/184,1/184)
                self.student[2].weight.uniform_(-math.sqrt(6/128)/30,math.sqrt(6/128)/30)
            # Keep same zero initial output/public base as T559; only hidden sine/init changes.
            torch.nn.init.zeros_(self.student[-1].weight);torch.nn.init.zeros_(self.student[-1].bias)
        self.requires_grad_(False)
    def location(self,h,n):return Base.Actor.location(self,h,n)+self.student(torch.cat([h,n],-1))
    def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))

def initial(p):
    saved=P.checkpoint(p['student_initial']['path']);assert sha(p['student_initial']['path'])==p['student_initial']['sha256']
    q=Actor(p);q.load_state_dict(saved['actor']);assert digest(q.state_dict())==digest(saved['actor']) and q.phash()==saved['policy_hash'];return q

def frozen(p,path):
    s=P.checkpoint(path)
    if s.get('architecture')!=ARCH:return Prior.Old.frozen(p,path)
    q=Actor(p);q.load_state_dict(s['actor']);assert q.phash()==s['policy_hash'];return q.eval()

def variables(q):return list(q.encoder.parameters())+list(q.core.parameters())+list(q.student.parameters())
def make(p):
    q=initial(p);q.encoder.requires_grad_(True);q.core.requires_grad_(True);q.student.requires_grad_(True);opt=torch.optim.SGD(variables(q),lr=1.,momentum=0.,weight_decay=0.,foreach=False,fused=False);assert sum(x.numel() for x in variables(q))==93186;return q,opt

def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])
def fixed_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith(('student.','encoder.','core.'))})

def dataset(p,limit=None):
    teacher=Prior.Old.frozen(p,p['teacher']['path']);q=initial(p);data=[]
    with torch.no_grad():
        for kind,rows in [('keep',p['keep_rows']),('neighbor',p['neighbor_rows'])]:
            for row in rows:
                assert sha(row['path'])==row['sha256'] and row['success'] and not row['collision'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);f=torch.cat([h[:,0],n],-1);base=Base.Actor.location(q,h[:,0],n)
                target=torch.tensor(a['locations']);assert float((teacher.sequence(x)-target).abs().max())<1e-10
                mask=torch.tensor(~a['ownership'].astype(bool));assert mask.sum()>=2
                if limit is not None:mask[limit:]=False
                data.append(dict(id=row['job_id'],kind=kind,f=f[mask],base=base[mask],target=target[mask],command=torch.tensor(a['actions'])[mask],source_sha256=row['sha256'],x=x,mask=mask))
    assert len(data)==27 and sum(g['kind']=='keep' for g in data)==23
    return data

def locations(q,g):return q.sequence(g['x'])[g['mask']]
def measure(q,data):
    rows=[]
    with torch.no_grad():
        for g in data:
            loc=locations(q,g);cmd=loc.clamp(-1,1).float().double();delta=cmd-g['command'].double();err=loc-g['target']
            rows.append(dict(id=g['id'],kind=g['kind'],location_RMS=float(err.square().mean().sqrt()),command_RMS=float(delta.square().mean().sqrt()),command_max=float(delta.abs().max())))
    return dict(rows=rows,worst_command_RMS=max(v['command_RMS'] for v in rows),worst_command_max=max(v['command_max'] for v in rows),mean_location_MSE=sum(v['location_RMS']**2 for v in rows)/len(rows))

def project_direction(d0,A,b,p):
    # b>=0 guarantees zero is feasible; solve projection dual, no feasibility certificate claimed for failed tolerance.
    assert torch.isfinite(A).all() and torch.isfinite(b).all() and bool((b>=-1e-12).all())
    norms=A.norm(dim=1);valid=norms>1e-12;assert bool((b[~valid]>=0).all());A=A[valid]/norms[valid,None];b=b[valid]/norms[valid];H=A@A.T;c=A@d0-b;lam=torch.zeros_like(c);hl=torch.zeros_like(c)
    for sweep in range(p['projection_sweeps']):
        for i in range(len(c)):
            new=torch.clamp(lam[i]+(c[i]-hl[i])/H[i,i],min=0);change=new-lam[i];lam[i]=new;hl+=change*H[:,i]
    d=d0-A.T@lam;slack=b-A@d;primal=float(torch.relu(-slack).max()) if len(slack) else 0.;dualgrad=hl-c;kkt=float(torch.where(lam>1e-10,dualgrad.abs(),torch.relu(-dualgrad)).max()) if len(c) else 0.
    return d,dict(rows=len(A),sweeps=p['projection_sweeps'],primal_violation=primal,dual_KKT_residual=kkt,stationarity=float((d-d0+A.T@lam).norm()),dual_hash=digest(lam),pass_tolerance=primal<=p['projection_tolerance'] and kkt<=p['projection_tolerance'])

def fit(obj,data,p,guard,event,steps=None):
    q,opt=obj;fixed=fixed_hash(q);trace=[];directions=[];before=measure(q,data);current=before;protected=set(p['protected_fit_ids']);assert len(protected)==24
    def qualified(row):return row['command_RMS']<=p['fit_gate_RMS'] and row['command_max']<=p['fit_gate_max']
    assert all(qualified(g) for g in before['rows'] if g['id'] in protected) and before['worst_command_RMS']<=p['fit_gate_RMS']
    packed=torch.nn.utils.rnn.pad_sequence([g['x'] for g in data]);accepted=0;reason='JOINT_DIRECTION_CAP';solve_count=0;projections=0
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();old=state(obj);weights0=[z.detach().clone() for z in variables(q)];opt.zero_grad(set_to_none=True);pred=q.sequence(packed);errors=[];mse=[];protected_indices=[];offset=0
        for i,g in enumerate(data):
            ei=(pred[:len(g['x']),i][g['mask']].clamp(-1,1).float().double()-g['command'].double()).reshape(-1);errors.append(ei);mse.append(ei.square().mean())
            if g['id'] in protected:protected_indices.extend((torch.nonzero(ei.detach().abs()>=p['protected_coordinate_threshold'],as_tuple=False).flatten()+offset).tolist())
            offset+=len(ei)
        e=torch.cat(errors);bad=int((e.detach().abs()>p['fit_gate_max']).sum());n=max(p['minimum_active_coordinates'],bad);assert n<=p['maximum_active_coordinates'] and len(protected_indices)<=p['maximum_protected_coordinates'], ('FIXED_COORDINATE_CAP',bad,len(protected_indices));indices=e.detach().abs().topk(min(n,len(e))).indices;rows=[]
        def gradient(value):
            guard();return torch.cat([a.reshape(-1) for a in torch.autograd.grad(value,variables(q),retain_graph=True)])
        for i in indices:rows.append(gradient(e[i]))
        J=torch.stack(rows);res=e.detach()[indices];rhs=res.sign()*torch.relu(res.abs()-p['joint_internal_target']);H=J@J.T;damping=max(p['minimum_damping'],p['relative_damping']*float(H.diag().mean()));event('BEFORE_SOLVE',k);alpha=torch.linalg.solve(H+torch.eye(len(indices),dtype=H.dtype)*damping,rhs);event('AFTER_SOLVE',k);solve_count+=1;d0=-(J.T@alpha);Ar=[];br=[]
        for v in mse:Ar.append(gradient(v));br.append(max(0.,p['constraint_slack_fraction']*(p['fit_gate_RMS']**2-float(v.detach()))))
        for i in protected_indices:Ar.append(gradient(e[i].abs()));br.append(max(0.,p['constraint_slack_fraction']*(p['fit_gate_max']-float(e[i].detach().abs()))))
        A=torch.stack(Ar);b=e.new_tensor(br);event('BEFORE_PROJECTION',k);delta,certificate=project_direction(d0,A,b,p);event('AFTER_PROJECTION',k);projections+=1;raw_norm=float(delta.norm());assert torch.isfinite(delta).all()
        directions.append(dict(indices=indices.tolist(),protected_indices=protected_indices,jacobian_hash=digest(J),constraint_hash=digest(A),bound_hash=digest(b),gram_hash=digest(H),rhs_hash=digest(rhs),unprojected_direction_hash=digest(d0),direction_hash=digest(delta),damping=damping,raw_norm=raw_norm,certificate=certificate))
        if not certificate['pass_tolerance']:reason='PROJECTION_RESIDUAL_STOP';break
        if raw_norm<=1e-12:reason='ZERO_CONSTRAINED_DIRECTION_STOP';break
        delta=delta*min(1.,p['parameter_radius_cap']/raw_norm);offset=0
        for z in variables(q):z.grad=-delta[offset:offset+z.numel()].reshape_as(z).clone();offset+=z.numel()
        assert offset==len(delta);event('BEFORE',k);opt.step();event('AFTER',k);proposal=state(obj);actual_delta=[z.detach()-w for z,w in zip(variables(q),weights0)];trials=[-1.]*(4*len(p['step_factors']));chosen=0.
        for i,factor in enumerate(p['step_factors']):
            guard();load(obj,proposal)
            with torch.no_grad():
                for z,w,d in zip(variables(q),weights0,actual_delta):z.copy_(w+factor*d)
            candidate=measure(q,data);lost=sum(g['id'] in protected and not qualified(g) for g in candidate['rows']);gate=candidate['worst_command_RMS']<=p['fit_gate_RMS'] and candidate['worst_command_max']<=p['fit_gate_max'];improved=candidate['worst_command_max']<=current['worst_command_max']-p['minimum_peak_decrease'];ok=not lost and candidate['worst_command_RMS']<=p['fit_gate_RMS'] and (gate or improved)
            trials[i*4:i*4+4]=[candidate['worst_command_max'],candidate['worst_command_RMS'],float(lost),float(ok)]
            if ok:chosen=factor;current=candidate;accepted+=1;break
        if not chosen:load(obj,old)
        trace.append([float(e.detach().abs().max()),float(len(indices)),damping,raw_norm,float(delta.norm()),chosen,float(accepted),*trials]);assert fixed_hash(q)==fixed
        if steps is None:
            if not chosen:reason='NO_ACCEPTED_CONSTRAINED_STEP_STOP';break
            if current['worst_command_RMS']<=p['fit_gate_RMS'] and current['worst_command_max']<=p['fit_gate_max']:reason='ALL27_OFFLINE_GATE_EARLY_STOP';break
    after=measure(q,data);assert after==current
    return dict(steps=len(trace),linear_solves=solve_count,projection_solves=projections,direction_evidence=directions,trace=np.asarray(trace,dtype=np.float64).reshape(-1,7+4*len(p['step_factors'])),before=before,after=after,fit_gate=after['worst_command_RMS']<=p['fit_gate_RMS'] and after['worst_command_max']<=p['fit_gate_max'],new_RL=0,new_Adam=0,optimizer='LINEARIZED_HISTORY_CONSTRAINED_RESIDUAL_STEP',teacher_runtime=False,full_current_history_BPTT=True,retained_proposals=accepted,rejected_proposals=len(trace)-accepted,stop_reason=reason,protected_fit24_preserved=True,shadow_is_numerical_repeat_not_independent_data=True)

def export(obj,p,path):
    q,opt=obj;s=state(obj);b=io.BytesIO();torch.save(dict(architecture=ARCH,actor=s['actor'],actor_opt=s['actor_opt'],joint_hash=digest(s),policy_hash=q.phash(),single_actor=True,no_runtime_teacher=True,no_compact_banks=True),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    with Path(path).open('xb') as h:h.write(blob);h.flush();__import__('os').fsync(h.fileno())
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),joint_hash=digest(s),bytes=len(blob))
