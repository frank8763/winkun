"""Original T501 command heads, full384 balanced retention plus one recovery.

This is supervised adaptation of an RL-derived policy, not a new PPO update,
not CLEAR reproduction, and not an original lifelong-learning algorithm.
"""
import copy,gzip,io
import t637_data as D
L,P,E,torch,np=D.L,D.P,D.E,D.torch,D.np
read,save,sha,digest=D.read,D.save,D.sha,D.digest


def make(p,device='cuda'):
    q=L.frozen(p,p['initial']['path']).to(device)
    q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    opt=torch.optim.Adam(L.variables(q),lr=p['fit_lr'],foreach=False,fused=False)
    assert sum(v.numel() for v in L.variables(q))==11906 and not opt.state
    return q,opt


def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])


def pack(data,device):
    # Every history contributes equally, independently of episode duration.
    lengths=[len(g['f']) for g in data]
    return dict(f=torch.cat([g['f'] for g in data]).to(device),logits=torch.cat([g['logits'] for g in data]).to(device),
         target=torch.cat([g['target'] for g in data]).to(device),
         weights=torch.cat([torch.full((n,),1/(len(data)*n),dtype=torch.float64) for n in lengths]).to(device))


def location(q,data):
    res=torch.cat([q.drive_adapter(data['f']),q.steer_adapter(data['f'])],-1)
    return 1.25*torch.tanh(data['logits']+.3*torch.tanh(res))


def correction(p,a,k):
    q=L.frozen(p,p['initial']['path']);x=torch.tensor(a['states'])
    with torch.no_grad():
        n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0]
        return dict(f=torch.cat([h,n],-1)[k:],logits=q.head(h)[k:],target=torch.tensor(a['actions'][k:]).double())


def objective(q,keep,new,p):
    retention=((location(q,keep)-keep['target'])/.2).square().mean(-1)
    protection=(retention*keep['weights']).sum()
    loss=P.action_loss(location(q,new),new['target']).mean(-1)/.2**2
    repair=.5*loss.mean()+.5*loss.topk(max(1,(len(loss)+4)//5)).values.mean()
    return repair+p['keep_weight']*protection,repair,protection


def fit(obj,keep,new,p,guard,event):
    q,opt=obj;fixed=L.fixed_hash(q);trace=[]
    for k in range(p['fit_steps']):
        guard();opt.zero_grad(set_to_none=True);loss,r,s=objective(q,keep,new,p);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append(dict(step=k,loss=float(loss.detach()),repair=float(r.detach()),keep=float(s.detach()),norm=norm))
        assert all(torch.isfinite(v).all() for v in L.variables(q))
    assert fixed==L.fixed_hash(q)
    with torch.no_grad():
        error=(location(q,keep)-keep['target']).abs()
        report=dict(max_keep_raw_error=float(error.max()),max_keep_command_error=float((location(q,keep).clamp(-1,1)-keep['target'].clamp(-1,1)).abs().max()),
            recovery_command_RMSE=float((location(q,new).clamp(-1,1)-new['target']).square().mean().sqrt()))
    return dict(Actor_Adam=len(trace),Value_Adam=0,full384_history_equal_weight=True,trainable_parameters=11906,trace=trace,metrics=report)


def export(obj,p,path):
    q,opt=obj;q.cpu();s=state(obj);old=P.checkpoint(p['initial']['path'])['joint_state']
    # Export the original full checkpoint format, so inference uses the exact
    # original loader and architecture. The frozen Value remains historical.
    joint=dict(actor=s['actor'],actor_opt=s['actor_opt'],value=old['value'],value_opt=old['value_opt'])
    b=io.BytesIO();torch.save(dict(joint_state=joint,policy_hash=q.phash(),joint_hash=digest(joint),
        architecture='T501_single120_64carry_two_neural_command_residuals',single_actor=True,teacher_at_deployment=False),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    with path.open('xb') as f:f.write(blob)
    assert L.frozen(p,path).phash()==q.phash()
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),joint_hash=digest(joint),bytes=len(blob))
