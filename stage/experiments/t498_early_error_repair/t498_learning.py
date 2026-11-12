"""Early/tail-error repair with zero-initial-loss own-success retention.

The deployed Actor is the byte-identical T496 architecture. Only its learned
longitudinal adapter changes; mature Adam is preserved across this pilot.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'))
import speed_actor as S
G,P,torch,np=S.G,S.P,S.torch,S.np
read,sha,digest=G.read,G.sha,G.digest

def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    checkpoint=P.checkpoint(p['initial']['path']);old=checkpoint['joint_state'];q=S.Actor(p);q.load_state_dict(old['actor'])
    assert q.phash()==p['initial']['policy_hash'];q.requires_grad_(False);q.drive_adapter.requires_grad_(True)
    assert sum(v.numel() for v in q.parameters() if v.requires_grad)==5953
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(q.drive_adapter.parameters(),lr=.001,foreach=False,fused=False);ao.load_state_dict(old['actor_opt'])
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    assert digest(P.state(q,v,ao,vo))==p['initial']['joint_hash']
    assert len(ao.state)==4 and all(int(s['step'])==512 for s in ao.state.values())
    return q,v,ao,vo

def groups(p):
    q=S.frozen(p,p['initial']['path']);data=[]
    with torch.no_grad():
        for kind,rows in [('repair',[p['repair_row']]),('keep',p['keep_rows'])]:
            for r in rows:
                assert r['success'] and not r['collision'] and sha(r['path'])==r['sha256']
                a=G.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
                x=torch.tensor(a['states']);n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0];logit=q.head(h)[:,0]
                actual=q.location(h,n)[:,0];target=torch.tensor(a['actions'][:,0],dtype=torch.float64) if kind=='repair' else torch.tensor(a['locations'][:,0],dtype=torch.float64)
                if kind=='keep':assert float((actual-target).abs().max())<1e-10
                data.append(dict(kind=kind,id=r['job_id'],features=torch.cat([h,n],-1),logit=logit,target=target,early=torch.tensor(np.r_[0.,np.cumsum(a['native_counts']*.002)[:-1]]<20),source_sha256=r['sha256']))
    assert len(data)==54 and sum(g['kind']=='keep' for g in data)==53
    return data

def location(q,g):return 1.25*torch.tanh(g['logit']+.3*torch.tanh(q.drive_adapter(g['features'])[:,0]))
def measure(q,data):
    with torch.no_grad():
        repair=data[0];err=P.action_loss(location(q,repair)[:,None],repair['target'][:,None])[:,0];keep=[(location(q,g)-g['target']).square().mean() for g in data[1:]]
        return dict(repair_all_MSE=float(err.mean()),repair_first20_MSE=float(err[repair['early']].mean()),repair_first_drive_error=float(location(q,repair)[0]-repair['target'][0]),keep_mean_MSE=float(torch.stack(keep).mean()),keep_max_episode_MSE=float(torch.stack(keep).max()))

def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=S.backbone(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());trace=[];initial=measure(q,data)
    # All keep targets are un-clipped actual initial locations, not commands.
    assert initial['keep_max_episode_MSE']<1e-20
    for k in range(p['repair_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);g=data[0]
        err=P.action_loss(location(q,g)[:,None],g['target'][:,None])[:,0];early=err[g['early']]
        repair=.5*err.mean()+.25*early.mean()+.25*early.topk(max(1,math.ceil(.2*len(early)))).values.mean()
        (.5*repair).backward();keep=0.
        for g in data[1:]:
            loss=(location(q,g)-g['target']).square().mean()/53;(.5*loss).backward();keep+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(q.drive_adapter.parameters(),1.,error_if_nonfinite=True))
        event('BEFORE',k);ao.step();event('AFTER',k);trace.append(dict(step=k,repair_loss=float(repair.detach()),keep_MSE=keep,norm=norm))
        assert S.backbone(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
    return dict(Actor_Adam=len(trace),Value_Adam=0,trainable_parameters=5953,mature_initial_step=512,final_step=512+len(trace),initial=initial,final=measure(q,data),trace=trace,teacher_at_deployment=False)
