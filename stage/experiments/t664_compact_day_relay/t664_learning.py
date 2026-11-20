"""Clean unique-ID BC and fresh original-start PPO; no learned Q guidance."""
from pathlib import Path
import copy
import gzip
import io
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'))
import t654_learning as Base
L,P,E,torch,np=Base.L,Base.P,Base.E,Base.torch,Base.np
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest
location=Base.location


class Value(torch.nn.Module):
    """Frozen-causal-feature baseline; no action input or actor gradient."""
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net=torch.nn.Sequential(torch.nn.Linear(184,64),torch.nn.Tanh(),torch.nn.Linear(64,1)).double()
            torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,f):return self.net(f.detach())[...,0]


def make(p,device):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=L.frozen(p,p['initial']['path']).to(device).train().requires_grad_(False)
    q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad]
    assert sum(v.numel() for v in variables)==11906
    v=Value(p['baseline_seed']).to(device)
    ao=torch.optim.Adam(variables,lr=p['fit_lr'],foreach=False,fused=False)
    vo=torch.optim.Adam(v.parameters(),lr=p['baseline_lr'],foreach=False,fused=False)
    return q,v,ao,vo


def state(obj):
    q,v,ao,vo=obj
    return copy.deepcopy(dict(actor=q.state_dict(),value=v.state_dict(),actor_opt=ao.state_dict(),value_opt=vo.state_dict()))


def load(obj,s):
    for target,key in zip(obj,['actor','value','actor_opt','value_opt']):
        target.load_state_dict(copy.deepcopy(s[key]))


def variant_payload(p):
    source=p['initial'];saved=copy.deepcopy(P.checkpoint(source['path']))
    q=L.Actor(p).eval().requires_grad_(False);q.load_state_dict(saved['joint_state']['actor'])
    weights=digest(q.state_dict());saved['policy_hash']=q.phash()
    saved['exploration_variant_only']=dict(source_sha256=source['sha256'],sigma=p['raw_sigma'],weights_sha256=weights)
    buf=io.BytesIO();torch.save(saved,buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    return blob,q.phash(),weights

def initial_variant(p, folder):
    """Change declared exploration distribution, preserving all weights."""
    source=p['initial'];blob,policy_hash,weights=variant_payload(p)
    if policy_hash==source['policy_hash']:return source
    path=Path(folder)/'initial_sigma.pt.gz'
    if not path.exists():
        with path.open('xb') as handle:handle.write(blob)
    assert digest(L.frozen(p,path).state_dict())==weights
    return dict(source,path=str(path),sha256=sha(path),policy_hash=policy_hash,bytes=path.stat().st_size)


def cpu_state(obj):
    def cpu(x):
        if torch.is_tensor(x):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        if isinstance(x,tuple):return tuple(cpu(v) for v in x)
        return x
    return cpu(state(obj))


def features(q,a,key,source_sha):
    device=next(q.parameters()).device
    with torch.no_grad():
        x=torch.tensor(a['states'],device=device)
        n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0]
        g=dict(id=key,f=torch.cat([h,n],-1),logits=q.head(h),target=torch.tensor(a['actions'],device=device).double(),
               source_sha256=source_sha)
        assert float((location(q,g)-q.sequence(x)).abs().max())<1e-10
    return g


def references(p,q,guard):
    contract=read(p['reference_contract'])
    assert len(contract['reference_rows'])==387
    answer=[]
    for row in contract['reference_rows']:
        guard();assert sha(row['source_path'])==row['sha256']
        a=E.N.S.arrays(row['source_path'])
        assert not a['ownership'].any() and not a['stochastic'].any() and a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
        answer.append(features(q,a,row['task_id'],row['sha256']))
    assert len({g['id'] for g in answer})==387
    return answer


def pack(groups):
    lengths=[len(g['f']) for g in groups]
    return dict(f=torch.cat([g['f'] for g in groups]),logits=torch.cat([g['logits'] for g in groups]),
                target=torch.cat([g['target'] for g in groups]),lengths=lengths,ids=[g['id'] for g in groups])


def retention(q,data,tail=False):
    err=P.action_loss(location(q,data),data['target']).mean(-1)/.2**2
    episodes=[]
    for e in err.split(data['lengths']):
        term=e.mean()
        if tail:term=.5*term+.5*e.topk(max(1,(len(e)+4)//5)).values.mean()
        episodes.append(term)
    losses=torch.stack(episodes)
    result=losses.mean()
    if tail:result=.5*result+.5*losses.topk(max(1,(len(losses)+9)//10)).values.mean()
    return result,losses


def outcome(row,kind='binary'):
    assert row['complete'] and not row['administrative_censor']
    success=float(row['success'] and not row['collision'] and
                 (not ('fullfield_start_pose' in row['request']['job']) or not row.get('field_boundary',False)))
    assert kind in ['binary','collision_cost']
    return success-float(kind=='collision_cost' and row['collision'])


def fresh_items(obj,groups,p):
    q,v,ao,vo=obj;items=[]
    for key,values in groups:
        episodes=[]
        assert len(values)==p['episodes_per_task']
        for a,row in values:
            assert row['complete'] and not row['administrative_censor'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
            assert not a['ownership'].any() and a['stochastic'].all() and row['seed'] is not None
            assert row['policy_hash']==q.phash()
            g=features(q,a,key,row['sha256']);device=g['f'].device
            with torch.no_grad():
                loc=location(q,g);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device)
                old=torch.tensor(a['logp'],device=device)
                density=P.L.density(loc,raw,atom,p['raw_sigma'])
                assert float((density-old).abs().max())<1e-9
                assert float((loc-torch.tensor(a['locations'],device=device)).abs().max())<1e-10
                baseline=v(g['f']).detach().clone()
                target=loc.new_full((len(loc),),outcome(row,p['objective_kind']))
            episodes.append(dict(**g,raw=raw,atom=atom,old=old,initial=loc.detach().clone(),
                                 baseline=baseline,target_value=target,adv=(target-baseline).detach()))
        items.append(episodes)
    return items


def pg_loss(q,episodes,p):
    loss=next(q.parameters()).new_zeros(())
    for g in episodes:
        ratio=(P.L.density(location(q,g),g['raw'],g['atom'],p['raw_sigma'])-g['old']).exp()
        loss-=torch.minimum(ratio*g['adv'],ratio.clamp(1-p['ratio_clip'],1+p['ratio_clip'])*g['adv']).sum()/(len(episodes)*p['score_scale'])
    return loss


def fit(obj,refs,groups,p,guard,event):
    q,v,ao,vo=obj;fixed=L.fixed_hash(q);tail=p['arm'].endswith('tail');rl=p['arm'].startswith('ppo')
    keep=pack(refs);items=fresh_items(obj,groups,p) if rl else []
    frozen_adv=[digest([g['baseline'] for g in episodes]) for episodes in items]
    trace=[];steps=p['ppo_epochs'] if rl else p['bc_steps']
    for k in range(steps):
        guard();before=state(obj);ao.zero_grad(set_to_none=True);pg=0.
        if rl:
            for episodes in items:
                loss=pg_loss(q,episodes,p)/len(items);loss.backward();pg+=float(loss.detach())
        keep_loss,losses=retention(q,keep,tail)
        ((p['replay_weight'] if rl else 1.)*keep_loss).backward()
        variables=[x for x in q.parameters() if x.requires_grad]
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('actor','BEFORE',k);ao.step();event('actor','AFTER',k)
        assert all(torch.isfinite(x).all() for x in variables) and L.fixed_hash(q)==fixed
        kl=[]
        with torch.no_grad():
            for episodes in items:
                kl.append(sum(float((.5*((location(q,g)-g['initial'])/.2).square().sum(-1)).mean()) for g in episodes)/len(episodes))
        okay=not kl or max(kl)<=p['max_task_latent_KL']
        if not okay:load(obj,before)
        trace.append(dict(step=k,pg=pg,keep=float(keep_loss.detach()),worst_task_loss=float(losses.max().detach()),norm=norm,kl=kl,accepted=okay))
        if not okay:break
    # Value from BEFORE the block generated every advantage. Current data fit
    # is exclusively for the NEXT fresh block; it never guides action choice.
    value_trace=[]
    if rl:
        assert frozen_adv==[digest([g['baseline'] for g in episodes]) for episodes in items]
        f=torch.cat([g['f'] for episodes in items for g in episodes]);targets=torch.cat([g['target_value'] for episodes in items for g in episodes])
        weights=torch.cat([g['old'].new_full((len(g['old']),),1/(len(items)*len(episodes)*len(g['old']))) for episodes in items for g in episodes])
        for k in range(p['value_steps']):
            guard();vo.zero_grad(set_to_none=True);loss=((v(f)-targets).square()*weights).sum();loss.backward()
            torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True)
            event('value','BEFORE',k);vo.step();event('value','AFTER',k)
            value_trace.append(float(loss.detach()))
        assert L.fixed_hash(q)==fixed and all(torch.isfinite(param).all() for param in v.parameters())
    return dict(Actor_Adam=len(trace),Value_Adam=len(value_trace),trace=trace,value_trace=value_trace,
                fresh_original_start_block=rl,previous_block_value_used=True,no_Q_guidance=True,
                unique_reference_count=387,tail_loss=tail,trainable_Actor_parameters=11906,
                backbone_hash=fixed,no_deterministic_success_guarantee=True)


def export(obj,p,path):
    s=cpu_state(obj);old=P.checkpoint(p['initial']['path'])['joint_state']
    # Legacy Value is retained only to keep the established deploy loader
    # format. The new baseline and both active optimizers are separate state.
    joint=dict(actor=s['actor'],actor_opt=s['actor_opt'],value=old['value'],value_opt=old['value_opt'])
    buf=io.BytesIO();torch.save(dict(joint_state=joint,policy_hash=obj[0].phash(),joint_hash=digest(joint),
         architecture='T501_single120_64carry_two_neural_command_residuals',single_actor=True,teacher_at_deployment=False),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(joint),bytes=len(blob))


def save_state(path,s,limit):
    buf=io.BytesIO();torch.save(s,buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=limit
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=sha(path),state_hash=digest(s),bytes=len(blob))


def read_state(row):
    assert sha(row['path'])==row['sha256']
    state=torch.load(io.BytesIO(gzip.decompress(Path(row['path']).read_bytes())),map_location='cpu',weights_only=False)
    assert digest(state)==row['state_hash'];return state
