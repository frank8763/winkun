"""Matched SGD replay versus Reptile-style meta updates; no new policy gradient.

Fixed successful teacher trajectories and initial-function history retention.
This is a supervised transfer pilot inspired by MER, not a DQN/MER replication.
"""
from pathlib import Path
import copy
import gzip
import io
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t577_offline_recovery_night'))
import t577_learning as Prior
torch,np,read,save,sha,digest=Prior.torch,Prior.np,Prior.read,Prior.save,Prior.sha,Prior.digest
ORIGINAL_FROZEN=Prior.frozen
ARCH='T580_COMPACT_SGD_HEAD_ON_T576_BACKBONE'


def frozen(p,path):
    s=Prior.P.checkpoint(path)
    if s.get('architecture')!=ARCH:return ORIGINAL_FROZEN(p,path)
    assert s['optimizer_class']=='SGD_NO_MOMENTUM' and s['static_model']==p['static_model']
    assert sha(s['static_model']['path'])==s['static_model']['sha256']
    q=ORIGINAL_FROZEN(p,s['static_model']['path']);q.continual.load_state_dict(s['continual'])
    assert q.phash()==s['policy_hash'];return q.eval()


def make(p):
    q=frozen(p,p['initial']['path']);q.requires_grad_(False);q.continual.requires_grad_(True)
    assert sum(v.numel() for v in q.parameters() if v.requires_grad)==5986
    return q,torch.optim.SGD(q.continual.parameters(),lr=p['inner_lr'],momentum=0,weight_decay=0)


def state(obj):
    return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))


def load(obj,s):
    obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])


def fixed(q):
    return digest({k:v for k,v in q.state_dict().items() if not k.startswith('continual.')})


def dataset(p):
    q=frozen(p,p['initial']['path']);data=[]
    with torch.no_grad():
        for kind,rows in [('keep',p['keep_rows']),('repair',p['repair_rows'])]:
            for row in rows:
                assert sha(row['path'])==row['sha256']
                a=Prior.G.E.N.S.arrays(row['path']);x=torch.tensor(a['states'])
                f,b=Prior.Prior.Prior.feature(q,x);initial=b+q.continual(f)
                assert float((q.sequence(x)-initial).abs().max())<1e-9
                if kind=='keep':target=initial.clone()
                else:
                    assert row['success'] and not row['collision'] and row['field_detail']['field_qualified_success']
                    command=torch.tensor(a['actions'],dtype=torch.float64)
                    # Fixed saturation-aware targets. Do not pull a valid >1 location down to 1.
                    target=torch.where(command>=1,torch.maximum(initial,torch.ones_like(initial)),
                        torch.where(command<=-1,torch.minimum(initial,-torch.ones_like(initial)),command))
                data.append(dict(id=row['job_id'],kind=kind,f=f,base=b,target=target))
    assert len(data)==27 and sum(g['kind']=='keep' for g in data)==23
    return data


def schedule(data,p):
    rng=np.random.default_rng(p['batch_seed']);keep=[i for i,g in enumerate(data) if g['kind']=='keep'];repair=[i for i,g in enumerate(data) if g['kind']=='repair'];result=[]
    for block in range(p['blocks']):
        entries=[]
        for k in range(p['inner_steps']):
            idx=repair[(2*block+k//4)%len(repair)] if k%4==0 else keep[(6*block+k-1-k//4)%len(keep)]
            chosen=rng.choice(len(data[idx]['f']),size=min(p['rows_per_chunk'],len(data[idx]['f'])),replace=False)
            entries.append((idx,chosen.tolist()))
        result.append(entries)
    return result


def measure(q,data):
    with torch.no_grad():
        return [dict(id=g['id'],kind=g['kind'],location_RMS=float((g['base']+q.continual(g['f'])-g['target']).square().mean().sqrt()),command_RMS=float(((g['base']+q.continual(g['f'])).clamp(-1,1)-g['target'].clamp(-1,1)).square().mean().sqrt())) for g in data]


def fit(obj,data,p,arm,guard,event,blocks=None,gamma_override=None):
    q,opt=obj;body=fixed(q);plan=schedule(data,p);trace=[]
    gamma=p['meta_gamma'] if arm=='MER' else 1.
    if gamma_override is not None:gamma=gamma_override
    lr=p['inner_lr']*(p['meta_gamma'] if arm=='ER_MATCHED_STEP' else 1.)
    for g in opt.param_groups:g['lr']=lr
    before=measure(q,data)
    for bi,entries in enumerate(plan[:p['blocks'] if blocks is None else blocks]):
        start={k:v.detach().clone() for k,v in q.continual.state_dict().items()}
        losses=[]
        for si,(idx,indices) in enumerate(entries):
            guard();g=data[idx];opt.zero_grad(set_to_none=True)
            loss=(g['base'][indices]+q.continual(g['f'][indices])-g['target'][indices]).square().mean()
            assert torch.isfinite(loss);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True))
            step=bi*p['inner_steps']+si;event('SGD_BEFORE',step);opt.step();event('SGD_AFTER',step)
            assert not opt.state
            losses.append([float(loss.detach()),norm])
        if arm=='MER':
            event('META_BEFORE',bi)
            with torch.no_grad():
                for name,v in q.continual.named_parameters():
                    v.copy_(start[name]+gamma*(v-start[name]))
            event('META_AFTER',bi)
        trace.append(dict(block=bi,losses=losses))
    assert fixed(q)==body
    return dict(trace=trace,SGD_steps=len(trace)*p['inner_steps'],meta_interpolations=len(trace) if arm=='MER' else 0,
        inner_lr=lr,outer_gamma=gamma,within_batch_beta=1.,before=before,after=measure(q,data),
        fixed_body_hash=body,schedule_digest=digest(plan),new_RL=0,
        fixed_bank_NOT_online_reservoir_MER_replication=True)


def export(obj,p,path):
    q,opt=obj;buf=io.BytesIO();torch.save(dict(architecture=ARCH,static_model=p['static_model'],
        continual=q.continual.state_dict(),optimizer_class='SGD_NO_MOMENTUM',
        optimizer_state=opt.state_dict(),policy_hash=q.phash(),joint_hash=digest(state(obj)),
        single_actor=True,teacher_at_deployment=False),buf)
    b=gzip.compress(buf.getvalue(),mtime=0);assert len(b)<=p['model_bytes']
    with Path(path).open('xb') as f:f.write(b);f.flush();__import__('os').fsync(f.fileno())
    assert frozen(p,path).phash()==q.phash()
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),joint_hash=digest(state(obj)),bytes=len(b))
