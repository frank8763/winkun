"""Frozen-Actor behavior-return calibration, not policy-Q certification."""
from pathlib import Path
import sys,copy,io,gzip,hashlib,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t621_acceleration_probe'))
import accelerated as A
L,torch,np=A.L,A.torch,A.L.np
HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261117/t622';OUT=ROOT/'stage/runs/t622_gpu_critic_calibration'

def config():
    p=L.read(HERE/'protocol.json');c=L.read(p['actor_configuration']);c.update(p);return c

def setup():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()

def data(p):
    rows=L.read(p['online_summary'])['result']['online_rows'];prior,online=L.dataset(p,rows);result=[];seen=set()
    for e in prior+online:
        if e['sha256'] in seen:continue
        seen.add(e['sha256']);ret=e['r'].clone();acc=0.
        for k in range(e['n']-1,-1,-1):acc=e['r'][k]+e['discount'][k]*e['bootstrap'][k]*acc;ret[k]=acc
        held=e['id'] in p['critical_ids'] or int(hashlib.sha256(e['id'].encode()).hexdigest()[:8],16)%5==0
        result.append(dict(e,ret=ret,heldout=held))
    train=[e for e in result if not e['heldout']];test=[e for e in result if e['heldout']]
    assert train and test and not ({e['id'] for e in train}&{e['id'] for e in test})
    return train,test

def make(p):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p['learner_seed']);q=torch.nn.ModuleList([L.Critic(p),L.Critic(p)]).cuda()
    return q,torch.optim.Adam(q.parameters(),lr=p['critic_lr'],foreach=False,fused=False)

def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(z) for k,z in v.items()}
    if isinstance(v,list):return [cpu(z) for z in v]
    if isinstance(v,tuple):return tuple(cpu(z) for z in v)
    return v

def state(obj):return cpu(dict(critics=obj[0].state_dict(),optimizer=obj[1].state_dict()))

def pack(episodes,rng,p):
    selected=[episodes[int(rng.integers(len(episodes)))] for _ in range(4)];x=torch.nn.utils.rnn.pad_sequence([e['x'] for e in selected]).cuda();ii=[(int(k),i) for i,e in enumerate(selected) for k in rng.integers(e['n'],size=p['points_per_episode'])];t=torch.tensor([k for k,i in ii],device='cuda');b=torch.tensor([i for k,i in ii],device='cuda');u=torch.stack([selected[i]['u'][k] for k,i in ii]).cuda();y=torch.stack([selected[i]['ret'][k] for k,i in ii]).cuda();return x,t,b,u,y

def loss(q,pack,arm):
    x,t,b,u,y=pack
    if arm=='blind':u=torch.zeros_like(u)
    return sum(torch.nn.functional.mse_loss(m.values(m.features(x)[t,b],u),y) for m in q)

def evaluate(q,episodes,arm):
    records=[]
    with torch.no_grad():
        for e in episodes:
            x=e['x'][:,None].cuda();u=e['u'].cuda()
            if arm=='blind':u=torch.zeros_like(u)
            pred=torch.stack([m.values(m.features(x)[:-1,0],u) for m in q]).mean(0).cpu();err=(pred-e['ret']).square();records.append(dict(id=e['id'],sha256=e['sha256'],points=e['n'],MSE=float(err.mean()),return0=float(e['ret'][0]),prediction0=float(pred[0]),min_return=float(e['ret'].min()),max_return=float(e['ret'].max())))
    return dict(episode_mean_MSE=sum(z['MSE'] for z in records)/len(records),episodes=records)

def fit(obj,episodes,p,arm,guard,event):
    rng=np.random.default_rng(p['learner_seed']);trace=[]
    for k in range(p['updates_per_arm']):
        guard();pp=pack(episodes,rng,p);obj[1].zero_grad(set_to_none=True);value=loss(obj[0],pp,arm);value.backward();norm=float(torch.nn.utils.clip_grad_norm_(obj[0].parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);obj[1].step();event('AFTER',k);trace.append(dict(step=k,loss=float(value.detach()),gradient_norm=norm))
    return trace

def write_checkpoint(obj,p,path):
    s=state(obj);buf=io.BytesIO();torch.save(dict(state=s,state_sha=L.digest(s)),buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['checkpoint_bytes']
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),bytes=len(blob),sha256=L.sha(path),state_sha=L.digest(s))
