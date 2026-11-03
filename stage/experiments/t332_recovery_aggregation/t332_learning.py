"""One successful-demo residual initialization and one fresh score-PG proposal.

Frozen 5m base; no learned critic, expert logp, runtime teacher or routing.
"""
from pathlib import Path
import copy,gzip,io,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t332_recovery_aggregation';DOC=ROOT/'stage/docs/distance_20261103'
sys.path.insert(0,str(ROOT/'stage/experiments/t319_imperfect_demo_head'))
import t319_core as H
sys.path.insert(0,str(ROOT/'stage/experiments/frozen_base_residual'))
from actor import FrozenBaseResidual
from mem_actor import Actor,R
sys.path.insert(0,str(ROOT/'stage/experiments/t308_iid_native_goal_value'))
import t308_core as B
np,torch,K=H.np,H.torch,H.K
read=H.read

def save(f,v):Path(f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
def model(p):
    base=K.policy(read(p['base_protocol'])).eval().requires_grad_(False)
    assert K.S.digest(base.state_dict())==p['frozen_policy_hash']
    return FrozenBaseResidual(base,R.center,Actor.mode,torch.tensor(p['input_mean']),torch.tensor(p['input_scale']),p['seed']).eval()
def optimizer(q,p,kind):return torch.optim.Adam(q.correction.parameters(),lr=p['bc_lr'] if kind=='bc' else p['pg_lr'])
def state(q):return {k:v.detach().clone() for k,v in q.state_dict().items() if not k.startswith('frozen.')}
def load(q,v):
    assert set(v)==set(state(q));q.load_state_dict(dict(q.state_dict(),**v),strict=True)
def patch(q,opt):
    b=io.BytesIO();torch.save(dict(residual=state(q),optimizer=opt.state_dict(),policy_hash=K.S.digest(q.state_dict()),optimizer_hash=K.S.digest(opt.state_dict()),frozen_policy_hash=K.S.digest(q.frozen.state_dict())),b)
    z=gzip.compress(b.getvalue(),mtime=0);assert len(z)<512*1024;return z
def read_patch(f):return torch.load(io.BytesIO(gzip.decompress(Path(f).read_bytes())),weights_only=False,map_location='cpu')
def packed(a):
    assert np.array_equal(a['states'][1:],a['next_states'][:-1])
    return dict({k:v for k,v in a.items() if k!='next_states'},last_next_state=a['next_states'][-1:])
def arrays(f):
    with np.load(f) as z:a={k:z[k].copy() for k in z.files}
    if 'last_next_state' in a:a['next_states']=np.concatenate([a['states'][1:],a.pop('last_next_state')])
    return a
def mode(q,x):return torch.cat([q.step(row[None])[0] for row in x])
def means(q,x):return torch.cat([q.step(row[None])[1] for row in x])
def bc_data(p,q,arm):
    assert arm in ['control','recovery']
    original=arrays(p['demo']);correction=arrays(p['recovery_demo'])
    assert len(correction['states'])==318 and p['human_start_decision']==150
    groups=[(original['states'],original['actions'])]
    if arm=='recovery':
        groups.append((correction['states'][150:],correction['actions'][150:]))
    xs=[];ys=[];ws=[]
    for x,y in groups:
        changed=y[:,1]!=x[:,-2];assert changed.any() and (~changed).any()
        weights=np.where(changed,.5/changed.sum(),.5/(~changed).sum())/len(groups)
        xs.append(x);ys.append(y);ws.append(weights)
    x=torch.tensor(np.concatenate(xs));y=torch.tensor(np.concatenate(ys));weights=torch.tensor(np.concatenate(ws).astype(np.float32))
    with np.load(read(p['base_protocol'])['retention_states']) as z:
        old=torch.tensor(z['states'].copy());offsets=z['offsets'].copy();tasks=z['task'].copy()
    assert len(old)==28629 and len(tasks)==211
    oldtasks=np.concatenate([np.full(int(offsets[k+1]-offsets[k]),int(tasks[k]),np.int64) for k in range(211)])
    counts=np.bincount(oldtasks,minlength=8);assert (counts>0).all()
    oldweights=torch.tensor((1/(8*counts[oldtasks])).astype(np.float32))
    with torch.no_grad():
        raw=torch.cat([q.frozen.step(v[None])[3] for v in x]);oldraw=torch.cat([q.frozen.step(v[None])[3] for v in old]);oldmode=q.mode_fn(oldraw)
    return dict(x=x,y=y,weights=weights,raw=raw,old=old,oldraw=oldraw,oldmode=oldmode,oldweights=oldweights,
        human_correction_rows=0 if arm=='control' else 168,policy_prefix_supervised_rows=0)

def normalized(q,x):return ((x-q.input_mean)/q.input_scale).clamp(-10,10)

def bc_backward(q,d,p):
    q.zero_grad(set_to_none=True);losses=[0.,0.]
    for start in range(0,len(d['x']),64):
        sl=slice(start,start+64);pred=q.mode_fn(d['raw'][sl]+q.correction(normalized(q,d['x'][sl])))
        loss=(((pred-d['y'][sl]).square().mean(-1))*d['weights'][sl]).sum();loss.backward();losses[0]+=float(loss.detach())
    for start in range(0,len(d['old']),512):
        sl=slice(start,start+512);pred=q.mode_fn(d['oldraw'][sl]+q.correction(normalized(q,d['old'][sl])))
        loss=((pred-d['oldmode'][sl]).square().mean(-1)*d['oldweights'][sl]).sum()*p['retention_weight'];loss.backward();losses[1]+=float(loss.detach())
    assert all(v.grad is None for v in q.frozen.parameters())
    return losses

def fit(p,q,opt,guard,event,arm,data=None):
    d=bc_data(p,q,arm) if data is None else data;rows=[]
    for k in range(p['bc_steps']):
        guard();losses=bc_backward(q,d,p);norm=float(torch.nn.utils.clip_grad_norm_(q.correction.parameters(),1.,error_if_nonfinite=True));opt.step();opt.zero_grad(set_to_none=True)
        r=dict(kind='bc',arm=arm,step=k,losses=losses,gradient_norm=norm,policy_hash=K.S.digest(q.state_dict()),optimizer_hash=K.S.digest(opt.state_dict()));event(r);rows.append(r)
    return rows
