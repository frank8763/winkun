"""One successful-demo residual initialization and one fresh score-PG proposal.

Frozen 5m base; no learned critic, expert logp, runtime teacher or routing.
"""
from pathlib import Path
import copy,gzip,io,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t329_success_residual';DOC=ROOT/'stage/docs/distance_20261103'
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
def bc_data(p,q):
    a=arrays(p['demo']);x=torch.tensor(a['states']);y=torch.tensor(a['actions']);old,_=H.anchors(read(p['base_protocol']));old=torch.tensor(old)
    change=(a['actions'][:,1]!=a['states'][:,-2]);weights=np.where(change,.5/change.sum(),.5/(~change).sum()).astype(np.float32)
    with torch.no_grad():
        raw=torch.cat([q.frozen.step(v[None])[3] for v in x]);oldraw=torch.cat([q.frozen.step(v[None])[3] for v in old]);oldmode=q.mode_fn(oldraw)
    return dict(x=x,y=y,weights=torch.tensor(weights),raw=raw,old=old,oldraw=oldraw,oldmode=oldmode)
def normalized(q,x):return ((x-q.input_mean)/q.input_scale).clamp(-10,10)
def bc_backward(q,d,p):
    q.zero_grad(set_to_none=True);losses=[0.,0.]
    for start in range(0,len(d['x']),64):
        sl=slice(start,start+64);pred=q.mode_fn(d['raw'][sl]+q.correction(normalized(q,d['x'][sl])))
        loss=(((pred-d['y'][sl]).square().mean(-1))*d['weights'][sl]).sum();loss.backward();losses[0]+=float(loss.detach())
    for start in range(0,len(d['old']),64):
        sl=slice(start,start+64);pred=q.mode_fn(d['oldraw'][sl]+q.correction(normalized(q,d['old'][sl])))
        loss=(pred-d['oldmode'][sl]).square().sum()/(2*len(d['old']))*p['retention_weight'];loss.backward();losses[1]+=float(loss.detach())
    assert all(v.grad is None for v in q.frozen.parameters())
    return losses
def fit(p,q,opt,guard,event):
    d=bc_data(p,q);rows=[]
    for k in range(p['bc_steps']):
        guard();losses=bc_backward(q,d,p);norm=float(torch.nn.utils.clip_grad_norm_(q.correction.parameters(),1.,error_if_nonfinite=True));opt.step();opt.zero_grad(set_to_none=True)
        r=dict(kind='bc',step=k,losses=losses,gradient_norm=norm,policy_hash=K.S.digest(q.state_dict()),optimizer_hash=K.S.digest(opt.state_dict()));event(r);rows.append(r)
    return rows
def pg(p,q,opt,items,guard,event):
    assert len(items)==8;returns=[]
    for item in items:
        a=item['arrays'];disc=K.R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        returns.append(float(disc@a['rewards']))
    returns=np.asarray(returns);adv=returns-(returns.sum()-returns)/7;scale=max(.05,float(np.sqrt(np.mean(adv**2))));coef=adv/scale
    q.zero_grad(set_to_none=True);initial=[];max_mean=0.;max_lp=0.
    for c,item in zip(coef,items):
        guard();a=item['arrays'];m=means(q,torch.tensor(a['states']));lp=B.D.terms(m,torch.tensor(a['actions']))[1].sum(-1)
        max_mean=max(max_mean,float((m.detach()-torch.tensor(a['means'])).abs().max()));max_lp=max(max_lp,float((lp.detach()-torch.tensor(a['logp'])).abs().max()))
        assert max_mean<=3e-6 and max_lp<=.005
        (-float(c)*lp.sum()/8).backward();initial.append(lp.detach())
    d=bc_data(p,q)
    # Same fixed old anchor soft retention, all rows, no artificial expert logp.
    for start in range(0,len(d['old']),64):
        sl=slice(start,start+64);pred=q.mode_fn(d['oldraw'][sl]+q.correction(normalized(q,d['old'][sl])))
        ((pred-d['oldmode'][sl]).square().sum()/(2*len(d['old']))*p['retention_weight']).backward()
    assert all(v.grad is None for v in q.frozen.parameters())
    norm=float(torch.nn.utils.clip_grad_norm_(q.correction.parameters(),1.,error_if_nonfinite=True));opt.step();opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        delta=[]
        for old,item in zip(initial,items):
            a=item['arrays'];lp=B.D.terms(means(q,torch.tensor(a['states'])),torch.tensor(a['actions']))[1].sum(-1);delta.append(lp-old)
        delta=torch.cat(delta);k3=float((torch.exp(-delta)-1+delta).mean());maxratio=float(delta.abs().max())
    r=dict(kind='pg',step=0,returns=returns.tolist(),coefficients=coef.tolist(),scale=scale,gradient_norm=norm,
        mean_error=max_mean,logp_error=max_lp,k3=k3,max_abs_logratio=maxratio,numeric_pass=k3<=.02 and maxratio<=1,
        policy_hash=K.S.digest(q.state_dict()),optimizer_hash=K.S.digest(opt.state_dict()))
    event(r);return r
