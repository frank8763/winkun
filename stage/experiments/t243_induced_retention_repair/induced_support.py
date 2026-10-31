"""Own-success regression with dense current-policy induced-state retention.

Supervised auxiliary consolidation, not additional RL or an external teacher.
The initial Actor itself is the T239 genuinely RL-updated research candidate.
"""
from pathlib import Path
import sys,os,json,time,io,gzip,copy,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t243_induced_retention_repair'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
B,S,C,R,np,torch=V.B,V.S,V.C,V.R,V.np,V.torch
digest,used,state_np=S.digest,S.used,S.state_np
def initial(plan):
    state=torch.load(io.BytesIO(gzip.decompress(Path(plan['initial_checkpoint']).read_bytes())),map_location='cpu',weights_only=False)['policy'];assert digest(state)==plan['initial_policy_hash'];return S.policy(state)
def groups(plan):
    p=initial(plan).eval().requires_grad_(False)
    with np.load(plan['induced_states']) as z:allx=z['states'].copy();off=z['offsets'];ok=z['success'];task=z['task'];ids=z['case_ids']
    assert len(ok)==211 and int(ok.sum())==210;gg=[]
    for t in range(8):
        x=torch.tensor(np.concatenate([allx[off[k]:off[k+1]] for k in range(211) if ok[k] and task[k]==t]))
        with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach()
        gg.append((x,y))
    for name in ['dev_task6_12','dev_task6_21']:
        k=int(np.flatnonzero(ids==name)[0]);assert ok[k];x=torch.tensor(allx[off[k]:off[k+1]])
        with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach()
        gg.append((x,y))
    with np.load(plan['repair_states']) as z:x=torch.tensor(z['states'])
    other=torch.load(io.BytesIO(gzip.decompress(Path(plan['own_success_checkpoint']).read_bytes())),map_location='cpu',weights_only=False)['policy'];other=S.policy(other).eval().requires_grad_(False)
    with torch.no_grad():y=C.ParentPolicy.forward(other,x).detach()
    gg.append((x,y));assert len(gg)==11;return gg
def train(plan,g,report,check):
    torch.manual_seed(plan['seed']);p=initial(plan);opt=torch.optim.Adam([q for q in p.parameters() if q.requires_grad],lr=plan['lr']);rng=np.random.default_rng(plan['seed']);sizes=[16]*8+[32]*3
    for step in range(plan['updates']):
        if step%50==0:check()
        ind=[rng.integers(0,len(x),n) for (x,y),n in zip(g,sizes)];x=torch.cat([q[0][j] for q,j in zip(g,ind)]);y=torch.cat([q[1][j] for q,j in zip(g,ind)]);errors=(C.ParentPolicy.forward(p,x)-y).square().mean(-1)
        loss=errors[:128].mean()+sum(errors[k:k+32].mean() for k in [128,160,192]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],1.,error_if_nonfinite=True);opt.step()
        if (step+1)%500==0:report(step+1,digest(p.state_dict()),digest(opt.state_dict()),float(loss.detach()))
    return p,opt
def worker(pipe,index,deadline):V.OUT=OUT;V.worker(pipe,index,deadline)
