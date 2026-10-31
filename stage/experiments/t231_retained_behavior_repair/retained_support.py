"""Warm-start behavior repair with complete successful frozen-state retention."""
from pathlib import Path
import sys,os,json,time,traceback,copy,io,gzip,math
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t231_retained_behavior_repair'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
B,S,C,R,np,torch=V.B,V.S,V.C,V.R,V.np,V.torch
digest,used,state_np=S.digest,S.used,S.state_np

def initial(plan):
    state=torch.load(io.BytesIO(gzip.decompress(Path(plan['initial_checkpoint']).read_bytes())),map_location='cpu',weights_only=False)['policy'];assert digest(state)==plan['initial_policy_hash'];return S.policy(state)

def groups(plan):
    p=initial(plan);result=[]
    with np.load(plan['retention_targets']) as z:x=torch.tensor(z['states']);offset=z['offsets']
    with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach()
    for i in range(7):result.append((x[offset[i]:offset[i+1]],y[offset[i]:offset[i+1]]))
    # OriginalLEFT input kept, but labels preserve the already-deployed T228
    # neural function rather than forcing its old noisy exploration actions.
    with np.load(plan['original_left_states']) as z:x=torch.tensor(z['states'])
    with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach()
    result.append((x,y))
    for r in plan['retained_sources']:
        assert C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:x=torch.tensor(z['states'])
        with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach()
        i=r['task'];result[i]=(torch.cat([result[i][0],x]),torch.cat([result[i][1],y]))
    cstate=torch.load(io.BytesIO(gzip.decompress(Path(plan['complementary_checkpoint']).read_bytes())),map_location='cpu',weights_only=False)['policy'];complement=S.policy(cstate).eval().requires_grad_(False)
    for r in plan['correction_sources']:
        assert C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:x=torch.tensor(z['states']);y=torch.tensor(z['actions'])
        if r['target_kind']=='T221_preclip':
            with torch.no_grad():y=C.ParentPolicy.forward(complement,x).detach()
        i=r['task'];result[i]=(torch.cat([result[i][0],x]),torch.cat([result[i][1],y]))
    return result

def train(plan,g,report,check):
    torch.manual_seed(plan['seed']);p=initial(plan);opt=torch.optim.Adam([q for q in p.parameters() if q.requires_grad],lr=plan['lr']);rng=np.random.default_rng(plan['seed'])
    for step in range(plan['updates']):
        if step%50==0:check()
        idx=[rng.integers(0,len(x),plan['batch_each_task']) for x,y in g];x=torch.cat([q[0][i] for q,i in zip(g,idx)]);y=torch.cat([q[1][i] for q,i in zip(g,idx)])
        loss=(C.ParentPolicy.forward(p,x)-y).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],1.,error_if_nonfinite=True);opt.step()
        if (step+1)%500==0:report(step+1,digest(p.state_dict()),digest(opt.state_dict()),float(loss.detach()))
    return p,opt

def worker(pipe,index,deadline):
    V.OUT=OUT;V.worker(pipe,index,deadline)
