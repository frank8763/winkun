"""Stage full MLP, exact successful exploration data and reconstruction recipe."""
from pathlib import Path
import sys,os,json,time,math,traceback,copy
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t221_left_success'
sys.path.insert(0,str(ROOT/'stage/experiments/t219_rnd_runtime'))
import support as S
sys.path.insert(0,str(ROOT/'stage/experiments/t220_temporal_noise'))
import temporal_support as TS
C,R,np,torch=S.C,S.R,S.np,S.torch
digest,state_np,used=S.digest,S.state_np,S.used

def worker(pipe,index,deadline,temporal):
    S.OUT=TS.OUT=OUT
    (TS.worker if temporal else S.worker)(pipe,index,deadline)

def groups(plan):
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p=S.policy(parent)
    with np.load(plan['retention_targets']) as z:x=torch.tensor(z['states']);offset=z['offsets']
    with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach()
    result=[(x[offset[i]:offset[i+1]],y[offset[i]:offset[i+1]]) for i in range(7)]
    with np.load(OUT/'verified_success.npz') as z:result.append((torch.tensor(z['states']),torch.tensor(z['actions'])))
    return parent,result

def train_recipe(plan,groups,steps,check=lambda:None,report=lambda *args:None):
    torch.manual_seed(plan['seed']);parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p=S.policy(parent)
    optimizer=torch.optim.Adam([q for q in p.parameters() if q.requires_grad],lr=plan['lr']);rng=np.random.default_rng(plan['seed'])
    counts=0
    for step in range(steps):
        if step%50==0:check()
        inds=[rng.integers(0,len(x),plan['batch_each_task']) for x,y in groups]
        x=torch.cat([g[0][i] for g,i in zip(groups,inds)]);y=torch.cat([g[1][i] for g,i in zip(groups,inds)])
        loss=(C.ParentPolicy.forward(p,x)-y).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],1.,error_if_nonfinite=True);optimizer.step();counts+=1
        if (step+1)%500==0:report(step+1,digest(p.state_dict()),digest(optimizer.state_dict()),float(loss.detach()))
    return p,optimizer,counts

def reconstruct():
    plan=json.loads((HERE/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text());recipe=json.loads((OUT/'recipe.json').read_text())
    assert summary['reconstruction_verified'] and recipe['completed_supervised_steps']==plan['updates']
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h
    assert C.sha(OUT/'verified_success.npz')==recipe['data_sha256']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    _,g=groups(plan);p,opt,count=train_recipe(plan,g,plan['updates'])
    assert digest(p.state_dict())==recipe['final_policy_sha256'] and digest(opt.state_dict())==recipe['final_optimizer_sha256']
    return p.eval().requires_grad_(False)
