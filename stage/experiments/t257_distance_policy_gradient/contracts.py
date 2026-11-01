"""Disposable synthetic proposal/shadow integration, no navigation interaction."""
from distance_learning import *
import json,math
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t257_distance_policy_gradient'
def main():
    plan=json.loads((HERE/'protocol.json').read_text());assert not (OUT/'contracts.json').exists()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=policy(plan);original=S.digest(p.state_dict());parent=copy.deepcopy(p).eval().requires_grad_(False);x,y,offset=anchors(plan,parent);mse,mx=anchor_measure(p,x,y,offset);assert float(mse)==float(mx)==0
    rng=np.random.default_rng(9101);items=[]
    for task in range(4):
        for k in range(4):
            xx=x[task*24:(task+1)*24].numpy();means=[];acts=[];logps=[];latent=None
            for row in xx:
                first=latent is None;eps=rng.normal(size=2);latent=eps if first else L.AR.RHO*latent+math.sqrt(1-L.AR.RHO**2)*eps
                with torch.no_grad():
                    _,mean,_,_=p.step(torch.tensor(row[None]));mean=mean[0];a=L.AR.sample(mean,latent);lp=L.AR.logp(mean[None],a[None],(mean if first else means[-1])[None],(a if first else acts[-1])[None],torch.tensor([first]))
                means.append(mean);acts.append(a);logps.append(float(lp))
            reward=np.zeros(24);reward[-1]=1 if k%2 else -1;items.append(dict(row=dict(source_reset_id=str(task)),arrays=dict(states=xx,means=torch.stack(means).numpy(),actions=torch.stack(acts).numpy(),logp=np.asarray(logps),native_counts=np.full(24,100,dtype=np.uint8),rewards=reward)))
    data=L.training_batch(items,p);q=copy.deepcopy(p);opt=L.optimizer(p,plan);qo=L.optimizer(q,plan);testplan=dict(plan,actor_attempts_per_round=1);live=update(p,opt,data,x,y,offset,testplan,0,lambda:None);audit=update(q,qo,data,x,y,offset,testplan,0,lambda:None);assert live==audit and S.digest(p.state_dict())==S.digest(q.state_dict()) and S.digest(opt.state_dict())==S.digest(qo.state_dict())
    result=dict(state='PASS',protocol_sha256=C.sha(HERE/'protocol.json'),physics_steps=0,navigation_RL_updates=0,disposable_synthetic_optimizer_proposals=len(live)+len(audit),initial_policy_hash=original,retention_states=len(x),initial_full_retention_MSE=0,initial_full_retention_max=0,density=data['density'],synthetic_proposals_shadow_exact=True,synthetic_report=live,old_models_unchanged=True)
    C.save(OUT/'contracts.json',result);print(json.dumps(dict(state='PASS',retention_states=len(x),shadow_exact=True,density=data['density'])),flush=True)
if __name__=='__main__':main()
