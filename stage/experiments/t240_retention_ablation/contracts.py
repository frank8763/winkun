"""Zero-physics integration/gradient checks; synthetic actions aren't episodes."""
from ablation_support import *
def main():
    plan=json.loads((HERE/'protocol.json').read_text());assert not (OUT/'synthetic_contracts.json').exists()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=S.policy(read(plan['checkpoint']));original=digest(p.state_dict());items=[];rng=np.random.default_rng(plan['seed'])
    for reset in ['dev_task6_00','dev_task6_24']:
        with np.load(ROOT/'stage/runs/t236_closed_loop_audit'/(reset+'.npz')) as z:x=z['states'][:16].copy()
        for k in range(4):
            acts=[];means=[];lps=[];latent=None
            for state in x:
                first=latent is None;noise=rng.normal(size=2);latent=noise if first else AR.RHO*latent+math.sqrt(1-AR.RHO**2)*noise
                with torch.no_grad():m=p.physical_mean(torch.tensor(state[None]))[0];a=AR.sample(m,latent);lp=AR.logp(m[None],a[None],(m if first else means[-1])[None],(a if first else acts[-1])[None],torch.tensor([first]))
                acts.append(a);means.append(m);lps.append(float(lp))
            rew=np.zeros(16);rew[-1]=1 if k%2 else -1
            item=dict(row=dict(source_reset_id=reset),arrays=dict(states=x,actions=torch.stack(acts).numpy(),means=torch.stack(means).numpy(),logp=np.asarray(lps),native_counts=np.full(16,100,dtype=np.uint8),rewards=rew));items.append(item)
    data=batch(items,p,plan);m=p.physical_mean(data['x']);lp=AR.logp(m,data['actions'],m[data['prev']],data['actions'][data['prev']],data['reset']);loss=-(lp*data['advantage']*data['weights']).sum()/8/234;loss.backward();gn=float(p.body.net[0].weight.grad.norm());assert math.isfinite(gn) and gn>0 and digest(p.state_dict())==original
    r=dict(state='PASS',protocol_sha256=C.sha(HERE/'protocol.json'),physics_steps=0,optimizer_calls=0,synthetic_sequences=8,synthetic_states=128,first_layer_gradient_norm=gn,all_gradients_finite=all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in p.parameters()),parameters_unchanged=True,density_check=data['density_check']);assert r['all_gradients_finite'];C.save(OUT/'synthetic_contracts.json',r);print(json.dumps(r))
if __name__=='__main__':main()
