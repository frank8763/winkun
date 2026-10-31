"""Zero-physics recurrent density, reset, causality and gradient contracts."""
from learning import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t255_overnight'
def main():
    plan=json.loads((HERE/'protocol.json').read_text());assert not (OUT/'synthetic_contracts.json').exists()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);parent=AS.policy(read(plan['checkpoint'])['policy']).eval().requires_grad_(False);anchor=retention(plan,parent)
    p=Actor(parent.state_dict(),True,plan['arms'][0]['seed']);off=Actor(parent.state_dict(),False,plan['arms'][0]['seed']);assert digest(p.state_dict())==digest(off.state_dict())
    x=anchor['x'][:32,:4];base=parent(x[...,:117].reshape(-1,117).contiguous()).reshape(32,4,2)
    with torch.no_grad():
        init_error=float((p.sequence(x)[0]-base).abs().max());off_error=float((off.sequence(x)[0]-base).abs().max())
        assert init_error==off_error==0
    q=copy.deepcopy(p)
    with torch.no_grad():q.residual.weight.normal_(0,.01)
    hidden=None;steps=[]
    with torch.no_grad():
        for t in range(len(x)):mode,_,hidden,_=q.step(x[t],hidden);steps.append(mode)
        seq=q.sequence(x)[0];stream_error=float((seq-torch.stack(steps)).abs().max());assert stream_error<3e-6
        changed=x.clone();changed[16:]+=2;future_error=float((seq[:16]-q.sequence(changed)[0][:16]).abs().max());assert future_error==0
        reset_error=float((q.step(x[0],None)[0]-q.step(x[0],None)[0]).abs().max());assert reset_error==0
    loss=q.sequence(x)[0].square().mean();loss.backward();core_gradient=float(q.core.weight_hh_l0.grad.norm());assert core_gradient>0
    reports={}
    for flag in [True,False]:
        actor=p if flag else off;items=[];rng=np.random.default_rng(901)
        for task in range(4):
            for k in range(4):
                states=x[:,task].numpy();acts=[];means=[];lps=[];latent=None;h=None
                for state in states:
                    first=latent is None;eps=rng.normal(size=2);latent=eps if first else AR.RHO*latent+math.sqrt(1-AR.RHO**2)*eps
                    with torch.no_grad():
                        _,m,h,_=actor.step(torch.tensor(state[None]),h);m=m[0];a=AR.sample(m,latent);lp=AR.logp(m[None],a[None],(m if first else means[-1])[None],(a if first else acts[-1])[None],torch.tensor([first]))
                    means.append(m);acts.append(a);lps.append(float(lp))
                rewards=np.zeros(32);rewards[-1]=1 if k%2 else -1
                items.append(dict(row=dict(source_reset_id=str(task)),arrays=dict(states=states,actions=torch.stack(acts).numpy(),means=torch.stack(means).numpy(),logp=np.asarray(lps),native_counts=np.full(32,100,dtype=np.uint8),rewards=rewards)))
        data=training_batch(items,actor);lp,_=likelihood(actor,data);(-lp*data['advantage']).mean().backward();assert float(actor.residual.weight.grad.norm())>0
        reports[str(flag)]=data['density']
    r=dict(state='PASS',physics_steps=0,optimizer_calls=0,protocol_sha256=C.sha(HERE/'protocol.json'),initial_mode_error=init_error,matched_parameter_state_hash=digest(p.state_dict()),stream_sequence_error=stream_error,future_prefix_error=future_error,episode_reset_error=reset_error,recurrent_gradient_reachable=core_gradient,trainable_parameters=sum(q.numel() for q in p.parameters() if q.requires_grad),retention_ids=anchor['ids'],density=reports)
    C.save(OUT/'synthetic_contracts.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
