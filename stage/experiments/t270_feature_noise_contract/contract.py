"""Zero-physics, zero-optimizer sequence density and gradient contract."""
from pathlib import Path
import hashlib
import io
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t270_feature_noise_contract'
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,np,torch,policy
import feature_noise as F


def main():
    start=time.monotonic();plan=json.loads((HERE/'protocol.json').read_text())
    assert not (OUT/'summary.json').exists(),'No expanded prefix/repeated contract'
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name,h in plan['input_hashes'].items():assert sha(name)==h,name
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(2026110114)
    features=torch.tensor(rng.normal(size=(17,32)),dtype=torch.float64);features/=features.norm(dim=1,keepdim=True)
    # Nonzero nearby-state similarities test persistent latent conditioning.
    features[1:]=.2*features[1:]+features[0];features/=features.norm(dim=1,keepdim=True)
    means=torch.tensor(rng.uniform(-.3,.3,size=(17,2)),dtype=torch.float64)
    z=torch.tensor(rng.normal(size=(17,2)),dtype=torch.float64)
    actions=torch.stack([F.AR.sample(m,a,native=False) for m,a in zip(means,z)])
    lp=F.sequence_logp(means,actions,features,block=7)
    dense=means.new_zeros(())
    for off in range(0,len(means),7):
        f=features[off:off+7];m=means[off:off+7];a=actions[off:off+7]
        zz,mar=F.AR.terms(m,a);K=F.LATENT_VAR*(f@f.T)+F.RESIDUAL_VAR*torch.eye(len(f),dtype=f.dtype)
        dist=torch.distributions.MultivariateNormal(torch.zeros(len(f),dtype=f.dtype),covariance_matrix=K)
        dense+=sum(dist.log_prob(zz[:,j]) for j in range(2))+(mar+.5*zz.square()+.5*F.AR.LOG2PI).sum()
    err=float((lp.sum()-dense).abs());assert err<1e-9
    prefix=F.sequence_logp(means[:5],actions[:5],features[:5],block=7)
    assert torch.equal(prefix,lp[:5])
    theta=torch.tensor(.01,dtype=torch.float64,requires_grad=True)
    direction=torch.tensor(rng.normal(size=(17,2)),dtype=torch.float64)
    value=F.sequence_logp(means+theta*direction,actions,features,block=7).sum()
    grad=float(torch.autograd.grad(value,theta)[0]);eps=1e-6
    fn=lambda v:float(F.sequence_logp(means+v*direction,actions,features,block=7).sum())
    finite=(fn(.01+eps)-fn(.01-eps))/(2*eps)
    relative=abs(grad-finite)/max(1.,abs(finite));assert relative<2e-6
    white=F.sequence_logp(means,actions,features,block=7,latent_var=0.,residual_var=1.)
    marginal=F.AR.terms(means,actions)[1].sum(-1)
    assert float((white-marginal).abs().max())<1e-10
    wrong=float((lp.sum()-marginal.sum()).abs());assert wrong>1e-3
    p=policy(plan).eval().requires_grad_(False);assert S.digest(p.state_dict())==plan['policy_hash']
    with np.load(plan['real_source']) as data:states=data['states'][:64].copy()
    actual_means=[];actual_features=[]
    with torch.no_grad():
        for row in states:
            mode,mean,hidden,raw=p.step(torch.tensor(row[None]));actual_means.append(mean[0]);actual_features.append(F.unit_feature(hidden))
    actual_means=torch.stack(actual_means);actual_features=torch.stack(actual_features)
    def sample(seed):
        random=np.random.default_rng(seed);out=[];stored=[]
        for k,(m,f) in enumerate(zip(actual_means,actual_features)):
            if k%F.BLOCK==0:
                weights=torch.tensor(random.normal(size=(32,2)),dtype=torch.float64)*np.sqrt(F.LATENT_VAR)
                pm,P=F.prior(f)
            latent=F.latent_sample(f,weights,random.normal(size=2));action=F.AR.sample(m,latent)
            val,pm,P=F.logp_step(m,action,f,pm,P);out.append(action);stored.append(val)
        return torch.stack(out),torch.stack(stored)
    a,record=sample(2026111001);b,again=sample(2026111001)
    assert torch.equal(a,b) and torch.equal(record,again)
    recovered=F.sequence_logp(actual_means,a,actual_features)
    real_error=float((record-recovered).abs().max());assert real_error==0
    buf=io.BytesIO();np.savez_compressed(buf,actions=a.numpy(),features=actual_features.numpy(),logp=record.numpy())
    with np.load(io.BytesIO(buf.getvalue())) as z:
        assert np.array_equal(z['features'],actual_features.numpy()) and np.array_equal(z['logp'],record.numpy())
    result=dict(state='PASS',physics_steps=0,optimizer_calls=0,navigation_RL_updates=0,
        dense_joint_vs_history_filter_logp_error=err,prefix_no_future_information=True,
        finite_difference_relative_error=relative,autograd=grad,finite_difference=finite,
        white_limit_matches_marginal=True,incorrect_marginal_joint_gap=wrong,
        actual64_rows_recorded_vs_recovered_density_error=real_error,
        exact_seed_replay=True,feature_dimension=32,block_decisions=50,latent_variance=.98,residual_variance=.02,
        native_action_bounds=[float(a.min()),float(a.max())],serialization_exact=True,
        frozen_actor_unchanged=S.digest(p.state_dict())==plan['policy_hash'],
        elapsed_s=time.monotonic()-start,protocol_sha256=sha(HERE/'protocol.json'),
        limitations=['Standard latent-Gaussian filtering and bounded-copula density, not a new theorem or trained algorithm.',
            'Frozen features only; actor gradients must retain past recovered latents/posterior means.',
            'No physics, old-task gate, collision avoidance or six-metre success demonstrated.'])
    assert result['elapsed_s']<plan['wall_seconds']
    OUT.mkdir(exist_ok=True);C= L.C;C.save(OUT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
