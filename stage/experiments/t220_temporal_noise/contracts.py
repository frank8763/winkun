"""Offline AR/copy-map/reset/marginal checks, no simulator stepping."""
from temporal_support import *
def main():
    plan=json.loads((HERE/'protocol.json').read_text());assert not (OUT/'contracts.json').exists()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h
    result={}
    for label,rho in plan['arms'].items():
        rng=np.random.default_rng(plan['seed']);n=4096;length=128
        z=rng.normal(size=(n,2));seq=[z.copy()]
        for _ in range(length-1):z=rho*z+math.sqrt(1-rho*rho)*rng.normal(size=(n,2));seq.append(z.copy())
        z=np.stack(seq,axis=1);corr=float(np.corrcoef(z[:,:-1].ravel(),z[:,1:].ravel())[0,1]);mean=float(z.mean());std=float(z.std())
        assert abs(mean)<.035 and abs(std-1)<.035 and abs(corr-rho)<.015
        m=torch.tensor([[1.2,-.3],[-1.2,1.3]],dtype=torch.float32);latent=torch.tensor([[0.,1.],[-1.,.5]],dtype=torch.float64)
        a=copula_action(m,latent);assert torch.isfinite(a).all() and (a.abs()<=1).all()
        u=torch.special.ndtr(latent);sigma=torch.tensor(R.SIGMA,dtype=torch.float64)
        lo=torch.special.ndtr((-1-m.double())/sigma);hi=torch.special.ndtr((1-m.double())/sigma)
        back=(torch.special.ndtr((a.double()-m.double())/sigma)-lo)/(hi-lo)
        assert float((back-u).abs().max())<2e-6
        # Initial latent is drawn identically for every arm; no leaked state.
        assert np.array_equal(np.random.default_rng(plan['seed']).normal(size=2),seq[0][0])
        result[label]={'rho':rho,'empirical_latent_mean':mean,'empirical_latent_std':std,'lag1_correlation':corr,'CDF_roundtrip_max_error':float((back-u).abs().max())}
    C.save(OUT/'contracts.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,protocol_sha256=C.sha(HERE/'protocol.json'),checks=result))
    print(json.dumps(result))
if __name__=='__main__':main()
