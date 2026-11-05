"""Finite-data linear steering-head preservation rank; no parameter fit or native."""
from t371_core import *


def main():
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Capacity audit60s')));signal.alarm(60)
    torch.set_num_threads(1);p=read(HERE/'protocol.json');q=NightActor(p).eval().requires_grad_(False);a=arrays(p['repair_source']['path']);x=torch.tensor(a['states'])
    with torch.no_grad():
        z=q.encoder(((x.double()-q.mean)/q.scale).clamp(-10,10));h=q.core(z[:,None])[0][:,0].numpy()
    design=np.column_stack([h,np.ones(len(h))]);prefix=design[:248];repair=design[248:253];u,sv,vt=np.linalg.svd(prefix,full_matrices=False)
    tolerance=max(prefix.shape)*np.finfo(np.float64).eps*sv[0];rank=int(np.sum(sv>tolerance))
    before=read_model(p['repair_parent']['path']);s=read(OUT/'summary.json');after=read_model(s['model']['path'])
    delta=np.r_[after['head.weight'][1].numpy()-before['head.weight'][1].numpy(),float(after['head.bias'][1]-before['head.bias'][1])]
    result=dict(state='READONLY_CAPACITY_AUDIT_COMPLETE',new_native=0,parameter_fits=0,gradient_calls=0,optimizer=0,elapsed_s=time.time()-start,
        source_sha256=K.C.sha(p['repair_source']['path']),prefix_decisions=248,features=17,singular_values=sv.tolist(),rank_tolerance=float(tolerance),numerical_rank=rank,condition_number=float(sv[0]/sv[-1]),exact_preservation_nullity=17-rank,
        t371_fitted_logit_delta_prefix_rms=float(np.sqrt(np.mean((prefix@delta)**2))),t371_fitted_logit_delta_repair_rms=float(np.sqrt(np.mean((repair@delta)**2))),
        interpretation='At the displayed finite-precision SVD tolerance, a full-column-rank prefix design has no nonzero linear-head change preserving all248prefix logits exactly. This does not prove approximate preservation impossible, nonlinear adaptation sufficient, or unique cause of collision. No new coefficients were fitted.')
    save(DOC/'T371_LINEAR_HEAD_CAPACITY.json',result);signal.alarm(0);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
