"""Check fixed hidden-feature duplicate conflicts, not new expert coverage."""
from t371_core import *


def main():
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Representation audit60s')));signal.alarm(60)
    torch.set_num_threads(1);p=read(HERE/'protocol.json');q=NightActor(p).eval().requires_grad_(False);old,g,factual=dataset(p,q,lambda:None)
    def features(ref):
        with torch.no_grad():
            x=ref['x'];z=q.encoder(((x.double()-q.mean)/q.scale).clamp(-10,10));return q.core(z[:,None])[0][:,0].numpy()
    gh=features(g);groups=[]
    for ref in [g,*old,*factual]:
        h=gh if ref is g else features(ref);idx=np.arange(len(h))
        if ref is g:idx=idx[~g['mask'].numpy()]
        groups.append((ref['id'],h[idx],ref['y'][idx,1].numpy(),idx))
    pool=np.concatenate([v[1] for v in groups]);targets=np.concatenate([v[2] for v in groups]);ids=np.concatenate([np.repeat(v[0],len(v[1])) for v in groups]);indices=np.concatenate([v[3] for v in groups]);items=[]
    for k in np.flatnonzero(g['mask'].numpy()):
        distance=np.sqrt(np.mean((pool-gh[k])**2,axis=1));near=int(np.argmin(distance));exact=np.max(np.abs(pool-gh[k]),axis=1)<=1e-12
        items.append(dict(repair_decision=int(k),closest_protected_history=str(ids[near]),closest_protected_decision=int(indices[near]),hidden_rms_distance=float(distance[near]),protected_target=float(targets[near]),repair_target=float(g['target'][k]),exact_duplicate_count=int(exact.sum()),incompatible_exact_duplicate_count=int(np.sum(exact & (np.abs(targets-float(g['target'][k]))>1e-10)))))
    result=dict(state='READONLY_REPRESENTATION_CHECK_COMPLETE',new_native=0,gradient_calls=0,parameter_fits=0,optimizer=0,elapsed_s=time.time()-start,protected_states=len(pool),repair_states=5,hidden_features=16,exact_duplicate_tolerance=1e-12,items=items,source_protocol_sha256=K.C.sha(HERE/'protocol.json'),source_repair_sha256=p['repair_source']['sha256'],scope='No exact duplicate conflict on this finite set does not establish sufficient nonlinear capacity, robustness or labeled coverage of new learner states.')
    save(DOC/'T371_FROZEN_REPRESENTATION.json',result);signal.alarm(0);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
