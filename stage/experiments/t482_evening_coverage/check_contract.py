"""Zero-native/zero-Adam actual-history anchor equivalence and finite max test."""
from pathlib import Path
import json
import t482_learning as L

def main():
    L.torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[3]
    p=L.read(root/'stage/runs/t481_fresh16_rehearsal/protocol.json');p=dict(p,anchor_mode='targeted_mean')
    obj=L.P.make(p,device='cpu');initial=L.P.checkpoint(p['initial']['path'])['joint_state'];L.P.load(obj,initial)
    q=obj[0];before=L.digest(L.P.state(*obj));refs=dict(old=[],own=[],actual_old=[])
    with L.np.load(p['old_states']) as z:
        for task in range(8):
            i=L.np.flatnonzero(z['task']==task)[0]
            x=L.torch.tensor(z['states'][z['offsets'][i]:z['offsets'][i+1]])
            with L.torch.no_grad():target=q.sequence(x).clone()
            refs['old'].append(dict(x=x,target=target,task=task))
    for key,rows in [('own',p['retention_rows']),('actual_old',p['replay_rows'])]:
        for r in rows:
            assert L.sha(r['path'])==r['sha256'];a=L.E.N.S.arrays(r['path']);x=L.torch.tensor(a['states'])
            with L.torch.no_grad():target=q.sequence(x).clone()
            refs[key].append(dict(id=r['job_id'],x=x,target=target,side='LEFT' if '_LEFT_' in r['job_id'] else 'RIGHT'))
    with L.torch.no_grad():
        z=L.keep_loss(q,refs,0);assert all(abs(float(v))<1e-24 for v in z)
        # Discarded in-RAM perturbation, no optimizer and no file export.
        next(q.parameters()).add_(1e-4)
        old=L.F.keep_loss(q,refs,0);new=L.keep_loss(q,refs,0)
        for a,b in zip(old,new):L.torch.testing.assert_close(a,b,rtol=0,atol=1e-12)
        q.p=dict(p,anchor_mode='targeted_max');mx=L.keep_loss(q,refs,0)
        assert float(mx[2])>=float(new[2]) and float(mx[0])>=float(new[0])
        L.P.load(obj,initial)
    assert L.digest(L.P.state(*obj))==before and all(t.grad is None for t in q.parameters())
    x=L.torch.tensor([1.,4.,2.,3.,0.,1.,2.,3.],dtype=L.torch.float64,requires_grad=True)
    y=L.aggregate(x,'targeted_max');g=L.torch.autograd.grad(y,x)[0]
    assert y==4 and L.torch.equal(g,L.torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.],dtype=x.dtype))
    z=L.torch.zeros(8,dtype=x.dtype,requires_grad=True);g=L.torch.autograd.grad(L.aggregate(z,'targeted_max'),z)[0]
    assert L.torch.equal(g,L.torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.],dtype=x.dtype))
    print(json.dumps(dict(state='PASS_REAL_HISTORY_MEAN_EQUIVALENCE_MAX_GRADIENT',native=0,optimizer_steps=0,
        actual_model_backward=0,discarded_in_RAM_manual_perturbation_restored=True,initial_joint_hash_unchanged=True,
        actual_history_mean_equals_T481=True,finite_max_not_less_than_mean=True,synthetic_grad_calls=2,
        original_all_parameters_Adam_preserved=True,formal_two_mode_preflight_not_executed=True)))
if __name__=='__main__':main()
