"""Zero-native/zero-Adam test of batch scaling on real immutable histories."""
from pathlib import Path
import json
import t481_learning as L


def main():
    L.torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[3]/'stage/runs/t480_old_task_rehearsal'
    p=L.read(root/'protocol.json');p=dict(p,fresh_groups=[0]*8+[1]*8)
    samples=L.read(root/'capture/summary.json')['result']['samples']
    arrays=[L.E.N.S.arrays(r['path']) for r in samples]
    obj=L.P.make(p,device='cpu');L.P.load(obj,L.P.checkpoint(p['initial']['path'])['joint_state'])
    state_before=L.digest(L.P.state(*obj))
    with L.torch.no_grad():
        four,oldstats=L.B.C.batch(obj,arrays,p)
        sixteen,newstats=L.batch(obj,arrays[:2]*4+arrays[2:]*4,p)
    index=list(range(2))*4+list(range(2,4))*4
    for i,j in enumerate(index):
        L.torch.testing.assert_close(sixteen[i]['adv'],four[j]['adv'],rtol=0,atol=1e-12)
        L.torch.testing.assert_close(sixteen[i]['target'],four[j]['target'],rtol=0,atol=1e-12)
    # Independent synthetic gradient: repeat4 must not multiply loss/gradient4x.
    t=L.torch.tensor(.1,dtype=L.torch.float64,requires_grad=True)
    losses=[(t-x)**2 for x in [.3,-.4,.6,-.8]]
    loss4=sum(losses)/4;g4=L.torch.autograd.grad(loss4,t)[0]
    losses=[(t-[.3,-.4,.6,-.8][j])**2 for j in index]
    loss16=sum(losses)/16;g16=L.torch.autograd.grad(loss16,t)[0]
    L.torch.testing.assert_close(loss4,loss16,rtol=0,atol=1e-14)
    L.torch.testing.assert_close(g4,g16,rtol=0,atol=1e-14)
    assert L.digest(L.P.state(*obj))==state_before and all(z.grad is None for z in obj[0].parameters())
    print(json.dumps(dict(state='PASS_REAL_HISTORY_BATCH_SCALING_ZERO_ADAM',native=0,optimizer=0,
        actual_model_backward=0,synthetic_grad_calls=2,real4_repeated16_NOT_independent_data=True,
        pooled_advantages_and_targets_match=True,loss_and_gradient_NOT_scaled4x=True,
        full_16fresh_native_and_shadow_preflight_still_required=True)))


if __name__=='__main__':main()
