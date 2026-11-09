"""One final serialization/full-length backward contract, zero Adam/native."""
from pathlib import Path
import time
import t441_learning as L

def main():
    root=L.ROOT;out=root/'stage/runs/t441_joint_actor_critic'
    p=L.read(out/'protocol.json');assert L.read(out/'preflight.json')['state']=='PASS'
    assert not (out/'final_contract.json').exists()
    start=time.monotonic();objects=L.make(p,'cuda');q,v,ao,vo=objects
    assert q.phash()==p['expected_policy_hash']
    model=L.export(objects,out/'preflight_initial.pt.gz',384*1024)
    mirror=L.C.frozen(p,model['path']).requires_grad_(False)
    assert mirror.phash()==q.phash()
    saved=L.read_state(model['path']);assert L.digest(saved)==model['joint_hash']
    xx=L.np.repeat(mirror.mean.numpy()[None],400,axis=0).astype(L.np.float32)
    xx[:,112]=L.np.linspace(1.,.0025,400)
    x=L.torch.tensor(xx,device='cuda')
    with L.torch.no_grad():
        b=mirror.base_actions(L.torch.tensor(xx))
        expected=mirror.sequence(L.torch.tensor(xx),b)
    actual=q.sequence(x,b.cuda())
    error=float((actual.detach().cpu()-expected).abs().max());assert error<1e-10
    # Verify both full400-step recurrent graphs, without an optimizer call.
    ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True)
    old=L.state(*objects)
    (actual.square().mean()+v(x).square().mean()).backward()
    assert all(L.torch.isfinite(t.grad).all() for obj in [q,v] for t in obj.parameters() if t.grad is not None)
    assert L.digest(old)==L.digest(L.state(*objects))
    L.torch.cuda.synchronize()
    L.save(out/'final_contract.json',dict(state='PASS',native=0,Adam_steps=0,
        serialized_new_format_native_decoder_policy_exact=True,model=model,
        full400_CPU_GPU_mean_max_error=error,full400_Actor_V_backward_finite=True,
        elapsed_s=time.monotonic()-start,peak_torch_allocated_bytes=L.torch.cuda.max_memory_allocated()))
    print('T441_FINAL_CONTRACT_PASS',flush=True)

if __name__=='__main__':main()
