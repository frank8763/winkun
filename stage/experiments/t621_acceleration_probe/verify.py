"""Compare the old loss and GPU adaptation; do not apply an optimizer."""
from pathlib import Path
import copy,time,json
import accelerated as A
from probe import gradients
L,torch=A.L,A.torch
ROOT=A.ROOT;DOC=ROOT/'stage/docs/day_20261117/t621'

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    p=L.read(ROOT/'stage/experiments/t620_recurrent_prior_td3/protocol.json');c=L.read(p['actor_configuration']);c.update(p);p=c
    rows=L.read(ROOT/'stage/runs/t620_recurrent_prior_td3/direct_00/summary.json')['result']['online_rows']
    pack=L.batch(L.dataset(p,rows),p,L.np.random.default_rng(p['learner_seed']));values=[];gg=[]
    for implementation in ['immutable_original','new_cpu','new_gpu','new_gpu_repeat']:
        device='cuda' if 'gpu' in implementation else 'cpu';obj=L.make(p);before=L.digest(L.state(obj));A.move(obj,device);ref=L.frozen(p,p['student_initial']['path']).to(device);pp=A.pack_to(pack,device);vv=[];gr=[]
        for actor in [False,True]:
            for q in obj['critics']:q.requires_grad_(not actor)
            obj['actor_opt'].zero_grad(set_to_none=True);obj['critic_opt'].zero_grad(set_to_none=True);gen=torch.Generator().manual_seed(p['learner_seed']+1)
            loss=L.losses(obj,ref,pp,p,gen,actor)[0] if implementation=='immutable_original' else A.loss(obj,ref,pp,p,gen,actor)
            loss.backward();vv.append(float(loss.detach().cpu()));gr.append(gradients(obj,actor))
        A.move(obj,'cpu');assert L.digest(L.state(obj))==before;values.append(vv);gg.append(gr)
    for vv,gr in zip(values[1:],gg[1:]):
        assert max(abs(a-b) for a,b in zip(vv,values[0]))<1e-10
        assert max(float((a-b).abs().max()) for a,b in zip(gr,gg[0]))<1e-9
    assert values[-1]==values[-2] and all(torch.equal(a,b) for a,b in zip(gg[-1],gg[-2]))
    report=dict(state='PASS',original_CPU_and_adapted_CPU_and_GPU_losses_and_gradients_match=True,GPU_repeated_gradients_bitwise_identical=True,no_parameter_optimizer_or_target_changes=True,new_native=0,new_optimizer=0,losses=values,gradient_digests=[L.digest(z) for z in gg],epoch=time.time());L.save(DOC/'EQUIVALENCE.json',report);print('ORIGINAL_CPU_GPU_GRADIENT_EQUIVALENCE_PASS')

if __name__=='__main__':main()
