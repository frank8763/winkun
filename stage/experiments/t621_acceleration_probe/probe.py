"""Actual recorded full-prefix T620 loss/gradient benchmark, zero updates."""
from pathlib import Path
import os,time,json,traceback,copy
import accelerated as A
L,torch=A.L,A.torch
ROOT=A.ROOT;DOC=ROOT/'stage/docs/day_20261117/t621'

def gradients(obj,actor):
    params=L.variables(obj['actor']) if actor else list(obj['critics'].parameters())
    return torch.cat([v.grad.detach().cpu().flatten() for v in params])

def main():
    p=L.read(ROOT/'stage/experiments/t620_recurrent_prior_td3/protocol.json');c=L.read(p['actor_configuration']);c.update(p);p=c
    rows=L.read(ROOT/'stage/runs/t620_recurrent_prior_td3/direct_00/summary.json')['result']['online_rows']
    data=L.dataset(p,rows);pack=L.batch(data,p,L.np.random.default_rng(p['learner_seed']))
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=[];reference_grad=None;reference_value=None
    for device,threads in [('cpu',1),('cpu',2),('cpu',4),('cuda',4)]:
        torch.set_num_threads(threads);item=dict(device=device,threads=threads,new_optimizer=0,new_native=0)
        try:
            obj=L.make(p);before=L.digest(L.state(obj));fixed=L.fixed(obj['actor']);A.move(obj,device);ref=L.frozen(p,p['student_initial']['path']).to(device);pp=A.pack_to(pack,device)
            if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            values=[];grads=[];timings=[]
            for actor in [False,True]:
                for q in obj['critics']:q.requires_grad_(not actor)
                obj['actor_opt'].zero_grad(set_to_none=True);obj['critic_opt'].zero_grad(set_to_none=True)
                gen=torch.Generator().manual_seed(p['learner_seed']+1)
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();loss=A.loss(obj,ref,pp,p,gen,actor);loss.backward()
                if device=='cuda':torch.cuda.synchronize()
                timings.append(time.perf_counter()-start);values.append(float(loss.detach().cpu()));grads.append(gradients(obj,actor))
            if reference_grad is None:reference_grad=grads;reference_value=values
            errors=[float((a-b).abs().max()) for a,b in zip(grads,reference_grad)]
            scale=[max(1.,float(g.abs().max())) for g in reference_grad]
            assert all(e/s<1e-9 for e,s in zip(errors,scale)),errors
            assert max(abs(a-b) for a,b in zip(values,reference_value))<1e-10
            if device=='cuda':item['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
            A.move(obj,'cpu');assert L.digest(L.state(obj))==before and L.fixed(obj['actor'])==fixed
            item.update(state='PASS',critic_gradient_seconds=timings[0],actor_gradient_seconds=timings[1],losses=values,gradient_max_errors=errors,all_parameters_optimizer_and_targets_unchanged=True)
        except BaseException:item.update(state='FAILED',error=traceback.format_exc())
        records.append(item);L.save(DOC/'BENCHMARK.json',dict(state='MEASURING',records=records,new_optimizer=0,new_native=0));print(json.dumps(item),flush=True)
    assert records[0]['state']=='PASS'
    valid=[z for z in records if z['state']=='PASS'];best=min(valid,key=lambda z:z['critic_gradient_seconds']+.5*z['actor_gradient_seconds'])
    L.save(DOC/'BENCHMARK.json',dict(state='COMPLETE_ZERO_UPDATES',torch=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,affinity=sorted(os.sched_getaffinity(0)),records=records,best_measured=dict(device=best['device'],threads=best['threads']),single_gradient_timing_not_full_training_speed=True,new_optimizer=0,new_native=0,epoch=time.time()))

if __name__=='__main__':main()
