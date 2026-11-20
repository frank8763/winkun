"""Describe observed steering-rate differences; no causal attribution."""
from pathlib import Path
import json
import os
import time
import analyze as A


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    A.Resource.Raw.resources()
    os.sched_setaffinity(0,{14})
    A.torch.set_num_threads(1)
    p=A.M.read(A.ROOT/'stage/runs/t661r_value_state_isolation/protocol.json')
    summary=A.M.read(A.OUT/'RESULTS.json')
    source=A.ROOT/'stage/runs/t661r_value_state_isolation'
    refs=A.M.read(p['reference_contract'])['reference_rows']
    rows=A.read_rows(source)
    q=A.M.L.frozen(p,summary['models']['T501']['path'])
    qnew=A.M.L.frozen(p,summary['models']['retained_Value']['path'])
    assert A.M.L.fixed_hash(q)==A.M.L.fixed_hash(qnew)
    output={}
    for key,values in summary['results'].items():
        ref=next(x for x in refs if x['task_id']==key)
        row=next(x for x in rows if x['job_id']==key and x['full'])
        a=A.M.E.N.S.arrays(ref['source_path']);b=A.M.E.N.S.arrays(row['path'])
        index=values['retained_Value']['first_hidden_L2_001']['decision']
        delta=b['states'][index]-a['states'][index]
        indices=A.np.argsort(abs(delta))[-5:][::-1]
        with A.torch.no_grad():
            def hidden(arrays):
                x=A.torch.as_tensor(arrays['states'])
                h,_=q.core(q.encoder(q.normalized(x))[:,None])
                h=h[:,0].cpu().numpy()
                return A.np.vstack([A.np.zeros((1,64)),h[:-1]])
            errors=[float(abs(hidden(x)-x['hidden_before']).max()) for x in [a,b]]
        assert max(errors)<1e-10
        output[key]=dict(decision=index,seconds=index*.2,
            exact_recorded_hidden_reconstruction_errors=errors,
            highest_input_differences=[dict(index=int(j),reference=float(a['states'][index,j]),candidate=float(b['states'][index,j]),delta=float(delta[j])) for j in indices],
            LF_steering_rate_difference_rad_per_s=float(delta[115]*3),
            RF_steering_rate_difference_rad_per_s=float(delta[116]*3))
    report=dict(state='COMPLETE_ZERO_NATIVE_ZERO_OPTIMIZER',results=output,
        fixed_encoder_GRU_head_identical=True,
        channel_source='t251_actuator_recovery/actuator_support.py:22 plus t255_overnight/mem_actor.py:30',
        not_a_causal_steering_sensor_ablation=True,new_native=0,new_optimizer=0,new_GPU=0)
    A.M.save(A.OUT/'OBSERVATION_DIFFERENCES.json',report)
    A.M.save(A.DOC/'OBSERVATION_DIFFERENCES.json',report)
    print(json.dumps(dict(state=report['state'],hidden_reconstruction=True)))


if __name__=='__main__':main()
