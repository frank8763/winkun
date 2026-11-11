"""CPU-only frozen replay-support snapshot; never steps either optimizer."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t480_old_task_rehearsal'))
import t480_learning as L


def main():
    L.torch.set_num_threads(1)
    root=ROOT/'stage/runs/t480_old_task_rehearsal';p=L.read(root/'protocol.json')
    cap=L.read(root/'capture/summary.json')['result']
    output=ROOT/'stage/docs/review_20261111/T480_REPLAY_SIGNAL_POSTHOC.json'
    assert not output.exists() and L.read(root/'capture/audit.json')['state']=='PASS'
    data=L.memory(p,cap['replay_rows'],'cpu');models={'initial':Path(p['initial']['path']),'proposal':root/'pair/proposal_B.pt.gz'}
    result={}
    with L.torch.no_grad():
        for label,f in models.items():
            obj=L.P.make(p,device='cpu');L.P.load(obj,L.P.checkpoint(f)['joint_state'])
            al,vl,stats=L.replay_losses(obj[0],obj[1],data,p,assert_behavior=label=='initial')
            result[label]=dict(model_sha256=L.sha(f),actor_auxiliary=float(al),value_auxiliary=float(vl),
                               task_stats=stats,all_behavior_densities_real_stochastic=True)
    out=dict(state='COMPLETE_FROZEN_REPLAY_SUPPORT_POSTHOC_NOT_NAVIGATION',epoch=time.time(),
             native=0,optimizer_steps=0,backward=0,CUDA=False,models=result,
             protocol_sha256=L.sha(root/'protocol.json'),source_sha256=L.sha(__file__),
             caveats=['Eight successful stochastic centers do not cover all275 old perturbations.',
                      'Auxiliary loss/support is descriptive, not causal or a success guarantee.',
                      'Formal whole383 validation remains in the unmodified active pipeline.'])
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(state=out['state'],native=0,optimizer_steps=0,backward=0,
        initial_value_loss=result['initial']['value_auxiliary'],proposal_value_loss=result['proposal']['value_auxiliary'],
        proposal_mean_weight=sum(s['weight_mean'] for s in result['proposal']['task_stats'])/8)))


if __name__=='__main__':main()
