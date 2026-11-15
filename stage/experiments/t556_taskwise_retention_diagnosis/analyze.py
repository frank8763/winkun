"""Saved-checkpoint diagnosis of individual retention histories; no optimizer/native."""
from pathlib import Path
import json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'))
import t552_actor as A
torch,np=A.torch,A.np
OUT=ROOT/'stage/runs/t556_taskwise_retention_diagnosis'
DOC=ROOT/'stage/docs/day_20261115'

def main():
    assert not OUT.exists();start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
    root=ROOT/'stage/runs/t555_adam_direction_control';p0=A.read(root/'protocol.json');p=A.read(p0['actor_configuration']);p.update(p0)
    seal=A.read(root/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(i)).exists() for i in seal['scientific_pids'])
    data,pairs=A.cached_dataset(p);reports=[];inputs={str(root/'protocol.json'):A.sha(root/'protocol.json')}
    arms=[('T555_project',root/'round_00_project'),('T555_scalar',root/'round_00_scalar'),('T552_rl',ROOT/'stage/runs/t552_joint_robust_night/round_01_rl')]
    for name,folder in arms:
        s=A.read(folder/'summary.json')['result'];assert A.read(folder/'audit.json')['state']=='PASS'
        check=s['check'];before=set(p['before_check']['success_ids']);now=set(check['success_ids']);lost=sorted(before-now)
        models={stage:A.frozen(p,m['path']) for stage,m in [('initial',p['initial']),('behavior',s['behavior']),('candidate',s['proposal'])]}
        delta=torch.cat([(a-b).detach().flatten() for a,b in zip(models['candidate'].extra.parameters(),models['behavior'].extra.parameters())]);stage_metrics={}
        for stage,q in models.items():
            q.extra.requires_grad_(True);metrics=[]
            for g in data:
                loc=A.location(q,g);e=A.P.action_loss(loc,g['target']).mean(-1) if g['kind']=='restore' else (loc-g['target']).square().mean(-1)
                loss=.5*e.mean()+.5*e.topk(max(1,math.ceil(.2*len(e)))).values.mean()
                v=torch.cat([z.flatten() for z in torch.autograd.grad(loss,list(q.extra.parameters()))])
                d=(loc-g['target']).detach();command=(loc.clamp(-1,1)-g['target'].clamp(-1,1)).detach()
                metrics.append(dict(id=g['id'],kind=g['kind'],loss=float(loss.detach()),command_RMS=float(command.square().mean().sqrt()),command_absmax=float(command.abs().max()),location_RMS=float(d.square().mean().sqrt()),gradient_norm=float(v.norm()),gradient_dot_actual_RL_delta=float(v@delta),known_lost=g['id'] in lost))
            stage_metrics[stage]=metrics;q.extra.requires_grad_(False)
        comparisons=[]
        for a,b in zip(stage_metrics['behavior'],stage_metrics['candidate']):
            assert a['id']==b['id']
            comparisons.append(dict(id=a['id'],kind=a['kind'],known_lost=a['known_lost'],loss_BC=a['loss'],loss_after_RL=b['loss'],loss_delta_RL=b['loss']-a['loss'],linear_prediction=a['gradient_dot_actual_RL_delta'],command_RMS_BC=a['command_RMS'],command_RMS_after_RL=b['command_RMS']))
        retention=[x for x in comparisons if x['kind']!='restore'];lostrows=[x for x in retention if x['known_lost']]
        meta=dict(arm=name,closed_loop_check=check,lost_success_ids=lost,gained_success_ids=sorted(now-before),models={k:s[k] for k in ['behavior','proposal']},stages=stage_metrics,per_history_RL_change=comparisons,retention_histories=len(retention),retention_histories_RL_loss_increase=sum(x['loss_delta_RL']>1e-18 for x in retention),lost_histories_RL_loss_increase=sum(x['loss_delta_RL']>1e-18 for x in lostrows),retention_per_kind_average_RL_change={k:sum(x['loss_delta_RL'] for x in retention if x['kind']==k)/sum(x['kind']==k for x in retention) for k in ['keep','neighbor']})
        reports.append(meta)
        for f in [folder/'summary.json',folder/'audit.json',folder/'cases.jsonl']:inputs[str(f)]=A.sha(f)
        for m in [s['behavior'],s['proposal']]:assert A.sha(m['path'])==m['sha256'];inputs[m['path']]=m['sha256']
        print(name,'lost',len(lost),'retention increases',meta['retention_histories_RL_loss_increase'],'of',len(retention),'averages',meta['retention_per_kind_average_RL_change'],flush=True)
    # At zero output layer, hidden-layer gradients from the output are zero.
    # Audit only the tangent expressiveness of initial final linear output.
    q=models['initial'];keep=[g for g in data if g['kind'] in ['keep','neighbor']]
    with torch.no_grad():
        X=torch.cat([torch.cat([q.extra[:-1](g['f']),torch.ones(len(g['f']),1,dtype=torch.float64)],1) for g in keep]);sv=torch.linalg.svdvals(X)
    assert len(sv)==33 and torch.isfinite(sv).all()
    tangent=dict(rows=len(X),columns=33,singular_values=sv.tolist(),relative_smallest=float(sv[-1]/sv[0]),numerical_rank_descriptive={str(tol):int((sv>sv[0]*tol).sum()) for tol in [1e-4,1e-6,1e-8,1e-10]},last_layer_zero_init=True,not_full_network_capacity_or_impossibility_proof=True,meaning='Only tangent output-layer matrix on consumed old histories. No claim old/new behaviors physically incompatible.')
    for g in p['restore_rows']+p['keep_rows']+p['neighbor_rows']:assert A.sha(g['path'])==g['sha256'];inputs[g['path']]=g['sha256']
    out=dict(state='COMPLETE_SAVED_ONLY',epoch=time.time(),pid=os.getpid(),elapsed_s=time.time()-start,new_native=0,new_optimizer=0,new_rollouts=0,gradient_evaluations=3*3*len(data),source_sha256=A.sha(__file__),inputs=inputs,reports=reports,initial_output_tangent=tangent,limitations='Saved state errors and finite endpoint checkpoints do not locate exact native divergence or identify BC-only closed-loop outcomes. No behavior closed-loop evaluation was executed. No causal claim or novelty/cross-layout claim.')
    OUT.mkdir();A.save(OUT/'diagnostic.json',out)
    raw=(OUT/'diagnostic.json').read_text();A.save(DOC/'T556_RAW_EVIDENCE.json',dict(original_path=str(OUT/'diagnostic.json'),original_sha256=A.sha(OUT/'diagnostic.json'),original_utf8=raw,reversible=True,new_native=0,new_optimizer=0))
    print('T556_COMPLETE',len(raw),tangent['numerical_rank_descriptive'],flush=True)
if __name__=='__main__':main()
