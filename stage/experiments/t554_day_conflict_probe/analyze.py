"""Read-only saved T552 gradients and actual parameter displacement; no optimizer/native."""
from pathlib import Path
import os,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'))
import t552_actor as A
sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'))
import t519_learning as Credit
torch,np=A.torch,A.np
OUT=ROOT/'stage/runs/t554_day_conflict_probe'

def main():
    start=time.time();assert not (OUT/'diagnostic.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert set(os.sched_getaffinity(0))<={14,15}
    prior=ROOT/'stage/runs/t552_joint_robust_night';r=prior/'round_01_rl'
    root=A.read(prior/'protocol.json');p=A.read(root['actor_configuration']);p.update(root);p.update(A.read(r/'protocol.json'))
    summary=A.read(r/'summary.json')['result'];rows=[A.read_json if False else __import__('json').loads(x) for x in (r/'cases.jsonl').read_text().splitlines()]
    before=set(p['before_check']['success_ids']);after={x['job_id'] for x in rows[4:] if x['success'] and not x['collision']}
    lost=sorted(before-after);gained=sorted(after-before);assert len(lost)==3 and len(gained)==5
    data,pairs=A.cached_dataset(p)
    models={name:A.frozen(p,m['path']) for name,m in [('initial',p['initial']),('behavior',summary['behavior']),('candidate',summary['proposal'])]}
    def vector(q):return torch.cat([v.detach().flatten() for v in q.extra.parameters()])
    delta=vector(models['candidate'])-vector(models['behavior'])
    arrays=[A.G.E.N.S.arrays(x['path']) for x in rows[:4]]
    items,diags=Credit.peer_batch((models['behavior'],None),arrays,p)
    cached=[]
    for b in items:
        f,base=A.feature_mean(models['behavior'],b['x']);cached.append(dict(b,f=f,base=base))
    reports=[]
    for name,q in models.items():
        q.extra.requires_grad_(True);counts={k:sum(g['kind']==k for g in data) for k in ['restore','keep','neighbor']};loss={k:0. for k in counts};per=[]
        for g in data:
            loc=A.location(q,g);err=A.P.action_loss(loc,g['target']).mean(-1) if g['kind']=='restore' else (loc-g['target']).square().mean(-1)
            v=.5*err.mean()+.5*err.topk(max(1,__import__('math').ceil(.2*len(err)))).values.mean();loss[g['kind']]+=v/counts[g['kind']]
            d=loc.clamp(-1,1)-g['target'].clamp(-1,1)
            per.append(dict(id=g['id'],kind=g['kind'],loss=float(v.detach()),command_absmax=float(d.abs().max().detach()),command_RMS=float(d.square().mean().sqrt().detach()),lost_field=g['id'] in lost))
        policy=0.
        for b in cached:
            ratio=(A.P.L.density(A.location(q,b),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            policy-=torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(cached)
        components={'restore':loss['restore'],'retention':loss['keep']+loss['neighbor'],'ppo_same_saved_behavior':policy}
        grads={k:torch.cat([v.flatten() for v in torch.autograd.grad(v,list(q.extra.parameters()),retain_graph=True)]) for k,v in components.items()}
        grads['restore_plus_ppo']=grads['restore']+grads['ppo_same_saved_behavior']
        norms={k:float(v.norm()) for k,v in grads.items()}
        cosine={a+'__'+b:float(grads[a]@grads[b]/(grads[a].norm()*grads[b].norm())) if norms[a]*norms[b]>1e-24 else None for a,b in [('restore','retention'),('ppo_same_saved_behavior','retention'),('restore_plus_ppo','retention')]}
        reports.append(dict(model=name,losses={k:float(v.detach()) for k,v in components.items()},gradient_norms=norms,cosines=cosine,gradient_dot_actual_four_step_Adam_delta={k:float(v@delta) for k,v in grads.items()},per_history=per))
        q.extra.requires_grad_(False);print(name,norms,cosine,flush=True)
    inputs={str(prior/'protocol.json'):A.sha(prior/'protocol.json'),str(r/'protocol.json'):A.sha(r/'protocol.json'),str(r/'summary.json'):A.sha(r/'summary.json'),str(r/'cases.jsonl'):A.sha(r/'cases.jsonl')}
    for m in [p['initial'],summary['behavior'],summary['proposal']]:assert A.sha(m['path'])==m['sha256'];inputs[m['path']]=m['sha256']
    for g in p['restore_rows']+p['keep_rows']+p['neighbor_rows']+rows[:4]:inputs[g['path']]=g['sha256']
    assert all(A.sha(f)==h for f,h in inputs.items())
    OUT.mkdir(exist_ok=True)
    A.save(OUT/'diagnostic.json',dict(state='SAVED_ONLY_COMPLETE',epoch=time.time(),elapsed_s=time.time()-start,pid=os.getpid(),new_native=0,new_optimizer=0,new_training_trajectories=0,new_gradient_evaluations=9,lost=lost,gained=gained,actual_behavior_to_candidate_Adam_delta_norm=float(delta.norm()),reports=reports,source_sha256=A.sha(__file__),inputs=inputs,limitations='Saved-history gradients/counterfactual loss are not causal navigation proof. Initial PPO is off-behavior diagnostic only. No parameter updates, new native or replay. Three states do not establish persistent conflict over all optimizer steps.'))
    print('T554_SAVED_DIAG_COMPLETE',flush=True)
if __name__=='__main__':main()
