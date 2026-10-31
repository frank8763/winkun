"""Supplementary posthoc frozen guard check plus two toy Adam rollback probes."""
from pathlib import Path
import sys, json, copy
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage/experiments'))
sys.path.insert(0,str(ROOT/'stage/experiments/t210_full_parameter_progress'))
from support import *
from native_update_guard import truncated_gaussian_kl, select_proposal, guarded_adam_step

def main():
    torch.set_num_threads(1)
    out=OUT/'update_guard_contract.json';assert not out.exists()
    summary=json.loads((OUT/'summary.json').read_text());assert summary['all_workers_stopped']
    plan=json.loads((OUT/'protocol.json').read_text())
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']
    with np.load(plan['retention_targets']) as z: anchors=torch.tensor(z['states'])
    rows=[json.loads(s) for s in (OUT/'rollouts.jsonl').read_text().splitlines()]
    x=torch.tensor(np.concatenate([np.load(r['trace'])['states'] for r in rows if r['role']=='training' and r['round']==0 and r['arm']=='original']))
    baseline=policy(parent).requires_grad_(False)
    with torch.no_grad(): old_mean=baseline.physical_mean(x); anchor_mean=baseline.physical_mean(anchors)
    reports=[]
    for arm in plan['arms']:
        proposed=torch.load(OUT/'models'/f'{arm}_round0.pt',map_location='cpu',weights_only=False)['policy']
        p=policy(parent).requires_grad_(False)
        def evaluate(q):
            kl=truncated_gaussian_kl(old_mean,q.physical_mean(x))
            anchor_kl=truncated_gaussian_kl(anchor_mean,q.physical_mean(anchors))
            return {'accept':float(kl.mean())<=.01 and float(anchor_kl.max())<=.01,
                    'training_mean_KL':float(kl.mean()),'old7_anchor_max_KL':float(anchor_kl.max())}
        result=select_proposal(p,parent,proposed,evaluate)
        assert result['accepted'] and result['factor']==.125
        # All-reject branch restores every full Actor tensor, including unused scalars.
        rejected=select_proposal(p,parent,proposed,lambda q:{'accept':False})
        assert not rejected['accepted'] and all(torch.equal(v,p.state_dict()[k]) for k,v in parent.items())
        reports.append({'arm':arm,**result,'full_actor_rejection_restores_all_tensors':True})
    # Stateful failure checks are intentionally a tiny diagnostic, not navigation training.
    torch.manual_seed(210)
    toy=torch.nn.Linear(2,1);opt=torch.optim.Adam(toy.parameters(),lr=.01)
    def backward():
        opt.zero_grad();toy(torch.ones(1,2)).square().sum().backward()
    backward();opt.step()  # Populate Adam moments before probing rollback.
    old={k:v.clone() for k,v in toy.state_dict().items()};old_opt=copy.deepcopy(opt.state_dict())
    backward();result=guarded_adam_step(toy,opt,lambda q:{'accept':False})
    def same(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    assert not result['accepted'] and same(old,toy.state_dict()) and same(old_opt,opt.state_dict())
    backward()
    def fail(q):raise RuntimeError('Intentional evaluator failure')
    try:guarded_adam_step(toy,opt,fail)
    except RuntimeError:pass
    else:raise AssertionError('Expected evaluator error')
    assert same(old,toy.state_dict()) and same(old_opt,opt.state_dict())
    report={'state':'POSTHOC_GUARD_CONTRACT_PASS','physics_steps':0,'navigation_optimizer_updates':0,
            'toy_optimizer_step_calls':3,'toy_retained_updates':1,'full_actor_selection':reports,
            'nonempty_Adam_state_rejection_and_exception_rollback_verified':True,
            'models_exported':False,'next_online_experiment_started':False,
            'scope':'Local KL checks only, not full surrogate-improvement/online trainer integration or closed-loop retention. Factor1/8 is selected posthoc, not a preregistered result. No claim of five-meter success.',
            'source_sha256':{str(p):C.sha(p) for p in [Path(__file__),ROOT/'stage/experiments/native_update_guard.py']}}
    C.save(out,report);print(json.dumps(report))
if __name__=='__main__':main()
