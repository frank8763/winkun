"""Parent and updated-policy numerical identity plus disposable full-batch guard."""
from support import *
from learner import *
import fcntl,copy
def main():
    lock=(OUT/'contracts.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'contracts.json').exists()
    start=time.monotonic();plan=json.loads((HERE/'protocol.json').read_text())
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    checks=[];data=[]
    for row in plan['preflight_rows']:
        p=policy(torch.load(row['weights'],map_location='cpu',weights_only=False)['policy'])
        with np.load(row['trace']) as z:
            x=torch.tensor(z['states']);actions=torch.tensor(z['actions']);means=torch.tensor(z['means']);logp=torch.tensor(z['logp'])
            result=verify_behavior(p,x,actions,means,logp,plan);checks.append({'id':row['id'],**result})
            if row['round']==0:data.append((x,actions,logp,torch.tensor(monte_carlo(z['rewards'],z['native_counts']))))
        try:verify_behavior(p,x,actions,means+.001,logp,plan)
        except AssertionError:pass
        else:raise AssertionError('Perturbed behavior identity not rejected')
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p=policy(parent)
    x,actions,oldlogp,returns=[torch.cat([d[i] for d in data]) for i in range(4)]
    with np.load(plan['retention_targets']) as z:anchor_x=torch.tensor(z['states']);anchor_y=torch.tensor(z['actions']);offsets=z['offsets'].copy()
    with torch.no_grad():
        assert float((p(anchor_x)-anchor_y).abs().max())==0
        parent_anchor_mean=p.physical_mean(anchor_x).clone()
        advantage=(returns-returns.mean())/returns.std(unbiased=False).clamp_min(1e-8)
    opt=torch.optim.Adam([q for q in p.parameters() if q.requires_grad],lr=plan['actor_lr'])
    one=copy.deepcopy(plan);one['actor_attempts_per_round']=1
    result=actor_update(p,opt,x,actions,oldlogp,advantage,anchor_x,anchor_y,offsets,parent_anchor_mean,one,lambda:None)
    assert result['accepted_updates']==1 and result['first_layer_gradient_norm']>0
    accepted=[r for r in result['attempts'][0]['attempts'] if r['accept']]
    assert len(accepted)==1
    row={'state':'PASS','protocol_sha256':C.sha(HERE/'protocol.json'),'physics_steps':0,'navigation_optimizer_updates':0,
         'disposable_preflight_Adam_proposals':1,'disposable_preflight_accepted':1,'online_initialization_remains_parent':True,
         'behavior_checks':checks,'mutated_means_rejected':True,'fullbatch_candidate':result,
         'elapsed_s':time.monotonic()-start,'model_exported':False}
    assert time.monotonic()-start<90 and used(OUT)<1024**2
    C.save(OUT/'contracts.json',row);print(json.dumps(row))
if __name__=='__main__':main()
