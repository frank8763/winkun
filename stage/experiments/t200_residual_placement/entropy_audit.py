"""Post-hoc offline physical-reward/entropy scale accounting, no optimizer/physics."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import re,math
from common import *
from replay import build,GAMMA

def main():
    torch.set_num_threads(1)
    plan=json.loads((HERE/'protocol.json').read_text());assert json.loads((OUT/'summary.json').read_text())['state']=='COMPLETE'
    episodes=[json.loads(s) for s in (OUT/'episodes.jsonl').read_text().splitlines()];cache={};rows=[]
    for row in episodes:
        assert sha(row['path'])==row['sha256'] and sha(row['behavior_checkpoint'])==row['behavior_sha256']
        with np.load(row['path']) as z:a={k:z[k].copy() for k in z.files}
        a['end_flags']=a['end_flags'].astype(float);transitions=build(a)
        path=row['behavior_checkpoint']
        if path not in cache:
            with np.load(path) as z:cache[path]={k:torch.tensor(z[k]) for k in z.files}
        p=Policy(row['placement']).eval().requires_grad_(False);p.load_state_dict(cache[path])
        with torch.no_grad():mu=torch.cat([p.latent_mean(x) for x in torch.tensor(a['states']).split(64)]).numpy().astype(float);sigma=p.logstd.clamp(-3,0).exp().numpy().astype(float)
        batch,index=map(int,re.search(r'_b(\d+)_w(\d+)$',row['id']).groups());epsilon=np.random.default_rng(plan['seed']+batch*100+index).normal(size=mu.shape);z=mu+sigma*epsilon
        log_normal=-.5*epsilon**2-np.log(sigma)-.5*math.log(2*math.pi)
        log_jacobian=2*(math.log(2)-z-np.logaddexp(0,-2*z));logp=(log_normal-log_jacobian).sum(1)
        weights=GAMMA**a['steps'];entropy=-plan['alpha']*float(weights@logp)
        value=0.
        for r,d in zip(transitions['rewards'][::-1,0],transitions['discounts'][::-1,0]):value=float(r)+float(d)*value
        steps=row['physics_steps'];time_cost=-.000002*(1-GAMMA**steps)/(1-GAMMA)
        rows.append({'id':row['id'],'arm':row['arm'],'group':'near_LEFT' if row['snapshot_side'] else 'original_LEFT','event':'GOAL' if row['success'] else 'CONTACT' if row['collision'] else 'TIMEOUT','environment_complete_return':value,'sampled_behavior_entropy_contribution_including_current_action':entropy,'environment_plus_sampled_behavior_entropy':value+entropy,'discounted_time_cost_only':time_cost,'entropy_magnitude_over_time_cost':abs(entropy/time_cost),'mean_sample_log_density':float(logp.mean())})
    grouped={}
    for arm in ['native','latent']:
        grouped[arm]={}
        for group in ['original_LEFT','near_LEFT']:
            subset=[r for r in rows if r['arm']==arm and r['group']==group]
            grouped[arm][group]={'n':len(subset),'mean_environment_return':float(np.mean([r['environment_complete_return'] for r in subset])),'mean_sampled_entropy_contribution':float(np.mean([r['sampled_behavior_entropy_contribution_including_current_action'] for r in subset])),'entropy_over_time_cost_median':float(np.median([r['entropy_magnitude_over_time_cost'] for r in subset]))}
    result={'state':'COMPLETE','analysis_type':'POST_HOC_OFFLINE_ONLY','physics_steps':0,'optimizer_updates':0,'alpha_per_decision':plan['alpha'],'nominal_decision_dt_s':.2,'normalized_time_cost_rate_per_s':.001,'maximum_bounded2D_action_entropy_nats':math.log(4),'entropy_rate_upper_bound_per_s':plan['alpha']*math.log(4)/.2,'upper_bound_over_time_cost_rate':plan['alpha']*math.log(4)/.2/.001,'grouped':grouped,'episodes':rows,'scope':'Reconstructs sampled on-policy behavior entropy with saved latent noise/actors, not a counterfactual next-policy target-Q calibration. Includes current entropy like Actor soft objective; does not claim these totals equal logged training-Q or prove entropy caused waiting. Differential entropy can be negative near saturated actions. Fixed action frequency and finite absorbing task make physical-time accounting relevant. Post-hoc hypothesis only; no new training or navigation success.'}
    save(OUT/'entropy_scale_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))

if __name__=='__main__':main()
