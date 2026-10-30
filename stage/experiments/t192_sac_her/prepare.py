"""Lock the paired pilot and verify reward/first-hit semantics without physics."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *
from replay import data,build,GAMMA

def main():
    assert not OUT.exists()
    previous=json.loads((REPO/'stage/experiments/t190_adaptive_start_curriculum/protocol.json').read_text())
    audit=json.loads((REPO/'stage/runs/t191_her_contract_audit/summary.json').read_text());assert audit['state']=='COMPLETE' and audit['max_original_observation_rebuild_error']==0
    checkpoint=REPO/'stage/runs/t190_adaptive_start_curriculum/checkpoint_latest.pt'
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'];old=ParentPolicy().eval();new=Policy().eval();old.load_state_dict(parent);new.load_state_dict(parent)
    with np.load(audit['cases'][0]['path']) as z:states=z['states']
    with torch.no_grad():error=float((old(torch.tensor(states))-new(torch.tensor(states))).abs().max())
    assert error<1e-6
    reward_errors=[]
    for row in audit['cases']:
        assert sha(row['path'])==row['sha256'];a=data(row['path']);tr=build(a);value=0.
        for reward,discount in zip(tr['rewards'][::-1,0],tr['discounts'][::-1,0]):value=float(reward)+float(discount)*value
        n=row['physics_steps'];event=1. if row['success'] else -1. if row['collision'] else -.5
        expected=-.000002*(1-GAMMA**n)/(1-GAMMA)+GAMMA**(n-1)*event+.03*float(np.linalg.norm(a['poses'][0,:2]-a['goal_xy']))
        reward_errors.append(abs(value-expected));assert abs(value-expected)<2e-5
    jobs=[]
    for side in ['LEFT','RIGHT']:
        c=next(c for c in previous['primary_cases'][-2:] if c['side']==side);jobs.append({'id':'near_'+side,'case':c,'snapshot_side':side})
    jobs.extend({'id':c['id'],'case':c} for c in previous['primary_cases'][-2:])
    plan={'experiment':'T192','seed':2026103092,'checkpoint':str(checkpoint),'primary_cases':previous['primary_cases'],'anchor_sources':previous['anchor_sources'],'retention_demonstrations':previous['retention_demonstrations'],'training_jobs':jobs,'seed_episodes':[r['path'] for r in audit['cases']],'initial_std':[.5,.5],'alpha':.001,'warmup_updates':1500,'updates_per_batch':500,'batches_per_arm':8,'episodes_per_arm':64,'max_physics_per_arm':64*23334,'wall_seconds':1800,
          'controlled_difference':'Only HER goal-relabel replay augmentation: no_her samples original transitions; her samples50% original50% retrospectively relabeled. Both use same four fixed T191 seed episodes, fresh twin256-256 Critics/targets, fresh optimizers, same T190 Actor parent, task schedule, exploration seeds and update counts. Fixed terminal-balanced25% replay in both. Single paired seed.',
          'algorithm':'SAC-style squashed Gaussian Actor with fixedalpha.001, twinQmin target, tau.005, normalized finite-horizon reward/10, exp(-.002/60) physics discount. 1500 critic-only warmup updates then500critic+500actor per batch. Actor bodylr1e-5, heads1e-4,logstd1e-4; criticlr3e-4. BC retention10 on six old tasks, history action columns frozen0. No T174/T175 or other historical Critic reuse.',
          'reward_contract':'Same task events +10GOAL,-10collision,-5finite timeout,-.01/sec and Phi=-.3distance, allnormalized/10. Recompute complete action-duration reward; future goals truncate at first physical radius hit, zero bootstrap; missing exact interior-terminal next observation excluded from target-network/entropy calls. No relabeled-goal actor execution or runtime waypoints.',
          'critic_gate':'Before first Actor update, observed seed terminal MAE of bothQ <.15 in each sampled group. This verifies terminal targets only, not complete Critic calibration or unseen action ranking. Full algorithm performance judged by frozen native outcomes, not this sanity gate.',
          'scope':'Bounded exploratory SAC+BC/HER pilot, not exact reproduction of canonical SAC or HER papers. One warm-start parent and seed; old6/original5m are consumed training tasks. No algorithm superiority, cross-map generalization or novelty claim. No automatic promotion; T159 retained. Max128 complete new training episodes, no automatic extension.',
          'budget':'Both arms64completeepisodes<=2986752physics; fixed final28evalcases; wall1800s. All newruns1GiB/free40GiB, owned latest/previous only, keep evaluated final and raw data. No Phase.'}
    files=[checkpoint,REPO/'stage/runs/t191_her_contract_audit/summary.json',Path(plan['retention_demonstrations'])]+list(HERE.glob('*.py'))+[Path(p) for p in plan['seed_episodes']]+[Path(v['snapshot_path']) for v in plan['anchor_sources'].values()]
    plan['input_hashes']={str(p):sha(p) for p in files};OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    save(OUT/'preparation.json',{'state':'PASS','initial_mean_action_error':error,'original_reward_mc_max_error':max(reward_errors),'physics_steps':0,'optimizer_updates':0})
    print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json'),'initial_action_error':error,'reward_error':max(reward_errors)}))

if __name__=='__main__':main()
