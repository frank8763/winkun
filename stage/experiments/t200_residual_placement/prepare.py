"""Lock equal-size residual placement ablation and explicit peak storage reserves."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *
from replay import build

def main():
    assert not OUT.exists()
    audit=REPO/'stage/runs/t199_sac_mean_gradient/summary.json';summary=json.loads(audit.read_text());assert summary['state']=='COMPLETE' and max(summary['initial_function_errors'].values())==0
    old=json.loads((REPO/'stage/experiments/t197_timeout_progress/protocol.json').read_text())
    checkpoint=REPO/'stage/runs/t199_sac_mean_gradient/initial_residual_policy.pt';assert sha(checkpoint)==summary['initial_checkpoint_sha256']
    previous=json.loads((REPO/'stage/experiments/t196_left_exploration/protocol.json').read_text())
    eval_rng=np.random.default_rng(2026103300);evaluation=[];left=old['primary_cases'][6]
    for i in range(12):
        dx,dy=eval_rng.uniform(-.03,.03,2);yaw=float(eval_rng.uniform(-2,2));evaluation.append({'id':f'left_T200_holdout_{i}','case':left,'group':'left_fresh_reset','perturbation':[float(dx),float(dy),yaw]})
    plan={k:old[k] for k in ['primary_cases','anchor_sources','initial_std','alpha','warmup_updates','updates_per_batch','seed_episodes','training_jobs']}
    plan.update(experiment='T200',seed=2026103300,checkpoint=str(checkpoint),retention_targets=previous['retention_targets'],evaluation_resets=evaluation,batches_per_arm=4,episodes_per_arm=32,max_physics_per_arm=32*23334,wall_seconds=1200,peak_bytes=64*1024**2,max_bytes=40*1024**2,pending_batch_bytes=20*1024**2,
                controlled_difference='Two models eachadd same1202parameters, zero initialized: extra_head600to2 added before nativeclip+atanh(native), or afteratanh(latent). Same initial physical output, parentweights, Gaussianstd,seeds,data,SAC/retention/update budgets. This tests residual placement/conditioning, not sole causalrole of one gradient mechanism.',
                gradient_evidence='T199 shows nativeinverseclip gives exactzero localmean gradients in saturatedsamples; extra latentbranch restores entropy mean gradient while zerohead keeps initial behavior identical4096states. Shared oldparams may still shift saturatedstates indirectly; no claim of permanent architecture impossibility or sole LEFTfailure cause.',
                training_contract='Bothordinary independenttanhGaussian exploration only, no extra parameter noise or timeoutdistancepenalty/HER. Same freshCritic/target and1500warmup,500Actor+Critic/batch,32completeepisodes/arm (6originalLEFT+2nearLEFT perbatch). Seven frozenT194 retention groups32states each, weight10. Bodylr1e-5,native/direct/extraheads1e-4,logstd1e-4,Q3e-4,alpha.001,tau.005,terminalbalanced25%. GateobservedterminalMAE<.15 only basic targetcheck.',
                evaluation_contract='Initialnativezerohead must old6+RIGHT pass. Fixed finalbothcentralActors on old8+12freshLEFTreset cases, no best-midpoint selection or automatic promotion. Scene/goal/0.20mradius/deadlines unchanged. Deployment singleActor with explicitplacement metadata.',
                storage_contract='Before batch reserve20MiB:behavior<3MiB,8episodes<8*256KiB,partialfullcheckpoint<12MiB plusmetadata. Ownpeak<=64MiB/globalnewruns<=1GiB/free40GiB. Finalown<=40MiB. After complete budget and final validation retain exactActor+bothCritics/targets, remove only optimizer payloads with tensor-equality and hash receipt; no claim exactoptimizerresume. Behavioractors/data retained.',
                scope='One paired pilot seed, warm-start consumed fixedtasks, same capacity extrahead placement. DifferentfromT196/T197parent/noise/budget; causalcomparison withinT200 only. No algorithm superiority, originality/generalization or automaticpromotion. T159 and verifiedT194 preserved. Final bounded comparison, no automatic extension.',
                references=['https://proceedings.mlr.press/v80/haarnoja18b.html'])
    for path in plan['seed_episodes']:
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        if 'end_flags' in a:a['end_flags']=a['end_flags'].astype(float)
        build(a)
    files=[checkpoint,audit,Path(plan['retention_targets'])]+[Path(p) for p in plan['seed_episodes']]+list(HERE.glob('*.py'))+[Path(plan['anchor_sources']['LEFT']['snapshot_path'])]
    plan['input_hashes']={str(p):sha(p) for p in files};OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    save(OUT/'preparation.json',{'state':'PASS','initial_function_errors':summary['initial_function_errors'],'verification_states':4096,'seed_episodes':len(plan['seed_episodes']),'physics_steps':0,'optimizer_updates':0,'pending_batch_reserve_bytes':plan['pending_batch_bytes']})
    print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json'),'final_own_cap_MiB':40,'peak_own_cap_MiB':64}))

if __name__=='__main__':main()
