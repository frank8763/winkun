from t308_core import *
import ast

def main():
    assert not OUT.exists();base=read(ROOT/'stage/runs/t299_conditional_goal_likelihood/protocol.json')
    contract=read(ROOT/'stage/runs/t307_iid_goal_contract/protocol.json');inputs=dict(contract['input_hashes'])
    assert read(DOC/'T307_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T307.json';assert receipt.exists()
    jobs=[]
    ph='31b694d1ecf92dbcf7909f6ba6e73a808bc9725aeeacfb7584d0aab8fb162d54'
    for side in ['LEFT','RIGHT']:
        for reset,split in [(13,'train'),(1,'test')]:
            name=f'dev6_{side}_{reset:02d}';original=next(j for j in base['development_jobs'] if j['id']==name)
            inputs[original['exact_snapshot']]=C.sha(original['exact_snapshot'])
            for k in range(4):
                seed=2026110300+(0 if side=='LEFT' else 100)+reset*4+k
                family=name+f'_noise{k}'
                for arm,offset in ([('center',0.)] if split=='train' else [('center',0.),('minus',-1.),('plus',1.)]):
                    jobs.append(dict(original,id=family+'_'+arm,source_reset_id=name,family=family,side=side,split=split,arm=arm,seed=seed,
                        mode='iid',aux_eta=0.,first_steer_innovation_offset=offset,policy_hash=ph))
    for f in [ROOT/'stage/runs/t307_iid_goal_contract/protocol.json',ROOT/'stage/runs/t307_iid_goal_contract/contracts.json',
        DOC/'T307_AUDIT.json',receipt,DOC/'T306_AUDIT.json',DOC/'T308_PLAN.md',
        ROOT/'stage/experiments/t305_goal_conditioned_contract/gc_contract.py',
        ROOT/'stage/experiments/t295_hindsight_support/run.py',ROOT/'stage/experiments/t291_hard_guard_ablation/t291_codec.py',
        ROOT/'stage/experiments/t278_fixed_goal_starts/t278_guard.py',*HERE.glob('*.py')]:inputs[str(f)]=C.sha(f)
    p=dict(experiment='T308',policy_protocol=contract['policy_protocol'],policy_hash=ph,input_hashes=inputs,jobs=jobs,
        seed=2026110211,evaluation_seed=2026110212,workers=4,exploration_rho=0.,sigma=[.035,.03],
        batch_size=64,head_steps=1500,head_learning_rate=.001,validation_batches=16,kernel_width_m=.2,value_margin=1e-8,
        max_commands=64,max_physics=1792000,wall_seconds=5400,final_reserve_seconds=900,audit_seconds=1800,
        night_absolute_end_epoch=1793665200.,final_bytes=13*1024**2,payload_bytes=int(9.75*1024**2),metadata_bytes=2*1024**2,
        episode_bytes=256*1024,model_bytes=256*1024,evidence_bytes=256*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,
        Actor_updates=0,navigation_gate=False,automatic_next_training=False,
        contract='One newly registered iid marginal boundedGaussian fixedsafeT255 data/goalvalue CONTROL pilot, not T304/T305 continuation. Same deterministic120hiddenreset Actor and physical cap/GOAL/contact geometry. 16 original6m native iid centers (LEFT/RIGHT reset13train and01heldout, four fixed noise streams each); heldout8 centers plus16 first-steer-innovation +/-1sigma branches followed by SAME iid policy with common future innovation stream. All32unique+32exact12array/history/integration replays charged, not64 independent samples. Original goal remains6m; absoluteposes/futurecoordinates only offline labels, never runtime goals/teacher/waypoints/planner/router/map/Phase. Future finite discounted DECISIONendpoint occupancy, safeGOALcoordinate absorption, collisionpriority token, truetaskcap timeouttoken, administrativecensorunknown; not safe firsthitQ/originaleta.1reward or continuingMDP unbiasedCRL theorem. Branches nevertraining data. Fit only8train centers, episode/anchoruniform, futurelaw weighted. Matchedsameinit Critic124-64-32/query6-32-32 actionvsblindnewactionzero, previouslatent2zero, factualpreviousactionremainsin120; original120 scaling, bodyqueries/10, tokenbodyzero; equalexactworldcoordinates/terminaltokens allpositive, unknownmasked. 1500steps64batch/head lr.001clip5,3000realheadAdam and3000independentdisposableaudit; no bestcheckpoint/grid. Heldout16fixedbatches perALL/LEFT/RIGHT samequeries actionblind. Strict full NCE<=.95blind and eachside<=1.02blind, eachside>=6informative nativecontrolpairs(absdifference>1e-8), actionrank>=.90 and topchoice realkerneloccupancy>tieuniformbaseline; allconjunctive. 3offlinecoordinatequeries atfloor20/50/80% centerdecisioncount (clipped0..N-2), require3distinctnonterminal else invalid STOP, kernelwidth.2m. Paired single-noise control samples, smoothing/observationstate aliasing/goalcontext-to-querypolicy transfer are limitations, not Qcertificate. NO Actoroptimizer/navigationpromotion permitted EVENIF headpasses: separatelypreregister one conditionalActor pilot with remaining capacity after audit/backups. Original275/54/fresh16/reload78 NOT executed in this value-only run, no7or8. 64commands1.792Mphysics5400s last900fit/reserve, nightabsolute03Nov08:20 earlierfromstrictwall. own13MiB:32full256KiB+4interruptions256KiB+2heads256KiB+evidence256KiB=9.75MiBpayload, metadata2MiB,buffer1.25MiB; global1GiB/free40GiB. Source hashes/zero-native0Adam preflight/exactroot sync before unique launch; singleStagewriter/inhibitor/timeout allstop before independent raw/3000headAdam exactaudit/fullRAMremoteeverymemberSHA. No failed oldhead connected, no repeatedseed/roundgrid, all failures/models preserved; safeT255/T299/HDvideo bytes protected.')
    assert len(jobs)==32 and p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    for f,h in inputs.items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:C.save(f,p)
    print('T308 registered finite native iid goalvalue/control pilot; zero native/Adam so far')

if __name__=='__main__':main()
