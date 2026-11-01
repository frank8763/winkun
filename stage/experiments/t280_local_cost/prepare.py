from t280_core import *
import ast,gzip,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t280_local_cost'


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    receipt=ROOT/'stage/docs/long_distance_20261101/T279_LOG_ENCODING_RECEIPT.json'
    assert json.loads(receipt.read_text())['state']=='LOSSLESS_LOG_ENCODING_COMPLETE'
    basepath=ROOT/'stage/experiments/t267_causal_credit_course/protocol.json';base=json.loads(basepath.read_text())
    kept=['checkpoint','checkpoint_sha256','initial_base','policy_hash','retention_states','retention_states_sha256',
        'base_lr','memory_lr','retention_weight','max_conditional_k3','max_abs_log_ratio','max_anchor_MSE','max_anchor_error',
        'surrogate_horizon_decisions','old_development','old_regression64','acceptance_guard_jobs','old_centers','development_jobs','heldout_jobs']
    p={k:base[k] for k in kept};p['centers']=[j for j in p['development_jobs'] if not any(j[k] for k in ['delta_x_m','delta_y_m','delta_yaw_deg'])];assert len(p['centers'])==2
    initial={r['id']:r for r in map(json.loads,(ROOT/'stage/runs/t278_fixed_goal_starts/initials.jsonl').read_text().splitlines())}
    t278=json.loads((ROOT/'stage/experiments/t278_fixed_goal_starts/protocol.json').read_text());local=[]
    for side in ['LEFT','RIGHT']:
        j=next(r for r in t278['jobs'] if r['side']==side and r['remaining_distance_m']==1.)
        item=dict(j['base_job'],id='local1m_'+side,source_reset_id='local1m_'+side,group='fixed6m_goal_local1m',
            exact_snapshot=str(ROOT/'stage/runs/t278_fixed_goal_starts'/(side+'_remaining1m_snapshot.npz')),
            expected_initial120_sha256=initial[side+'_remaining1m']['initial120_sha256'])
        assert item['case']['step_cap']==28000;local.append(item)
    p['local_centers']=local;p['training_jobs']=local+p['centers'];assert len(p['training_jobs'])==4
    refs=list(map(json.loads,(ROOT/'stage/runs/t271_frozen_feature_exploration/exploration.jsonl').read_text().splitlines()))
    paths=[basepath,receipt,ROOT/'stage/docs/long_distance_20261101/release_receipt_T279.json',ROOT/'stage/docs/long_distance_20261101/T278_AUDIT.json',
        ROOT/'stage/runs/t278_fixed_goal_starts/initials.jsonl',ROOT/'stage/experiments/t278_fixed_goal_starts/t278_guard.py',*map(Path,[j['exact_snapshot'] for j in local]),*HERE.glob('*.py')]
    for j in p['centers']:
        ref=next(r for r in refs if r['id']==f"feature_copula_dev6_{j['case']['side']}_13_0")
        paths.append(Path(ref['path']))
        with np.load(ref['path']) as z:j['expected_initial120_sha256']=S.digest(z['states'][0])
    inputs={};encodings=[];known={}
    for ledger in ['T268_LOG_ENCODING_RECEIPT.json','T279_LOG_ENCODING_RECEIPT.json']:
        file=ROOT/'stage/docs/long_distance_20261101'/ledger;paths.append(file)
        for entry in json.loads(file.read_text())['records']:known[str(ROOT/entry['original_path'])]=entry
    for name,expected in base['input_hashes'].items():
        file=Path(name)
        if file.exists():assert C.sha(file)==expected;inputs[name]=expected
        else:
            record=known[name];encoded=ROOT/record['encoded_path'];original=gzip.decompress(encoded.read_bytes())
            assert hashlib.sha256(original).hexdigest()==expected==record['original_sha256']
            inputs[str(encoded)]=C.sha(encoded);encodings.append(dict(original_path=name,original_sha256=expected,encoded_path=str(encoded),encoded_sha256=C.sha(encoded)))
    inputs.update({str(f):C.sha(f) for f in paths});p['original_dependency_encodings']=encodings
    p.update(experiment='T280',primary_distance_m=6,arms=dict(original_MC=0.,bounded_endpoint_cost=.1),seed=2026111801,
        rounds=2,draws_per_reset=4,actor_attempts_per_round=2,step_factors=[1.,.25,.0625],
        wall_seconds=4800,final_reserve_seconds=1200,max_commands=1600,max_physics=45000000,
        training_episode_bytes=384*1024,max_training_payloads=64,compact_episode_bytes=12288,max_compact_payloads=1380,model_bytes=8*1024**2,
        metadata_bytes=7*1024**2,payload_bytes=64*384*1024+1380*12288+2*8*1024**2,final_bytes=64*1024**2,
        global_bytes=1024**3,min_free_bytes=40*1024**3,input_hashes=inputs,
        preflight_clone_optimizer_max_calls=6,preflight_total_clone_optimizer_max_calls=12,postrun_clone_optimizer_max_calls=16,
        contract='Paired reward-only ablation withinT280, not single-factor versusT267. Fresh retainedsafeT255+freshAdam in both arms. Fixed four training resets: true6mgoal LEFT/RIGHT at originalstart and verifiedT278 remaining1m snapshots. No adaptive goal/start/noise grid. Same one common seed schedule; first16 perarm all11physical/environment/cost arrays and integration must be exact; duplicated trajectories count actual physics, not32 independent trajectories. Everygroup4draws;2rounds/32unique perarm, plus exact native replay everytrainingrecord; max2Actor proposals perround,3backoff factors, canonicalStepActor/causaltimebaseline/PCGrad/soft100/fixed20 same. The only arm difference is eta0/.1 subtracting normalized bounded SMDP endpoint proximity cost; uses actual next120 currentrange48 and actual nativeholdcount, not invented per-native integral or future/map/goal relabel. Full trajectory added discounted penalty is in[0,.1]; objective changes, no policy-invariance/safety/novelty claim. Firstcostbatch must distinguish >=2equalcap timeouts with originalG0 spread<1e-10 and newG0 spread>1e-5, else stop no replacement. Complete64trainingfull120 twelvearrays; otheractualactions/counts/flags/3rewardarrays compact with all12array hashes; incomplete prefixes retained before stop, no learning from administrative censoring. Exact-policy old20cache includes shadow/commonActor reuse, not independent safety repetition. Rejection/interruption rolls wholeActor+Adam round back. Allactual live/shadowoptimizerinvocations logged, disposablepreflight/postrunclones separate. Allcommands inclreplays/oldguards/roundcenters/final old275+6m54 perarm; only completepass then16fresh and78newenvironment exactmodelreload. No7/8 or automaticnexttraining. Single120hiddenresetActor, noCritic/BC/planner/map/waypoint/teacher/router/Phase. Full64MiB reserved underglobal1GiB, cooperativewriterlock,caseboundary scans, no recursive hot scans. Learning3600s total4800s absolute frommain, no extension/restart. Frozen5m andallfailures retained.')
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    for f,h in inputs.items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();C.save(HERE/'protocol.json',p);C.save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',own_bytes=p['final_bytes'],max_physics=p['max_physics'],max_training_episodes=64)),flush=True)


if __name__=='__main__':main()
