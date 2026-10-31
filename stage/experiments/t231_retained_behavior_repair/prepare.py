from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t231_retained_behavior_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicateT231'
    source=ROOT/'stage/runs/t232_complementary_sources';s=json.loads((source/'summary.json').read_text());assert s['state']=='COMPLETE' and s['success_source_gate'] and s['all_workers_stopped']
    orig=json.loads((ROOT/'stage/experiments/t229_valid_start_acceptance/protocol.json').read_text());base=json.loads((ROOT/'stage/experiments/t228_balanced_stability/protocol.json').read_text());oldvalid=list(map(json.loads,(ROOT/'stage/runs/t229_valid_start_acceptance/validity.jsonl').read_text().splitlines()));valid={r['id'] for r in oldvalid if r['valid_start']}
    kept=[];correction=[]
    for r in map(json.loads,(ROOT/'stage/runs/t229_valid_start_acceptance/fresh_cases.jsonl').read_text().splitlines()):
        if r['complete'] and r['success'] and not r['collision']:kept.append(dict(id=r['id'],task=int(r['group'][4:]),path=r['path'],sha256=r['sha256']))
    # Deterministic old-policy success is preferred over noisy correction in5m.
    # All successful4mrecovery data are retained because no complete deterministic
    # source has yet covered those3failedresets. This type rule is locked now.
    det_rows=list(map(json.loads,(ROOT/'stage/runs/t232_complementary_sources/episodes.jsonl').read_text().splitlines()))
    deterministic_safe_resets={r['source_reset_id'] for r in det_rows if r['complete'] and r['success'] and not r['collision']}
    for folder in ['t230_remaining_sources','t232_complementary_sources']:
        for r in map(json.loads,(ROOT/'stage/runs'/folder/'episodes.jsonl').read_text().splitlines()):
            if not(r['complete'] and r['success'] and not r['collision']):continue
            task=int(r['job']['group'][4:])
            if folder=='t230_remaining_sources' and (task!=4 or r['source_reset_id'] in deterministic_safe_resets):continue
            correction.append(dict(id=r['id'],task=task,path=r['path'],sha256=r['sha256'],reset=r['source_reset_id'],target_kind='T221_preclip' if folder=='t232_complementary_sources' else 'actual_success_command'))
    required=json.loads((ROOT/'stage/experiments/t230_remaining_sources/protocol.json').read_text())['failed_reset_ids']
    assert set(required).issubset({r['reset'] for r in correction})
    assert set(r['task'] for r in correction).issubset({4,6})
    plan=dict(experiment='T231',seed=2026103251,initial_checkpoint=orig['checkpoint'],initial_policy_hash=orig['policy_hash'],retention_targets=base['retention_targets'],original_left_states=str(ROOT/'stage/runs/t221_left_success/verified_success.npz'),complementary_checkpoint=json.loads((ROOT/'stage/experiments/t232_complementary_sources/protocol.json').read_text())['checkpoint'],retained_sources=kept,correction_sources=correction,
        development_jobs=[j for j in orig['jobs'] if j['id'] in valid],heldout_jobs=base['heldout_jobs'],updates=3000,batch_each_task=32,lr=1e-5,max_commands=339,max_physics=6500000,wall_seconds=1000,final_reserve_seconds=600,final_bytes=6*1024**2,peak_bytes=6*1024**2,global_bytes=1024**3,
        contract='Warm-start frozenT228 already204/211validsuccess, notfromT194: preserveownpreclipfunction atold7anchors/originalLEFTinputs+ALLsafeT229freshtrajectory states (includingRIGHT27/27 andLEFTsuccessfulneighbors). AddALLT232safedeterministicT221precliptargets; T230noisy4mcommandsONLYforresetswithoutverifieddeterministicT221source. ExcludeT230noisy5mtargets. Predeclaredpreference bysourceTYPE perreset, notbestseedpick. Actualselectedcorrections mustcoverall7failedresetids. T230failedgateunchanged; T232verifieduniongateprerequisite. Eighttaskbalanced32states each/3000Adam/lr1e-5/nativepreclipMSE/norm1. Multipleengineeringchanges fromT228(initial/LR/retentionsource); do notattributecausally toonefactor/newRLalgo. Independent3000shadow fromsameinputs/seeds every500policy+Adamhashmustmatch, onlyfinalActor gzip/readbackusedforeval. Zero RLupdates/rollouttrainingphysics; warmstartdeepRL+selfsuccesssupervisedconsolidation. Freeze211prevalidateddevelopmentresetsunchanged, allpassedrequired; preflight64predeclarednewheldoutoffsets physicallybeforetraining, keepinvalidheldoutids/noresampling. Onlyif211allsafeevaluateALLphysicallyvalidheldoutcountH, reportHoutof64andinvalidseparately; nevercall216/216or64/64ifinvalid. Eightoriginalcentersmustsafe, finalphysicalGOAL/radius/timeunchanged. Max64zerostepvaliditycommands+211dev+64heldout=339/6.5Mallphysics/1000s/6MiB/global1GiB/free40GiB. No automaticnexttrial/longdistance/promotion; MuJoCovideoonlyaftervalid-domainlocalgate and all8centers+newheldoutpass. NoPhase/maps/planner/waypoint/runtimeTeacher.',
        references=['https://proceedings.mlr.press/v78/laskey17a.html','https://arxiv.org/abs/2306.02437','https://arxiv.org/abs/2108.13264'])
    paths=list(HERE.glob('*.py'))+[Path(plan['initial_checkpoint']),Path(plan['retention_targets']),Path(plan['original_left_states']),Path(plan['complementary_checkpoint']),source/'summary.json',ROOT/'stage/runs/t230_remaining_sources/summary.json',ROOT/'stage/runs/t229_valid_start_acceptance/summary.json',ROOT/'stage/runs/t229_valid_start_acceptance/validity.jsonl',ROOT/'stage/experiments/t229_valid_start_acceptance/valid_support.py',ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py']+[Path(r['path']) for r in kept+correction]+[Path(j['exact_snapshot']) for j in plan['development_jobs']]
    plan['input_hashes']={str(p):sha(p) for p in paths};size=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',kept=len(kept),corrections=len(correction),dev=len(plan['development_jobs']),runs_MiB=size/1024**2)))
if __name__=='__main__':main()
