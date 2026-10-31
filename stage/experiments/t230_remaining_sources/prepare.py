from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t230_remaining_sources'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicateT230'
    prior=ROOT/'stage/runs/t229_valid_start_acceptance';s=json.loads((prior/'summary.json').read_text());assert s['state']=='COMPLETE' and s['all_workers_stopped']
    rows=json.loads((prior/'merged_valid_cases.json').read_text());failed=[r for r in rows if not(r['complete'] and r['success'] and not r['collision'])];assert 1<=len(failed)<=12,'New bounded protocol required if >12failedcases'
    old=json.loads((ROOT/'stage/experiments/t229_valid_start_acceptance/protocol.json').read_text());jobs=[];initial={j['id']:j for j in old['jobs']}
    for row in failed:
        for seed in range(2026103241,2026103247):
            j=dict(initial[row['id']]);j.update(id=f"{row['id']}_seed{seed}",source_reset_id=row['id'],arm='frozenT228',round=-1,role='remaining_valid_failure_source',seed=seed,mode='correlated_diagnostic');jobs.append(j)
    plan=dict(experiment='T230',checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],candidate_hash=old['policy_hash'],jobs=jobs,failed_reset_ids=[r['id'] for r in failed],sigma=[.035,.03],rho=.98,max_commands=len(jobs)*2,max_physics=len(jobs)*2*23334,wall_seconds=600,final_bytes=12*1024**2,peak_bytes=12*1024**2,global_bytes=1024**3,
        contract='Single bounded remainingvalidfailurecoverage afterT229zero-stepvalidityofall216. IncludeALLfailedphysicallyvalidmeasuredresets, maximum12elsefailregistration; sixpredeclaredseeds2026103241–46 EACH, no adaptivechoice/retry/bestseed/extension. FrozenexactT228nativeActor (newlylearned5mretainingcandidate), sigma(.035,.03) AR.98validatedcopula. Keepallfailures/successfullarrays usinglosslessnextstatesencoding; eachfreshrolloutindependentlyregeneratedsecondtime all10arrays/finalintegrationexact. Everyfailedreset needs>=1safeGOALbeforeseparatelearningprotocol, no sourcegate relaxation. Zero optimizer/physicsmodelchanges/rewardshaping/planner/waypoints/runtimeTeacher/Phase. NativeGOAL/radius/time/scenariofixed. Up to144sourceepisodes+replays if12failures, max600s/12MiBown/global1GiB/free40GiB; actualcommand/physicsbudgetlockedfromfailedcount. No automatictraining/promotion/preview/longdistance.',
        references=['https://proceedings.mlr.press/v78/laskey17a.html','https://arxiv.org/abs/2306.02437','https://arxiv.org/html/2312.11091v2'])
    files=list(HERE.glob('*.py'))+[Path(plan['checkpoint']),prior/'summary.json',prior/'merged_valid_cases.json',prior/'validity.jsonl',ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t220_temporal_noise/temporal_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py']+[Path(j['exact_snapshot']) for j in jobs]
    plan['input_hashes']={str(f):sha(f) for f in files};size=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',failed=len(failed),new_source_episodes=len(jobs),runs_MiB=size/1024**2)))
if __name__=='__main__':main()
