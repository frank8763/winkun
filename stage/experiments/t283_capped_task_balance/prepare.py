"""Register one new candidate against a declared, already consumed T280 control."""
from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t283_capped_task_balance'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    old=ROOT/'stage/experiments/t280_local_cost/protocol.json';p=json.loads(old.read_text());p['experiment']='T283'
    inputs=dict(p['input_hashes']);paths=[old,ROOT/'stage/runs/t280_local_cost/summary.json',ROOT/'stage/runs/t280_local_cost/cases.jsonl',
        ROOT/'stage/docs/long_distance_20261101/T280_AUDIT.json',ROOT/'stage/experiments/t282_capped_balance/balanced_learning.py',
        ROOT/'stage/experiments/t282_capped_balance/protocol.json',ROOT/'stage/runs/t281_task_gradient_audit/summary.json',ROOT/'stage/runs/t282_capped_balance/contract.json',*HERE.glob('*.py')]
    rr=list(map(json.loads,(ROOT/'stage/runs/t280_local_cost/cases.jsonl').read_text().splitlines()))
    history=[r for r in rr if r['arm']=='bounded_endpoint_cost' and r['role']=='training' and r['id'].startswith('bounded_endpoint_cost_r0_')];assert len(history)==16
    paths.extend(Path(r['path']) for r in history);inputs.update({str(f):sha(f) for f in paths})
    for key in ['capacity_final_relocked','preflight_engineering_repair','preflight_completion_contract','preflight_total_clone_optimizer_max_calls']:
        p.pop(key,None)
    p.update(arms=dict(capped_task_balance=.1),balance_cap=32.,balance_relative_noise_floor=1e-8,
        historical_first16=history,historical_control=dict(experiment='T280',arm='bounded_endpoint_cost',consumed_development=True,
            concurrent_independent_replication=False,summary_path=str(ROOT/'stage/runs/t280_local_cost/summary.json'),summary_sha256=sha(ROOT/'stage/runs/t280_local_cost/summary.json')),
        max_commands=800,max_physics=22400000,max_training_payloads=32,max_compact_payloads=700,
        metadata_bytes=5*1024**2,payload_bytes=32*384*1024+700*12288+8*1024**2,final_bytes=35*1024**2,
        preflight_clone_optimizer_max_calls=4,postrun_clone_optimizer_max_calls=8,input_hashes=inputs,
        contract='Newcappedtaskbalancing32mechanism, notT280budgetcontinuation/restart. Oneactualcandidate fromsafeT255+freshAdam; historicalT280boundedcost control explicitlyconsumeddevelopment, notnewconcurrentarm/independentsuccessclaim. Same2round32trainingfourfixedresets/alltrainingexactreplay, ARseed2026111801/eta.1/basearchitecture/noise/caps/reward/PCGrad/soft100/fixed20/2proposalsround/3factors. Onlynewmechanism: taskgradnormweightsup32,relativefloor1e-8andabs1e-12 noamplifyroundoff, whole-roundconstantsweightedproxy, commontaskgradientrescale preserves initialPCGradnorm, oldretentionunchanged. First16all12arrays/integration exactlyhistoricalbeforefirstupdate. Completeold275+6m54 thengate16fresh+78independentreload,no7/8automatictraining. 800allcommands/22.4Mallphysics/4800stotalreservefinal1200/own35MiB/global1GiB/free40GiB. 32full12012array384KiB,700compact12KiB,oneActorAdam8MiB,metadata5MiB worstcaseallincluded. max4live+4shadowinvocations; rejected/interruptedwholeActorAdamrollback. Samepolicyhashguardcachefortheshadowonly actual20recorded, notsecondphysicalrepeat. Allpartialcounts/models/failuresretained. Zero-nativepreflightmax4disposablecloneAdam separate, postrun<=8reconstructionclone. Scientificsource/protocolcommittedandfullremoteexacttreeverifybeforeonlinenavigation. UserGitHubreauthlater; onlylocalpreparationalloweduntilrestored. NoPhase/map/planner/waypoint/teacher/router/newCritic/BC.')
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    for f,h in inputs.items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();text=json.dumps(p,indent=2)+'\n';(HERE/'protocol.json').write_text(text);(OUT/'protocol.json').write_text(text)
    print(json.dumps(dict(state='REGISTERED_LOCAL_PREPARATION_ONLY',full_reserve=p['final_bytes'],global_if_full=used+p['final_bytes'])),flush=True)


if __name__=='__main__':main()
