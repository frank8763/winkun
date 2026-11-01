"""Register one new candidate against a declared, already consumed T280 control."""
from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t287_time_aligned_credit'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    old=ROOT/'stage/experiments/t283_capped_task_balance/protocol.json';p=json.loads(old.read_text());p['experiment']='T287'
    inputs=dict(p['input_hashes']);paths=[old,ROOT/'stage/runs/t280_local_cost/summary.json',ROOT/'stage/runs/t280_local_cost/cases.jsonl',
        ROOT/'stage/docs/long_distance_20261101/T283_AUDIT.json',ROOT/'stage/experiments/t282_capped_balance/balanced_learning.py',
        ROOT/'stage/experiments/t282_capped_balance/protocol.json',ROOT/'stage/runs/t281_task_gradient_audit/summary.json',ROOT/'stage/runs/t282_capped_balance/contract.json',*HERE.glob('*.py')]
    rr=list(map(json.loads,(ROOT/'stage/runs/t280_local_cost/cases.jsonl').read_text().splitlines()))
    history=[r for r in rr if r['arm']=='bounded_endpoint_cost' and r['role']=='training' and r['id'].startswith('bounded_endpoint_cost_r0_')];assert len(history)==16
    paths.extend([ROOT/'stage/runs/t283_capped_task_balance/summary.json',ROOT/'stage/runs/t283_capped_task_balance/cases.jsonl',ROOT/'stage/docs/long_distance_20261101/T285_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T285.json',ROOT/'stage/docs/long_distance_20261101/T286_CHECKPOINT_ENCODING_RECEIPT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T286.json']);paths.extend(Path(r['path']) for r in history);inputs.update({str(f):sha(f) for f in paths})
    for key in ['capacity_final_relocked','preflight_engineering_repair','preflight_completion_contract','preflight_total_clone_optimizer_max_calls']:
        p.pop(key,None)
    p.update(arms=dict(time_aligned_credit=.1),balance_cap=32.,balance_relative_noise_floor=1e-8,
        historical_first16=history,historical_control=dict(experiment='T283',arm='capped_task_balance',consumed_development=True,
            concurrent_independent_replication=False,summary_path=str(ROOT/'stage/runs/t283_capped_task_balance/summary.json'),summary_sha256=sha(ROOT/'stage/runs/t283_capped_task_balance/summary.json')),
        rounds=4,max_commands=1400,max_physics=39200000,max_training_payloads=64,max_compact_payloads=1100,
        metadata_bytes=5*1024**2,payload_bytes=64*384*1024+1100*12288+8*1024**2,final_bytes=50*1024**2,external_group_watchdog_seconds=7200,wall_seconds=7200,final_reserve_seconds=1800,
        preflight_clone_optimizer_max_calls=4,postrun_clone_optimizer_max_calls=16,input_hashes=inputs,
        contract='T287 one physical-time-aligned independent-peer reward-to-go RLOO baseline pilot. New standardcredit mechanism notalgorithmoriginality claim. Same safeT255freshAdam/fourfixedstarts/eta.1/ARnoise/PCGrad/capped32balance/soft100/fixed20/nativethresholds/actorLR and2proposals3factors perround. Prior causalbaseline weightedcoeffconstantG0-B0 replaced Gt-mean3peersGt at exactsamestartnativeclock; peerpastterminal0, no interpolation/preterminalmissingtime; same originalMCbatchRMS/.05 scale. First16full12arrays/integrationmustmatchT280T283parent, notnewindependentdata. HistoricalT283consumedfirst2round32control only; new4round64capmeansnotmatchedtotalbudgetcausaltrial. Stop2consecutivezeroacceptedrounds oroldcenterloss; noadditionalroundseednoisegrid. 1400allcommands39.2Mallphysics7200sreserve1800/own50MiB/global1GiB/free40GiB. Fourworkers CPU1to4, parent0; sourcefullremotesyncandT286lossless43checkpointreadback requiredbeforelaunch. Rejected/interruptedwholeActorAdamrollback; shadow exactcacheold20 notnewphysicaltest. All64full120statesnextstatesandall12arrays saved384KiB,1100compact12KiB,oneActorAdam8MiB,metadata5MiB. max8live8shadow; preflight4disposablecloneAdam separate, postrun16clone. Old275six54 allsafe then16fresh78independentreload, onlythenfrozen7m2diagnostic; no7/8trainingautomatic. Absoluteovernightlearningstop20261102T0800+08 totalstop0820, earlier2hourdeadlinewins. Endallprocessesabsent auditActorAdam/rawhash/replays/budgets, fullremoteRAMbackup andsourcesync. NoPhase/map/planner/waypoint/teacher/router/newCritic/BC/observationorcapacity.')
    from datetime import datetime
    from zoneinfo import ZoneInfo
    p.update(overnight_stop_epoch=datetime(2026,11,2,8,20,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp(),overnight_learning_stop_epoch=datetime(2026,11,2,8,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp())
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    for f,h in inputs.items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();text=json.dumps(p,indent=2)+'\n';(HERE/'protocol.json').write_text(text);(OUT/'protocol.json').write_text(text)
    print(json.dumps(dict(state='REGISTERED_LOCAL_PREPARATION_ONLY',full_reserve=p['final_bytes'],global_if_full=used+p['final_bytes'])),flush=True)


if __name__=='__main__':main()
