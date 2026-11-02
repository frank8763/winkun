"""Register an isolated learning-capability ablation, never a baseline replacement."""
from pathlib import Path
import ast,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t291_hard_guard_ablation'
sys.path.insert(0,str(ROOT/'stage/experiments/t290_exact_duplicate_store'))
from store import resolve
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    p=json.loads((ROOT/'stage/experiments/t287_time_aligned_credit/protocol.json').read_text())
    inputs={str(resolve(f)):h for f,h in p['input_hashes'].items()}
    for row in p['historical_first16']:row['path']=str(resolve(row['path']))
    prior=ROOT/'stage/runs/t287_time_aligned_credit/summary.json'
    files=[*HERE.glob('*.py'),ROOT/'stage/experiments/t287_time_aligned_credit/protocol.json',prior,
           ROOT/'stage/docs/distance_20261102/T289_AUDIT.json',ROOT/'stage/docs/distance_20261102/T289_FINDINGS.md',
           ROOT/'stage/docs/long_distance_20261101/release_receipt_T289.json',
           ROOT/'stage/docs/distance_20261102/T290_ENCODING_RECEIPT.json',ROOT/'stage/docs/distance_20261102/T290_AUDIT.json',
           ROOT/'stage/docs/long_distance_20261101/release_receipt_T290.json',ROOT/'stage/experiments/t290_exact_duplicate_store/store.py']
    inputs.update({str(f):sha(f) for f in files})
    p.update(experiment='T291',rounds=2,max_commands=900,max_physics=25200000,
             max_training_payloads=32,max_compact_payloads=719,
             training_episode_bytes=192*1024,model_bytes=int(7.25*1024**2),metadata_bytes=4*1024**2,
             payload_bytes=32*192*1024+719*12288+int(7.25*1024**2),final_bytes=int(25.75*1024**2),
             wall_seconds=4800,final_reserve_seconds=1200,external_group_watchdog_seconds=4800,
             overnight_stop_epoch=10**12,overnight_learning_stop_epoch=10**12,
             preflight_clone_optimizer_max_calls=5,postrun_clone_optimizer_max_calls=8,
             hard_old20_acceptance_required=False,research_only_not_deployment=True,
             historical_control=dict(experiment='T287',first_two_completed_rounds=True,consumed_development=True,
                concurrent_independent_replication=False,summary_path=str(prior),summary_sha256=sha(prior)),input_hashes=inputs,
             contract='T291 new independently bounded research-only learning-capability ablation, not continuation/restart T287. Same safeT255+freshAdam/fourfixedstarts/seed/first16/eta.1/timealignedGtRLOO/AR/PCGradcapped32/soft100/LR/numerictrust/factors/2proposals andoldcenterstop; only old20 hard accept disabled. Every numeric-pass proposed Actor still physically evaluated on true old20, actual all_safe/collision/timeout untouched in cache/reports, acceptance=numeric only. Existing T287 first two completed rounds32consumed control, not randomized independent replication. Two rounds at most32fulltraining+32native exact replay/max4live4shadow; twozeroaccepted orold8centerloss stops. No changing rounds/noise/seed grid; no7/8training or modelpromotion. All original12arrays includingfull120state/nextstate exactly recoverable via redundant statechain codec192KiB; compact six arrays12KiB,32full+719compact+7.25MiBActorAdam+4MiBmetadata withinown25.75MiB/global1GiB/free40GiB. 900allcommands25.2Mallphysics4800seconds final1200evaluation, daytime new absolute deadlines atmain and external watchdog earlier wins, no extension. Preflight4disposablecloneAdam syntheticold20failure not real native acceptance, fullrollback/shadow/density/codec/resource/sourcecontracts; postrun atmost8disposableexactreconstruction calls. Fullold275 and6m54; onlybothallGOALthen16fresh78reload, unmet/invalid/censored separate. Preserve safeT255/currentinputs/allfailedmodels, native20 cache only identical policyhash shadow reuse not independentphysics. Endallprocessesabsent audit exactActorAdam/source/rawarrays/counts, RAM fullremoteeverymemberSHA backup then sourcesync. NoPhase/map/planner/waypoints/teacher/router/newCritic/BC/observations/capacity. Current Actor120hiddenreset unchanged.')
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    for f,h in inputs.items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    assert used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();text=json.dumps(p,separators=(',',':'))+'\n'
    (HERE/'protocol.json').write_text(text);(OUT/'protocol.json').write_text(text)
    print(json.dumps(dict(state='REGISTERED_NO_NATIVE',global_if_full=used+p['final_bytes'],own_reserved=p['final_bytes'])))


if __name__=='__main__':main()
