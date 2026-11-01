from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t284_feedback_drift'


def main():
    assert not OUT.exists();OUT.mkdir()
    src=ROOT/'stage/runs/t283_capped_task_balance';base=json.loads((src/'protocol.json').read_text());rows=[json.loads(l) for l in (src/'cases.jsonl').read_text().splitlines()]
    candidate='e63afdb61ae5a59bdfdc9d74df885d17053b51a3f75462f8e25a9c187cd3c859'
    references={name:next(r for r in rows if r['source_reset_id']=='dev_task6_24' and r['policy_hash']==ph and r['role']==role)
        for name,ph,role in [('parent',base['policy_hash'],'old_regression'),('candidate',candidate,'old20_guard')]}
    hashes=dict(base['input_hashes'])
    paths=list(HERE.glob('*.py'))+[src/n for n in ['protocol.json','cases.jsonl','round_updates.jsonl','native_guard_cache.json','summary.json']]
    paths += [ROOT/'stage/docs/long_distance_20261101'/n for n in ['T283_AUDIT.json','release_receipt_T283.json']]
    for r in rows:
        if r['role']=='training' and r['id'].startswith('capped_task_balance_r0_'):paths.append(Path(r['path']))
    for p in paths:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    mib=1024**2
    plan=dict(experiment='T284',source_protocol=str(src/'protocol.json'),source_cases=str(src/'cases.jsonl'),source_updates=str(src/'round_updates.jsonl'),source_cache=str(src/'native_guard_cache.json'),candidate_policy_hash=candidate,
        job=next(j for j in base['acceptance_guard_jobs'] if j['id']=='dev_task6_24'),references=references,input_hashes=hashes,
        max_commands=4,max_physics=93336,wall_seconds=900,model_bytes=3*mib,episode_bytes=384*1024,analysis_bytes=128*1024,
        payload_bytes=3*mib+4*384*1024+128*1024,metadata_bytes=mib,final_bytes=6*mib,global_bytes=1024**3,min_free_bytes=40*1024**3,
        max_clone_optimizer_calls=2,new_navigation_training_updates=0,
        contract='Single fixed historical rejected T283 round0 factor0.0625 counterexample at old5m dev_task6_24, selected posthoc explicitly. Not new RL/new seed/grid/extension. Reconstruct original round exactly with2 disposable live/shadowAdam calls and exact recorded reports/cache; capture only known candidate policy hash, parent and Adam rollback exact. Frozen parent/candidate each one full120 native reconstruction plus one exact replay, all12arrays and integration mustmatchhistorical rawhash. 4commands/93336allphysics/900s/own6MiB/global1GiB/free40GiB. Retain allpartials/failures. Same-state cross-policy action queries permit exact two-order algebraic parameter/state-feedback decomposition at aligned decision starts, not physical interventions/stabilityproof/newsuccess. Fixed XY descriptive thresholds .001/.01/.05/.1m, no thresholds tuned to pass. No new optimizer learning, model promotion,7/8, map/planner/waypoint/teacher/router or Phase. Source exact remote sync before native. End audit no optimizer and fullremote backup.')
    blob=json.dumps(plan,ensure_ascii=False,indent=2)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    print('REGISTERED',hashlib.sha256(blob.encode()).hexdigest())


if __name__=='__main__':main()
