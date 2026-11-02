from pathlib import Path
import json,hashlib,shutil,ast
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t289_rejected_candidates'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    src=ROOT/'stage/runs/t287_time_aligned_credit';base=json.loads((src/'protocol.json').read_text());updates=[json.loads(l) for l in (src/'round_updates.jsonl').read_text().splitlines()];rows=[json.loads(l) for l in (src/'cases.jsonl').read_text().splitlines()]
    candidates=[]
    for record in updates:
        for update in record['updates']:
            for trial in update['attempts']:
                if trial.get('native_old_task_guard') is not None:
                    g=trial['native_old_task_guard'];assert not g['all_safe']
                    candidates.append(dict(id=f'r{record["round"]}_factor'+str(trial['factor']).replace('.','p'),round=record['round'],factor=trial['factor'],policy_hash=g['policy_hash'],historical_old20=g))
    assert len(candidates)==4 and [c['factor'] for c in candidates]==[.25,.0625,.25,.0625]
    jobs=[dict(j) for j in base['centers']+base['local_centers']]
    for j in jobs:
        for key in ['policy_hash','memory_on','model_seed']:j.pop(key,None)
        j.update(mode='deterministic',seed=0,aux_eta=0.)
    parent={j['id']:next(r for r in rows if r['role']=='round_centers' and r['source_reset_id']==j['id']) for j in jobs}
    paths=[*HERE.glob('*.py'),*[src/n for n in ['protocol.json','cases.jsonl','round_updates.jsonl','native_guard_cache.json','summary.json']],ROOT/'stage/docs/long_distance_20261101/T287_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T287.json']
    paths += [Path(r['path']) for r in rows if r['role']=='training'];paths += [Path(j['exact_snapshot']) for j in jobs]
    hashes=dict(base['input_hashes']);hashes.update({str(f):sha(f) for f in paths})
    p=dict(experiment='T289',source_protocol=str(src/'protocol.json'),source_cases=str(src/'cases.jsonl'),source_updates=str(src/'round_updates.jsonl'),source_cache=str(src/'native_guard_cache.json'),
        candidates=candidates,candidate_hashes=[c['policy_hash'] for c in candidates],jobs=jobs,historical_parent_rows=parent,input_hashes=hashes,
        max_commands=32,max_physics=896000,wall_seconds=1200,model_bytes=3*1024**2,episode_bytes=384*1024,partial_replay_bytes=12288,
        payload_bytes=12*1024**2+16*384*1024+16*12288,metadata_bytes=1024**2,final_bytes=20*1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        max_historical_optimizer_calls=4,postrun_optimizer_calls=0,navigation_training_updates=0,
        contract='User20261102 reauthorized continueafterreport. Four andonlyfour existing T287 numeric-pass old20-rejected candidate Actors, source rounds0/1 factors.25/.0625 fixedbeforecurrentoutcomes. Reconstruct original2rounds with2live2shadow disposable oldAdam calls, exactreports/cache/ActorAdamrollback, notnewlearning. All4 frozen on original6mLEFT/RIGHT andfixed6mgoalnew1mLEFT/RIGHT, eachoneunique+oneexactnative12array/integrationreplay. Historicalparentfourcontrols explicitlyconsumedreuse no independentnewcontrolclaim. Native32commands896kphysics1200s20MiBglobal1GiBfree40GiB, full16x120statesnextstatesandall12arrays384KiB,4Actors3MiB,partialreplay16x12KiB,metadata1MiB. No new noise/seed/factor/algorithmgrid, no candidatepromotion, full27554fresh16reload78notperformed, no7/8. PrimarysafeGOALonly; smallerremainingatidenticalcapandoutcomeonlydescriptive, contactnevercountsimprovedsuccess. Allpartials/models/sourceSHApreserved. Sourcecommit/fullremoteexact andzero-nativepreflightbeforeuniquecontrolledlaunch. Endallprocessesabsent0optimizer0nativeaudit, completeRAMremoteeverymemberSHAbackup/sourceexact. SingleActor120hiddenreset noPhase/map/planner/waypoint/teacher/router.')
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    for f,h in hashes.items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();text=json.dumps(p,indent=2)+'\n';(HERE/'protocol.json').write_text(text);(OUT/'protocol.json').write_text(text);print(json.dumps(dict(state='REGISTERED',global_if_full=used+p['final_bytes'])))


if __name__=='__main__':main()
