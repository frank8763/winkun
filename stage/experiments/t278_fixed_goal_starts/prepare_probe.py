from t278_native import ROOT,C,np
from pathlib import Path
import ast,json,shutil
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t278_fixed_goal_starts'


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    previous=ROOT/'stage/runs/t256_distance_frontier/protocol.json';base=json.loads(previous.read_text())
    source_rows=ROOT/'stage/runs/t271_frozen_feature_exploration/exploration.jsonl'
    rr=list(map(json.loads,source_rows.read_text().splitlines()));jobs=[]
    for side in ['LEFT','RIGHT']:
        ref=next(r for r in rr if r['id']==f'feature_copula_dev6_{side}_13_0')
        for d in [1.,3.]:
            jobs.append(dict(id=f'{side}_remaining{int(d)}m',side=side,remaining_distance_m=d,base_job=ref['job'],
                reference_initial_path=ref['path'],reference_initial_sha256=ref['sha256']))
    paths=[previous,source_rows,ROOT/'stage/docs/long_distance_20261101/T277_PARTIAL_AUDIT.json',
        ROOT/'stage/docs/long_distance_20261101/release_receipt_T277.json',
        ROOT/'stage/docs/long_distance_20261101/T277_ENGINEERING_AUDIT.json',
        ROOT/'stage/experiments/t255_overnight/mem_runtime.py',*map(Path,[j['reference_initial_path'] for j in jobs]),*HERE.glob('*.py')]
    inputs=dict(base['input_hashes']);inputs.update({str(f):C.sha(f) for f in paths})
    p={k:base[k] for k in ['checkpoint','checkpoint_sha256','policy_hash','initial_base']}
    p.update(experiment='T278',scope='Frozen safe5m Actor, fixed true6m final endpoint, four reset-only shorter-start capability probes',
        jobs=jobs,input_hashes=inputs,max_commands=8,max_physics=224000,wall_seconds=600,
        episode_bytes=384*1024,snapshot_bytes=64*1024,payload_bytes=5*384*1024+4*64*1024,
        metadata_bytes=768*1024,final_bytes=3*1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        contract='New scientific question, not continuation/restart of T277 partial8 recovery. Frozen retainedT255s2_reset/zeroActorAdam updates. True6mLEFT/RIGHT original goal and scene, .2radius/.002physics/100hold/56s cap fixed; reset starts exactly1m/3m from final goal on original goal bearing, preserve original full physical state exceptqposXY, fresh sensor history at reset, no waypoint route supplied to Actor. Everybase original120 zero-translation matches authenticated original source. Fullbody initial contact=INVALID, retained/skipped without replacement or seed search. Four unique starts each valid one independent full snapshot reload/replay; invalid gets no navigation/replay. Full12010array payloads and native snapshots, at most1partial extra saved before failure checks. Same single120Actor hidden reset each decision, no runtime head/teacher/planner/router/map/Phase. Stage cooperative exclusive writer flock plus entire own3MiB reserved from global1GiB at startup; sourceoutside bytes scanned/reconciled onlycase boundaries; eachholdO(1)wall/physics/free, payload/pending accounted beforewrites, allmetadata reserved. Noncooperatingoutside writes abort at next boundary; noclaim instantaneous filesystemquota. NootherStage writer launched by this chat whileactive; externalCPUaffinity2 fixedbeforeimports; no change to oldprotocols. Allphysicsincluding reload/censored, full8command224kphysics600s max. Near-start success is not original6m-from-original-start success, reversecurriculum benefit, forgetting cure or novelalgorithm. No automatic training/7m/seed or start grid.')
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    used=lambda folder:sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes'] and used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();C.save(HERE/'protocol.json',p);C.save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',max_physics=224000,max_commands=8,own_reserve=p['final_bytes'],initial_policy=p['policy_hash'])),flush=True)


if __name__=='__main__':main()
