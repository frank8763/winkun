"""Register exactly two fixed T280 first-batch gradient measurements."""
from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t281_task_gradient_audit'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    basefile=ROOT/'stage/experiments/t280_local_cost/protocol.json';base=json.loads(basefile.read_text())
    cases=ROOT/'stage/runs/t280_local_cost/cases.jsonl';rows=list(map(json.loads,cases.read_text().splitlines()));selected={}
    paths=[basefile,cases,ROOT/'stage/docs/long_distance_20261101/T280_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/T280_MECHANISM_DESCRIPTION.json',*HERE.glob('*.py')]
    for arm in base['arms']:
        selected[arm]=[r for r in rows if r['arm']==arm and r['role']=='training' and r['id'].startswith(arm+'_r0_')]
        assert len(selected[arm])==16 and len({r['source_reset_id'] for r in selected[arm]})==4
        for r in selected[arm]:assert r['complete'] and r['policy_hash']==base['policy_hash'] and sha(r['path'])==r['sha256'];paths.append(Path(r['path']))
    inputs=dict(base['input_hashes']);inputs.update({str(f):sha(f) for f in paths})
    p=dict(experiment='T281',source_protocol=str(basefile),source_protocol_sha256=sha(basefile),input_hashes=inputs,
        first_batch_ids={arm:[r['id'] for r in records] for arm,records in selected.items()},
        checkpoint=base['checkpoint'],checkpoint_sha256=base['checkpoint_sha256'],policy_hash=base['policy_hash'],
        wall_seconds=600,max_physics=0,max_optimizer_calls=0,final_bytes=1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        decision_weak_ratio=.1,PCGrad_seed=base['seed'],
        contract='One offline audit, exactlyT280twoarmsfirst16each at samefrozenT255; no data selection/seed/grid/fit/newactor. Gradient means actual original surrogate, fourtaskgradients and exact existingPCGrad inFloat32, contractions additionallyFloat64, pervector hashes/Grams/projection factors. Oldsoft100 gradient at unchangedparent audited, notproof nonlinear old20safe. Measure sensitivity of olddev_task6_24 deterministic commands to resultant descent direction viaexactautogradJVP contractions, no parameter step/line search/model export. New coefficient scales alone are not parameter gradient norms. Weak screen boundedcost right/local norm<.1left AND directional dot reporting, not guaranteedtraininggain. No actualGradNorm/CAGrad or proof, no navigating/nativeenv/optimizer; patch both prohibited entrypoints to fail. Full600s/1MiB/global1GiB/free40GiB, exactlytwofixedbatches, stop onfailurewithoutrepeat. Alloldsource/data/models immutable. Localcode/protocolcommit beforeexecution; remote backupcurrently401, explicitly no claimremoteverified. No7/8/Phase/planner/waypoints/router/teacher.')
    for f,h in inputs.items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();text=json.dumps(p,indent=2)+'\n';(HERE/'protocol.json').write_text(text);(OUT/'protocol.json').write_text(text)
    print(json.dumps(dict(state='REGISTERED',physics=0,optimizer=0,fixed_batches=2)),flush=True)


if __name__=='__main__':main()
