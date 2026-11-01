from pathlib import Path
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t282_capped_balance'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    source=ROOT/'stage/experiments/t281_task_gradient_audit/protocol.json';base=json.loads(source.read_text())
    inputs=dict(base['input_hashes'])
    for f in [source,ROOT/'stage/runs/t281_task_gradient_audit/summary.json',ROOT/'stage/runs/t281_task_gradient_audit/bounded_endpoint_cost.json',*HERE.glob('*.py')]:inputs[str(f)]=sha(f)
    p=dict(experiment='T282',input_hashes=inputs,balance_cap=32.,relative_noise_floor=1e-8,seed=2026111801,
        wall_seconds=300,max_physics=0,max_optimizer_calls=0,final_bytes=1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        contract='One zero-optimizer contract of capped32 task weighting followingT281actualgradientmeasurement, not navigation/training. Synthetic fixed4vectors verify exactsamePCGrad, zero/tinyvectors not amplified, commonrescale preserves initial taskdirectionnorm, fixedweights invariant withinround and compatible with weighted surrogate. T281real4x4Gram only: independently rebuild baselinePCGrad coefficients/compare recordedFloat32norm then analyze one predeclaredcapped32weighted direction; no grid/norm search/learnedweights/savedActor. Gram contraction not newphysical or optimizer proof, no claimbetteroldretention/navigation. CPU/RAM only,300s/1MiB/global1GiB/free40GiB. Native/Adam entrypoints forbidden. Source localcommit, remote currently401/userwillhandlelater; noautomaticDRL/7m/8m/Phase.')
    for f,h in inputs.items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();text=json.dumps(p,indent=2)+'\n';(HERE/'protocol.json').write_text(text);(OUT/'protocol.json').write_text(text);print(json.dumps(dict(state='REGISTERED',physics=0,optimizer=0)),flush=True)


if __name__=='__main__':main()
