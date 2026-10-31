"""Bounded exact success verification and own-exploration policy consolidation."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t221_left_success'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T221'
    t220=ROOT/'stage/runs/t220_temporal_noise';summary=json.loads((t220/'summary.json').read_text());assert summary['state']=='COMPLETE' and summary['all_workers_stopped']
    success=[r for r in map(json.loads,(t220/'rollouts.jsonl').read_text().splitlines()) if r['success'] and not r['collision']];assert len(success)==1
    old=json.loads((ROOT/'stage/experiments/t219_rnd_runtime/protocol.json').read_text())
    plan={k:old[k] for k in ('checkpoint','retention_targets','primary_cases','task_snapshots')}
    plan.update(experiment='T221',seed=2026103221,source_success=success[0],updates=3000,batch_each_task=16,lr=3e-5,
        max_commands=10,max_physics=233340,wall_seconds=360,final_reserve_seconds=90,
        final_bytes=320*1024,peak_bytes=320*1024,global_bytes=1024**3,
        contract='First reproduce T220 originalLEFT successful trajectory exactly using frozen sensor Actor and recorded AR innovations, all10arrays/finalintegration equality mandatory. This establishes sampled original-scene feasibility only. Supervised consolidation of own successful exploration, not new PPO/SAC/RL improvement: 8 balanced tasks (7 old T194 deterministic action anchors + actual sampledLEFT success actions), all574210MLP parameters except unusedlogstd, 3000 Adam steps lr3e-5 preclip native-output MSE against actual successful actions; seven oldtargets are parent preclip outputs to keep initial functions, seed fixed. Oldactions from retained T194 anchor inputs. Train2 independent candidates LIVE+SHADOW fromsameinputs/seeds; aftereach500 steps parameter+optimizer hashes identical. Every optimizer call separatelyreported; no density/PPO use of zero diagnosticlogp. Reconstruct finalmodel from initialreference+fullnewsuccessdata+deterministicrecipe, hashverifiedidentical, shadowreconstructedActor old8 finalonly. Explicit no binary finalmodel export due global1GiB; model losslessly recoverable via this recipe in matchedversions, not direct hotresume. No bestintermediatecheckpoint or biassearch. Initial+posttraining action fit and oldanchors separatelyreported, no trainingfit=closedloop claim. At most2successreplays+finalold8=10commands/233340allphysics/360s, last90sevalreserve, own320KiB/global1GiB/free40GiB. Missing finaleval not failure0/8. No automatic54grid/training/promotion/extension. T159/T194 untouched, Phaseisolated.',
        references=['https://arxiv.org/abs/1806.05635','https://arxiv.org/abs/1802.05313','https://arxiv.org/abs/1511.06295'])
    files=list(HERE.glob('*.py'))+[ROOT/'stage/experiments/t220_temporal_noise/temporal_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',ROOT/'stage/experiments/t216_left_specialist/support.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py',Path(success[0]['trace'])]+[Path(plan[k]) for k in ('checkpoint','retention_targets')]+list(map(Path,plan['task_snapshots']))
    plan['input_hashes']={str(f):sha(f) for f in files};size=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','all_runs_MiB':size/1024**2}))
if __name__=='__main__':main()
