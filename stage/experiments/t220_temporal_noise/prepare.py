from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t220_temporal_noise'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T220'
    prior=json.loads((ROOT/'stage/runs/t219_rnd_runtime/summary.json').read_text())
    assert prior['state']=='COMPLETE' and prior['all_workers_stopped'] and prior['training_episodes']==32
    assert not any(r['success'] and not r['collision'] for r in prior['final_saved_reload_old8'] if r['case']['distance_m']==5 and r['case']['angle_deg']<0)
    p=json.loads((ROOT/'stage/experiments/t219_rnd_runtime/protocol.json').read_text())
    plan=dict(experiment='T220',seed=2026103220,checkpoint=p['checkpoint'],case=p['primary_cases'][6],snapshot=p['task_snapshots'][6],
        arms={'white':0.,'ar09':.9,'ar098':.98},episodes_per_arm=6,workers=4,max_commands=21,max_physics=490014,wall_seconds=240,
        final_bytes=640*1024,peak_bytes=640*1024,global_bytes=1024**3,
        references=['https://arxiv.org/html/2312.11091v2','https://onno-eberhard.appspot.com/pink'],
        contract='Frozen T194 MLP, exact original LEFT full reset/native physics and original GOAL. Only latent temporal correlation differs; same per-seed Gaussian innovations, sigma(.35,.30), stationary z0. AR(1) copula via normal-CDF/truncated-normal-inverse-CDF; no action script, planner, waypoints, teacher, scene edits, RND reward or optimization. Not pink-noise/PPO reproduction or learned navigation. Noise latent marginal N(0,1), feedback means conditional action law is history dependent. Stored logp is zero placeholder, not behavior likelihood; prohibited as on-policy learning data. All10arrays hashed; compact original actions retained. First complete seed each arm must exact-replay in RAM before further batch. If any sampledGOAL, stop after finishing current paired 3-arm seed block; keep evidence, not deterministic GOAL claim. No automatic training/promotion. At most18rollouts+3replays=21commands/490014allphysics/240s, own640KiB/global1GiB/free40GiB, no duplicate/resume/extension.')
    files=list(HERE.glob('*.py'))+[ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',ROOT/'stage/experiments/t217_rnd_preflight/model.py',ROOT/'stage/experiments/t216_left_specialist/support.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py',Path(plan['checkpoint']),Path(plan['snapshot'])]
    plan['input_hashes']={str(f):sha(f) for f in files}
    size=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for f in (HERE/'protocol.json',OUT/'protocol.json'):f.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','all_runs_MiB':size/1024**2,'protocol_sha256':sha(HERE/'protocol.json')}))
if __name__=='__main__':main()
