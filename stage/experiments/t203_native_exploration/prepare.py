from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t203_native_exploration'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists()
    old=json.loads((ROOT/'stage/experiments/t200_residual_placement/protocol.json').read_text());parent=ROOT/'stage/runs/t194_drive_sensitivity/selected_policy.pt';near=Path(old['anchor_sources']['LEFT']['snapshot_path'])
    plan={'experiment':'T203','checkpoint':str(parent),'near_snapshot':str(near),'case':old['primary_cases'][6],'seed':2026103503,'sigma_native':[.35,.30],'wall_seconds':360,'max_physics':373344,'max_bytes':4*1024**2,'peak_bytes':8*1024**2,'optimizer_updates':0,
          'deltas':[[.05,0,0],[-.05,0,0],[0,.05,0],[0,-.05,0],[0,0,5]],
          'contract':'One deterministic source gate, then5small-reset-neighbors each deterministic/legacy tanhGaussian(.5latent)/native truncatedGaussian(.35,.30),16episodes maximum. Two replay witnesses count separately inside total373344physics cap. Source failure stops. Original goal/scene/.20radius/cap unchanged; histories refilled consistently for all curriculum resets. Zero training and no planner/waypoints/runtime teacher. This is a distribution/amplitude engineering preflight, not an isolated coordinate-only algorithm comparison.',
          'gate':'Source safeGOAL; native law numeric mass/gradient checks; native behavior>=3/5neighbor safeGOAL; in clip states observed native throttle<.9 fraction>=.05. At least one native and one legacy episode exact full-state/action replay. No fullLEFT5m success claim. Failure stops proposed T204 launch; no automatic alternative distributions.',
          'references':['https://arxiv.org/abs/1707.06347','https://proceedings.mlr.press/v78/florensa17a.html']}
    files=[parent,near,ROOT/'stage/experiments/t200_residual_placement/common.py']+list(HERE.glob('*.py'));plan['input_hashes']={str(p):sha(p) for p in files}
    assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print('T203 PREPARED',sha(HERE/'protocol.json'))
if __name__=='__main__':main()
