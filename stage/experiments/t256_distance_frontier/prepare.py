from pathlib import Path
import json,hashlib,shutil,math
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t256_distance_frontier'
def main():
    assert not OUT.exists(),'No duplicate T256'
    old=json.loads((ROOT/'stage/experiments/t255_overnight/protocol.json').read_text());s=json.loads((ROOT/'stage/runs/t255_overnight/summary.json').read_text());assert s['all_workers_stopped'] and s['stable_5m_gate']
    parent=s['arms']['s2_reset']['export'];basecases=[]
    for distance in [6.,7.]:
        for oldjob in old['center_jobs'][6:8]:
            side=oldjob['case']['side'];case=dict(oldjob['case'],id=f'{int(distance)}m_{side}',distance_m=distance,step_cap=math.ceil(distance/3*14000));basecases.append(dict(id=f'base{int(distance)}_{side}',source_reset_id=f'base{int(distance)}_{side}',group=f'{int(distance)}m_{side}',case=case,delta_x_m=0.,delta_y_m=0.,delta_yaw_deg=0.,seed=0,mode='deterministic'))
    eight=[r['job'] for r in map(json.loads,(ROOT/'stage/runs/t255_overnight/8m.jsonl').read_text().splitlines())]
    p=dict(experiment='T256',checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],policy_hash=parent['policy_hash'],initial_base=old['checkpoint'],old_development=old['development_jobs'],reference_rows=str(ROOT/'stage/runs/t255_overnight/s2_reset_development.jsonl'),eight_jobs=eight,eight_reference=str(ROOT/'stage/runs/t255_overnight/8m.jsonl'),basecases=basecases,
      max_commands=320,max_physics=7500000,wall_seconds=1000,final_bytes=28*1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
      contract='Frozen accepted T255 s2_reset. Zero optimizer calls. Four factory base goals6m/7m LEFT/RIGHT; native start/goal collision checks, actual seed/scene/goal and exact snapshot recorded; factory retry is reported, not a same-state distance-only claim. Nine predeclared dx/dy(-.03,0,.03),yaw0 starts per base, invalid states retained separately. No new layouts and no global reachability proof. Restore all211 previously safe development episodes complete120 inputs, every ten array and final integration hash must exactly match T255; not new independent data. Rebuild original8m2 failed episodes once with full120states and exact original hashes, not new independent tests. Extra offline same-initial-state radial goal-input queries at5/6/7/8m are causal sensitivity diagnostics only, zero physical steps, never a deployed goal clip/cap, waypoint, planner or success claim. Retain a compact full current-policy state dataset for later fixed5m function retention, all failure results retained. Identical100native hold/.002physics/.2radius and proportionaldistance horizon. No Phase, planning research, external planner, waypoints, runtime teacher or routing. No training in this run, no seed/quantile/temperature/frequency grid. New userNov1 authorization to continue longer distance supersedes old stop-after-overnight scope; old directories never resumed. Next new bounded RL protocol only after this frontier evidence.',
      references=['https://proceedings.mlr.press/v80/florensa18a.html','https://arxiv.org/abs/2112.12490','https://horizon-generalization.github.io/'])
    paths=list(HERE.glob('*.py'))+[Path(p[k]) for k in ['checkpoint','initial_base','reference_rows','eight_reference']]
    p['input_hashes']={**old['input_hashes'],**{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}}
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>p['min_free_bytes']
    OUT.mkdir();data=json.dumps(p,indent=2)+'\n';(HERE/'protocol.json').write_text(data);(OUT/'protocol.json').write_text(data);print(json.dumps(dict(state='REGISTERED',runs_MiB=used/1024**2,new_rollouts=36,recovery_episodes=213)))
if __name__=='__main__':main()
