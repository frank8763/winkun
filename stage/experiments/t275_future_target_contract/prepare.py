from future_data import *
import ast,shutil
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t275_future_target_contract'


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    parent=ROOT/'stage/runs/t274_action_identifiability/protocol.json';base=json.loads(parent.read_text())
    env=ROOT/'stage/runtime/mk-mini/training/e2e_phase10_lidar_v1/phase10_scene_e2e_env.py'
    inputs=[parent,env,ROOT/'stage/experiments/t190_adaptive_start_curriculum/common.py',
        ROOT/'stage/experiments/t251_actuator_recovery/actuator_support.py',ROOT/'stage/experiments/t255_overnight/mem_actor.py',
        ROOT/'stage/docs/long_distance_20261101/T274_COMPARISON.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T274.json',*HERE.glob('*.py')]
    p=dict(experiment='T275',scope='Zero-physics/optimizer explicit-goal removal and realized future endpoint timing contract',records=base['records'],
        environment_source=str(env),input_hashes={str(f):sha(f) for f in inputs},anchor_stride_decisions=4,
        requested_physics_horizons=[500,1000,2000,4000,8000,16000],wall_seconds=120,
        pairs_bytes=768*1024,metadata_bytes=256*1024,final_bytes=1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        contract='Only fixed96 prior full120 records, no new physical or optimizer or Actor update. Source-audited13 direct current/history goal-coordinate fields removed to107 training-side state, not a reconstructed120 Actor input. Retain motion, previous actual actions, encoders, hold, original cap clock; behavioral/history statistics can depend on original goals and this is NOT statistical goal independence. Real future next_pose after each fixed stride4 anchor and requested1/2/4/8/16/32s, choose first existing endpoint at-or-after requested time, no replacement when missing/contact. Relative body coordinates/10 are query goals from exact logged achieved future, no future sensors/absolute position/case IDs passed as features. Actual delta physical steps is positive-time label, not first-hit/shortest distance or reachability probability. Contact endpoints excluded, unseen suffix masked, real timeout endpoint is observed. Preserve same80/16 family split and all original success/failure labels. One semantic contract only, no grids/restarts. Phase isolated, no map/planner/waypoint/teacher/router.')
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    assert used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',reserve_bytes=p['final_bytes'])),flush=True)


if __name__=='__main__':main()
