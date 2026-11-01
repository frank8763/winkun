"""Lock one scientific action-availability ablation; no fitting grid."""
from blind_model import *
import ast,shutil
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t274_action_identifiability'


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    p=json.loads((ROOT/'stage/runs/t272_event_value_pilot/protocol.json').read_text())
    p['experiment']='T274';p['scope']='Capacity- and schedule-matched current-action masking ablation of T272'
    p['contract']='One new bounded offline ablation, NOT another T272 seed/step/radius search. Exactly same96 records,80/16 family split, normalization computed only on train,13,202 parameter values at initialization, seed2026111201,500 Adam updates/lr.001/batch64/episode-row random schedule, all labels/loss/horizon settings. Only a fixed input mask sets the last2 standardized CURRENT-action coordinates to zero in training and inference. Original120 causal state including previous actual action remains. T272 source trainer reused unchanged through an explicit model-class/output/protocol adapter; source hashes locked. Same logged-mixture event forecasting, not causal action ranking, Q for a changed Actor, safety proof or six-metre success. Full matching/reread Adam/holdout probability evidence retained. Test evaluated only initial/final; no best checkpoint, seed/step/architecture/noise grid, no native simulation or Actor updates. A lower heldout loss without action would support state-forecasting explanation but not prove actions physically irrelevant, universal action identifiability failure or no better controlled-learning method. All failures/models retained, Phase isolated, no planner/waypoints/teacher/router.'
    inputs=set(p['input_hashes'])
    inputs.update(map(str,HERE.glob('*.py')))
    inputs.update(str(ROOT/s) for s in ['stage/runs/t272_event_value_pilot/protocol.json','stage/runs/t272_event_value_pilot/summary.json',
        'stage/runs/t272_event_value_pilot/event_checkpoint.pt.gz','stage/runs/t272_event_value_pilot/test_predictions.npz',
        'stage/docs/long_distance_20261101/T272_AUDIT.json','stage/docs/long_distance_20261101/T273_AUDIT.json',
        'stage/docs/long_distance_20261101/T273_EFFECTS.json','stage/runs/t273_value_direction_probe/summary.json',
        'stage/experiments/t272_event_value_pilot/audit.py'])
    p['input_hashes']={path:sha(path) for path in sorted(inputs)}
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    assert used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',full_reserve=p['final_bytes'],global_if_full=used(ROOT/'stage/runs')+p['final_bytes'])),flush=True)


if __name__=='__main__':main()
