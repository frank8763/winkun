"""No-step descriptive comparison of already completed T656 and its origin."""
from pathlib import Path
import json,gzip,sys
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261119/t656'
sys.path.insert(0,str(ROOT/'stage/experiments/t656_qualified_recovery_pg'))
import t656_runtime as T
R,np,torch=T.R,T.np,T.torch

def rows(folder):return [json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]

def main():
    root=ROOT/'stage/runs/t656_qualified_recovery_pg';p=T.read(root/'protocol.json');sealed=T.read(root/'SEALED.json')
    assert sealed['all_scientific_processes_stopped'] and all(not Path('/proc',str(v)).exists() for v in sealed['scientific_pids'])
    assert T.read(root/'FINAL_REPORT_FINAL.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    R.B.setup(p);origin=ROOT/'stage/runs/t653_afternoon_outcome_constraints/mean';s=T.read(root/'pg/summary.json');before=T.read(origin/'summary.json')
    parent=R.L.frozen(p,p['initial']['path']);child=R.L.frozen(p,s['proposal']['path']);assert R.L.fixed_hash(parent)==R.L.fixed_hash(child)
    source={r['job_id']:r for r in rows(origin) if r['full']};target={r['job_id']:r for r in rows(root/'pg') if r['full']};report=[]
    for key in ['dev_task6_00','dev_task6_02','dev_task6_03','dev_task6_26']:
        arow,brow=target[key],source[key];assert T.sha(arow['path'])==arow['sha256'] and T.sha(brow['path'])==brow['sha256']
        a=R.E.N.S.arrays(arow['path']);b=R.E.N.S.arrays(brow['path'])
        assert np.array_equal(a['initial_native_integration'],b['initial_native_integration']) and np.array_equal(a['states'][0],b['states'][0])
        assert np.array_equal(a['scene_sha256'],b['scene_sha256']) and np.array_equal(a['task_goal_xy'],b['task_goal_xy'])
        departure=T.O.select_departure(a,b)
        with torch.no_grad():
            x=torch.tensor(a['states']);xb=torch.tensor(b['states']);ca=child.sequence(x).clamp(-1,1);pa=parent.sequence(x).clamp(-1,1);cb=child.sequence(xb).clamp(-1,1);pb=parent.sequence(xb).clamp(-1,1)
        assert float((child.sequence(x).detach()-torch.tensor(a['locations'])).abs().max())<1e-10
        assert float((parent.sequence(xb).detach()-torch.tensor(b['locations'])).abs().max())<1e-10
        report.append(dict(id=key,origin_safe=T.safe(brow),candidate_safe=T.safe(arow),origin_seconds=brow['physics']*.002,candidate_seconds=arow['physics']*.002,
            origin_remaining=brow['remaining_m'],candidate_remaining=arow['remaining_m'],first_actual_departure=departure,
            parameter_command_RMS_on_origin_history=float((cb-pb).square().mean().sqrt()),parameter_command_peak_on_origin_history=float((cb-pb).abs().max()),
            parameter_command_RMS_on_candidate_history=float((ca-pa).square().mean().sqrt()),
            origin_path=brow['path'],origin_sha256=brow['sha256'],candidate_path=arow['path'],candidate_sha256=arow['sha256'],not_causal_attribution=True))
    trace=s['update']['trace'];pg0=trace[0]['pg_loss'];pg1=trace[-1]['pg_loss']
    old=set(p['old_screen_ids']);bmap={r['job_id']:r for r in before['screen']['rows']};amap={r['job_id']:r for r in s['screen']['rows']}
    value=dict(state='READ_ONLY_NO_NATIVE_NO_OPTIMIZER',comparisons=report,
        newly_recovered=sorted(k for k in old if not T.safe(bmap[k]) and T.safe(amap[k])),newly_lost=sorted(k for k in old if T.safe(bmap[k]) and not T.safe(amap[k])),
        origin_lost=sorted(k for k in old if not T.safe(bmap[k])),candidate_lost=sorted(k for k in old if not T.safe(amap[k])),
        all8_field_retained=s['screen']['all8_mean_field_retained'],origin_field24=before['screen']['field24_success'],candidate_field24=s['screen']['field24_success'],
        max_recorded_per_task_KL=max(max(v['per_task_KL']) for v in trace),registered_KL_cap=p['max_task_latent_KL'],
        PG_pre_step0=pg0,PG_pre_step7=pg1,relative_pre_step_surrogate_change=(pg1-pg0)/abs(pg0),
        pre_step_metrics_not_final_objective=True,full407=s['final407'],new_native=0,new_optimizer=0)
    T.save(DOC/'ACTUAL_T656_REVIEW.json',value);print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()
