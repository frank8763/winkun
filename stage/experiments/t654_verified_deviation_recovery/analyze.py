"""Read-only fixed departure anchors on fully recorded T653 failures."""
from pathlib import Path
import json,gzip,sys,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261119/t654'
sys.path.insert(0,str(ROOT/'stage/experiments/t653_conflict_aware_afternoon'))
import t653_runtime as T
R,np,torch=T.R,T.np,T.torch

def main():
    root=ROOT/'stage/runs/t653_afternoon_outcome_constraints';p=T.read(root/'protocol.json');T.R.B.setup(p)
    assert T.read(root/'FINAL_REPORT_FINAL.json')['state']=='COMPLETE'
    sealed=T.read(root/'SEALED.json');assert sealed['all_scientific_processes_stopped'] and all(not Path('/proc',str(v)).exists() for v in sealed['scientific_pids'])
    assert T.read(root/'mean/audit.json')['state']=='PASS'
    q=R.L.frozen(p,p['initial']['path']);mean=T.read(root/'mean/summary.json')['proposal'];student=R.L.frozen(p,mean['path'])
    assert R.L.fixed_hash(q)==R.L.fixed_hash(student)
    original=T.read(ROOT/'stage/runs/t646_offline_night/OFFLINE_INPUTS.json');by={r['id']:r for r in original['members']}
    rows=[json.loads(v) for v in gzip.decompress((root/'mean/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    result=[]
    for key in ['dev_task6_00','dev_task6_03','dev_task6_26']:
        row=next(r for r in rows if r['role']=='exact20' and r['job_id']==key);assert row['collision'] and not row['success'] and T.sha(row['path'])==row['sha256']
        a=R.E.N.S.arrays(row['path']);ref=by[key];assert T.sha(ref['offline_path'])==ref['sha256'];b=R.E.N.S.arrays(ref['offline_path'])
        assert np.array_equal(a['states'][0],b['states'][0]) and np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
        assert np.array_equal(a['scene_sha256'],b['scene_sha256']) and np.array_equal(a['task_goal_xy'],b['task_goal_xy'])
        dep=T.O.select_departure(a,b);assert dep and dep['decision']>0
        k=dep['decision'];x=torch.tensor(a['states']);xb=torch.tensor(b['states'])
        with torch.no_grad():
            same_original=q.sequence(x);same_student=student.sequence(x);old_original=q.sequence(xb);old_student=student.sequence(xb)
        assert float((same_student-torch.tensor(a['locations'])).abs().max())<1e-10
        assert float((old_original-torch.tensor(b['locations'])).abs().max())<1e-10
        job=dict(row['request']['job'],source_path=row['path'],source_sha256=row['sha256'],recovery_first=k)
        result.append(dict(id=key,departure=dep,job=job,source=row,original=ref,
            initial120_sha256=hashlib.sha256(a['states'][0].tobytes()).hexdigest(),scene_sha256=bytes(a['scene_sha256']).hex(),
            same_student_history_parent_command_at_anchor=same_original[k].clamp(-1,1).tolist(),
            same_student_history_candidate_command_at_anchor=same_student[k].clamp(-1,1).tolist(),
            parent_history_mean_parameter_command_RMS=float((old_student.clamp(-1,1)-old_original.clamp(-1,1)).square().mean().sqrt()),
            actual_candidate_collision_remaining_m=row['remaining_m']))
    T.save(DOC/'ACTUAL_DEPARTURES.json',dict(state='ZERO_NATIVE_ZERO_OPTIMIZER_READ_ONLY',initial_research_model=mean,teacher=p['initial'],
        fixed_3_first_departures=result,not_causal_proof=True,new_native=0,new_optimizer=0))
    print('T654_READ_ONLY_DEPARTURES',[(v['id'],v['departure']['seconds']) for v in result])
if __name__=='__main__':main()
