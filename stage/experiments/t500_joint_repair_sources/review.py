"""Zero-native/zero-Adam joint failure reproduction and success-source review."""
from pathlib import Path
import gzip,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'));import speed_actor as S
G,P,torch,np=S.G,S.P,S.torch,S.np

def main():
    prior=ROOT/'stage/runs/t499_frozen_joint_validation';p=G.read(prior/'protocol.json');assert G.read(prior/'regression/audit.json')['state']=='PASS'
    rr=G.rows(prior/'regression');failed=[r for r in rr if not r['success']];assert len(failed)==43
    old=ROOT/'stage/runs/t449_offline_joint_recovery';historical=[json.loads(x) for x in gzip.decompress((old/'cases.jsonl.t481.gz').read_bytes()).decode().splitlines()]
    success={r['job_id']:r for r in historical if r['role']=='FINAL_recovery' and r['success'] and not r['collision']}
    overlap=[r for r in failed if r['job_id'] in success];missing=[r for r in failed if r['job_id'] not in success];assert len(overlap)==42 and [r['job_id'] for r in missing]==['dev_task6_00']
    recovery=next(r for r in historical if r['job_id']=='dev_task6_00' and r['role']=='RECOVERY_QUERY');assert recovery['success'] and Path(recovery['path']).exists() and G.sha(recovery['path'])==recovery['sha256']
    a=G.E.N.S.arrays(recovery['path']);assert all(G.digest(v)==recovery['array_hashes'][k] for k,v in a.items()) and a['end_flags'][-1].tolist()==[0,1,0]
    student=S.frozen(p,p['initial']['path']);olddata=P.old_data(p);errors=[]
    with torch.no_grad():
        for name in ['dev_task4_00','dev_task6_00']:
            g=next(g for g in olddata if g['id']==name);pred=student.sequence(torch.tensor(g['x'])).clamp(-1,1).float().numpy();err=pred-g['y'];errors.append(dict(id=name,successful_T255_history_drive_steer_RMSE=np.sqrt(np.mean(err**2,axis=0)).tolist(),max_abs_error=np.abs(err).max(0).tolist()))
    result=dict(state='RAW_JOINT_FAILURE_REPRODUCED_SOURCES_IDENTIFIED',native=0,Adam=0,actual_old_success=233,actual_six_success=53,failed43_ids=[r['job_id'] for r in failed],T449_same_task_success_reconstruction_available42=[r['job_id'] for r in overlap],T449_known_success_teacher_model=next(v['candidate']['model'] for v in G.read(old/'summary.json')['phases'] if v['phase']=='recovery'),verified_actual_recovery_source_for_missing=recovery,student_on_successful_old_histories=errors,causal_intervention_NOT_yet_executed=True)
    d=ROOT/'stage/docs/review_20261112/T500_CAUSE_AND_SOURCE_REVIEW.json';d.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['verified_actual_recovery_source_for_missing','T449_same_task_success_reconstruction_available42','failed43_ids']},ensure_ascii=False))
if __name__=='__main__':main()
