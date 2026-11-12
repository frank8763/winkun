"""Read-only representational support check for a future two-command repair.

No optimizer/native calls; success histories do not prove autonomous recovery.
"""
from pathlib import Path
import json,sys,time,shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'));import speed_actor as S
G,np,torch=S.G,S.np,S.torch

def main():
    root=ROOT/'stage/runs/t500_joint_repair_sources';p=G.read(root/'protocol.json');assert G.read(root/'sources/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(v)).exists() for v in G.read(root/'SEALED.json')['scientific_pids'])
    refs=G.read(root/'sources/summary.json')['result']['successful_repair_sources'];q=S.frozen(p,p['initial']['path']);reports=[];allreq=[]
    with torch.no_grad():
        for row in refs:
            assert G.sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0];base=q.head(h).numpy();target=a['actions'].astype(float)
            required=np.arctanh(np.clip(target,-1,1)/1.25)-base
            # Executed boundary action has an inequality feasible set. Minimal
            # bounded logit displacement needed to reproduce that command.
            required=np.where(target>=1,np.maximum(required,0),np.where(target<=-1,np.minimum(required,0),required))
            after=q.location(h,n).clamp(-1,1).float().numpy();err=after-target
            allreq.append(required)
            reports.append(dict(id=row['job_id'],decisions=len(x),drive_steer_RMSE=np.sqrt(np.mean(err**2,axis=0)).tolist(),max_required_drive_steer_logit_shift=np.abs(required).max(0).tolist(),outside_bound03_fraction=(np.abs(required)>.3).mean(0).tolist()))
    flat=np.concatenate(allreq);keep_old=[r for r in G.rows(ROOT/'stage/runs/t499_frozen_joint_validation/regression') if r['success'] and not r['collision']];keep8=G.rows(ROOT/'stage/runs/t498_early_error_repair/repair')
    assert len(refs)==43 and len(keep_old)==286 and len(keep8)==54
    assert len({r['job_id'] for r in refs+keep_old+keep8})==383
    initial=G.read(ROOT/'stage/runs/t499_frozen_joint_validation/protocol.json');jobs=initial['old275_jobs']+initial['six_jobs']+initial['wide8_jobs']
    percap=sum(j['case']['step_cap'] for j in jobs);size=sum(1352*int(np.ceil(j['case']['step_cap']/100))+4480+65536 for j in jobs)
    result=dict(state='READ_ONLY_REPRESENTATION_AND_DATA_REVIEW_COMPLETE',native=0,Adam=0,teacher_sources=43,own_success_references=340,distinct_task_ids=383,required_logit_shift_max_abs=np.abs(flat).max(0).tolist(),outside03_fraction=(np.abs(flat)>.3).mean(0).tolist(),rows=reports,next_full383_max_physics=percap,full383_full_history_ZIP_shape_bound=size,global_logical_bytes=G.M.used(),free_bytes=shutil.disk_usage(ROOT).free,offset_bounds_are_representational_NOT_control_safety_proof=True,fit_or_navigation_NOT_executed=True)
    (ROOT/'stage/docs/review_20261112/T501_DATA_AND_ACTION_SUPPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
