"""Read-only analysis of already audited T646 trajectories, zero native/Adam."""
from pathlib import Path
from collections import Counter
import gzip,json,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t646_offline_night';DOC=ROOT/'stage/docs/day_20261119/t647'

def main():
    assert json.loads((OUT/'coverage/audit.json').read_text())['state']=='PASS'
    rows=[json.loads(x) for x in gzip.decompress((OUT/'coverage/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    fresh=[r for r in rows if r['role']=='coverage_fresh'];assert len(fresh)==608
    result=[]
    for key in sorted({r['job_id'] for r in fresh}):
        group=[r for r in fresh if r['job_id']==key];success=[r for r in group if r['success'] and not r['collision'] and not r.get('field_boundary',False)]
        mins=[];times=[];turns=[];pathlength=[]
        for r in group:
            path=Path(r['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
            with np.load(path,allow_pickle=False) as a:
                xy=a['poses'][:,:2];goal=a['task_goal_xy'];mins.append(float(np.linalg.norm(xy-goal,axis=1).min()))
                times.append(r['physics']*.002);turns.append(float(np.rad2deg(np.unwrap(a['poses'][:,2])[-1]-a['poses'][0,2])))
                pathlength.append(float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()))
        result.append(dict(id=key,n=len(group),safe_success=len(success),contact=sum(r['collision'] for r in group),timeout=sum(r.get('timeout',False) for r in group),boundary=sum(r.get('field_boundary',False) for r in group),
            remaining_m_quantiles=np.quantile([r['remaining_m'] for r in group],[0,.5,1]).tolist(),closest_recorded_m_quantiles=np.quantile(mins,[0,.5,1]).tolist(),elapsed_s_quantiles=np.quantile(times,[0,.5,1]).tolist(),net_unwrapped_yaw_deg_quantiles=np.quantile(turns,[0,.5,1]).tolist(),path_length_m_quantiles=np.quantile(pathlength,[0,.5,1]).tolist()))
    report=dict(state='READ_ONLY_ANALYSIS_COMPLETE',native=0,optimizer=0,tasks=result,all_successes=sum(v['safe_success'] for v in result),positive_tasks=sum(v['safe_success']>0 for v in result),
       limitations='closest distance from recorded decision states, not continuous native minimum; no geometric impossibility or single root cause proven')
    (DOC/'COVERAGE_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
    for v in result:print(v['id'], 'safe',v['safe_success'],'contact',v['contact'],'timeout',v['timeout'],'closest',v['closest_recorded_m_quantiles'],'remain',v['remaining_m_quantiles'],flush=True)

if __name__=='__main__':main()
