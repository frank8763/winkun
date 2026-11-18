"""Read-only T631 outcome/reward review. No model import, physics or update."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/day_20261118/after_t631'


def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def arrays(row):
    assert sha(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def main():
    root=ROOT/'stage/runs/t631_explored_recovery'
    summary=read(root/'summary.json');p=read(root/'protocol.json');gamma=p['gamma_native']
    assert summary['state']=='COMPLETE' and read(root/'audit.json')['state']=='PASS'
    assert summary['Actor_updates_started']=={'live':0,'shadow':0}
    records=[];error=0.
    old=read(ROOT/'stage/runs/t630_early_departure_recovery/summary.json')
    ref=next(v['baseline'] for v in old['results'] if v['id']=='wide117_8m_dev6_LEFT_13')
    baseline=arrays(ref);bn=baseline['native_counts'].astype(np.int64)
    bt=np.r_[0,np.cumsum(bn)[:-1]]
    for row in summary['result']['exploration']:
        a=arrays(row);n=a['native_counts'].astype(np.int64)
        first=int(a['student_takeover_first']);v=0.
        for k in range(len(n)-1,first-1,-1):
            v=float(a['rewards'][k])+gamma**int(n[k])*v
        total=int(n[first:].sum(dtype=np.int64))
        distance=float(np.linalg.norm(a['poses'][first,:2]-a['task_goal_xy']))
        event=1. if row['success'] else -1. if row['collision'] else -.5
        cost=-.000002*(1-gamma**total)/(1-gamma)
        closed=.03*distance+cost+gamma**(total-1)*event
        error=max(error,abs(v-closed));assert abs(v-closed)<1e-10
        minimum=float(np.linalg.norm(a['next_poses'][:,:2]-a['task_goal_xy'],axis=1).min())
        contacts=sorted({name for c in row['terminal_contacts']
                         for name in [c['geom1'],c['geom2']]
                         if name and name.startswith(('real_circle','real_rect'))})
        times=np.r_[0,np.cumsum(n)[:-1]]
        gaps={}
        if row['job_id']=='wide117_8m_dev6_LEFT_13':
            for seconds in [7.4,15.,25.,30.]:
                step=int(round(seconds/.002));bi=np.flatnonzero(bt==step);ai=np.flatnonzero(times==step)
                if len(bi) and len(ai):
                    gaps[str(seconds)]=float(np.linalg.norm(a['poses'][ai[0],:2]-baseline['poses'][bi[0],:2]))
        records.append(dict(job_id=row['job_id'],seed_index=row['seed'][-1],
            success=row['success'],collision=row['collision'],timeout=row['timeout'],
            boundary_crossed=row['field_detail']['field_boundary_crossed'],
            original_clock_terminal_s=float(n.sum(dtype=np.int64))*.002,
            suffix_seconds=total*.002,suffix_return=v,closed_form_return=closed,
            terminal_distance_m=row['remaining_m'],minimum_logged_endpoint_distance_m=minimum,
            contact_obstacles=contacts,position_gap_to_nominal_by_clock_s=gaps,
            path=row['path'],sha256=row['sha256']))
    report=dict(state='OFFLINE_REVIEW_COMPLETE',native_steps=0,optimizer_updates=0,
        source_summary_sha256=sha(root/'summary.json'),source_audit_sha256=sha(root/'audit.json'),
        T631_physics=summary['physics'],T631_commands=summary['commands'],
        T631_Actor_updates=0,full49=None,retained_model=summary['result']['model'],
        baseline_8m_physics=int(bn.sum(dtype=np.int64)),
        baseline_8m_goal_seconds=float(bn.sum(dtype=np.int64))*.002,
        gamma_native=gamma,discount_efold_seconds=-.002/math.log(gamma),
        maximum_return_reconstruction_error=error,
        reward_inference='For this fixed-start all-collision group, potential terms telescope; total return distinguishes collision time, not an independent terminal-distance preference. This is an intended property of potential shaping, not proof of an implementation bug or the unique failure cause.',
        records=records,
        dtype_note='Widen packed uint native_counts to int64 before sums/cumsums. An earlier ad-hoc built-in sum printed an overflowed nominal duration; only this int64 result is authoritative. Existing native counters/audit evidence were not modified.')
    DOC.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(report,indent=2)+'\n').encode();assert len(data)<128*1024
    (DOC/'OUTCOME_AND_REWARD.json').write_bytes(data)
    print(json.dumps(dict(state=report['state'],baseline_8m_goal_seconds=report['baseline_8m_goal_seconds'],
                         maximum_return_reconstruction_error=error,records=len(records),
                         native_steps=0,optimizer_updates=0)))


if __name__=='__main__':main()
