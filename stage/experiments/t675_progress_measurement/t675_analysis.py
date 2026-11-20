"""Validate recorded radial-progress measurements; no new reward or policy update."""
from pathlib import Path
import fcntl,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t675_progress_measurement';DOC=ROOT/'stage/docs/weekend_20261120_23/t675'
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
from status_io import save
np,read,sha=B.np,B.read,B.sha

def measure(row,a):
    poses=a['poses'];ends=a['next_poses'];goal=a['task_goal_xy']
    assert poses.ndim==2 and poses.shape[1]==3 and poses.shape==ends.shape and goal.shape==(2,)
    assert np.isfinite(poses).all() and np.isfinite(ends).all() and np.isfinite(goal).all()
    assert np.array_equal(ends[:-1],poses[1:])
    assert int(a['native_counts'].sum())==row['physics']
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
    distances=np.linalg.norm(np.vstack([poses[0,:2],ends[:,:2]])-goal,axis=1)
    initial=float(distances[0]);terminal=float(distances[-1]);error=abs(terminal-row['remaining_m']);assert initial>.2 and error<=1e-9
    safe=bool(B.M.outcome(row,'binary'))
    if safe:assert terminal<=.2+1e-9
    progress=float(np.clip((initial-terminal)/max(initial,.2),-1.,1.))
    return dict(id=row['job_id'],source_sha256=row['sha256'],initial_m=initial,terminal_m=terminal,remaining_error=error,
                recorded_min_m=float(distances.min()),closest_recorded_decision=int(distances.argmin()),decisions=len(ends),
                normalized_terminal_progress=progress,safe=safe,collision=bool(row['collision']),timeout=bool(row['timeout']),
                outcome=B.M.outcome(row,'collision_cost'))

def compute(p,limit):
    start=time.time();rows=[];tasks={};last=[0.]
    for key,group in B.dataset(p).items():
        B.A.guard(p,start,limit);items=[]
        for row in group:
            assert row['full'] and not row.get('teacher') and row['policy_hash']==p['initial']['policy_hash'] and sha(row['path'])==row['sha256']
            a=B.T.R.E.N.S.arrays(row['path']);assert not a['ownership'].any() and a['stochastic'].all()
            item=measure(row,a);rows.append(item);items.append(item)
        values=[x['normalized_terminal_progress'] for x in items];ys=[x['outcome'] for x in items]
        tasks[key]=dict(n=len(items),outcomes=ys,progress=values,progress_range=max(values)-min(values),constant_terminal=len(set(ys))==1,
                        all_failed=all(not x['safe'] for x in items))
        if time.time()-last[0]>3:save(OUT/'status.json',dict(state='ACTIVE',action=sys.argv[1],task=key,completed=len(rows),child_pid=os.getpid(),epoch=time.time(),new_native=0,new_optimizer=0));last[0]=time.time()
    failed={k:v for k,v in tasks.items() if k.startswith('t502_field_') and v['constant_terminal'] and v['all_failed']}
    informative=[k for k,v in failed.items() if v['progress_range']>=p['minimum_progress_range']]
    checks=dict(all407_and1724=len(tasks)==407 and len(rows)==1724,all_geometry_remaining_chain_checked=True,
                at_least3_failed_field_progress_variation=len(informative)>=3)
    return dict(state='PROGRESS_MEASUREMENT_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,rows=rows,tasks=tasks,
                failed_field_same_terminal_count=len(failed),failed_field_informative=informative,maximum_remaining_error=max(x['remaining_error'] for x in rows),
                progress_not_obstacle_free_distance=True,no_safety_or_navigation_improvement_claim=True,no_new_reward_defined=True,new_native=0,new_optimizer=0)

def execute(action):
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    for folder in p['sealed_sources']:
        seal=read(Path(folder)/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
        assert not Path('/proc',str(read(Path(folder)/'DRIVER_STARTED.json')['pid'])).exists()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();result=compute(p,p[action+'_seconds'])
    if action=='main':save(OUT/'RESULTS.json',result)
    else:
        assert result==read(OUT/'RESULTS.json');save(OUT/'audit.json',dict(state='PASS',all1724_geometry_and_progress_exact=True,new_native=0,new_optimizer=0))
    assert B.T.R.used(OUT)<p['max_bytes'] and B.T.R.used(ROOT/'stage/runs')<p['global_bytes']
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',elapsed_s=time.time()-start,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':execute(sys.argv[1])
