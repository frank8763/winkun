"""Read saved histories to separate imitation error from rollout input shift.

Same-time action decomposition is algebra, not a physical causal intervention.
No native environment, optimizer, new policy or navigation test is executed.
"""
from pathlib import Path
import sys,time,json,os,datetime
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t600_task_risk_pair'))
import t600_learning as L
read,sha,digest,torch,np=L.read,L.sha,L.digest,L.torch,L.np
OLD=ROOT/'stage/runs/t600_task_risk_pair'
OUT=ROOT/'stage/runs/t601_takeover_distribution_diagnosis'
DOC=ROOT/'stage/docs/day_20261116/t601'


def save(p,obj):
    b=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n';assert len(b.encode())<128*1024
    with p.open('x') as f:f.write(b)


def register():
    assert not OUT.exists() and read(OLD/'audit.json')['state']=='PASS'
    sealed=read(OLD/'SEALED.json');assert all(not Path('/proc',str(pid)).exists() for pid in sealed['scientific_pids'])
    p=read(OLD/'protocol.json');s=read(OLD/'summary.json');assert s['state']=='COMPLETE'
    files=[OLD/'protocol.json',OLD/'summary.json',OLD/'audit.json',Path(__file__)]
    files += [Path(p['joint_recovery_row']['path'])]
    for a in s['result']['arms']:files += [Path(a['proposal']['path']),Path(a['recovery']['path'])]
    inputs={str(f):sha(f) for f in files}
    model=p['base_model']
    while model:inputs[model['path']]=model['sha256'];model=model.get('parent')
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):inputs[str(Path(f).resolve())]=sha(f)
    cap=128*1024;assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())+cap<2**30
    end=datetime.datetime.fromisoformat('2026-11-16T21:00:00+08:00').timestamp();assert time.time()+300<end
    protocol=dict(id='T601_SAVED_TAKEOVER_ERROR_DECOMPOSITION',inputs=inputs,seconds=300,all_stop=end,own_bytes=cap,new_native=0,new_optimizer=0,new_gradients=0,CPU=[14],GPU=False,
        descriptive_thresholds=dict(position_m=.03,yaw_deg=3.,action_abs=.05),fixed_windows_seconds=[[0,2],[2,5],[5,10]],
        scope='Same successful source and2failed candidates; compare first takeover action on equal observation histories; success-source prediction error and live-input shift. Not a controller, policy update or causal intervention.')
    OUT.mkdir();DOC.mkdir(parents=True,exist_ok=True);save(OUT/'protocol.json',protocol);save(DOC/'PLAN.json',protocol)
    print('REGISTERED_ZERO_NATIVE_ZERO_GRADIENT',flush=True)


def first(mask,k):
    ids=np.flatnonzero(mask & (np.arange(len(mask))>=k));return int(ids[0]) if len(ids) else None


def main():
    p=read(OUT/'protocol.json');assert not (OUT/'launch.json').exists();start=time.time()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,protocol_sha256=sha(OUT/'protocol.json')))
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert set(os.sched_getaffinity(0))=={14} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    op=read(OLD/'protocol.json');s=read(OLD/'summary.json');source=op['joint_recovery_row']
    assert sha(source['path'])==source['sha256'];a=L.G.E.N.S.arrays(source['path']);k=int(a['student_takeover_first'])
    assert source['success'] and not source['collision'] and a['ownership'][:k].all() and not a['ownership'][k:].any()
    reports=[]
    for arm in s['result']['arms']:
        assert time.time()-start<p['seconds'] and time.time()<p['all_stop']
        row=arm['recovery'];b=L.G.E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256']
        assert int(b['student_takeover_first'])==k
        for key in ['states','poses','history_poses','history_goals','history_times']:
            assert np.array_equal(a[key][:k+1],b[key][:k+1]),key
        assert np.array_equal(a['actions'][:k],b['actions'][:k])
        q=L.frozen(op,arm['proposal']['path']);before=digest(q.state_dict())
        with torch.no_grad():
            source_loc=q.sequence(torch.tensor(a['states'])).numpy()
            live_loc=q.sequence(torch.tensor(b['states'])).numpy()
        assert float(np.abs(live_loc-b['locations']).max())<1e-12 and before==digest(q.state_dict())
        source_command=np.clip(source_loc,-1,1).astype(np.float32).astype(np.float64)
        live_command=np.clip(live_loc,-1,1).astype(np.float32).astype(np.float64)
        teacher=np.asarray(a['actions'],np.float64);n=min(len(a['states']),len(b['states']))
        off=source_command[:n]-teacher[:n];shift=live_command[:n]-source_command[:n];total=live_command[:n]-teacher[:n]
        assert np.max(np.abs(off+shift-total))<1e-14
        assert np.array_equal(live_command[k],source_command[k])
        time0=np.r_[0,np.cumsum(b['native_counts'])[:-1]]*.002
        distance=np.linalg.norm(b['poses'][:n,:2]-a['poses'][:n,:2],axis=1)
        yaw=np.abs(np.arctan2(np.sin(b['poses'][:n,2]-a['poses'][:n,2]),np.cos(b['poses'][:n,2]-a['poses'][:n,2])))*180/np.pi
        ix=first(distance>=.03,k);iy=first(yaw>=3,k);ia=first(np.max(np.abs(total),axis=1)>=.05,k)
        windows=[]
        for lo,hi in p['fixed_windows_seconds']:
            mask=(time0[:n]-time0[k]>=lo)&(time0[:n]-time0[k]<hi)
            if mask.any():windows.append(dict(after_takeover_seconds=[lo,hi],decisions=int(mask.sum()),source_history_action_RMS=float(np.sqrt(np.mean(off[mask]**2))),live_input_shift_action_RMS=float(np.sqrt(np.mean(shift[mask]**2))),total_action_RMS=float(np.sqrt(np.mean(total[mask]**2))),mean_source_drive_steer=teacher[:n][mask].mean(0).tolist(),mean_candidate_live_drive_steer=live_command[:n][mask].mean(0).tolist(),max_xy_difference_m=float(distance[mask].max()),max_yaw_difference_deg=float(yaw[mask].max())))
        reports.append(dict(method=arm['method'],takeover_decision=k,takeover_clock_s=float(time0[k]),same_inputs_at_takeover=True,
            teacher_takeover_drive_steer=teacher[k].tolist(),candidate_takeover_drive_steer=live_command[k].tolist(),takeover_action_error=(live_command[k]-teacher[k]).tolist(),
            first_3cm_decision=ix,first_3cm_clock_s=None if ix is None else float(time0[ix]),first_3deg_decision=iy,first_3deg_clock_s=None if iy is None else float(time0[iy]),
            first_action05_decision=ia,first_action05_clock_s=None if ia is None else float(time0[ia]),
            collision_clock_s=float(np.sum(b['native_counts'])*.002),collision_remaining_m=row['remaining_m'],
            source_success_clock_s=float(np.sum(a['native_counts'])*.002),windows=windows,
            descriptive_no_physical_intervention=True))
    for f,h in p['inputs'].items():assert sha(f)==h,f
    save(OUT/'result.json',dict(state='COMPLETE_READ_ONLY',new_native=0,new_optimizer=0,new_gradients=0,models_and_sources_unchanged=True,arms=reports,elapsed_s=time.time()-start))
    print('T601_COMPLETE_READ_ONLY',time.time()-start,flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
