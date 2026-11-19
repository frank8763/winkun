"""Recover exactly six T651 actual failure paths, not new policy search."""
import fcntl,gzip,json,os,traceback
import t652_runtime as T
R,O,OUT=T.R,T.O,T.OUT


def main():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'reconstruction'
    lock=(T.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (folder/'launch.json').exists();T.save(folder/'launch.json',dict(pid=os.getpid(),epoch=T.time.time()))
    rt=O.Runtime(p);error=None;rows=[]
    try:
        rt.pool=R.N.Pool(dict(p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['reconstruction_jobs'],p['reconstruction_model'],'parent_full6',True);rows=[r for a,r in vv]
        for (a,row),ref in zip(vv,p['reconstruction_references']):
            assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256'] and row['policy_hash']==ref['policy_hash'],'T651 actual failure path reconstruction mismatch'
        rt.request(p['reconstruction_jobs'],p['reconstruction_model'],'parent_exact6',True,rows)
    except BaseException:error=traceback.format_exc();T.save(folder/'failure.json',dict(error=error,raw_retained=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        T.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,rows=rows,new_optimizer=0,exact_regeneration_not_new_independent=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def divergence(a,b):
    ta=T.np.r_[0,T.np.cumsum(a['native_counts'])[:-1]];tb=T.np.r_[0,T.np.cumsum(b['native_counts'])[:-1]];n=min(len(ta),len(tb));assert T.np.array_equal(ta[:n],tb[:n])
    xy=T.np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1)
    dy=a['poses'][:n,2]-b['poses'][:n,2];yaw=abs(T.np.arctan2(T.np.sin(dy),T.np.cos(dy)))*180/T.np.pi
    ix=T.np.flatnonzero((xy>=.02)|(yaw>=1.))
    if not len(ix):return None
    k=int(ix[0]);return dict(decision=k,seconds=float(ta[k]*.002),XY_m=float(xy[k]),yaw_degrees=float(yaw[k]))


def audit():
    p=T.plan();T.verify(p);T.setup(p);folder=OUT/'reconstruction';s=T.read(folder/'summary.json');assert s['state']=='COMPLETE'
    assert not T.Path('/proc',str(s['pid'])).exists()
    for g in T.read(folder/'worker_generations.json'):assert all(not T.Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows};assert len(by)==len(rows)==12
    refs={r['job_id']:r for r in p['reconstruction_references']};original={r['id']:r for r in T.read(T.ROOT/'stage/runs/t646_offline_night/OFFLINE_INPUTS.json')['members']}
    details=[]
    for row in rows:
        assert T.sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path']);O.N.inspect(p,a,row);ref=refs[row['job_id']]
        assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        if row.get('exact'):assert row['array_hashes']==by[row['reference_path']]['array_hashes']
        else:
            source=original[row['job_id']];assert T.sha(source['offline_path'])==source['sha256'];b=R.E.N.S.arrays(source['offline_path'])
            assert T.np.array_equal(a['scene_sha256'],b['scene_sha256']) and T.np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
            assert T.np.array_equal(a['states'][0],b['states'][0]) and T.np.array_equal(a['task_goal_xy'],b['task_goal_xy'])
            details.append(dict(id=row['job_id'],first_actual_pose_divergence=divergence(a,b),collision=row['collision'],timeout=row['timeout'],remaining_m=row['remaining_m'],reference_T501_safe=True))
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and s['commands']==18
    T.save(folder/'DIVERGENCE.json',dict(state='READ_ONLY_ACTUAL_TRACE_COMPARISON',cases=details,new_optimizer=0,no_causal_stability_claim=True))
    T.save(folder/'audit.json',dict(state='PASS',all_six_actual_paths_exact_T651=True,all_six_replays_exact=True,initial_geometry_120_integration_goal_match_T501=True,new_native=0,new_optimizer=0))

if __name__=='__main__':globals()[T.sys.argv[1]]()
