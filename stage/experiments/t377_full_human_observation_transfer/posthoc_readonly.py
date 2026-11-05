"""Fixed failed native/human comparison; no native, gradients or optimizer."""
from t377_core import *


def main():
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T377posthoc30s')));signal.alarm(30)
    p=read(HERE/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and read(OUT/'audit.json')['state']=='PASS';r=read(OUT/'cases.json')[0];a=arrays(r['path']);d=arrays(p['demonstration']);n=min(len(a['states']),len(d['states']))
    assert K.C.sha(r['path'])==r['sha256'];delta=np.linalg.norm(a['poses'][:n,:2]-d['poses'][:n,:2],axis=1);times=np.r_[0,np.cumsum(a['native_counts'])[:-1]]*.002
    divergence={}
    for limit in [.01,.05,.1]:
        ks=np.flatnonzero(delta>limit);k=None if not len(ks) else int(ks[0]);divergence[str(limit)]=dict(decision=k,time_s=None if k is None else float(times[k]))
    mean=np.asarray(p['input_mean']);scale=np.asarray(p['input_scale']);goal=(a['states'][:,48:53]-mean[48:53])/scale[48:53]
    before=s['result']['initial_fit_metrics']['demonstrations'];after=s['result']['final_fit_metrics']['demonstrations'];changes=[]
    for x,y in zip(before,after):changes.append(dict(id=x['id'],initial_RMSE=x['drive_steer_RMSE'],final_RMSE=y['drive_steer_RMSE'],relative_reduction_percent=(100*(1-np.asarray(y['drive_steer_RMSE'])/np.asarray(x['drive_steer_RMSE']))).tolist()))
    result=dict(state='READONLY_POSTHOC_COMPLETE',new_native=0,gradient_calls=0,optimizer=0,elapsed_s=time.time()-start,source_native_sha256=r['sha256'],source_human_sha256=K.C.sha(p['demonstration']),whole_route_fit_changes=changes,native_first_action=a['actions'][0].tolist(),human_first_action=d['actions'][0].tolist(),native_vs_human_same_time_position_divergence=divergence,native_end_seconds=r['physics_steps']*.002,native_end_remaining_m=r['remaining_m'],initial_goal_raw_columns48_to52=a['states'][0,48:53].tolist(),initial_goal_preclip_normalized=goal[0].tolist(),goal_column_clip_fraction=(np.mean(np.abs(goal)>=10,axis=0)).tolist(),scope='Same-time paths are different states, not interchangeable expert labels or a unique-cause test. Clip statistics do not prove information sufficiency. No new parameter fit or native rollout.')
    save(DOC/'T377_POSTHOC_READONLY.json',result);signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
