"""Descriptive T371 locality check: fixed saved evidence, zero native/Adam/gradient."""
from t371_core import *


def main():
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Read-only analysis60s')));signal.alarm(60)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=read(HERE/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and read(OUT/'audit.json')['state']=='PASS'
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    sources=[HERE/'protocol.json',OUT/'summary.json',OUT/'audit.json',p['repair_parent']['path'],s['model']['path'],p['repair_source']['path'],OUT/'cases.json']
    initial=NightActor(p).eval().requires_grad_(False);final=NightActor(p).eval().requires_grad_(False);final.load_learned(read_model(s['model']['path']))
    assert fixed_hash(initial)==fixed_hash(final)
    def predict(q,a):
        x=torch.tensor(a['states'])
        with torch.no_grad():return q.sequence(x,q.base_actions(x)).numpy()
    a=arrays(p['repair_source']['path']);v=predict(initial,a);w=predict(final,a);mask=a['override_mask'].astype(bool)
    assert not np.any(w[:,0]!=v[:,0])
    stats=[]
    for lo,hi in [(0,20),(20,100),(100,200),(200,248),(248,253),(253,len(v))]:
        d=w[lo:hi,1]-v[lo:hi,1]
        stats.append(dict(decisions=[lo,hi],steer_delta_rmse=float(np.sqrt(np.mean(d*d))),steer_delta_max=float(np.max(np.abs(d)))))
    repair=dict(initial_patch_rmse=float(np.sqrt(np.mean((v[mask,1]-a['actions'][mask,1])**2))),final_patch_rmse=float(np.sqrt(np.mean((w[mask,1]-a['actions'][mask,1])**2))),nonpatch_steer_change_rmse=float(np.sqrt(np.mean((w[~mask,1]-v[~mask,1])**2))),drive_on_identical_histories_exact=True,windows=stats)
    rows=read(OUT/'cases.json');before=next(r for r in rows if r['role']=='initial' and r['job_id']=='narrow070_dev_0');after=next(r for r in rows if r['role']=='final' and r['job_id']=='narrow070_dev_0')
    for r in [before,after]:assert K.C.sha(r['path'])==r['sha256'];sources.append(r['path'])
    x,y=arrays(before['path']),arrays(after['path']);n=min(len(x['states']),len(y['states']))
    def first(z):
        found=np.flatnonzero(z);return None if not len(found) else int(found[0])
    step_times=np.concatenate([[0.],np.cumsum(x['native_counts'])[:-1]*.002]);action_diff=np.max(np.abs(x['actions'][:n]-y['actions'][:n]),axis=1);position_diff=np.linalg.norm(x['poses'][:n,:2]-y['poses'][:n,:2],axis=1)
    divergences={}
    for name,values,threshold in [('action_any',action_diff,0.),('action_0_01',action_diff,.01),('action_0_05',action_diff,.05),('position_0_05m',position_diff,.05),('position_0_1m',position_diff,.1)]:
        k=first(values>threshold);divergences[name]=dict(decision=k,time_s=None if k is None else float(step_times[k]))
    cv=predict(initial,x);cw=predict(final,x);identical_history_drive_error=float(np.max(np.abs(cv[:,0]-cw[:,0])))
    assert identical_history_drive_error==0.
    centre=dict(initial_end_seconds=before['physics_steps']*.002,final_end_seconds=after['physics_steps']*.002,initial_outcome='timeout' if before['timeout'] else 'contact',final_outcome='contact' if after['collision'] else 'other',initial_distance=before['remaining_m'],final_distance=after['remaining_m'],first_initial_action=x['actions'][0].tolist(),first_final_action=y['actions'][0].tolist(),divergences=divergences,repair_label_start_seconds=49.6,final_terminated_before_repair_state=after['physics_steps']*.002<49.6,identical_history_drive_error=identical_history_drive_error)
    before_ids=set(s['result']['initial_success_ids']);after_ids=set(s['result']['final_success_ids'])
    result=dict(state='READONLY_POSTHOC_COMPLETE',new_native=0,optimizer=0,gradient_calls=0,elapsed_s=time.time()-start,source_hashes={str(f):K.C.sha(f) for f in sources},repair=repair,centre=centre,retention=dict(lost_ids=sorted(before_ids-after_ids),gained_ids=sorted(after_ids-before_ids)),scope='Descriptive fixed evidence, no counterfactual new native test; not unique causal attribution or proof of required architecture.')
    save(DOC/'T371_POSTHOC_LOCALITY.json',result);signal.alarm(0);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
