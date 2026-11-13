"""Identify one causal model from fixed existing train blocks, then temporal test."""
from pathlib import Path
import fcntl,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t533_causal_vehicle_identification';DOC=ROOT/'stage/docs/day_20261113'
sys.path.insert(0,str(ROOT/'stage/experiments/t532_physics_forward_contract'));import t532_contract as C
sys.path.insert(0,str(HERE));import t533_model as V
read,save,sha,used=C.read,C.save,C.sha,C.used
def verify(p):
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json') and time.time()<p['absolute_all_stop']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))<={14,15}
    assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t532_physics_forward_contract';p0=read(prior/'protocol.json')
    assert read(prior/'audit.json')['state']=='PASS' and not read(prior/'summary.json')['result']['forward_gate']
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T532.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    A=C.backend();blocks=[]
    for window in p0['windows']:
        r=window['row'];a=A.G.E.N.S.arrays(r['path']);n=len(a['states']);cut=int(.4*n)
        # Split before fitting: first40%, then25decision=5second embargo;
        # validation uses nonoverlapping complete1second windows, no terminal.
        starts=[i for i in range(cut+25,n-5+1,5) if not a['end_flags'][i:i+5].any() and (a['native_counts'][i:i+5]==100).all()]
        assert cut>=25 and len(starts)>=5
        blocks.append(dict(id=window['id'],row=r,train_end=cut,validation_starts=starts,n=n))
    p=dict(id='T533_CAUSAL_VEHICLE_IDENTIFICATION',blocks=blocks,ridge=.001,holds=5,max_xy_error_m=.02,max_yaw_error_deg=1.,max_bytes=1024*1024,main_seconds=300,audit_seconds=300,transport_seconds=300,source_seconds=180,absolute_all_stop=p0['absolute_all_stop'],new_native=0,new_Actor_Adam=0,new_Value_Adam=0,model_fits=1,CPUQuota='100%',GPU=False,scope='One seven-feature linear causal next-body-velocity model. First40percent per existing source trajectory train,5second embargo,rest fixed1second validation. Existing temporal data only,not unseen map/independent trajectory. Recorded controls remain fixed,zero sensor/GRU forward modeling,zero native directional tests. No refit/grid/follow-on Actor training regardless result.')
    p['inputs']={f:h for f,h in p0['inputs'].items() if Path(f).suffix=='.py'}
    for b in blocks:p['inputs'][b['row']['path']]=b['row']['sha256']
    for f in [*HERE.glob('*.py'),DOC/'T533_PLAN.md',prior/'summary.json',prior/'audit.json',prior/'protocol.json']:p['inputs'][str(f)]=sha(f)
    assert time.time()+1200<p['absolute_all_stop'] and used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p)
    save(DOC/'T533_BUDGET.json',dict(own_bytes=p['max_bytes'],global_reserved=used(ROOT/'stage/runs')-used(OUT)+p['max_bytes'],new_native=0,new_Actor_Adam=0,new_Value_Adam=0,model_closed_form_fit=1,audit_closed_form_reconstruction=1,max_seconds=1080,all_stop=p['absolute_all_stop']))
    print('T533_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();save(OUT/'preflight.json',dict(V.fixtures(),pid=os.getpid(),real_fit_NOT_executed=True));print('T533_PREFLIGHT_PASS',flush=True)
def calculate(p):
    A=C.backend();np=A.np;arrays=[];xs=[];ys=[]
    for b in p['blocks']:
        a=A.G.E.N.S.arrays(b['row']['path']);assert all(A.digest(v)==b['row']['array_hashes'][k] for k,v in a.items())
        v=V.velocity(a['states']);nv=V.velocity(a['next_states']);n=b['train_end'];assert not a['end_flags'][:n].any() and (a['native_counts'][:n]==100).all()
        xs.append(V.features(v[:n],a['actions'][:n]));ys.append(nv[:n]);arrays.append(a)
    x=np.concatenate(xs);y=np.concatenate(ys);m=V.fit(x,y,p['ridge']);metrics=[];payload={k:v for k,v in m.items()};payload.update(train_x=x,train_y=y)
    for bi,(b,a) in enumerate(zip(p['blocks'],arrays)):
        rows=[];v=V.velocity(a['states'])
        for si,start in enumerate(b['validation_starts']):
            verify(p);stop=start+5;dt=a['native_counts'][start:stop]*.002;pose=a['poses'][start];actual=a['next_poses'][start:stop]
            pred=V.rollout(pose,v[start],a['actions'][start:stop],dt,m)
            base=C.model.constant_twist(pose,v[start,:2],v[start,2],dt)
            e=C.model.errors(pred,actual);be=C.model.errors(base,actual)
            rows.append(dict(start=start,start_seconds=start*.2,error=e,constant_velocity=be,qualified=e['max_xy_m']<=p['max_xy_error_m'] and e['max_yaw_deg']<=p['max_yaw_error_deg']))
            payload[f'b{bi}w{si}_prediction']=pred;payload[f'b{bi}w{si}_actual']=actual
        metrics.append(dict(id=b['id'],train_transitions=b['train_end'],validation_windows=len(rows),qualified_windows=sum(r['qualified'] for r in rows),max_xy_m=max(r['error']['max_xy_m'] for r in rows),max_yaw_deg=max(r['error']['max_yaw_deg'] for r in rows),constant_max_xy_m=max(r['constant_velocity']['max_xy_m'] for r in rows),constant_max_yaw_deg=max(r['constant_velocity']['max_yaw_deg'] for r in rows),windows=rows))
    ok=all(m['qualified_windows']==m['validation_windows'] for m in metrics)
    return dict(temporal_forward_gate=ok,metrics=metrics,train_samples=len(x),model_fit=1,model_ridge=p['ridge'],coefficient=m['coefficient'].tolist(),feature_mean=m['mean'].tolist(),feature_scale=m['scale'].tolist(),reason='TEMPORAL_FORWARD_PASS_NATIVE_DIRECTION_UNTESTED' if ok else 'CAUSAL_ARX_TEMPORAL_GATE_FAILED_STOP_THIS_MODEL',Actor_learning_eligible=False,direction_gate=None,Actor_Adam=0,Value_Adam=0,new_native=0,no_new_navigation_result=True),payload
def main():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'summary.json').exists();start=time.time()
    with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result,payload=calculate(p);import numpy as np
        np.savez_compressed(OUT/'predictions.npz',**payload);assert (OUT/'predictions.npz').stat().st_size<256*1024
        save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,result=result,payload_sha256=sha(OUT/'predictions.npz')));verify(p)
    print(result['reason'],flush=True)
def audit():
    p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json');assert not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    result,payload=calculate(p);assert result==s['result'];import numpy as np
    with np.load(OUT/'predictions.npz') as z:
        assert set(z.files)==set(payload)
        for k,v in payload.items():assert np.array_equal(z[k],v),k
    assert sha(OUT/'predictions.npz')==s['payload_sha256']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),all_arrays_fit_split_coefficients_predictions_recomputed=True,independent_discarded_closed_form_fit=1,new_native=0,new_optimizer=0,scientific_temporal_gate=result['temporal_forward_gate'],Actor_learning_eligible=False));print('T533_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
