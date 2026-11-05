"""Three fixed actuator-aware prediction fits, before any MPPI teacher use."""
from pathlib import Path
import ast
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t380_teacher_model_audit'
DOC=ROOT/'stage/docs/narrow_20261105'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
def save(path,value,cap=32*1024):
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    assert len(data)<=cap,(str(path),len(data));Path(path).write_bytes(data)
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def absent(pid):assert not Path('/proc',str(pid)).exists(),pid
def wrap(a):return np.arctan2(np.sin(a),np.cos(a))


def local_delta(before,after):
    d=after[...,:2]-before[...,:2];c=np.cos(before[...,2]);s=np.sin(before[...,2])
    return np.stack([c*d[...,0]+s*d[...,1],-s*d[...,0]+c*d[...,1],wrap(after[...,2]-before[...,2])],axis=-1)


def advance(pose,delta):
    c=np.cos(pose[2]);s=np.sin(pose[2])
    return np.array([pose[0]+c*delta[0]-s*delta[1],pose[1]+s*delta[0]+c*delta[1],wrap(pose[2]+delta[2])])


def state(x):return np.c_[x[:,53:55],2*x[:,55],.5*x[:,113:115],3*x[:,115:117]].astype(np.float64)


def features(s,u):
    s=np.asarray(s,dtype=np.float64);u=np.asarray(u,dtype=np.float64)
    return np.r_[1.,s,u,s[0]*np.tan(.5*(s[3]+s[4]))]


def load(path):
    with np.load(path) as z:a={k:z[k] for k in z.files}
    x=a['states'];assert x.shape[1]==120 and np.isfinite(x).all()
    nxt=a['next_states'] if 'next_states' in a else np.concatenate([x[1:],a['last_next_state']],axis=0)
    assert nxt.shape==x.shape and np.array_equal(nxt[:-1],x[1:])
    a['q']=state(x);a['next_q']=state(nxt)
    a['valid']=(a['native_counts']==100)&~a['end_flags'][:,0].astype(bool)
    a['X']=np.stack([features(s,u) for s,u in zip(a['q'],a['actions'])])
    a['Y']=np.c_[a['next_q']-a['q'],local_delta(a['poses'],a['next_poses'])]
    assert a['X'].shape[1]==11 and a['Y'].shape[1]==10
    return a


def fit(data):
    x=np.concatenate([a['X'][a['valid']] for a in data]);y=np.concatenate([a['Y'][a['valid']] for a in data])
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);mean[0]=0.;scale[0]=1.
    z=(x-mean)/scale;regular=np.eye(11)*.001;regular[0,0]=0.
    coefficient=np.linalg.solve(z.T@z+regular,z.T@y)
    singular=np.linalg.svd(z,compute_uv=False)
    return dict(mean=mean,scale=scale,coefficient=coefficient),dict(transitions=len(x),
         feature_singular_values=singular.tolist(),standardized_rank=int(np.linalg.matrix_rank(z)),
         action_min=x[:,8:10].min(0).tolist(),action_max=x[:,8:10].max(0).tolist(),
         action_std=x[:,8:10].std(0).tolist())


def rollout(model,q,pose,actions):
    q=q.copy();pose=pose.copy()
    with np.errstate(over='ignore',invalid='ignore'):
        for u in actions:
            delta=((features(q,u)-model['mean'])/model['scale'])@model['coefficient']
            pose=advance(pose,delta[7:]);q=q+delta[:7]
            if not(np.isfinite(pose).all() and np.isfinite(q).all()):return np.full(3,np.nan)
    return pose


def persistence(q,pose,horizon):
    p=pose.copy();angle=q[2]*.2
    for _ in range(horizon):
        middle=p[2]+angle*.5;c=np.cos(middle);s=np.sin(middle)
        p=np.array([p[0]+.2*(c*q[0]-s*q[1]),p[1]+.2*(s*q[0]+c*q[1]),wrap(p[2]+angle)])
    return p


def statistics(prediction,target):
    finite=np.isfinite(prediction).all(1);e=prediction[finite]-target[finite]
    if not len(e):return dict(nonfinite_windows=len(prediction),position_RMSE=None,position_p95=None,yaw_p95_rad=None)
    distance=np.linalg.norm(e[:,:2],axis=1);angle=np.abs(wrap(e[:,2]))
    return dict(nonfinite_windows=int((~finite).sum()),position_RMSE=float(np.sqrt(np.mean(distance**2))),
                position_p95=float(np.percentile(distance,95)),position_max=float(distance.max()),
                yaw_p95_rad=float(np.percentile(angle,95)),yaw_max_rad=float(angle.max()))


def compute(p):
    verify(p);data={k:load(v) for k,v in p['datasets'].items()};models=[];model_arrays={};predictions={};evaluations=[]
    for label,train,test in [('A',['human361'],'human364'),('B',['human364'],'human361'),('C',['human361','human364'],'learner379')]:
        model,description=fit([data[k] for k in train]);models.append(dict(model=label,training=train,test=test,**description))
        for k,v in model.items():model_arrays[label+'_'+k]=v
        a=data[test]
        for h in [1,5,10]:
            starts=np.array([k for k in range(len(a['valid'])-h+1) if a['valid'][k:k+h].all()],dtype=np.int32)
            assert len(starts)>0
            pred=np.stack([rollout(model,a['q'][k],a['poses'][k],a['actions'][k:k+h]) for k in starts])
            base=np.stack([persistence(a['q'][k],a['poses'][k],h) for k in starts])
            target=a['next_poses'][starts+h-1];key=label+'_'+str(h)
            predictions.update({key+'_starts':starts,key+'_model':pred,key+'_persistence':base,key+'_target':target})
            actual=statistics(pred,target);control=statistics(base,target)
            checks=dict(no_nonfinite=actual['nonfinite_windows']==0,
                 position_p95_002=actual['position_p95'] is not None and actual['position_p95']<=.02,
                 yaw_p95_3deg=actual['yaw_p95_rad'] is not None and actual['yaw_p95_rad']<=np.deg2rad(3.),
                 RMSE_080_persistence=actual['position_RMSE'] is not None and actual['position_RMSE']<=.8*control['position_RMSE'])
            evaluations.append(dict(model=label,test=test,horizon_holds=h,windows=len(starts),eligible_start_candidates=max(0,len(a['valid'])-h+1),
                  overlapping_windows_not_independent=True,model_prediction=actual,persistence=control,
                  checks={k:bool(v) for k,v in checks.items()},passed_1second_filter=bool(h==5 and all(checks.values()))))
    gate=all(row['passed_1second_filter'] for row in evaluations if row['horizon_holds']==5)
    support=all(v>=.02 for v in models[2]['action_std']);gate=gate and support
    rows=[]
    for key,a in data.items():
        rows.append(dict(dataset=key,transitions=len(a['valid']),eligible_full_noncontact=int(a['valid'].sum()),
             short_holds=int((a['native_counts']!=100).sum()),contact_intervals=int(a['end_flags'][:,0].sum()),
             source_observation_shape=list(a['states'].shape),action_min=a['actions'].min(0).tolist(),action_max=a['actions'].max(0).tolist(),
             velocity_clip_counts=(np.abs(a['states'][:,53:55])>=2).sum(0).tolist(),
             steering_clip_counts=(np.abs(a['states'][:,113:117])>=3).sum(0).tolist()))
    report=dict(state='COMPLETE',native_commands=0,physics_steps=0,optimizer_calls=0,analytic_fits=3,
         datasets=rows,models=models,evaluations=evaluations,action_support_std_ge002=bool(support),
         prediction_filter_passed=bool(gate),MPPI_executed=False,teacher_attached_to_Actor=False,
         next_scope='Only a separately bounded native model/control check if passed; never a safety or teacher-success certificate',
         all_workers_stopped=True,old275=None,six54=None,fresh16=None,reload78=None,entered7or8=False)
    return report,model_arrays,predictions


def budget(p):
    assert used(OUT)<=p['final_bytes'] and used(ROOT/'stage/runs')<=p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert used(ROOT/'stage/runs')-used(OUT)+p['final_bytes']<=p['global_bytes'];budget(p)


def prepare():
    assert not OUT.exists()
    previous=ROOT/'stage/runs/t379_human_only_control';assert read(previous/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T379.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for pid in [read(previous/'summary.json')['pid'],read(previous/'launch.json')['pid'],*read(previous/'workers.json')['workers']]:absent(pid)
    datasets=dict(human361=str(ROOT/'stage/runs/t361_human_terminal_correction/attempt_01.npz'),
                  human364=str(ROOT/'stage/runs/t364_early_correction_retry/attempt_01.npz'),learner379=str(previous/'centre.npz'))
    files=[*map(Path,datasets.values()),previous/'summary.json',previous/'audit.json',DOC/'T380_PLAN.md',
           ROOT/'stage/experiments/t249_actuator_observation/actuator_support.py',
           ROOT/'stage/runtime/mk-mini/training/e2e_phase10_lidar_v1/phase10_scene_e2e_env.py',*HERE.glob('*.py')]
    p=dict(id='T380',datasets=datasets,inputs={str(f):sha(f) for f in sorted(files)},final_bytes=256*1024,
           global_bytes=1024**3,min_free_bytes=40*1024**3,main_seconds=300,audit_seconds=300,all_work_seconds=1200,
           analytic_fits_main=3,analytic_fits_audit=3,native_commands=0,optimizer_calls=0)
    before=used(ROOT/'stage/runs');assert before+p['final_bytes']<p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p,8*1024);save(OUT/'protocol.json',p,8*1024)
    save(DOC/'T380_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=before,own256KiB=p['final_bytes'],full_reservation=before+p['final_bytes'],global_limit=p['global_bytes'],native=0,optimizer=0,analytic_fits=3))
    print('REGISTERED_T380_NO_NATIVE_NO_ACTOR')


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    origin=np.array([1.,2.,.7]);delta=np.array([.12,-.03,.2]);target=advance(origin,delta)
    assert np.max(np.abs(local_delta(origin,target)-delta))<1e-14
    assert abs(abs(wrap(np.pi+.01))-(np.pi-.01))<1e-14
    q=np.arange(7)*.01;assert features(q,np.array([.9,.2])).shape==(11,)
    m=np.array([True,False,True]);assert [k for k in range(2) if m[k:k+2].all()]==[]
    save(OUT/'preflight.json',dict(state='PASS',native=0,optimizer=0,actual_data_fits=0,coordinate_wrap_feature_and_window_filter_contract=True),8*1024)
    print('PREFLIGHT_PASS_NO_NATIVE_NO_FITS')


def persist_arrays(path,arrays,cap):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);blob=buffer.getvalue();assert len(blob)<=cap,(str(path),len(blob));path.write_bytes(blob)


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    start=time.time();remaining=read(OUT/'launch.json')['main_deadline_epoch']-start-2;assert remaining>0
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T380 main deadline')));signal.alarm(max(1,int(min(295,remaining))))
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]),4*1024)
    try:
        report,models,predictions=compute(p)
        persist_arrays(OUT/'models.npz',models,16*1024);persist_arrays(OUT/'predictions.npz',predictions,128*1024);save(OUT/'results.json',report)
        s=dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,native_commands=0,physics_steps=0,optimizer_calls=0,
               analytic_fits=3,prediction_filter_passed=report['prediction_filter_passed'],all_workers_stopped=True,
               results_sha256=sha(OUT/'results.json'),model_sha256=sha(OUT/'models.npz'),predictions_sha256=sha(OUT/'predictions.npz'),MPPI_executed=False)
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),analytic_fits_completed=None,analytic_fit_upper_bound=3),16*1024)
        s=dict(state='FAILED',pid=os.getpid(),elapsed_s=time.time()-start,native_commands=0,physics_steps=0,optimizer_calls=0,
               analytic_fits_completed=None,analytic_fit_upper_bound=3,all_workers_stopped=True,MPPI_executed=False)
    save(OUT/'summary.json',s,4*1024);save(OUT/'status.json',s,4*1024);budget(p);signal.alarm(0);print(json.dumps(s))


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    absent(s['pid']);absent(read(OUT/'launch.json')['pid'])
    remaining=read(OUT/'launch.json')['all_work_deadline_epoch']-time.time()-10;assert remaining>0
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T380 audit deadline')));signal.alarm(max(1,int(min(295,remaining))))
    report,models,predictions=compute(p);assert report==read(OUT/'results.json') and sha(OUT/'results.json')==s['results_sha256']
    for name,arrays,key in [('models',models,'model_sha256'),('predictions',predictions,'predictions_sha256')]:
        path=OUT/(name+'.npz');assert sha(path)==s[key]
        with np.load(path) as z:
            assert set(z.files)==set(arrays)
            assert all(np.array_equal(z[k],v,equal_nan=True) and array_sha(z[k])==array_sha(v) for k,v in arrays.items())
    result=dict(state='PASS',native=0,optimizer=0,independent_analytic_fits=3,all_coefficients_predictions_metrics_and_gates_exact=True,all_original_processes_absent=True)
    save(OUT/'audit.json',result,8*1024);save(DOC/'T380_AUDIT.json',result,8*1024);signal.alarm(0);print(json.dumps(result))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'launch.json').exists() and read(OUT/'preflight.json')['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'main.log').open('xb') as log:
        proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=1s','300s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],
           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,main_deadline_epoch=epoch+300,all_work_deadline_epoch=epoch+1200,source_remote_commit=receipt['remote_commit'],
         root_tree=receipt['root_tree'],protocol_sha256=sha(HERE/'protocol.json')),4*1024);print(json.dumps(dict(state='LAUNCHED_ZERO_NATIVE_MODEL_CHECK',pid=proc.pid)))


if __name__=='__main__':
    assert sys.argv[1] in ['prepare','preflight','launch','main','audit'];globals()[sys.argv[1]]()
