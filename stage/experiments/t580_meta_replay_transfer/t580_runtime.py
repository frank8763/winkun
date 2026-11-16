"""Three predeclared matched transfer updates followed by autonomous development."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback
import t580_learning as L

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t580_meta_replay_transfer';DOC=ROOT/'stage/docs/day_20261116/t580'
sys.path.insert(0,str(ROOT/'stage/experiments/t577_offline_recovery_night'))
import t577_runtime as R
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np
Base,F,G,E,P=R.Base,R.F,R.G,R.E,R.P


def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes']
    assert used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15}
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f


def setup():
    R.HERE=HERE;R.verify=verify;R.L.frozen=L.frozen;R.setup(OUT);Base.verify=verify;P.frozen=L.frozen


def worker(pipe,index,p,counts):
    R.L.frozen=L.frozen;R.worker(pipe,index,p,counts)


class Pool(R.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
            save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise


def assess(p,rows):
    assert len(rows)==37 and [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']]
    ids={r['job_id'] for r in rows if r['success'] and not r['collision'] and r['field_detail']['field_qualified_success']}
    prior=set(p['initial_success_ids']);return dict(old8=G.R.M.tally(rows[:8]),field24=G.R.M.tally(rows[8:32]),
        extra5=G.R.M.tally(rows[32:]),success_ids=sorted(ids),lost_initial=sorted(prior-ids),
        gained=sorted(ids-prior),accepted=prior<=ids and len(ids)>len(prior),
        full407=None,fresh16=None,reload78=None,not_global_retention_guarantee=True)


def progress(r,a):
    dt=a['native_counts'].astype(float)*.002
    poses=np.vstack([a['poses'],a['next_poses'][-1]])
    d=np.linalg.norm(poses[:,:2]-a['task_goal_xy'],axis=1)
    ds=np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1)
    return dict(job_id=r['job_id'],role=r['role'],policy_hash=r['policy_hash'],
        seconds=float(dt.sum()),remaining_m=float(d[-1]),min_decision_distance_m=float(d.min()),
        endpoint_path_m=float(ds.sum()),mean_endpoint_speed_m_s=float(ds.sum()/dt.sum()),
        drive_command_mean=float(np.average(a['actions'][:,0],weights=dt)),
        full_pose_saved=r['full'],full_generated_arrays_digest=r['array_digest'])


def main():
    p=read(OUT/'protocol.json');setup();verify(p,True);E.N.offline();E.L.install_stop_handlers()
    rt=Base.Runtime(p);state='FAILED';error=None;result=None;arms={};objects={};initials={}
    try:
        data=L.dataset(p)
        for arm in p['arms']:
            rt.role='LEARNING_'+arm;obj=L.make(p);shadow=L.make(p);before=L.state(obj)
            event=lambda w:(lambda kind,k:Base.append('optimizer_calls.jsonl',[arm,w,kind,k]))
            u=L.fit(obj,data,p,arm,lambda:rt.guard(True),event('live'))
            v=L.fit(shadow,data,p,arm,lambda:rt.guard(True),event('shadow'))
            assert u==v and digest(L.state(obj))==digest(L.state(shadow))
            m=L.export(obj,p,OUT/(arm+'.pt.gz'));arms[arm]=dict(update=u,proposal=m,initial_joint_hash=digest(before))
            objects[arm]=obj;initials[arm]=before
        save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time(),stop=p['absolute_learning_stop']))
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for arm in p['arms']:
            a=arms[arm];keyjobs=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']]
            others=[j for j in p['evaluation_jobs'] if j['id'] not in p['key_replay_ids']]
            assert len(keyjobs)==4 and len(others)==33
            vv=rt.request(keyjobs,a['proposal'],'PURE_STUDENT_'+arm,full=True)
            keys=[r for x,r in vv]
            vv+=rt.request(others,a['proposal'],'PURE_STUDENT_'+arm)
            for x,r in vv:
                F.M.audit_full_history(p,r,x)
                Base.append('progress.jsonl',progress(r,x))
            byid={r['job_id']:r for x,r in vv};rr=[byid[j['id']] for j in p['evaluation_jobs']]
            rt.request(keyjobs,a['proposal'],'EXACT_KEY4_'+arm,full=True,reference=keys)
            a['check']=assess(p,rr)
            obj=objects[arm];assert digest(L.state(obj))==a['proposal']['joint_hash']
            if not a['check']['accepted']:L.load(obj,initials[arm])
            a['retained_joint_hash']=digest(L.state(obj));a['model']=a['proposal'] if a['check']['accepted'] else p['initial']
        result=dict(arms=arms,new_RL=0,new_Adam=0,live_SGD=768,shadow_SGD=768,
            no_global_model_promotion=True,
            supervised_meta_transfer_NOT_meta_RL=True)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,error=error,result=result,partial_arms=arms if state!='COMPLETE' else None))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists()
    s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);rep=G.rows(OUT,'replays.jsonl');data=L.dataset(p)
    assert len(rows)==111 and len(rep)==12
    for r in rows:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256']
        assert int(a['native_counts'].sum())==r['physics']
        assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r['full']:F.M.audit_full_history(p,r,a)
    by={r['path']:r for r in rows}
    for r in rep:
        assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes']
        assert r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    events=[]
    for arm in p['arms']:
        obj=L.make(p);before=L.state(obj);u=L.fit(obj,data,p,arm,lambda:None,lambda t,k:events.append([arm,'live',t,k]))
        a=s['result']['arms'][arm];assert u==a['update'] and digest(L.state(obj))==a['proposal']['joint_hash']
        assert sha(a['proposal']['path'])==a['proposal']['sha256']
        byid={r['job_id']:r for r in rows if r['role']=='PURE_STUDENT_'+arm}
        rr=[byid[j['id']] for j in p['evaluation_jobs']];assert assess(p,rr)==a['check']
        if not a['check']['accepted']:L.load(obj,before)
        assert digest(L.state(obj))==a['retained_joint_hash']
    recorded=G.rows(OUT,'optimizer_calls.jsonl')
    assert [x for x in recorded if x[1]=='live']==events
    assert [[a,'live',t,k] for a,w,t,k in recorded if w=='shadow']==events
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics']
    assert s['commands']==127<=p['max_commands']
    assert read(OUT/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
    metrics=G.rows(OUT,'progress.jsonl');assert len(metrics)==111
    md={(x['policy_hash'],x['job_id']):x for x in metrics}
    for r in rows:
        if r['full']:assert progress(r,E.N.S.arrays(r['path']))==md[(r['policy_hash'],r['job_id'])]
    verify(p,True);save(OUT/'audit.json',dict(state='PASS',new_native=0,new_Adam=0,new_RL=0,
        discarded_historical_SGD=768,all_source_data_model_SHA_verified=True,
        all_live_shadow_and_SGD_meta_rollback_rebuilt=True,all123_native_commands_accounted=True,
        key12_full_history_and_native_full_array_integrations_exact=True,compact99_not_full_pose_offline_reconstruction=True,
        physics=s['physics'],commands=s['commands'],full407_not_executed=True))
    print('T580_AUDIT_PASS',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
