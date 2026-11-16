"""Three predeclared matched transfer updates followed by autonomous development."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback
import t581_learning as L

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t581_preserved_t550_rl';DOC=ROOT/'stage/docs/day_20261116/t581'
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
    assert len(rows)==len(p['evaluation_jobs'])==49
    assert [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']]
    ids={r['job_id'] for r in rows if r['success'] and not r['collision'] and r['field_detail']['field_qualified_success']}
    prior=set(p['initial_success_ids']);return dict(success=len(ids),valid=49,
        contact=sum(r['collision'] for r in rows),timeout=sum(r['timeout'] for r in rows),
        success_ids=sorted(ids),lost_initial=sorted(prior-ids),gained=sorted(ids-prior),
        accepted=prior<=ids and len(ids)>len(prior),full407=None,fresh16=None,reload78=None,
        finite_development_NOT_global_retention=True)


def progress(r,a):
    dt=a['native_counts'].astype(float)*.002
    ds=np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1)
    return dict(role=r['role'],job_id=r['job_id'],policy_hash=r['policy_hash'],
        success=r['success'],remaining_m=r['remaining_m'],seconds=float(dt.sum()),
        mean_endpoint_speed_m_s=float(ds.sum()/dt.sum()),full_pose_saved=r['full'],array_digest=r['array_digest'])


def check_history(p,vals):
    for a,r in vals:
        F.M.audit_full_history(p,r,a);Base.append('progress.jsonl',progress(r,a))


def main():
    p=read(OUT/'protocol.json');setup();verify(p,True);E.N.offline();E.L.install_stop_handlers()
    rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
    try:
        obj=L.make(p);shadow=L.make(p);before=L.state(obj);data=L.references(p)
        assert all(float((L.loc(obj[0],g['f'],g['base'])-g['target']).abs().max())==0 for g in data)
        behavior=L.export(obj,p,OUT/'initial.pt.gz')
        assert L.frozen(p,behavior['path']).phash()==obj[0].phash()
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vals=rt.request(p['baseline_jobs'],behavior,'ZERO_RESIDUAL_T550_NATIVE_EQUIVALENCE',full=True)
        check_history(p,vals)
        for a,r in vals:
            old=next(v for v in p['historical_baseline'] if v['job_id']==r['job_id'])
            assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
        save(OUT/'INITIAL_FUNCTION_NATIVE_PASS.json',dict(two_exact_T579_histories=True,new_independent_samples=False))
        vals=rt.request([p['target_job']]*4,behavior,'FRESH_T550_RESIDUAL_POLICY_TRAIN',seeds=p['training_seeds'],full=True)
        check_history(p,vals);training=[r for a,r in vals]
        rt.request([p['target_job']]*4,behavior,'EXACT_FRESH_TRAIN',seeds=p['training_seeds'],full=True,reference=training)
        arrays=[E.N.S.arrays(r['path']) for r in training]
        rt.role='ONE_CURRENT_POLICY_RL_UPDATE'
        event=lambda w:(lambda t,k:Base.append('optimizer_calls.jsonl',[w,t,k]))
        u=L.fit(obj,arrays,data,p,lambda:rt.guard(True),event('live'))
        v=L.fit(shadow,arrays,data,p,lambda:rt.guard(True),event('shadow'))
        assert u==v and digest(L.state(obj))==digest(L.state(shadow))
        proposal=L.export(obj,p,OUT/'proposal.pt.gz')
        assert L.frozen(p,proposal['path']).phash()==obj[0].phash()
        save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time(),stop=p['absolute_learning_stop']))
        keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']]
        others=[j for j in p['evaluation_jobs'] if j['id'] not in p['key_replay_ids']]
        vals=rt.request(keys,proposal,'PURE_STUDENT_ALL49',full=True);keyrows=[r for a,r in vals]
        vals+=rt.request(others,proposal,'PURE_STUDENT_ALL49');check_history(p,vals)
        by={r['job_id']:r for a,r in vals};rows=[by[j['id']] for j in p['evaluation_jobs']]
        rt.request(keys,proposal,'EXACT_KEY4',full=True,reference=keyrows)
        check=assess(p,rows)
        if not check['accepted']:
            for z in [obj,shadow]:L.load(z,before)
        result=dict(update=u,behavior=behavior,proposal=proposal,check=check,training=G.R.M.tally(training),
            retained_joint_hash=digest(L.state(obj)),model=proposal if check['accepted'] else p['base_model'],
            accepted=check['accepted'],new_Actor_Adam=u['Actor_Adam'],shadow_Actor_Adam=v['Actor_Adam'],
            teacher_queries=0,BC=0,Critic=0,full407=None,fresh16=None,reload78=None,globally_promoted=False)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:
            for z in [obj,shadow]:L.load(z,before)
        save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True,whole_ActorAdam_rollback=before is not None))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');setup();verify(p,True);G.stopped(OUT)
    assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==55 and len(reps)==8
    for r in rows:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics']
        assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r['full']:F.M.audit_full_history(p,r,a)
    by={r['path']:r for r in rows}
    for r in reps:
        assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes']
        assert r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    for r in rows[:2]:
        old=next(v for v in p['historical_baseline'] if v['job_id']==r['job_id'])
        assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    obj=L.make(p);before=L.state(obj);data=L.references(p);events=[]
    u=L.fit(obj,[E.N.S.arrays(r['path']) for r in rows[2:6]],data,p,lambda:None,lambda t,k:events.append(['live',t,k]))
    res=s['result'];assert u==res['update'] and digest(L.state(obj))==res['proposal']['joint_hash']
    assert sha(res['proposal']['path'])==res['proposal']['sha256']
    rr={r['job_id']:r for r in rows[6:]}
    assert assess(p,[rr[j['id']] for j in p['evaluation_jobs']])==res['check']
    if not res['accepted']:L.load(obj,before)
    assert digest(L.state(obj))==res['retained_joint_hash']
    recorded=G.rows(OUT,'optimizer_calls.jsonl');assert [r for r in recorded if r[0]=='live']==events
    assert [['live',t,k] for w,t,k in recorded if w=='shadow']==events
    assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
    assert s['commands']==67<=p['max_commands']
    assert read(OUT/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
    save(OUT/'audit.json',dict(state='PASS',new_native=0,discarded_historical_Adam=u['Actor_Adam'],
        frozen_T550_body_and_whole_ActorAdam_rollback_rebuilt=True,all_arrays_SHA_verified=True,
        full10_history_and_exact8_verified=True,compact45_not_full_pose_offline_reconstruction=True,
        initial_two_histories_exact_T579=True,physics=s['physics'],commands=s['commands'],full407=None))
    print('T581_AUDIT_PASS',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
