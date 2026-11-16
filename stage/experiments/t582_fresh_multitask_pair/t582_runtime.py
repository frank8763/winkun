"""Three predeclared matched transfer updates followed by autonomous development."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback
import t582_learning as L

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t582_fresh_multitask_pair';DOC=ROOT/'stage/docs/day_20261116/t582'
sys.path.insert(0,str(ROOT/'stage/experiments/t577_offline_recovery_night'))
import t577_runtime as R
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np
Base,F,G,E,P=R.Base,R.F,R.G,R.E,R.P


def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def verify(p,full=False):
    root=read(OUT/'protocol.json');assert root['authorization_approved'] is True;assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
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




sys.path.insert(0,str(ROOT/'stage/experiments/t581_preserved_t550_rl'))
import t581_runtime as Previous
assess,progress=Previous.assess,Previous.progress


def inspect(vals,p):
    for a,r in vals:
        F.M.audit_full_history(p,r,a);Base.append('progress.jsonl',progress(r,a))


def main():
    p=read(OUT/'protocol.json');setup();verify(p,True);E.N.offline();E.L.install_stop_handlers()
    rt=Base.Runtime(p);state='FAILED';error=None;result=None;arms={};objects={};initials={}
    try:
        data=L.references(p)
        for arm in p['arms']:
            objects[arm]=(L.make(p),L.make(p));initials[arm]=L.state(objects[arm][0])
        assert digest(initials[p['arms'][0]])==digest(initials[p['arms'][1]])
        behavior=L.export(objects[p['arms'][0]][0],p,OUT/'initial.pt.gz')
        assert L.frozen(p,behavior['path']).phash()==objects[p['arms'][0]][0][0].phash()
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        collected=[]
        for task,job in enumerate(p['training_jobs']):
            vals=rt.request([job]*4,behavior,'SHARED_FRESH_TASK_'+str(task),seeds=p['training_seeds'][task],full=True)
            inspect(vals,p);rows=[r for a,r in vals];collected+=rows
            rt.request([job]*4,behavior,'EXACT_SHARED_FRESH_TASK_'+str(task),seeds=p['training_seeds'][task],full=True,reference=rows)
        assert len(collected)==24
        arrays=[E.N.S.arrays(r['path']) for r in collected]
        for arm in p['arms']:
            obj,shadow=objects[arm];rt.role='LEARN_'+arm
            event=lambda w:(lambda t,k:Base.append('optimizer_calls.jsonl',[arm,w,t,k]))
            u=L.fit(obj,arrays,data,p,arm,lambda:rt.guard(True),event('live'))
            v=L.fit(shadow,arrays,data,p,arm,lambda:rt.guard(True),event('shadow'))
            assert u==v and digest(L.state(obj))==digest(L.state(shadow))
            model=L.export(obj,p,OUT/(arm+'.pt.gz'));assert L.frozen(p,model['path']).phash()==obj[0].phash()
            arms[arm]=dict(update=u,proposal=model)
        save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time(),stop=p['absolute_learning_stop']))
        keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']]
        others=[j for j in p['evaluation_jobs'] if j['id'] not in p['key_replay_ids']]
        for arm in p['arms']:
            model=arms[arm]['proposal'];vals=rt.request(keys,model,'PURE_STUDENT_'+arm,full=True);keyrows=[r for a,r in vals]
            vals+=rt.request(others,model,'PURE_STUDENT_'+arm);inspect(vals,p)
            by={r['job_id']:r for a,r in vals};rows=[by[j['id']] for j in p['evaluation_jobs']]
            rt.request(keys,model,'EXACT_KEY4_'+arm,full=True,reference=keyrows)
            check=assess(p,rows)
            if not check['accepted']:
                for z in objects[arm]:L.load(z,initials[arm])
            arms[arm].update(check=check,retained_joint_hash=digest(L.state(objects[arm][0])),
                model=model if check['accepted'] else p['base_model'])
        result=dict(arms=arms,training_tasks=[dict(job=p['training_jobs'][k]['id'],**G.R.M.tally(collected[4*k:4*k+4])) for k in range(6)],
            shared24_unique_not48_independent=True,training_native_exact24=True,new_BC=0,new_Critic=0,
            full407=None,fresh16=None,reload78=None,globally_promoted=False)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        for arm in objects:
            for z in objects[arm]:L.load(z,initials[arm])
        save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True,whole_ActorAdam_rollback=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,error=error,result=result,partial_arms=arms if state!='COMPLETE' else None))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');setup();verify(p,True);G.stopped(OUT)
    assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==122 and len(reps)==32
    for r in rows:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics']
        assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r['full']:F.M.audit_full_history(p,r,a)
    by={r['path']:r for r in rows}
    for r in reps:
        assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes']
        assert r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    arrays=[E.N.S.arrays(r['path']) for r in rows[:24]];data=L.references(p);events=[];hist=0
    for arm in p['arms']:
        obj=L.make(p);before=L.state(obj)
        u=L.fit(obj,arrays,data,p,arm,lambda:None,lambda t,k:events.append([arm,'live',t,k]));hist+=u['Actor_Adam']
        res=s['result']['arms'][arm]
        assert u==res['update'] and digest(L.state(obj))==res['proposal']['joint_hash'] and sha(res['proposal']['path'])==res['proposal']['sha256']
        rr={r['job_id']:r for r in rows[24:] if r['role']=='PURE_STUDENT_'+arm}
        assert assess(p,[rr[j['id']] for j in p['evaluation_jobs']])==res['check']
        if not res['check']['accepted']:L.load(obj,before)
        assert digest(L.state(obj))==res['retained_joint_hash']
    recorded=G.rows(OUT,'optimizer_calls.jsonl')
    assert [r for r in recorded if r[1]=='live']==events
    assert [[arm,'live',t,k] for arm,w,t,k in recorded if w=='shadow']==events
    assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
    assert s['commands']==158<=p['max_commands']
    assert read(OUT/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
    save(OUT/'audit.json',dict(state='PASS',new_native=0,discarded_historical_Adam=hist,
        all_ActorAdam_live_shadow_and_rollback_rebuilt=True,all_saved_arrays_SHA_verified=True,
        full32_history_and_exact32_verified=True,compact90_not_full_pose_offline_reconstruction=True,
        shared_training24_not48=True,physics=s['physics'],commands=s['commands'],full407=None))
    print('T582_AUDIT_PASS',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
