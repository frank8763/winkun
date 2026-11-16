"""One supervised recovery pilot, conditional49, exact replay and rollback."""
from pathlib import Path
import sys,time,fcntl,traceback,os
import t598_learning as L
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t598_independent_recurrent_recovery';DOC=ROOT/'stage/docs/day_20261116/t598'
sys.path.insert(0,str(ROOT/'stage/experiments/t591_joint_representation_pair'))
import t591_runtime as R
Base,F,G,E,P,torch,np=R.Base,R.F,R.G,R.E,R.P,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest


def verify(p,full=False):return R.verify(p,full)
def setup(folder):
    R.OUT=OUT;R.HERE=HERE;R.DOC=DOC;R.L.frozen=L.frozen
    R.setup(folder);Base.verify=verify;P.frozen=L.frozen


def payload(s,tape=None):
    a=R.payload(s,tape)
    t=s.q.tape if tape is None else tape
    assert len(t)==len(a['states'])
    for k in ['extra_hidden_before','anchor_hidden_before']:
        a[k]=np.asarray([z[k] for z in t])
    a['last_extra_hidden_after']=s.q.extra_hidden[0,0].detach().numpy().copy()
    a['last_anchor_hidden_after']=s.q.anchor_hidden[0,0].detach().numpy().copy()
    return a


def worker(pipe,index,p,counts):
    G.T.Session=R.Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload
    G.cpu_worker(pipe,index,p,counts)


class Pool(R.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
            save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise


def inspect(p,vals):
    R.inspect(p,vals)
    for a,r in vals:
        q=L.frozen(p,r['request']['model']);q.sample=False;q.reset_noise();h=None
        with torch.no_grad():
            for i,x in enumerate(a['states']):
                command,loc,h,_=q.step(torch.tensor(x[None]),h)
                for k in ['extra_hidden_before','anchor_hidden_before']:
                    assert np.array_equal(q.tape[-1][k],a[k][i])
        assert np.array_equal(q.extra_hidden[0,0].numpy(),a['last_extra_hidden_after'])
        assert np.array_equal(q.anchor_hidden[0,0].numpy(),a['last_anchor_hidden_after'])


def main():
    p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True)
    E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
    try:
        data=L.dataset(p);obj=L.make(p);shadow=L.make(p);before=L.state(obj)
        rt.role='INDEPENDENT_RECURRENT_RECOVERY_LEARNING'
        event=lambda w:(lambda t,k:Base.append('optimizer_calls.jsonl',[w,t,k]))
        u=L.fit(obj,data,p,lambda:rt.guard(True),event('live'))
        v=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'))
        assert u==v and digest(L.state(obj))==digest(L.state(shadow))
        model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash()
        save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time(),steps=u['steps']))
        rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['handoff_jobs'],model,'AUTONOMOUS_RECOVERY3',full=True);inspect(p,vv);hh=[r for a,r in vv]
        rt.request(p['handoff_jobs'],model,'EXACT_RECOVERY3_COMPACT',reference=hh)
        check=None;rows=None
        if any(R.safe(r) for r in hh):
            vv=rt.request(p['evaluation_jobs'],model,'AUTONOMOUS_ORIGINAL49');inspect(p,vv);rows=[r for a,r in vv];check=R.assess(p,rows)
            keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];by={r['job_id']:r for r in rows}
            rt.request(keys,model,'EXACT_KEY4_COMPACT',reference=[by[j['id']] for j in keys])
        accepted=check is not None and not check['lost_initial'] and any(R.safe(r) for r in hh)
        result=dict(update=u,proposal=model,handoff=hh,check=check,evaluation_rows=rows,accepted=accepted,
            recovery_stage_only=True,new_RL=0,new_Critic=0,model=model if accepted else p['base_model'],
            full407=None,fresh16=None,reload78=None,globally_promoted=False)
        if not accepted:L.load(obj,before);L.load(shadow,before)
        result['retained_joint_hash']=digest(L.state(obj));state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:L.load(obj,before);L.load(shadow,before)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True,rollback=before is not None))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);G.stopped(OUT)
    assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';r=s['result']
    rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert rows[:3]==r['handoff']
    for x in rows:
        assert sha(x['path'])==x['sha256'];a=E.N.S.arrays(x['path'])
        assert int(a['native_counts'].sum())==x['physics'] and all(digest(v)==x['array_hashes'][k] for k,v in a.items())
        if x['full']:inspect(p,[(a,x)])
    by={x['path']:x for x in rows}
    for x in reps:
        assert x['exact'] and x['array_hashes']==by[x['reference_path']]['array_hashes'] and x['integration_sha256']==by[x['reference_path']]['integration_sha256']
    expected=r['check'] is not None
    assert len(rows)==(52 if expected else 3) and len(reps)==(7 if expected else 3)
    if expected:assert R.assess(p,rows[3:])==r['check']
    obj=L.make(p);before=L.state(obj);events=[]
    u=L.fit(obj,L.dataset(p),p,lambda:None,lambda t,k:events.append(['live',t,k]))
    assert u==r['update'] and digest(L.state(obj))==r['proposal']['joint_hash'] and sha(r['proposal']['path'])==r['proposal']['sha256']
    calls=G.rows(OUT,'optimizer_calls.jsonl')
    assert [e for e in calls if e[0]=='live']==events and [['live',t,k] for w,t,k in calls if w=='shadow']==events
    assert r['accepted']==(expected and not r['check']['lost_initial'] and any(R.safe(x) for x in r['handoff']))
    if not r['accepted']:L.load(obj,before)
    assert digest(L.state(obj))==r['retained_joint_hash']
    assert s['physics']==sum(x['physics'] for x in rows+reps)<=p['max_physics'] and s['commands']==len(rows)+len(reps)+4<=p['max_commands']
    save(OUT/'audit.json',dict(state='PASS',new_native=0,discarded_historical_Adam=u['steps'],
        whole_ActorAdam_exact=True,all_saved_arrays_SHA=True,all_replays_exact=True,
        independent_branch_and_reference_memories_checked=True,physics=s['physics'],commands=s['commands']))


if __name__=='__main__':globals()[sys.argv[1]]()
