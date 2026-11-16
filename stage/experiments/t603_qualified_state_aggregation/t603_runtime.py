"""One42-history aggregation pilot; shared accounting and durable records."""
from pathlib import Path
import sys,time,fcntl,traceback,os
import t603_learning as L
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t603_qualified_state_aggregation';DOC=ROOT/'stage/docs/day_20261116/t603'
sys.path.insert(0,str(ROOT/'stage/experiments/t598_independent_recurrent_recovery'))
import t598_runtime as Legacy
Base,F,G,E,P,torch,np=Legacy.Base,Legacy.F,Legacy.G,Legacy.E,Legacy.P,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
# Resolve the actual owner, not an absent alias on the wrapper.
assess=Legacy.R.assess


def setup(folder):
    Legacy.OUT=OUT;Legacy.HERE=HERE;Legacy.DOC=DOC;Legacy.setup(folder)


def verify(p,full=False):return Legacy.verify(p,full)


def sentinel_jobs(p):return [j for j in p['evaluation_jobs'] if j['id'] in p['sentinel_ids']]


def stages(p,level):
    rec=[r['request']['job'] for r in p['recovery_rows']]
    result=[(rec,False),(rec,True)]
    if level>=1:result.append((sentinel_jobs(p),False))
    if level>=2:
        result.append(([j for j in p['evaluation_jobs'] if j['id'] not in p['sentinel_ids']],False))
        result.append(([j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']],True))
    return result


def schedule_probe(p):
    """Run real Runtime.guard/request budget paths, stop at virtual pool dispatch.

    No pool process is spawned. The exception happens before payload/file writes.
    Counts are worst-case *virtual* step reservations, not physical execution.
    """
    import copy,itertools
    class Dispatched(Exception):pass
    def one(levels,cap):
        pc=copy.deepcopy(p);pc['max_commands']=cap
        rt=Base.Runtime(pc);rt.last=time.time()
        class VirtualPool:
            def request(self,requests):
                rt.counts[0]+=sum(z['job']['case']['step_cap'] for z in requests)
                raise Dispatched()
        rt.pool=VirtualPool();calls=0
        try:
            assert rt.commands==2
            for level in levels:
                for jobs,replay in stages(pc,level):
                    for off in range(0,len(jobs),2):
                        batch=jobs[off:off+2];reference=[dict(path='unused') for _ in batch] if replay else None
                        try:rt.request(batch,pc['base_model'],'ZERO_NATIVE_VIRTUAL',reference=reference)
                        except Dispatched:calls+=len(batch)
            rt.commands+=2
            assert rt.commands==calls+4<=cap and sum(rt.counts)<=pc['max_physics']
            return dict(levels=list(levels),commands=rt.commands,payload_commands=calls,virtual_max_physics=sum(rt.counts))
        finally:fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    reports=[one(v,p['max_commands']) for v in itertools.product(range(3),repeat=1)]
    try:one((2,),p['max_commands']-1)
    except AssertionError:negative=True
    else:raise AssertionError('Counter regression test did not reject undersized budget')
    assert max(r['commands'] for r in reports)==63 and max(r['virtual_max_physics'] for r in reports)==p['max_physics']
    return dict(state='PASS_REAL_RUNTIME_COUNTER_PATH',all3_conditional_schedules=reports,
        full_path_62_commands_rejected=negative,actual_new_native=0,actual_worker_processes=0,
        init_pool_start_charged_once=2,pool_close_reserved=2)


def main():
    p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True)
    E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None
    results=[];prepared=[];result=dict(arms=results,globally_promoted=False,full407=None,fresh16=None,reload78=None)
    try:
        data=L.dataset(p)
        # Finish both learning arms before spending any native validation window.
        for method in p['methods']:
            pc=dict(p,method=method);obj=L.make(pc);shadow=L.make(pc);before=L.state(obj)
            rec=dict(method=method,update=None,proposal=None,recovery=None,sentinels=None,check=None,accepted=False,model=p['base_model'])
            results.append(rec);rt.role='LEARNING_'+method
            ev=lambda who:(lambda t,k:Base.append('optimizer_calls.jsonl',[method+'_'+who,t,k]))
            try:
                u=L.fit(obj,data,pc,lambda:rt.guard(True),ev('live'))
                v=L.fit(shadow,data,pc,lambda:rt.guard(True),ev('shadow'))
                assert u==v and digest(L.state(obj))==digest(L.state(shadow))
                model=L.export(obj,pc,OUT/('proposal_'+method+'.pt.gz'))
                assert L.frozen(pc,model['path']).phash()==obj[0].phash()
                rec.update(update=u,proposal=model);save(OUT/('phase_'+method+'.json'),rec)
                prepared.append((pc,obj,shadow,before,rec))
            except BaseException:L.load(obj,before);L.load(shadow,before);raise
        # Runtime.__init__ already charged the only2-worker pool. Do not add2.
        rt.pool=Legacy.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for pc,obj,shadow,before,rec in prepared:
            method=pc['method'];model=rec['proposal'];job=p['joint_recovery_row']['request']['job']
            hh=[]
            for k,row in enumerate(p['recovery_rows']):
                jj=row['request']['job'];vv=rt.request([jj],model,method+'_RECOVERY_'+str(k),full=(k==0))
                Legacy.inspect(p,vv);r=vv[0][1];hh.append(r)
                save(OUT/('after_recovery_'+method+'.json'),dict(rec,recovery_rows=hh))
                rt.request([jj],model,method+'_EXACT_RECOVERY_'+str(k),reference=[r])
            rr=hh[0];rec['recovery']=rr;rec['recovery_rows']=hh
            save(OUT/('after_recovery_'+method+'.json'),rec)
            if Legacy.R.safe(rr):
                vv=rt.request(sentinel_jobs(p),model,method+'_SENTINELS15');Legacy.inspect(p,vv);sent=[r for a,r in vv]
                rec['sentinels']=sent;save(OUT/('after_sentinels_'+method+'.json'),rec)
                if all(Legacy.R.safe(r) for r in sent):
                    rest=[j for j in p['evaluation_jobs'] if j['id'] not in p['sentinel_ids']]
                    vv=rt.request(rest,model,method+'_REMAINING34');Legacy.inspect(p,vv);by={r['job_id']:r for r in sent+[r for a,r in vv]}
                    ordered=[by[j['id']] for j in p['evaluation_jobs']];rec['check']=assess(p,ordered)
                    save(OUT/('after49_'+method+'.json'),rec)
                    keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']]
                    rt.request(keys,model,method+'_EXACT_KEY4',reference=[by[j['id']] for j in keys])
            rec['accepted']=rec['check'] is not None and not rec['check']['lost_initial'] and Legacy.R.safe(rr)
            if not rec['accepted']:L.load(obj,before);L.load(shadow,before)
            rec.update(model=model if rec['accepted'] else p['base_model'],retained_joint_hash=digest(L.state(obj)))
            save(OUT/('final_'+method+'.json'),rec)
        eligible=[r for r in results if r['accepted']]
        if eligible:
            chosen=max(eligible,key=lambda r:(sum(i.startswith('t502_') for i in r['check']['success_ids']),r['check']['success'],-p['methods'].index(r['method'])))
            result['selected_research_model']=chosen['proposal']
        else:result['selected_research_model']=p['base_model']
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        for pc,obj,shadow,before,rec in prepared:L.load(obj,before);L.load(shadow,before)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);G.stopped(OUT)
    assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');by={r['path']:r for r in rows}
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert int(a['native_counts'].sum())==r['physics'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r['full']:Legacy.inspect(p,[(a,r)])
    for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    data=L.dataset(p);calls=G.rows(OUT,'optimizer_calls.jsonl');steps=0
    for rec in s['result']['arms']:
        method=rec['method'];pc=dict(p,method=method);obj=L.make(pc);before=L.state(obj);events=[]
        u=L.fit(obj,data,pc,lambda:None,lambda t,k:events.append([method+'_live',t,k]));steps+=u['steps']
        assert u==rec['update'] and digest(L.state(obj))==rec['proposal']['joint_hash']
        assert sha(rec['proposal']['path'])==rec['proposal']['sha256']
        assert [e for e in calls if e[0]==method+'_live']==events
        assert [[method+'_live',t,k] for w,t,k in calls if w==method+'_shadow']==events
        rr=[r for r in rows if r['role']==method+'_RECOVERY_0'];assert rr==[rec['recovery']]
        assert [r for r in rows if r['role'].startswith(method+'_RECOVERY_')]==rec['recovery_rows']
        assert len([r for r in reps if r['role'].startswith(method+'_EXACT_RECOVERY_')])==3
        original=[r for r in rows if r['role'] in [method+'_SENTINELS15',method+'_REMAINING34']]
        if rec['check'] is not None:
            ordered=[next(r for r in original if r['job_id']==j['id']) for j in p['evaluation_jobs']]
            assert assess(p,ordered)==rec['check'] and len(original)==49
            assert len([r for r in reps if r['role']==method+'_EXACT_KEY4'])==4
        else:assert len(original) in [0,15]
        assert rec['accepted']==(rec['check'] is not None and not rec['check']['lost_initial'] and Legacy.R.safe(rr[0]))
        if not rec['accepted']:L.load(obj,before)
        assert digest(L.state(obj))==rec['retained_joint_hash']
    assert steps==128 and s['commands']==len(rows)+len(reps)+4<=p['max_commands']
    assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
    save(OUT/'audit.json',dict(state='PASS',historical_discarded_Adam=steps,new_native=0,all_raw_payload_SHA=True,
        full_recovery_memories_and_histories=True,all_native_replays_exact=True,whole_ActorAdam_reconstruction_and_rollback_exact=True,
        actual_single_pool_command_accounting=True,physics=s['physics'],commands=s['commands']))


if __name__=='__main__':globals()[sys.argv[1]]()
