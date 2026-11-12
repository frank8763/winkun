"""Native teacher validation, then independently autonomous neural student."""
from pathlib import Path
import fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import speed_actor as S
G,P,E,M,torch,np=S.G,S.P,S.G.E,S.G.M,S.torch,S.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t496_neural_speed_profile';ACTIVE=OUT
BASE_SESSION=G.T.Session;BASE_PAYLOAD=E.payload

def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30 and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
    for r in p.get('teacher_rows',[]):assert sha(r['path'])==r['sha256']

def setup(folder):
    global ACTIVE
    ACTIVE=Path(folder);G.B.OUT,G.B.ACTIVE,G.B.verify=OUT,ACTIVE,verify;M.OUT,M.verify=ACTIVE,verify;M.setup_runtime()

class TeacherSession(BASE_SESSION):
    def step(self,*args,**kwargs):
        kwargs['owner']=1
        return super().step(*args,**kwargs)

def pulse_frozen(p,path):
    q=S.PLAIN_FROZEN(p,path);original=q.step
    def step(x,hidden=None):
        index=len(q.tape);action,loc,h,extra=original(x,hidden)
        action=action.clone()
        if index<p['pulse_decisions']:action[:,0]=(action[:,0]+action.new_tensor(p['raw_sigma'][0])).clamp(-1,1)
        return action,loc,h,extra
    q.step=step;return q

def teacher_payload(session):
    a=BASE_PAYLOAD(session);a['teacher_actions']=a['actions'].copy();return a

def worker(pipe,index,p,counts):
    if p['phase']=='teacher':
        G.T.Session=TeacherSession;P.frozen=pulse_frozen;E.payload=teacher_payload
    else:P.frozen=S.frozen
    G.cpu_worker(pipe,index,p,counts)

class Pool(G.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';old=read(f) if f.exists() else [];old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False));save(f,old)
        except BaseException:self.close();raise
    def receive(self,i):
        value=super().receive(i)
        if self.p['phase']=='teacher' and isinstance(value,tuple):
            a,r=value;r.update(teacher=True,teacher_first=0,teacher_policy_hash=digest(dict(base=r['policy_hash'],pulse_decisions=self.p['pulse_decisions'],drive_increment=self.p['raw_sigma'][0])),not_student_navigation=True)
        return value

def child_plan(folder):
    child=read(Path(folder)/'protocol.json');assert child.pop('_root_sha256')==sha(OUT/'protocol.json');p=read(OUT/'protocol.json');p.update(child);return p

def emit(which):
    def event(stage,k):
        with (ACTIVE/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([which,stage,k])+'\n')
    return event

def main(phase):
    folder=OUT/phase;p=child_plan(folder);setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        if phase=='teacher':
            rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
            first=rt.request(p['teacher_jobs'][:2],p['initial'],'PULSE_SENTINELS',full=True)
            eligible=all(r['success'] and not r['collision'] for a,r in first)
            rows=[r for a,r in first]
            if eligible:
                other=rt.request(p['teacher_jobs'][2:],p['initial'],'VERIFIED_RIGHT_PULSE',full=True);rows+=[r for a,r in other]
                rt.request(p['teacher_jobs'],p['initial'],'EXACT_RIGHT_PULSE',full=True,references=rows)
                eligible=all(r['success'] and not r['collision'] for r in rows)
            result=dict(eligible_for_student=eligible,teacher_only=True,student_navigation=None,right_teacher=G.R.M.tally(rows),rows=rows,not_run=27-len(rows),Actor_Adam=0,Value_Adam=0)
        else:
            assert read(OUT/'teacher/audit.json')['state']=='PASS' and read(OUT/'teacher/summary.json')['result']['eligible_for_student']
            P.frozen=S.frozen;obj=S.make(p);shadow=S.make(p);before=P.state(*obj)
            data=S.groups(p,p['teacher_rows']);save(folder/'training_data_manifest.json',[dict(id=g['id'],source_sha256=g['source_sha256'],features_sha256=digest(g['features']),target_sha256=digest(g['target']),all_actual_GOAL=True) for g in data])
            rt.role='LEARNED_NEURAL_DRIVE_FIT'
            aa=S.fit(obj,data,p,lambda:rt.guard(True),emit('live'));bb=S.fit(shadow,data,p,lambda:rt.guard(True),emit('shadow'))
            assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
            model=S.export(obj,folder/'proposal.pt.gz',p['model_bytes'])
            rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
            vv=rt.request(p['wide8_jobs'],model,'AUTONOMOUS_STUDENT_FULL54',full=True);rows=[r for a,r in vv]
            count=G.R.M.tally(rows);ids={r['job_id'] for r in rows if r['success'] and not r['collision']};initial=set(p['initial_success_ids'])
            accepted=initial<=ids and count['success']>40 and count['contact']==count['invalid']==count['censored']==0
            allpass=count['success']==54
            if allpass:
                rt.pool.close();rt.commands+=2;dest=folder/'independent_model.pt.gz';shutil.copyfile(model['path'],dest);assert sha(dest)==model['sha256']
                rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
                rt.request(p['wide8_jobs'],dict(model,path=str(dest)),'INDEPENDENT_FULL54_RELOAD',full=True,references=rows)
            else:
                by={r['job_id']:r for r in rows};jobs=[j for j in p['wide8_jobs'] if j['id'] in p['original_failure_ids']]
                rt.request(jobs,model,'EXACT_ORIGINAL14',full=True,references=[by[j['id']] for j in jobs])
            if not accepted:
                P.load(obj,before);P.load(shadow,before);assert digest(P.state(*obj))==digest(before)==digest(P.state(*shadow))
            result=dict(model=model,update=aa,accepted=accepted,wide8=count,sides={s:sum(s in x for x in ids) for s in ['LEFT','RIGHT']},gained_ids=sorted(ids-initial),lost_ids=sorted(initial-ids),success_ids=sorted(ids),targeted14_goals=sum(x in ids for x in p['original_failure_ids']),independent54_exact=allpass,eight_development_allpass=allpass,rollback_joint_hash=digest(P.state(*obj)),fresh16=None,old275=None,wide6=None,under8_gate=False,new_policy_gradient_steps=0)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:
            for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
        save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),phase=phase,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=child_plan(folder);setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    rr,rep=G.rows(folder),G.rows(folder,'replays.jsonl');P.frozen=S.frozen
    for r in rr:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
        if phase=='teacher':
            expected=torch.tensor(a['raw_actions']).clamp(-1,1).float();expected[:p['pulse_decisions'],0]=(expected[:p['pulse_decisions'],0]+expected.new_tensor(p['raw_sigma'][0])).clamp(-1,1)
            assert np.array_equal(expected.numpy(),a['actions']) and a['ownership'].all() and not a['stochastic'].any()
    by={r['path']:r for r in rr}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    rebuilt=0
    if phase=='teacher':
        assert len(rr) in [2,27] and len(rep)==(27 if len(rr)==27 else 0)
        eligible=len(rr)==27 and all(r['success'] and not r['collision'] for r in rr)
        assert eligible==s['result']['eligible_for_student']
    else:
        assert len(rr)==54 and len(rep)==(54 if s['result']['eight_development_allpass'] else 14)
        obj=S.make(p);before=P.state(*obj);data=S.groups(p,p['teacher_rows']);update=S.fit(obj,data,p,lambda:None,lambda *_:None)
        assert update==s['result']['update'] and digest(P.state(*obj))==s['result']['model']['joint_hash'];rebuilt=update['Actor_Adam']
        assert sha(s['result']['model']['path'])==s['result']['model']['sha256']
        calls=G.rows(folder,'optimizer_calls.jsonl')
        assert calls==[[which,stage,k] for which in ['live','shadow'] for k in range(rebuilt) for stage in ['BEFORE','AFTER']]
        ids={r['job_id'] for r in rr if r['success'] and not r['collision']};count=G.R.M.tally(rr)
        accepted=set(p['initial_success_ids'])<=ids and count['success']>40 and count['contact']==count['invalid']==count['censored']==0
        assert accepted==s['result']['accepted'] and count==s['result']['wide8'] and ids==set(s['result']['success_ids'])
        if not accepted:P.load(obj,before)
        assert digest(P.state(*obj))==s['result']['rollback_joint_hash']
    assert s['physics']==sum(r['physics'] for r in rr+rep)
    assert s['commands']==len(rr)+len(rep)+4+(4 if phase=='student' and s['result']['eight_development_allpass'] else 0)
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=rebuilt,all_full_history_density_rewards_and_physics_exact=True,teacher_controls_verified=phase=='teacher',physics=s['physics'],commands=s['commands']))
    print('T496_'+phase+'_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
