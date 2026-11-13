"""Native student-state recovery qualification followed by conditional rehearsal."""
from pathlib import Path
import copy,fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t525_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t525_student_state_recovery'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
BASE_PAYLOAD=E.payload
class RecoverySession(F.FieldSession):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs);self.teacher=F.L.frozen(self.q.p,self.q.p['neural_teacher']['path']);self.teacher.sample=False;self.teacher_hidden=None;self.teacher_labels=[];self.holds=int(job['teacher_holds'])
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0;state=self.packet().copy();before=E.N.K.R.integration(self.env).copy()
        with torch.no_grad():command,_,self.teacher_hidden,_=self.teacher.step(torch.tensor(state[None]),self.teacher_hidden)
        assert np.array_equal(before,E.N.K.R.integration(self.env));self.teacher_labels.append(command[0].numpy().copy());k=len(self.rows);first=self.q.p['teacher_first']
        use=k>=first and (self.holds<0 or k<first+self.holds)
        return super().step(command[0].numpy() if use else None,owner=int(use),actor_active=True)
def payload(session,tape=None):
    a=BASE_PAYLOAD(session,tape);a['teacher_actions']=np.asarray(session.teacher_labels[:len(a['states'])],np.float32);a['teacher_holds']=np.asarray(session.holds,np.int64);return a

def worker(pipe,index,p,counts):
    G.T.Session=RecoverySession;G.B.native_setup=F.field_setup;P.frozen=F.L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';old=read(f) if f.exists() else [];old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,old)
        except BaseException:self.close();raise
    def receive(self,i):
        v=super().receive(i)
        if isinstance(v,tuple):v[1].update(teacher=True,teacher_first=self.p['teacher_first'],teacher_model=self.p['neural_teacher'],training_mixture_NOT_autonomous=True)
        return v

def check_prefix(p,a):
    b=E.N.S.arrays(p['baseline_row']['path']);k=p['teacher_first'];assert len(a['states'])>k
    for key in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:
        assert np.array_equal(a[key][:k],b[key][:k]),key
    for key in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],b[key][k]),key
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration']) and a['native_counts'][:k].sum()==p['prefix_physics_steps']
    owner=np.zeros(len(a['states']),np.uint8);holds=int(a['teacher_holds']);owner[k:None if holds<0 else k+holds]=1
    assert np.array_equal(owner,a['ownership']) and not a['stochastic'].any()

def audit_recovery(p,row,a):
    check_prefix(p,a);teacher=F.L.frozen(p,p['neural_teacher']['path']);teacher.sample=False;th=None;q=F.L.frozen(p,p['initial']['path']);q.sample=False;h=None;own=[]
    with torch.no_grad():
        for i,x in enumerate(a['states']):
            u,_,h,_=q.step(torch.tensor(x[None]),h);v,_,th,_=teacher.step(torch.tensor(x[None]),th);own.append(u[0].numpy().copy())
            assert np.array_equal(v[0].numpy(),a['teacher_actions'][i])
            assert np.array_equal(a['actions'][i],a['teacher_actions'][i] if a['ownership'][i] else own[-1])
    # Audit the student's internal causal policy tape on the actual observed
    # mixed trajectory. The ORIGINAL executed actions are separately verified.
    b=dict(a,actions=np.asarray(own),ownership=np.zeros(len(own),np.uint8));P.frozen=F.L.frozen;M.audit_full_history(p,dict(row,teacher=False),b)
    return F.detail(p,row,a)

def evaluation(rt,p,model,reference=None):
    rows=[];rt.role='EXACT_FULL32' if reference else 'AUTONOMOUS_FULL32'
    for off in range(0,32,2):
        jobs=p['evaluation_jobs'][off:off+2];rt.guard();assert sum(rt.counts)+sum(j['case']['step_cap'] for j in jobs)<=p['max_physics'] and rt.commands+4<=p['max_commands'];rt.commands+=2
        req=[dict(model=model['path'],job=j,seed=None,query=False,full_path=None) for j in jobs]
        for i,((a,r),j,request) in enumerate(zip(rt.pool.request(req),jobs,req)):
            r.update(job_id=j['id'],role=rt.role,seed=None,request=request,loaded_model_sha256=sha(model['path']),full=False,full_arrays_generated_but_not_saved=True);assert r['policy_hash']==model['policy_hash'];r['field_detail']=F.detail(p,r,a) if j['id'].startswith('t502_') else None
            if reference:
                old=reference[off+i];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail'];r.update(exact=True,reference_path=old['path'],independent_new_sample=False);rt.replays.append(r);name='replays.jsonl'
            else:
                f=Path(p['out'])/'episodes'/('eval_'+str(off+i)+'.npz');f.parent.mkdir(exist_ok=True);E.N.write_npz(f,{k:a[k] for k in ['actions','native_counts','end_flags']},p['compact_bytes']);r.update(path=str(f),sha256=sha(f));rt.rows.append(r);name='cases.jsonl'
            with (Path(p['out'])/name).open('a') as f:f.write(json.dumps(r)+'\n')
            rows.append(r)
    return rows

def assess(p,rows):
    old=G.R.M.tally(rows[:8]);field=G.R.M.tally(rows[8:]);ids={r['job_id'] for r in rows[8:] if r['field_detail']['field_qualified_success']};before=set(p['initial_field_success_ids'])
    return dict(old8=old,field24=field,field_qualified_goals=len(ids),field_success_ids=sorted(ids),gained_ids=sorted(ids-before),lost_ids=sorted(before-ids),accepted=old['success']==8 and before<=ids and len(ids)>11 and field['contact']<=9 and field['invalid']==field['censored']==old['invalid']==old['censored']==0)

def main(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);P.frozen=F.L.frozen;E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        fulljob=dict(p['probe_job'],teacher_holds=-1);vv=rt.request([fulljob],p['initial'],'VALIDATE_NEURAL_TEACHER_FROM_STUDENT_STATE',full=True);a,row=vv[0];check_prefix(p,a);detail=F.detail(p,row,a)
        rt.request([fulljob],p['initial'],'EXACT_NEURAL_TEACHER',full=True,references=[row])
        teacher_ok=detail['field_qualified_success'];handoff=None;eligible=False
        if teacher_ok:
            job=dict(p['probe_job'],teacher_holds=20);values=rt.request([job],p['initial'],'FOUR_SECOND_CORRECTION_THEN_AUTONOMOUS',full=True);b,rr=values[0];check_prefix(p,b);rt.request([job],p['initial'],'EXACT_HANDOFF',full=True,references=[rr]);handoff=dict(tally=G.R.M.tally([rr]),autonomous_decisions_after_return=max(0,len(b['states'])-p['teacher_first']-20),detail=F.detail(p,rr,b))
        eligible=bool(teacher_ok and handoff and handoff['detail']['field_qualified_success'] and handoff['autonomous_decisions_after_return']>0)
        result=dict(recovery_teacher_qualified=teacher_ok,learning_eligible=eligible,teacher_detail=detail,handoff=handoff,Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,old8=None,field24=None,accepted=False,new_fullfield_autonomous_success_NOT_established=True,full_old275_six54_eight54_NOT_executed=True,fresh16=None,reload78=None)
        rt.pool.close();rt.pool=None;rt.commands+=2
        if eligible:
            obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);aa=L.fit(obj,a,p,refs,lambda:rt.guard(True),F.R.emit('live'));bb=L.fit(shadow,a,p,refs,lambda:rt.guard(True),F.R.emit('shadow'));assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
            model=L.A.export(obj,folder/'proposal.pt.gz',p['model_bytes']);rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
            rows=evaluation(rt,p,model);evaluation(rt,p,model,rows);result.update(assess(p,rows),model=model,update=aa,Actor_Adam=aa['Actor_Adam'])
            if not result['accepted']:
                for o in [obj,shadow]:P.load(o,before)
            result['final_joint_hash']=digest(P.state(*obj))
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:
            for z in [obj,shadow]:P.load(z,before)
        save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder);rows=G.rows(folder);rep=G.rows(folder,'replays.jsonl');result=s['result'];teachers=[r for r in rows if r['full']]
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        if r['full']:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());audit_recovery(p,r,a)
        else:
            assert set(a)=={'actions','native_counts','end_flags'} and int(a['native_counts'].sum())==r['physics']
            for k,v in a.items():assert digest(v)==r['array_hashes'][k]
    by={r['path']:r for r in rows}
    for r in rep:assert r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256'] and r['exact']
    a=E.N.S.arrays(teachers[0]['path']);teacher_ok=F.detail(p,teachers[0],a)['field_qualified_success'];assert teacher_ok==result['recovery_teacher_qualified'];eligible=bool(teacher_ok and len(teachers)==2 and F.detail(p,teachers[1],E.N.S.arrays(teachers[1]['path']))['field_qualified_success'] and len(E.N.S.arrays(teachers[1]['path'])['states'])>p['teacher_first']+20);assert eligible==result['learning_eligible'];steps=0
    if eligible:
        assert len(teachers)==2;obj=L.make(p);before=P.state(*obj);fit=L.fit(obj,a,p,L.references(p),lambda:None,lambda *_:None);steps=fit['Actor_Adam'];assert fit==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'];assert sha(result['model']['path'])==result['model']['sha256']
        erows=[r for r in rows if not r['full']];assert len(erows)==32
        for k,v in assess(p,erows).items():assert result[k]==v
        if not result['accepted']:P.load(obj,before)
        assert digest(P.state(*obj))==result['final_joint_hash']
    assert len(rows)==len(rep)==(34 if eligible else 2 if teacher_ok else 1)
    if teacher_ok:
        hand=E.N.S.arrays(teachers[1]['path']);hd=F.detail(p,teachers[1],hand);assert hd==result['handoff']['detail'];assert G.R.M.tally([teachers[1]])==result['handoff']['tally']
    assert F.detail(p,teachers[0],a)==result['teacher_detail']
    expected=[[w,st,k] for w in ['live','shadow'] for k in range(steps) for st in ['BEFORE','AFTER']];assert G.rows(folder,'optimizer_calls.jsonl')==expected
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==len(rows)+len(rep)+(8 if eligible else 4)
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,historical_discarded_Actor_Adam=steps,Value_Adam=0,all_student_and_teacher_causal_histories_rebuilt=True,actual_student_prefix_clock_and_integrations_preserved=True,full_ActorAdam_rebuilt_and_rollback_exact=True,compact_evaluation_NOT_full120_reconstructed_offline=True,all_native_replay_full_hashes_and_integrations_match=True,physics=s['physics'],commands=s['commands']));print('T525_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
