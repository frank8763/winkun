"""One child candidate, complete recorded failure screen then conditional407."""
from pathlib import Path
import copy,fcntl,gzip,io,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
ROOTOUT=ROOT/'stage/runs/t693_measure_valued_actor';ROOTDOC=ROOT/'stage/docs/weekend_20261120_23/t693'
NAME=sys.argv[2] if len(sys.argv)>2 else None
OUT=ROOTOUT/NAME if NAME else ROOTOUT;DOC=ROOTDOC/NAME if NAME else ROOTDOC
import t693_wait as Wait
from status_io import save
import t693_native as N
import t693_estimator as Est
import t693_learning as L
T=L.T;read,sha,digest,torch,np=T.read,T.sha,T.digest,T.torch,T.np
T.OUT,T.DOC,T.HERE,T.Wait=ROOTOUT,ROOTDOC,HERE,Wait

def plan():
    root=read(ROOTOUT/'protocol.json');p=dict(root);p.update(read(OUT/'protocol.json'))
    assert p['_root_sha256']==sha(ROOTOUT/'protocol.json');return p

def verify(p,*args):
    root=read(ROOTOUT/'protocol.json');assert root['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(ROOTOUT/'protocol.json')
    assert T.R.used(ROOTOUT)<root['max_bytes'] and T.R.used(OUT)<p['child_bytes'] and T.R.used(ROOT/'stage/runs')<root['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(ROOTDOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in root['inputs'].items():assert sha(f)==h,f

T.verify=verify

def count_commands(rows,commands):
    assert len(rows)+6==commands

def setup(p):
    T.setup(p)
    if p['arm']=='collection':
        T.R.N.worker=N.worker;T.R.N.inspect=T.O.N.inspect=T.N.inspect=N.inspect
        assert T.R.N.Pool.__init__.__globals__['worker'] is N.worker


def event(who):
    def emit(stage,k):
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,k,'actor',stage])+'\n')
    return emit

def export_state(state,p):
    old=L.M.P.checkpoint(p['initial']['path'])['joint_state'];q,temporary_opt=L.make(p);L.load(q,temporary_opt,state)
    joint=dict(actor=copy.deepcopy(state['actor']),actor_opt=copy.deepcopy(state['actor_opt']),value=old['value'],value_opt=old['value_opt'])
    buf=io.BytesIO();torch.save(dict(joint_state=joint,policy_hash=q.phash(),joint_hash=digest(joint),architecture='T501_single120_64carry_two_neural_command_residuals',single_actor=True,teacher_at_deployment=False),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<p['model_bytes'];path=OUT/'proposal.pt.gz'
    with path.open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),joint_hash=digest(joint),bytes=len(blob))

def preserve_failed_state(q,opt,shadow,shadowopt,before,proposal,p):
    interrupted=L.optimizer_state(q,opt)
    if proposal is None and digest(interrupted)!=digest(before):
        proposal=export_state(interrupted,p)
        save(OUT/'INTERRUPTED_MODEL.json',dict(model=proposal,rejected=True,never_promote=True))
    L.load(q,opt,before);L.load(shadow,shadowopt,before)
    assert digest(L.optimizer_state(q,opt))==digest(before)==digest(L.optimizer_state(shadow,shadowopt))
    save(OUT/'INTERRUPTION_ROLLBACK.json',dict(state='PASS',all_Actor_and_Adam_restored=True))
    return proposal

def collect():
    p=plan();verify(p);setup(p);T.R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p)
    error=None;fit=None;proposal=None;screen=[];full=None;rejected=None
    try:
        q,opt=L.make(p);shadow,shadowopt=L.make(p);before=L.optimizer_state(q,opt);assert digest(before)==digest(L.optimizer_state(shadow,shadowopt))
        start=time.time();last=[0.]
        def learning_guard():
            assert time.time()<min(start+p['learning_seconds'],p['absolute_learning_stop'])-10
            if time.time()-last[0]>5:rt.guard();last[0]=time.time()
        rows,unused,keep=L.data(p,q,learning_guard)
        fit,proposed=L.fit(p,q,opt,rows,unused,keep,learning_guard,event('live'))
        other,shadowproposed=L.fit(p,shadow,shadowopt,rows,unused,keep,learning_guard,event('shadow'))
        assert fit==other and digest(proposed)==digest(shadowproposed) and digest(L.optimizer_state(q,opt))==digest(L.optimizer_state(shadow,shadowopt))
        proposal=export_state(proposed,p) if fit['Actor_Adam'] else None;trained=digest(proposed)
        save(OUT/'LEARNING.json',dict(fit=fit,initial_hash=digest(before),trained_hash=trained,proposal=proposal))
        L.load(q,opt,before);L.load(shadow,shadowopt,before);assert digest(L.optimizer_state(q,opt))==digest(before)==digest(L.optimizer_state(shadow,shadowopt))
        save(OUT/'ROLLBACK.json',dict(state='PASS',all_Actor_Adam_restored=True))
        del q,opt,shadow,shadowopt,rows,unused,keep,before,proposed,shadowproposed
        if fit['accepted']:
            T.pool(rt);values=rt.request(p['screen_jobs'],proposal,'screen',True);screen=[r for a,r in values]
            rt.request(p['screen_jobs'],proposal,'exact_screen',False,screen)
            by={r['job_id']:r for r in screen};ids={k for k,r in by.items() if T.Prior.safe(r)}
            missing=sorted(set(p['required_screen_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
            save(OUT/'SCREEN.json',dict(lost_ids=missing,gained_ids=gained,field_success=len(ids&{j['id'] for j in p['field_jobs']}),model=proposal))
            if not missing and gained:
                assert time.time()+p['full_evaluation_reserve_seconds']<rt.deadline
                rest=[j for j in p['evaluation_jobs'] if j['id'] not in by]
                for idx in range(0,len(rest),48):
                    chunk=rest[idx:idx+48];assert time.time()+60+sum(j['case']['step_cap'] for j in chunk)/p['admission_physics_per_second']<rt.deadline-10
                    vv=rt.request(chunk,proposal,f'rest407_{idx//48:02d}',False);by.update({r['job_id']:r for a,r in vv})
                    save(OUT/'EVALUATION.json',dict(completed=len(by),total=407))
                ids={k for k,r in by.items() if T.Prior.safe(r)};lost=sorted(set(p['before_success_ids'])-ids)
                full=dict(success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=sorted(ids-set(p['before_success_ids'])),all387=set(p['required387_ids'])<=ids,accepted=not lost and bool(ids-set(p['before_success_ids'])))
                with gzip.open(OUT/'JOINT407.json.gz','xb') as f:f.write(json.dumps(dict(model=proposal,rows=[by[j['id']] for j in p['evaluation_jobs']])).encode())
        else:rejected=fit['reason'];save(OUT/'NO_NATIVE.json',dict(reason=rejected,proposal_preserved=True))
    except BaseException:
        error=traceback.format_exc()
        try:
            if 'before' in locals() and 'q' in locals():
                proposal=preserve_failed_state(q,opt,shadow,shadowopt,before,proposal,p)
        except BaseException:
            save(OUT/'interruption_preservation_error.json',dict(error=traceback.format_exc(),original_error=error))
        save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
            proposal=proposal,fit=fit,screen=screen,full407=full,rejected=rejected,retained=proposal if full and full['accepted'] else p['initial'],new_Value=0,paired_target_arm=p['target_arm']))
        fcntl.flock(lock,fcntl.LOCK_UN)
    assert error is None,error

def audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');learning=read(OUT/'LEARNING.json')
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    exported=L.M.P.checkpoint(s['proposal']['path'])['joint_state'] if s['proposal'] else L.M.P.checkpoint(p['initial']['path'])['joint_state'];initial=L.M.P.checkpoint(p['initial']['path'])['joint_state']
    assert digest(exported['value'])==digest(initial['value']) and digest(exported['value_opt'])==digest(initial['value_opt'])
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    allrows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (OUT/'cases.jsonl.gz').exists() else [];by={r['path']:r for r in allrows};assert len(by)==len(allrows)
    for r in allrows:
        assert sha(r['path'])==r['sha256'];a=T.R.E.N.S.arrays(r['path'])
        if r['full']:T.N.inspect(p,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            prev=by[r['reference_path']];assert r['array_hashes']==prev['array_hashes'] and r['integration_sha256']==prev['integration_sha256']
        assert r['policy_hash']==s['proposal']['policy_hash'] and r['loaded_model_sha256']==s['proposal']['sha256']
    assert sum(r['physics'] for r in allrows)==s['physics']<=p['max_physics']
    count_commands(allrows,s['commands']);assert s['commands']<=p['max_commands']
    q,opt=L.make(p);before=L.optimizer_state(q,opt);assert digest(before)==learning['initial_hash'];start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    rows,unused,keep=L.data(p,q,guard);again,proposed=L.fit(p,q,opt,rows,unused,keep,guard,event('independent_history'))
    assert again==s['fit'] and digest(proposed)==learning['trained_hash'];L.load(q,opt,before);assert digest(L.optimizer_state(q,opt))==digest(before)
    if s['full407']:
        rr=[r for r in allrows if r['role']=='screen' or r['role'].startswith('rest407_')];assert len(rr)==407
        ids={r['job_id'] for r in rr if T.Prior.safe(r)};assert s['full407']['success_ids']==sorted(ids) and s['full407']['lost_prior']==sorted(set(p['before_success_ids'])-ids)
    calls=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()] if (OUT/'optimizer_calls.jsonl').exists() else []
    for who in ['live','shadow','independent_history']:assert [(r[1],r[3]) for r in calls if r[0]==who]==([(0,'BEFORE'),(0,'AFTER')] if s['fit']['Actor_Adam'] else [])
    save(OUT/'audit.json',dict(state='PASS',independent_Actor=s['fit']['Actor_Adam'],new_Value=0,new_native=0,live_shadow_independent_exact=True,complete_parameter_and_Adam_restore=True,
        all_cases_and_exact_verified=True,full407=s['full407'],screen=read(OUT/'SCREEN.json') if (OUT/'SCREEN.json').exists() else None,target_arm=p['target_arm']))
    fcntl.flock(lock,fcntl.LOCK_UN)


def schedule(p):
    """Locked schedule, with invalid slots contributing no requests."""
    jobs=[];seeds=[];by={j['id']:j for j in p['evaluation_jobs']}
    for index,entry in enumerate(p['fixed_records']):
        if not entry['active']:continue
        task=Est.IDS.index(entry['task']);draws=Est.draws(task,entry['rep'])
        for arm in ['mvd','score']:
            for pair in range(2):
                for sign in range(2):
                    latent=np.asarray(entry['mu'])+Est.SIGMA*draws[arm][pair,sign]
                    jobs.append(dict(by[entry['task']],source_path=entry['source'],source_sha256=entry['source_sha256'],
                        branch_decision=entry['slot'],branch_latent=latent.tolist(),branch_command=latent.clip(-1,1).astype(np.float32).tolist(),
                        branch_record_index=index,method=arm,pair=pair,sign=sign))
                    seeds.append(draws['suffix_seeds'][pair])
    assert len(jobs)==72
    return jobs,seeds


def credits(p,rows):
    result=[]
    for index,entry in enumerate(p['fixed_records']):
        record=dict(entry,credit=dict(mvd=[0.,0.],score=[0.,0.]),outcomes=None)
        if entry['active']:
            draws=Est.draws(Est.IDS.index(entry['task']),entry['rep']);matrices={}
            for arm in ['mvd','score']:
                matrix=np.zeros((2,2));flags=[]
                for pair in range(2):
                    for sign in range(2):
                        rr=[r for r in rows if r['request']['job']['branch_record_index']==index and
                            r['request']['job']['method']==arm and r['request']['job']['pair']==pair and r['request']['job']['sign']==sign]
                        assert len(rr)==1;r=rr[0]
                        matrix[pair,sign]=int(T.Prior.safe(r))-int(r['collision'])
                        flags.append(dict(pair=pair,sign=sign,path=r['path'],safe=T.Prior.safe(r),collision=r['collision'],
                                          timeout=r['timeout'],boundary=r['field_boundary']))
                record['credit'][arm]=Est.credit(arm,matrix,draws[arm]).tolist()
                matrices[arm]=dict(Y=matrix.tolist(),flags=flags)
            record['outcomes']=matrices
        result.append(record)
    return dict(records=result,local8_ID_target_scale512=True,slot_weight=2.,denominator32=True,
                no_V_Q_or_variance_guarantee=True,all_zero_and_negative_outcomes_preserved=True)


def collection():
    p=plan();verify(p);setup(p);T.R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p);error=None;fresh=[]
    try:
        T.pool(rt);jobs,seeds=schedule(p)
        vv=rt.request(jobs,p['initial'],'credit_fresh',True,seeds=seeds);fresh=[r for a,r in vv]
        rt.request(jobs,p['initial'],'credit_exact',True,fresh,seeds=seeds)
        save(OUT/'CREDITS.json',credits(p,fresh))
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
                                    elapsed_s=time.time()-rt.start,rows=fresh,new_Actor=0,new_Value=0,full407=None,retained=p['initial']))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def collection_audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for generation in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(pid)).exists() for pid in [generation['parent'],*generation['workers']])
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by={r['path']:r for r in rows};assert len(by)==len(rows)==144
    jobs,seeds=schedule(p)
    for index,r in enumerate(rows):
        verify(p);assert sha(r['path'])==r['sha256'] and r['full']
        k=index%72;assert r['request']['job']==jobs[k] and r['seed']==seeds[k]
        assert r['loaded_model_sha256']==p['initial']['sha256'] and r['policy_hash']==p['initial']['policy_hash']
        a=T.R.E.N.S.arrays(r['path']);N.inspect(p,a,r)
        if index>=72:
            old=by[r['reference_path']];assert r.get('exact') and r['array_hashes']==old['array_hashes']
            assert r['integration_sha256']==old['integration_sha256']
    fresh=[r for r in rows if not r.get('exact')];assert fresh==s['rows']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    count_commands(rows,s['commands']);assert s['commands']<=p['max_commands']
    assert credits(p,fresh)==read(OUT/'CREDITS.json')
    assert not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'audit.json',dict(state='PASS',all72fresh72exact_full_arrays_integrals_owncarry_commands_and_RNG=True,
                              both_estimators_outcomes_and_credits_exact=True,independent_Actor=0,new_Value=0,new_native=0,
                              management_commands6=True,not_whole_policy_improvement_certificate=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    if NAME=='collection':{'collect':collection,'audit':collection_audit}[sys.argv[1]]()
    else:{'collect':collect,'audit':audit}[sys.argv[1]]()

