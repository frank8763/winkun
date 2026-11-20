"""One finite feasibility candidate, then original-start retention tests."""
from pathlib import Path
import copy,fcntl,gzip,io,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t671_joint_outcome_feasibility';DOC=ROOT/'stage/docs/weekend_20261120_23/t671'
import t671_wait as Wait
from status_io import save
import t671_learning as L
T=L.A.T;read,sha,digest,torch,np=T.read,T.sha,T.digest,T.torch,T.np
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait


def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<10*2**30
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f


T.verify=verify


def event(who):
    def emit(stage,k):
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,k,'actor',stage])+'\n')
    return emit


def export(q,opt,p):
    old=L.M.P.checkpoint(p['initial']['path'])['joint_state']
    joint=dict(actor=copy.deepcopy(q.state_dict()),actor_opt=copy.deepcopy(opt.state_dict()),value=old['value'],value_opt=old['value_opt'])
    buf=io.BytesIO();torch.save(dict(joint_state=joint,policy_hash=q.phash(),joint_hash=digest(joint),
        architecture='T501_single120_64carry_two_neural_command_residuals',single_actor=True,teacher_at_deployment=False),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<p['model_bytes']
    path=OUT/'proposal.pt.gz'
    with path.open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),joint_hash=digest(joint),bytes=len(blob))


def collect():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);T.R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p)
    error=None;fit=None;full=None;proposal=None;screen=None
    try:
        q,opt=L.make(p);shadow,shadowopt=L.make(p)
        before=L.optimizer_state(q,opt);assert digest(before)==digest(L.optimizer_state(shadow,shadowopt))
        start=time.time()
        def learning_guard():rt.guard();assert time.time()<min(start+p['learning_seconds'],p['absolute_learning_stop'])-10
        old,field,keep=L.data(p,q,learning_guard)
        fit=L.fit(p,q,opt,old,field,keep,learning_guard,event('live'))
        shadowfit=L.fit(p,shadow,shadowopt,old,field,keep,learning_guard,event('shadow'))
        assert fit==shadowfit and digest(L.optimizer_state(q,opt))==digest(L.optimizer_state(shadow,shadowopt))
        trained=L.optimizer_state(q,opt);proposal=export(q,opt,p)
        save(OUT/'LEARNING.json',dict(fit=fit,initial_hash=digest(before),trained_hash=digest(trained),proposal=proposal))
        L.load(q,opt,before);L.load(shadow,shadowopt,before)
        assert digest(L.optimizer_state(q,opt))==digest(before)==digest(L.optimizer_state(shadow,shadowopt))
        save(OUT/'ROLLBACK.json',dict(state='PASS',all_Actor_Adam_restored=True))
        del q,opt,shadow,shadowopt,old,field,keep,before,trained
        T.pool(rt);values=rt.request(p['screen_jobs'],proposal,'screen40',False)
        screen=[r for a,r in values];by={r['job_id']:r for r in screen}
        keys=[by[j['id']] for j in p['key_jobs']]
        rt.request(p['key_jobs'],proposal,'exact24',True,keys)
        ids={k for k,r in by.items() if T.Prior.safe(r)}
        lostscreen=sorted(set(p['required_screen_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
        save(OUT/'SCREEN.json',dict(lost_ids=lostscreen,gained_ids=gained,field_success=len(ids&{j['id'] for j in p['field_jobs']}),model=proposal))
        if not lostscreen and gained:
            assert time.time()+p['full_evaluation_reserve_seconds']<rt.deadline
            rest=[j for j in p['evaluation_jobs'] if j['id'] not in by];assert len(rest)==367
            for index in range(0,len(rest),48):
                chunk=rest[index:index+48]
                assert time.time()+60+sum(j['case']['step_cap'] for j in chunk)/p['admission_physics_per_second']<rt.deadline-10
                more=rt.request(chunk,proposal,f'rest407_{index//48:02d}',False);by.update({r['job_id']:r for a,r in more})
                save(OUT/'EVALUATION.json',dict(completed=len(by),total=407))
            ids={k for k,r in by.items() if T.Prior.safe(r)};lost=sorted(set(p['before_success_ids'])-ids)
            full=dict(success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=sorted(ids-set(p['before_success_ids'])),
                all387=set(p['required387_ids'])<=ids,accepted=not lost and bool(ids-set(p['before_success_ids'])))
            with gzip.open(OUT/'JOINT407.json.gz','xb') as f:f.write(json.dumps(dict(model=proposal,rows=[by[j['id']] for j in p['evaluation_jobs']])).encode())
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,proposal=proposal,fit=fit,screen=screen,full407=full,
            retained=proposal if full and full['accepted'] else p['initial'],new_Value=0,not_a_matched_algorithm_comparison=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);s=read(OUT/'summary.json');learning=read(OUT/'LEARNING.json')
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows}
    assert len(rows)==len(by)
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=T.R.E.N.S.arrays(r['path'])
        if r['full']:T.N.inspect(p,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert r['policy_hash']==s['proposal']['policy_hash'] and r['loaded_model_sha256']==s['proposal']['sha256']
        if r.get('exact'):
            prev=by[r['reference_path']];assert r['array_hashes']==prev['array_hashes'] and r['integration_sha256']==prev['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    assert len(rows)+6==s['commands']<=p['max_commands']
    q,opt=L.make(p);before=L.optimizer_state(q,opt);assert digest(before)==learning['initial_hash']
    start=time.time()
    def guard():verify(p);assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
    # Expensive source validation is throttled outside the 387-reference loop.
    last=[float('-inf')]
    def throttled():
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:guard();last[0]=time.time()
    old,field,keep=L.data(p,q,throttled);again=L.fit(p,q,opt,old,field,keep,throttled,event('independent_history'))
    assert again==s['fit'] and digest(L.optimizer_state(q,opt))==learning['trained_hash'] and q.phash()==s['proposal']['policy_hash']
    L.load(q,opt,before);assert digest(L.optimizer_state(q,opt))==digest(before)
    screen=[r for r in rows if r['role']=='screen40'];assert screen==s['screen'] and len(screen)==40
    if s['full407']:
        rr=screen+[r for r in rows if r['role'].startswith('rest407_')];assert len(rr)==407
        ids={r['job_id'] for r in rr if T.Prior.safe(r)};assert s['full407']['success_ids']==sorted(ids)
        assert s['full407']['lost_prior']==sorted(set(p['before_success_ids'])-ids)
        assert s['full407']['accepted']==(not s['full407']['lost_prior'] and bool(ids-set(p['before_success_ids'])))
    calls=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for who in ['live','shadow','independent_history']:
        assert [(r[1],r[3]) for r in calls if r[0]==who]==[(k,stage) for k in range(s['fit']['Actor_Adam']) for stage in ['BEFORE','AFTER']]
    save(OUT/'audit.json',dict(state='PASS',independent_Actor=s['fit']['Actor_Adam'],new_Value=0,new_native=0,
        live_shadow_independent_exact=True,complete_parameter_and_Adam_restore=True,all_cases_and_exact_verified=True,
        full407=s['full407'],single_candidate_not_algorithm_comparison=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
