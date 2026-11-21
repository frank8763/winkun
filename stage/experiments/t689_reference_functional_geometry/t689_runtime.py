"""One child candidate, complete recorded failure screen then conditional407."""
from pathlib import Path
import copy,fcntl,gzip,io,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
ROOTOUT=ROOT/'stage/runs/t689_reference_functional_geometry';ROOTDOC=ROOT/'stage/docs/weekend_20261120_23/t689'
NAME=sys.argv[2] if len(sys.argv)>2 else None
OUT=ROOTOUT/NAME if NAME else ROOTOUT;DOC=ROOTDOC/NAME if NAME else ROOTDOC
import t689_wait as Wait
from status_io import save
import t689_learning as L
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

def collect():
    p=plan();verify(p);T.setup(p);T.R.E.L.install_stop_handlers()
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
        proposal=export_state(proposed,p);trained=digest(proposed)
        save(OUT/'LEARNING.json',dict(fit=fit,initial_hash=digest(before),trained_hash=trained,proposal=proposal))
        L.load(q,opt,before);L.load(shadow,shadowopt,before);assert digest(L.optimizer_state(q,opt))==digest(before)==digest(L.optimizer_state(shadow,shadowopt))
        save(OUT/'ROLLBACK.json',dict(state='PASS',all_Actor_optimizer_restored=True))
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
        else:rejected='FINITE_OR_ACTUAL_SIGMA_KL_GATE';save(OUT/'NO_NATIVE.json',dict(reason=rejected,proposal_preserved=True))
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
            proposal=proposal,fit=fit,screen=screen,full407=full,rejected=rejected,retained=proposal if full and full['accepted'] else p['initial'],new_Value=0,paired_target_arm=p['target_arm']))
        fcntl.flock(lock,fcntl.LOCK_UN)
    assert error is None,error

def audit():
    p=plan();verify(p);T.setup(p);s=read(OUT/'summary.json');learning=read(OUT/'LEARNING.json')
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    exported=L.M.P.checkpoint(s['proposal']['path'])['joint_state'];initial=L.M.P.checkpoint(p['initial']['path'])['joint_state']
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
    assert (len(allrows)+6 if allrows else 3)==s['commands']<=p['max_commands']
    q,opt=L.make(p);before=L.optimizer_state(q,opt);assert digest(before)==learning['initial_hash'];start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    rows,unused,keep=L.data(p,q,guard);again,proposed=L.fit(p,q,opt,rows,unused,keep,guard,event('independent_history'))
    assert again==s['fit'] and digest(proposed)==learning['trained_hash'];L.load(q,opt,before);assert digest(L.optimizer_state(q,opt))==digest(before)
    if s['full407']:
        rr=[r for r in allrows if r['role']=='screen' or r['role'].startswith('rest407_')];assert len(rr)==407
        ids={r['job_id'] for r in rr if T.Prior.safe(r)};assert s['full407']['success_ids']==sorted(ids) and s['full407']['lost_prior']==sorted(set(p['before_success_ids'])-ids)
    calls=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for who in ['live','shadow','independent_history']:assert [(r[1],r[3]) for r in calls if r[0]==who]==[(0,'BEFORE'),(0,'AFTER')]
    save(OUT/'audit.json',dict(state='PASS',independent_Actor=1,new_Value=0,new_native=0,live_shadow_independent_exact=True,complete_parameter_and_optimizer_restore=True,
        all_cases_and_exact_verified=True,full407=s['full407'],screen=read(OUT/'SCREEN.json') if (OUT/'SCREEN.json').exists() else None,target_arm=p['target_arm']))
    fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
