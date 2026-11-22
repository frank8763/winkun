"""One supervised compatibility candidate and complete conditional407."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
import t696_actor as A
import t696_native as N
import t696_wait as Wait
from status_io import save
ROOT=A.ROOT;HERE=Path(__file__).resolve().parent
ROOTOUT=ROOT/'stage/runs/t696_existing_success_compatibility';ROOTDOC=ROOT/'stage/docs/weekend_20261120_23/t696'
NAME=sys.argv[2] if len(sys.argv)>2 else None
OUT=ROOTOUT/NAME if NAME else ROOTOUT;DOC=ROOTDOC/NAME if NAME else ROOTDOC
T,read,sha,digest,torch,np=A.L.T,A.read,A.sha,A.digest,A.torch,A.np
T.OUT,T.DOC,T.HERE,T.Wait=ROOTOUT,ROOTDOC,HERE,Wait


def plan():
    p=read(ROOTOUT/'protocol.json');child=read(OUT/'protocol.json')
    assert child['_root_sha256']==sha(ROOTOUT/'protocol.json')
    return dict(p,**child)


_manifest_stat = None

def verify(p,*unused,force=False):
    global _manifest_stat
    root=read(ROOTOUT/'protocol.json')
    assert root['authorization_approved'] and sha(HERE/'protocol.json')==sha(ROOTOUT/'protocol.json')
    assert time.time()<p['absolute_all_stop']
    assert T.R.used(ROOTOUT)<root['max_bytes'] and T.R.used(OUT)<p['child_bytes']
    assert T.R.used(ROOT/'stage/runs')-T.R.used(ROOTOUT)+root['max_bytes']<root['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(ROOTDOC/'AFFINITY_GUARD.json')
    assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    stamps={f:(Path(f).stat().st_ino,Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in root['inputs']}
    if _manifest_stat is None or force:
        for f,h in root['inputs'].items():assert sha(f)==h,f
    if _manifest_stat is not None:assert stamps==_manifest_stat
    _manifest_stat=stamps


T.verify=verify


def setup(p):
    T.setup(p);T.R.N.worker=N.worker;T.R.N.inspect=T.O.N.inspect=T.N.inspect=N.inspect
    N.P.frozen=A.frozen
    assert T.R.N.Pool.__init__.__globals__['worker'] is N.worker
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def event(who):
    def emit(stage,k):
        with (OUT/'fit_calls.jsonl').open('a') as f:f.write(json.dumps([who,k,'actor_closed_form',stage])+'\n')
    return emit


def check_commands(p,q,guard):
    try:return dict(passed=True,contracts=A.verify_commands(p,q,guard),error=None)
    except AssertionError as error:return dict(passed=False,contracts=None,error=str(error))


def fit():
    p=plan();verify(p);setup(p);T.R.E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()))
    started=time.time();error=None;model=None;commands=None;info=None;last=[0.]
    def guard():
        assert time.time()<min(started+p['learning_seconds'],p['absolute_learning_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    try:
        protected,sources,manifest=A.data(p,guard);save(OUT/'DATA_MANIFEST.json',manifest)
        q,info=A.fit(p,protected,sources,guard,event('live'))
        model=A.export(q,p,OUT/'proposal.pt.gz')
        shadow,other=A.fit(p,protected,sources,guard,event('shadow'))
        assert info==other and digest(q.state_dict())==digest(shadow.state_dict())
        assert model['policy_hash']==A.frozen(p,model['path']).phash()==shadow.phash()
        commands=check_commands(p,q,guard)
        save(OUT/'LEARNING.json',dict(model=model,fit=info,commands=commands,Actor_live=1,Actor_shadow=1,
                                    optimizer=0,new_RL=0,all_original_T501_parameters_unchanged=True,
                                    original_optimizer_provenance_NOT_new_head_optimizer=True))
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,model=model,no_restart=True,raw_preserved=True))
    finally:
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=0,commands=0,
            elapsed_s=time.time()-started,model=model,fit=info,command_gate=commands,no_native=True,full407=None,retained=p['initial']))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def fit_audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();last=[0.]
    def guard():
        assert time.time()<min(started+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    protected,sources,manifest=A.data(p,guard);assert manifest==read(OUT/'DATA_MANIFEST.json')
    q,info=A.fit(p,protected,sources,guard,event('independent_history'))
    saved=A.Base.P.checkpoint(s['model']['path'])
    assert sha(s['model']['path'])==s['model']['sha256'] and info==s['fit']
    assert digest(q.state_dict())==digest(saved['joint_state']['actor']) and q.phash()==s['model']['policy_hash']
    assert check_commands(p,q,guard)==s['command_gate']
    calls=[json.loads(x) for x in (OUT/'fit_calls.jsonl').read_text().splitlines()]
    for who in ['live','shadow','independent_history']:
        assert [r[3] for r in calls if r[0]==who]==['BEFORE','AFTER']
    verify(p,force=True)
    save(OUT/'audit.json',dict(state='PASS',independent_Actor=1,new_optimizer=0,new_native=0,
        all_sources_features_geometry_coefficients_and_model_exact=True,commands=s['command_gate'],
        original_T501_parameters_and_owncarry_unchanged=True,finite_compatibility_only_NOT_RL_or_generalization=True))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def evaluate():
    p=plan();verify(p);setup(p);T.R.E.L.install_stop_handlers()
    prior=read(ROOTOUT/'fit/summary.json');assert read(ROOTOUT/'fit/audit.json')['state']=='PASS'
    model=prior['model'];lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p)
    error=None;screen=[];full=None;gate='SOURCE_COMMAND_GATE_FAILED_NO_NATIVE'
    try:
        if prior['command_gate']['passed']:
            T.pool(rt);vv=rt.request(p['screen_jobs'],model,'screen',True);screen=[r for a,r in vv]
            rt.request(p['screen_jobs'],model,'exact_screen',False,screen)
            by={r['job_id']:r for r in screen};ids={key for key,r in by.items() if T.Prior.safe(r)}
            missing=sorted(set(p['required_screen_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
            save(OUT/'SCREEN.json',dict(lost_ids=missing,gained_ids=gained,model=model))
            gate='SCREEN_RETAIN_OR_SOURCE_GATE_FAILED'
            if not missing:
                assert time.time()+p['full_evaluation_reserve_seconds']<rt.deadline
                rest=[j for j in p['evaluation_jobs'] if j['id'] not in by]
                for off in range(0,len(rest),48):
                    chunk=rest[off:off+48]
                    assert time.time()+60+sum(j['case']['step_cap'] for j in chunk)/p['admission_physics_per_second']<rt.deadline-10
                    vv=rt.request(chunk,model,f'rest431_{off//48:02d}',False);by.update({r['job_id']:r for a,r in vv})
                ids={key for key,r in by.items() if T.Prior.safe(r)};core=ids&set(p['core407_ids'])
                lost=sorted(set(p['before_success_ids'])-core);new=sorted(core-set(p['before_success_ids']))
                lost403=sorted(set(p['required403_ids'])-ids)
                full=dict(success=len(core),success_ids=sorted(core),lost_prior=lost,gained_prior=new,
                          all387=set(p['required387_ids'])<=core,lost403=lost403,
                          all431_success_ids=sorted(ids),supplementary_safe_ids=sorted(ids-set(p['core407_ids'])),
                          all7_sources_pass=set(p['source_ids'])<=ids,
                          accepted=not lost403 and set(p['source_ids'])<=ids and bool(new))
                with gzip.open(OUT/'JOINT407.json.gz','xb') as f:
                    f.write(json.dumps(dict(model=model,rows=[by[j['id']] for j in p['evaluation_jobs']],core407_ids=p['core407_ids'])).encode())
                gate='FULL403_AND_CORE407_CANDIDATE_PENDING_AUDIT' if full['accepted'] else 'FULL403_OR_CORE407_GATE_FAILED'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        verify(p,force=True)
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,model=model,screen=screen,full407=full,gate=gate,
            retained=model if full and full['accepted'] else p['initial'],new_Actor=0,new_Value=0,new_optimizer=0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def evaluate_audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):
            assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (OUT/'cases.jsonl.gz').exists() else []
    by={r['path']:r for r in rows};assert len(by)==len(rows)
    for r in rows:
        verify(p);assert sha(r['path'])==r['sha256'] and r['loaded_model_sha256']==s['model']['sha256']
        assert r['policy_hash']==s['model']['policy_hash']
        a=T.R.E.N.S.arrays(r['path'])
        if r['full']:N.inspect(p,a,r)
        else:assert all(digest(v)==r['array_hashes'][key] for key,v in a.items())
        N.audit_source_hashes(p,r)
        if r.get('exact'):
            old=by[r['reference_path']]
            assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    assert len(rows)+6==s['commands'] and sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    if s['full407']:
        fresh=[r for r in rows if not r.get('exact')];assert len(fresh)==431
        ids={r['job_id'] for r in fresh if T.Prior.safe(r)};assert sorted(ids&set(p['core407_ids']))==s['full407']['success_ids']
        assert sorted(set(p['required403_ids'])-ids)==s['full407']['lost403']
        assert s['full407']['lost_prior']==sorted(set(p['before_success_ids'])-ids)
    verify(p,force=True)
    save(OUT/'audit.json',dict(state='PASS',all_actual_cases_and_exact_verified=True,all403_protected_and7_source_physical_hashes_compared=True,
                              full407=s['full407'],new_Actor=0,new_Value=0,new_optimizer=0,new_native=0,
                              learned_source_development_interpolation_NOT_independent_generalization=True))


if __name__=='__main__':
    {'fit':fit,'fit_audit':fit_audit,'evaluate':evaluate,'evaluate_audit':evaluate_audit}[sys.argv[1]]()
