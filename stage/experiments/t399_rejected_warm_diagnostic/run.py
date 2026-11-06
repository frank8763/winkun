"""Frozen candidate, four fixed task conditions, exact serialized repeats."""
from t399_contract import *

def main():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);forbid_learning()
    q=frozen(base,p);fixed=digest(q.state_dict());ref=N.S.arrays(REFERENCE)
    start=time.time();deadline=start+p['main_seconds'];physics=commands=0;last=0.;phase='SETUP';s=w=None;cases=[];replays=[];state_final='FAILED'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=deadline))
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_in_main=True))
    def guard(physical=False):
        nonlocal last
        C.check_stop();assert time.time()<deadline-5 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)==outside
            save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physics,commands=commands,unique=len(cases),exact=len(replays),Actor_updates=0,deadline=deadline))
    def charge():
        nonlocal physics
        physics+=1;assert physics<=p['max_physics']
    def command(label,fn):
        nonlocal commands
        guard();commands+=1;before=physics;begin=time.time();error=None
        try:return fn()
        except BaseException:error=traceback.format_exc();raise
        finally:
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands,label=label,physics=physics-before,elapsed_s=time.time()-begin,error=error),separators=(',',':'))+'\n')
    def execute(condition,reference=None):
        nonlocal s,phase
        first=condition['prefix_holds'];phase=condition['id']+('_SERIALIZED_EXACT' if reference else '_NEW')
        def body():
            nonlocal s
            q.load_learned(N.S.read_model(MODEL));q.reset_noise();q.sample=False
            assert q.phash()==p['policy_hash'] and digest(q.state_dict())==fixed
            m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=condition['job']['circle_shift_x_m']
            s=C.Session(condition['job'],native,create,w,guard,charge,q);C.install_stop_handlers()
            for i in range(first):
                row,_=s.step(ref['actions'][i],owner=1);assert not s.done()
                for k,name in enumerate(N.R.NAMES):assert equal(np.asarray(row[k]),ref[name][i]),(i,name)
            while not s.done():s.step(owner=0)
            a=payload(s);r=s.result();assert r['complete'] and not r['administrative_censor']
            rec=dict(r,id=condition['id'],prefix_holds=first,policy_hash=q.phash(),array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),measures=measures(a,first),new_condition=reference is None,independent_sample=False)
            if reference is None:
                f=OUT/'episodes'/(condition['id']+'.npz');N.write_npz(f,N.S.packed(a),p['full_bytes']);rec.update(path=str(f),sha256=sha(f));cases.append(rec);file='cases.jsonl'
            else:
                for name in ['id','array_digest','integration_sha256','measures','success','collision','timeout','physics_steps']:assert rec[name]==reference[name],name
                rec['exact']=True;replays.append(rec);file='replays.jsonl'
            with (OUT/file).open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
            assert digest(q.state_dict())==fixed;s.close();s=None;guard();return rec
        return command(phase,body)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(C.StageStop('T399 native deadline')));signal.alarm(p['main_seconds']-2)
    try:
        m,native,create,w,_=command('native_setup',lambda:N.S.P.C.B.setup(OUT/'native',base));C.install_stop_handlers();original=copy.deepcopy(m.ACTIVE)
        for condition in p['conditions']:
            row=execute(condition);execute(condition,row)
        state_final='COMPLETE'
    except BaseException:
        if s is not None:
            if s.rows:
                try:N.write_npz(OUT/'partial.npz',N.S.packed(payload(s)),p['full_bytes'])
                except BaseException:save(OUT/'partial_save_error.json',dict(error=traceback.format_exc()))
            save(OUT/'partial_case.json',dict(role=phase,actual_physics=s.steps,recorded_physics=sum(v[2] for v in s.rows),administrative_censor=True))
        save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=physics,commands=commands))
    finally:
        if s:s.close()
        if w:w.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        summary=dict(state=state_final,pid=os.getpid(),elapsed_s=time.time()-start,physics=physics,commands=commands,conditions=cases,new_conditions=len(cases),exact_replays=len(replays),Actor_updates=0,optimizer_updates=0,backward_calls=0,model_sha256=sha(MODEL),policy_hash=q.phash(),all_workers_stopped=True,all_native_sessions_closed=True,old275=None,main54=None,fresh16=None,reload78=None,full_six_passed=False,seven_eight_entered=False,historical_parent_controls=p['historical_parent_controls'],model_promoted=False)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state_final,updated_epoch=time.time(),physics=physics,commands=commands,Actor_updates=0));N.manifest(OUT)
    assert state_final=='COMPLETE'

def audit():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);summary=read(OUT/'summary.json')
    assert summary['state']=='COMPLETE' and not Path('/proc',str(summary['pid'])).exists() and not (OUT/'audit.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);forbid_learning();q=frozen(base,p);fixed=digest(q.state_dict())
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    assert len(cases)==len(reps)==len(p['conditions'])==4
    for condition,row,rep in zip(p['conditions'],cases,reps):
        assert row['id']==condition['id'] and row['prefix_holds']==condition['prefix_holds'] and sha(row['path'])==row['sha256']
        a=N.S.arrays(row['path']);audit_array(q,a,row['prefix_holds'])
        assert digest(a)==row['array_digest']==rep['array_digest'] and measures(a,row['prefix_holds'])==row['measures']==rep['measures']
        assert digest(a['final_native_integration'])==row['integration_sha256']==rep['integration_sha256']
        assert int(a['native_counts'].sum())==row['physics_steps']==rep['physics_steps']
        assert bool(a['end_flags'][-1,1])==row['success'] and bool(a['end_flags'][:,0].any())==row['collision'] and bool(a['end_flags'][-1,2])==row['timeout']
    commands=[json.loads(v) for v in (OUT/'commands.jsonl').read_text().splitlines()]
    assert summary['physics']==sum(v['physics_steps'] for v in [*cases,*reps])==sum(v['physics'] for v in commands)<=p['max_physics']
    assert summary['commands']==len(commands)==9 and summary['commands']+2<=p['max_commands']
    assert digest(q.state_dict())==fixed and summary['Actor_updates']==summary['optimizer_updates']==summary['backward_calls']==0
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,optimizer=0,backward=0,all_causal_history_hidden_actions_rewards_exact=True,all_serialized_native_repeats_exact=True,physics=summary['physics'],commands=summary['commands'],policy_hash=q.phash(),model_sha256=sha(MODEL),weights_unchanged=True,full_six_passed=False));N.manifest(OUT)
    print('T399_ZERO_NATIVE_ZERO_OPTIMIZER_AUDIT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
