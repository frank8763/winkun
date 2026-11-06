"""Two fixed teacher durations and exact serialized-model native repeats."""
from t397_contract import *

def main():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);freeze_optimizer()
    q=actor(base,p);fixed=digest(q.state_dict());teacher=N.S.arrays(TEACHER)
    start=time.time();deadline=start+p['main_seconds'];physical=commands=0;last=0.;phase='SETUP';s=w=None;rows=[];reps=[];status='FAILED'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT);assert outside+p['final_bytes']<=1024**3
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=deadline));save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_in_main=True))
    def guard(physical=False):
        nonlocal last
        C.check_stop();assert time.time()<deadline-5 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)==outside
            save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=count(),commands=commands,unique=len(rows),exact=len(reps),Actor_updates=0,deadline=deadline))
    def count():return physical
    def charge():
        nonlocal physical
        physical+=1;assert physical<=p['max_physics']
    def step_compare(i,forced=None,owner=0):
        row,hd=s.step(forced,owner=owner)
        for k,key in enumerate(N.R.NAMES):assert equal(np.asarray(row[k]),teacher[key][i]),(i,key)
        assert equal(hd,teacher['hold_sha256'][i])
        assert all(equal(a,b) for a,b in zip(s.hist[-1],[teacher[key][i] for key in ['history_poses','history_goals','history_times']]))
        assert equal(s.hb[-1],teacher['hidden_before'][i]);assert not s.done()
    def execute(holds,reference=None):
        nonlocal s,commands,phase
        guard();assert time.time()+40<deadline and physical+40000<=p['max_physics'];commands+=1
        phase='TEACHER_'+str(holds/5)+'s_'+('NEW' if reference is None else 'SERIALIZED_EXACT')
        q.load_learned(N.S.read_model(PARENT));assert q.phash()==p['parent_policy_hash'];q.sample=False;q.reset_noise()
        m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=p['shift_x_m']
        s=C.Session(p['job'],native,create,w,guard,charge,q);C.install_stop_handlers()
        assert equal(s.initial,teacher['initial_native_integration'])
        for i in range(265):step_compare(i,teacher['actions'][i],1)
        for i in range(265,267):step_compare(i)
        snapshot_hash=N.C.snapshot_hash(s);assert snapshot_hash==p['source_snapshot_sha256']
        first=267+holds
        for i in range(267,first):step_compare(i,teacher['actions'][i],1)
        handoff_hash=digest(dict(packet=s.packet(),history=w.get(s.env),integration=N.K.R.integration(s.env),hidden=s.hidden,previous=s.previous,pn=s.pn,clock=s.steps))
        while not s.done():s.step(owner=0)
        a=payload(s);r=s.result();assert r['complete'] and not r['administrative_censor']
        rec=dict(r,teacher_holds=holds,teacher_duration_s=holds/5,source_snapshot_sha256=snapshot_hash,handoff_sha256=handoff_hash,array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),policy_hash=q.phash(),handoff=handoff_metrics(a,first),independent_sample=False,new_condition=reference is None)
        if reference is None:
            f=OUT/'episodes'/('duration'+str(holds)+'.npz');N.write_npz(f,N.S.packed(a),p['full_bytes']);rec.update(path=str(f),sha256=sha(f));rows.append(rec);file='cases.jsonl'
        else:
            for key in ['array_digest','integration_sha256','source_snapshot_sha256','handoff_sha256','handoff']:assert rec[key]==reference[key],key
            rec.update(exact=True);reps.append(rec);file='replays.jsonl'
        with (OUT/file).open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
        assert digest(q.state_dict())==fixed;s.close();s=None;guard();return rec
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(C.StageStop('T397 native wall deadline')));signal.alarm(p['main_seconds']-2)
    try:
        m,native,create,w,_=N.S.P.C.B.setup(OUT/'native',base);commands+=1;C.install_stop_handlers();original=copy.deepcopy(m.ACTIVE)
        for holds in p['teacher_holds']:
            rec=execute(holds);execute(holds,rec)
        status='COMPLETE'
    except BaseException:
        if s is not None:
            if s.rows:
                try:N.write_npz(OUT/'partial.npz',N.S.packed(payload(s)),p['full_bytes'])
                except BaseException:save(OUT/'partial_save_error.json',dict(error=traceback.format_exc()))
            save(OUT/'partial_case.json',dict(role=phase,actual_physics=s.steps,recorded_physics=sum(v[2] for v in s.rows),administrative_censor=True))
        save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=physical,commands=commands))
    finally:
        if s:s.close()
        if w:w.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        result=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-start,physics=physical,commands=commands,native_setup_commands=1,new_conditions=len(rows),exact_replays=len(reps),conditions=rows,Actor_updates=0,optimizer_updates=0,backward_calls=0,parent_model_sha256=sha(PARENT),frozen_policy_hash=q.phash(),all_workers_stopped=True,all_native_sessions_closed=True,final_model_created=False,old275=None,main54=None,fresh16=None,reload78=None,full_six_passed=False,seven_eight_entered=False,historical_controls_reused=p['historical_controls'])
        save(OUT/'summary.json',result);save(OUT/'status.json',dict(state=status,updated_epoch=time.time(),physics=physical,commands=commands,new_conditions=len(rows),exact_replays=len(reps),Actor_updates=0));N.manifest(OUT)
    assert status=='COMPLETE'

def audit():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);freeze_optimizer();q=actor(base,p)
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    assert len(cases)==len(reps)==2 and [v['teacher_holds'] for v in cases]==p['teacher_holds']
    for row,rep in zip(cases,reps):
        assert sha(row['path'])==row['sha256'];a=N.S.arrays(row['path']);audit_actions(q,a,p,row['teacher_holds'])
        assert digest(a)==row['array_digest']==rep['array_digest']
        assert row['handoff']==handoff_metrics(a,267+row['teacher_holds'])==rep['handoff']
        assert row['handoff_sha256']==rep['handoff_sha256'] and row['source_snapshot_sha256']==p['source_snapshot_sha256']
        assert int(a['native_counts'].sum())==row['physics_steps']
        assert bool(a['end_flags'][-1,1])==row['success'] and bool(a['end_flags'][:,0].any())==row['collision']
    assert s['physics']==sum(v['physics_steps'] for v in [*cases,*reps])<=p['max_physics'] and s['commands']==len(cases)+len(reps)+1
    assert s['commands']+2<=p['max_commands'] and s['Actor_updates']==s['optimizer_updates']==s['backward_calls']==0
    for v in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/v['path'])==v['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,optimizer=0,backward=0,all_actual_prefix_and_teacher_actions_exact=True,causal_hidden_history_rewards_recomputed=True,all_serialized_model_native_repeats_exact=True,physics=s['physics'],commands=s['commands'],frozen_policy_hash=q.phash(),full_six_passed=False));N.manifest(OUT);print('T397_ZERO_NATIVE_ZERO_BACKWARD_AUDIT_PASS')
if __name__=='__main__':globals()[sys.argv[1]]()
