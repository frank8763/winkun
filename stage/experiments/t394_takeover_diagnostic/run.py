"""12 independent suffixes, 12 exact native repeats, no Actor update."""
from t394_contract import *

def main():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);freeze_optimizer()
    start=time.time();deadline=start+p['main_seconds'];physical=commands=0;last=0.;phase='SETUP';rows=[];reps=[];groups=[];state='FAILED';w=s=None
    q=actor(base,p);fixed=digest(q.state_dict());reference=N.S.arrays(REFERENCE)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT);assert outside+p['final_bytes']<=1024**3
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=deadline))
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_in_main=True))
    def guard(physical=False):
        nonlocal last
        C.check_stop();assert time.time()<deadline-5 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)==outside
            save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=count(),commands=commands,unique=len(rows),exact=len(reps),new_Actor_updates=0,deadline=deadline))
    def count():return physical
    def charge():
        nonlocal physical
        physical+=1;assert physical<=p['max_physics']
    def execute(first,peer,exact=None):
        nonlocal s,commands,phase
        assert time.time()+40<deadline and physical+40000<=p['max_physics']
        commands+=1;phase=f'PREFIX_{first}_PEER_{peer}_'+('NEW' if exact is None else 'EXACT');guard()
        m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=p['shift_x_m']
        # The repetition reloads the immutable serialized final model rather
        # than relying on in-memory weights; it is still not a new sample.
        if exact is not None:q.load_learned(N.S.read_model(PARENT))
        assert q.phash()==p['parent_policy_hash'];q.sample=False
        q.rng=np.random.default_rng(np.random.SeedSequence([p['seed'],first,peer]));q.reset_noise()
        s=C.Session(p['job'],native,create,w,guard,charge,q);C.install_stop_handlers()
        for k in range(first):
            s.step(forced=reference['actions'][k],owner=1)
            for j,name in enumerate(N.R.NAMES):assert equal(np.asarray(s.rows[-1][j]),reference[name][k]),(k,name)
            assert not s.done()
        assert s.steps==100*first and not s.done()
        takeover=digest(dict(packet=s.packet(),history=w.get(s.env),integration=N.K.R.integration(s.env),hidden=s.hidden))
        q.sample=True
        while not s.done():s.step(owner=0)
        a=payload(s);r=s.result();assert r['complete'] and not r['administrative_censor']
        row=dict(r,prefix_holds=first,prefix_seconds=first*.2,peer=peer,role=phase,seed=[p['seed'],first,peer],
            takeover_sha256=takeover,array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),
            policy_hash=q.phash(),measures=measures(a,first),independent_sample=exact is None)
        if exact is None:
            path=OUT/'episodes'/f'prefix{first}_peer{peer}.npz';N.write_npz(path,N.S.packed(a),p['full_bytes'])
            row.update(path=str(path),sha256=sha(path));rows.append(row);file='cases.jsonl'
        else:
            assert row['array_digest']==exact['array_digest'] and row['integration_sha256']==exact['integration_sha256']
            assert row['takeover_sha256']==exact['takeover_sha256'] and row['measures']==exact['measures']
            row['exact']=True;reps.append(row);file='replays.jsonl'
        with (OUT/file).open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
        assert digest(q.state_dict())==fixed;s.close();s=None;q.sample=False;guard()
        return row
    def expired(*_):raise C.StageStop('T394 wall deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(p['main_seconds']-2)
    try:
        m,native,create,w,_=N.S.P.C.B.setup(OUT/'native',base);C.install_stop_handlers();original=copy.deepcopy(m.ACTIVE);commands+=1
        for first in p['prefix_holds']:
            items=[]
            for peer in range(4):
                row=execute(first,peer);items.append(row);execute(first,peer,row)
            values=np.asarray([v['measures']['return_G0'] for v in items]);adv=values-(values.sum()-values)/3
            group=dict(prefix_holds=first,prefix_seconds=first*.2,valid=4,success=sum(v['success'] for v in items),
                contact=sum(v['collision'] for v in items),timeout=sum(v['timeout'] for v in items),invalid=0,censored=0,
                return_values=values.tolist(),return_spread=float(np.ptp(values)),max_abs_LOO_advantage=float(np.abs(adv).max()),
                success_and_failure_both_observed=0<sum(v['success'] for v in items)<4,
                deterministic_mode_evaluation=None,not_full_start_navigation=True)
            groups.append(group);save(OUT/'groups.json',groups)
        state='COMPLETE'
    except BaseException:
        if s is not None:
            if s.rows:
                try:N.write_npz(OUT/'partial.npz',N.S.packed(payload(s)),p['full_bytes'])
                except BaseException:save(OUT/'partial_save_error.json',dict(error=traceback.format_exc()))
            save(OUT/'partial_case.json',dict(role=phase,actual_physics=s.steps,recorded_physics=sum(v[2] for v in s.rows),administrative_censor=True))
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,actual_physics=physical,commands=commands))
    finally:
        if s is not None:s.close()
        if w is not None:w.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=physical,commands=commands,native_setup_commands=1,
            unique=len(rows),exact_replays=len(reps),groups=groups,Actor_updates=0,optimizer_updates=0,backward_calls=0,
            parent_model_sha256=sha(PARENT),frozen_policy_hash=q.phash(),all_native_sessions_closed=True,all_workers_stopped=True,
            final_model_created=False,old275=None,main54=None,fresh16=None,reload78=None,full_six_passed=False,seven_eight_entered=False))
        save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),physics=physical,commands=commands,unique=len(rows),exact=len(reps),new_Actor_updates=0))
        N.manifest(OUT)
    assert state=='COMPLETE'

def audit():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);freeze_optimizer();q=actor(base,p)
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()]
    assert len(rows)==len(reps)==12
    for row,rep in zip(rows,reps):
        assert sha(row['path'])==row['sha256'];a=N.S.arrays(row['path']);first=row['prefix_holds']
        assert digest(a)==row['array_digest']==rep['array_digest'] and row['takeover_sha256']==rep['takeover_sha256']
        assert int(a['native_counts'].sum())==row['physics_steps']
        C.audit_training(q,a,first,row['seed'],p)
        for k,xrow in enumerate(a['states']):
            rebuilt=N.K.H.rebuild(xrow,a['poses'][k],a['history_poses'][k],a['task_goal_xy'],a['history_goals'][k])
            assert np.abs(rebuilt-xrow).max()<=2e-6
        with torch.no_grad():
            x=torch.tensor(a['states']);loc=q.sequence(x,q.base_actions(x))
            assert float((loc-torch.tensor(a['locations'])).abs().max())<1e-10
            assert float((logp(loc,torch.tensor(a['raw_actions']),torch.tensor(a['atom']),p)[first:]-torch.tensor(a['logp'])[first:]).abs().max())<1e-9
        assert measures(a,first)==row['measures']
        assert bool(a['end_flags'][-1,1])==row['success'] and bool(a['end_flags'][:,0].any())==row['collision']
        assert row['measures']==rep['measures']
    for g in s['groups']:
        selected=[v for v in rows if v['prefix_holds']==g['prefix_holds']];assert len(selected)==4
        returns=np.asarray([v['measures']['return_G0'] for v in selected]);adv=returns-(returns.sum()-returns)/3
        assert g['valid']==4 and g['success']==sum(v['success'] for v in selected)
        assert g['contact']==sum(v['collision'] for v in selected) and g['timeout']==sum(v['timeout'] for v in selected)
        assert g['return_values']==returns.tolist() and g['return_spread']==float(np.ptp(returns))
        assert g['max_abs_LOO_advantage']==float(np.abs(adv).max()) and g['deterministic_mode_evaluation'] is None
    assert s['physics']==sum(v['physics_steps'] for v in [*rows,*reps]) and s['commands']==len(rows)+len(reps)+1
    assert s['Actor_updates']==s['optimizer_updates']==s['backward_calls']==0
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,optimizer_updates=0,backward=0,
        independent_raw_Actor_noise_history_density_reward_recompute=True,all_array_and_ledger_SHA_exact=True,
        physics=s['physics'],commands=s['commands'],frozen_policy_hash=q.phash(),exploration_only_not_stable_six=True))
    N.manifest(OUT);print('T394_INDEPENDENT_ZERO_NATIVE_ZERO_OPTIMIZER_AUDIT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
