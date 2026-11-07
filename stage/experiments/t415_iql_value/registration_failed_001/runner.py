"""Register/launch/finalize one finite offline IQL-value experiment."""
from core import *
import subprocess,signal,datetime,ast

def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t414_dependency_recovery';parent=read(prior/'protocol.json');summary=read(prior/'summary.json')
    assert summary['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS'
    assert read(DOC/'T414_FINAL_REPORT.json')['state']=='COMPLETE_AUDITED_REMOTE_VERIFIED'
    assert not Path('/proc',str(summary['pid'])).exists()
    sources=[]
    def add(path,split,role):sources.append(dict(path=str(path),split=split,role=role,sha256=sha(path)))
    for i in range(5):add(ROOT/f'stage/runs/t366_factual_history/episodes/wide_dev_{i}.npz','train' if i%2==0 else 'validation','frozen_LEFT_success_pose_family_'+str(i))
    add(Path(parent['right_history_file']),'train','verified_RIGHT_success')
    for row in [json.loads(line) for line in (prior/'cases.jsonl').read_text().splitlines()]:
        if row['role'].startswith('teacher_'):
            assert row['full'] and row['success'];index=int(row['role'].split('_')[2])
            add(Path(row['path']),'train' if index%2==0 else 'validation','mixed_control_actual_executed_only_'+str(index))
    for row in [json.loads(line) for line in (ROOT/'stage/runs/t410_day_offline_chain/recovery/cases.jsonl').read_text().splitlines()]:
        if row['role'].startswith('train_'):
            assert row['full'];round_index=int(row['role'].split('_')[1]);add(Path(row['path']),'validation' if round_index==2 else 'train','failed_student_round_'+str(round_index))
    assert len(sources)==22 and sum(x['split']=='train' for x in sources)==14
    source_files=set(parent['inputs'])|{s['path'] for s in sources}|{str(f) for f in HERE.glob('*.py')}
    source_files.update([str(prior/'protocol.json'),str(prior/'summary.json'),str(prior/'audit.json'),str(DOC/'T415_IQL_SOURCE_REVIEW.md'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T414.json')])
    p=dict(id='T415_PHYSICAL_TIME_IQL_VALUE_ONLY',sources=sources,actor_protocol=parent,frozen_actor_policy_hash=summary['result']['model']['policy_hash'],
        seed=20261107415,steps=3000,batch_size=256,lr=.0003,expectile=.7,target_tau=.005,inverse_temperature=3.,weight_cap=100.,
        own_bytes=6*2**20,model_bytes=4*2**20,main_seconds=900,audit_seconds=900,backup_seconds=300,all_seconds=2400,
        learning_stop_absolute=1794051000.,all_stop_absolute=1794052800.,native=0,Actor_steps=0,
        consumed_features='120 current causal normalized fields +16 frozen recurrent state after current observation; no pose/goaltruth/map features',
        reward='unchanged original physical interval reward; gamma_native**actual_hold; true GOAL/CONTACT/80sfinitecap terminal. No administrative censor transitions.',
        scope='IQL Q/V offline fitting only; no Actor extraction, autonomous navigation, new physics or six-metre acceptance in this pilot',
        data_split='Whole episodes: LEFT oddposefamilies1/3 heldout; lastfailed noise round2 heldout. Not novelmap or disjointstate proof; common prefixes possible; exacttrain/valduplicatedUID forbidden.',
        expected_train_episodes=14,expected_validation_episodes=8,head_updates_live=6000,head_updates_independent_discarded=6000,preflight_head_Adam=8,
        inputs={str(f):sha(f) for f in sorted(source_files)})
    assert time.time()+p['all_seconds']<p['all_stop_absolute']
    assert E.N.K.S.used(ROOT/'stage/runs')+p['own_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    d,checks=data(p);assert len(d['train']['episodes'])==14 and len(d['validation']['episodes'])==8
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T415_BUDGET.json',dict(state='REGISTERED_ZERO_NATIVE_ZERO_ACTOR',own_bytes=p['own_bytes'],global_reserved=E.N.K.S.used(ROOT/'stage/runs')+p['own_bytes'],live_Q=3000,live_V=3000,audit_Q=3000,audit_V=3000,preflight_discarded_head_Adam=8,Actor=0,native=0,all_seconds=2400,source_episodes=22,train_decisions=len(d['train']['x']),validation_decisions=len(d['validation']['x'])))
    print('T415_REGISTERED_NO_ACTOR',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);d,checks=data(p)
    short=dict(p,steps=2);one,s1=fit(short,d,lambda:None,lambda *_:None);two,s2=fit(short,d,lambda:None,lambda *_:None)
    assert one==two and {k:digest(v) for k,v in s1.items()}=={k:digest(v) for k,v in s2.items()}
    gamma=E.N.K.R.GAMMA
    assert abs(gamma**100*gamma**37-gamma**137)<1e-14
    assert torch.equal(torch.tensor([.25,.75]),torch.where(torch.tensor([-1.,1.])>=0,.75,.25))
    for split in d.values():
        for episode in split['episodes']:assert episode['discount'][-1]==0 and torch.isfinite(episode['x']).all()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Actor_Adam=0,discarded_head_Adam=8,
        QV_and_target_Adam_exact_clone_reconstruction=True,whole_episode_split_no_exact_cross_duplicate=True,
        original_reward_and_physical_time_finite_deadline_contract=True,only_actual_executed_action_pairs=True,
        actual_features_and_all_transitive_data_SHA=True,synthetic_discount_semigroup_and_expectile_sign=True))
    print('T415_PREFLIGHT_PASS_0NATIVE_0ACTOR_8DISCARDED_HEAD_ADAM',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==tree
    now=time.time();assert now+p['all_seconds']<p['all_stop_absolute'] and now+p['main_seconds']<p['learning_stop_absolute']
    with (DOC/'T415_CONSOLE.log').open('xb') as log:
        process=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T415','--why=Bounded offline IQL value test',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'runner.py'),'driver'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=process.pid,epoch=now,deadline=now+p['all_seconds'],source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),Actor=0,native=0))
    print('T415_ACTUALLY_LAUNCHED',process.pid,flush=True)

def driver():
    p=read(OUT/'protocol.json');launch=read(OUT/'launch.json');verify(p);deadline=launch['deadline'];start=time.time();active=None;state='FAILED'
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline))
    def stop(*args):
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        if args:raise TimeoutError('T415 absolute stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(deadline-time.time()-2)))
    try:
        for file,arg,seconds in [('core.py','run',p['main_seconds']),('core.py','audit',p['audit_seconds']),('transport.py','T415',p['backup_seconds'])]:
            with (DOC/('T415_'+arg+'.log')).open('xb') as log:
                active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),arg],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=active.wait(timeout=min(seconds,deadline-time.time()-5));assert code==0,(file,arg,code);active=None
        assert read(OUT/'audit.json')['state']=='PASS'
        assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T415.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        state='COMPLETE_AUDITED_REMOTE_VERIFIED'
    except BaseException:
        stop();save(DOC/'T415_DRIVER_ERROR.json',dict(error=traceback.format_exc(),native=0,Actor_updates=0,no_restart=True))
    finally:
        stop();signal.alarm(0);save(DOC/'T415_FINAL_REPORT.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,deadline=deadline,native=0,Actor_updates=0,no_six_navigation_acceptance=True))
    print('T415_DRIVER_FINISHED',state,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
