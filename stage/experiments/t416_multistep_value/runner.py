"""Register/launch/finalize one finite offline IQL-value experiment."""
from core import *
import core as WINDOW_CORE
from core import C
copy=C.copy
import subprocess,signal,datetime,ast,time,json,os,shutil,traceback

def prepare():
    assert not OUT.exists()
    original=read(ROOT/'stage/runs/t415_iql_value/protocol.json')
    assert read(ROOT/'stage/runs/t415_iql_value/audit.json')['state']=='PASS'
    assert read(DOC/'T415_FINAL_REPORT.json')['state']=='COMPLETE_AUDITED_REMOTE_VERIFIED'
    p=copy.deepcopy(original);p['id']='T416_FIXED20HOLD_BEHAVIOR_CONTINUATION_VALUE';p['n_step']=20
    p['scope']='Value-only fixed20hold actualbehavior continuation target; no importance correction, no unbiasedoptimalQ or originalIQL claim; 0Actor/0native'
    p['inputs'].update({str(f):sha(f) for f in HERE.glob('*.py')})
    p['inputs'].update({str(ROOT/'stage/runs/t415_iql_value/protocol.json'):sha(ROOT/'stage/runs/t415_iql_value/protocol.json'),str(DOC/'T415_FINDINGS.json'):sha(DOC/'T415_FINDINGS.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T415.json'):sha(ROOT/'stage/docs/long_distance_20261101/release_receipt_T415.json')})
    assert time.time()+p['all_seconds']<p['all_stop_absolute'] and E.N.K.S.used(ROOT/'stage/runs')+p['own_bytes']<2**30
    d,checks=data(p);assert len(d['train']['episodes'])==14 and len(d['validation']['episodes'])==8
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T416_BUDGET.json',dict(state='NEW_OPERATOR_SAME_BOUNDED_HEAD_COUNTS',n_step=20,own_bytes=p['own_bytes'],global_reserved=E.N.K.S.used(ROOT/'stage/runs')+p['own_bytes'],live_Q=3000,live_V=3000,audit_Q=3000,audit_V=3000,preflight_discarded_head_Adam=8,Actor=0,native=0,all_seconds=2400))
    print('T416_REGISTERED_NO_ACTOR',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);d,checks=data(p)
    short=dict(p,steps=2);one,s1=fit(short,d,lambda:None,lambda *_:None);two,s2=fit(short,d,lambda:None,lambda *_:None)
    assert one==two and {k:digest(v) for k,v in s1.items()}=={k:digest(v) for k,v in s2.items()}
    gamma=E.N.K.R.GAMMA
    assert abs(gamma**100*gamma**37-gamma**137)<1e-14
    assert torch.equal(torch.tensor([.25,.75]),torch.where(torch.tensor([-1.,1.])>=0,.75,.25))
    rr=torch.tensor([[1.],[2.],[3.]]);dd=torch.tensor([[.5],[.25],[0.]]);nx=torch.tensor([[10.],[20.],[30.]])
    R,D,nnext=WINDOW_CORE.window(rr,dd,nx,2)
    assert torch.equal(R,torch.tensor([[2.],[2.75],[3.]])) and torch.equal(D,torch.tensor([[.125],[0.],[0.]])) and torch.equal(nnext,torch.tensor([[20.],[30.],[30.]]))
    for split in d.values():
        for episode in split['episodes']:assert episode['discount'][-1]==0 and torch.isfinite(episode['x']).all()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Actor_Adam=0,discarded_head_Adam=8,
        QV_and_target_Adam_exact_clone_reconstruction=True,whole_episode_split_no_exact_cross_duplicate=True,
        original_reward_and_physical_time_finite_deadline_contract=True,only_actual_executed_action_pairs=True,
        actual_features_and_all_transitive_data_SHA=True,synthetic_discount_semigroup_and_expectile_sign=True))
    print('T416_PREFLIGHT_PASS_0NATIVE_0ACTOR_8DISCARDED_HEAD_ADAM',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==tree
    now=time.time();assert now+p['all_seconds']<p['all_stop_absolute'] and now+p['main_seconds']<p['learning_stop_absolute']
    with (DOC/'T416_CONSOLE.log').open('xb') as log:
        process=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T416','--why=Bounded offline IQL value test',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'runner.py'),'driver'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=process.pid,epoch=now,deadline=now+p['all_seconds'],source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),Actor=0,native=0))
    print('T416_ACTUALLY_LAUNCHED',process.pid,flush=True)

def driver():
    p=read(OUT/'protocol.json');launch=read(OUT/'launch.json');verify(p);deadline=launch['deadline'];start=time.time();active=None;state='FAILED'
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline))
    def stop(*args):
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        if args:raise TimeoutError('T416 absolute stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(deadline-time.time()-2)))
    try:
        for file,arg,seconds in [('core.py','run',p['main_seconds']),('core.py','audit',p['audit_seconds']),('transport.py','T416',p['backup_seconds'])]:
            with (DOC/('T416_'+arg+'.log')).open('xb') as log:
                active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),arg],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=active.wait(timeout=min(seconds,deadline-time.time()-5));assert code==0,(file,arg,code);active=None
        assert read(OUT/'audit.json')['state']=='PASS'
        assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T416.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        state='COMPLETE_AUDITED_REMOTE_VERIFIED'
    except BaseException:
        stop();save(DOC/'T416_DRIVER_ERROR.json',dict(error=traceback.format_exc(),native=0,Actor_updates=0,no_restart=True))
    finally:
        stop();signal.alarm(0);save(DOC/'T416_FINAL_REPORT.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,deadline=deadline,native=0,Actor_updates=0,no_six_navigation_acceptance=True))
    print('T416_DRIVER_FINISHED',state,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
