"""Recover existing RIGHT6m success full120 data, not new navigation skill."""
from pathlib import Path
import sys,json,os,io,time,contextlib,multiprocessing as mp,signal,traceback,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t301_right_success_recovery';DOC=ROOT/'stage/docs/distance_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t298_self_goal_transfer'))
from t298_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
from t280_worker import generate,LimitedLog
sys.path.insert(0,str(ROOT/'stage/experiments/t291_hard_guard_ablation'))
from t291_codec import pack_full,unpack_full
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget

def save(f,x):Path(f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify(p):
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert S.used(OUT)<p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
def actor(p):
    q=policy(json.loads(Path(p['source_protocol']).read_text()));q.load_state_dict(L.read(p['source_checkpoint'])['policy']);return q.eval().requires_grad_(False)

def preflight():
    assert not (OUT/'preflight.json').exists();p=json.loads((HERE/'protocol.json').read_text());verify(p);torch.set_num_threads(1);q=actor(p);assert S.digest(q.state_dict())==p['policy_hash']
    assert len(p['sources'])==10 and len({v['id'] for v in p['sources']})==10
    assert all(s['reference']['complete'] and s['reference']['success'] and not s['reference']['collision'] and s['reference']['policy_hash']==p['policy_hash'] for s in p['sources'])
    assert p['max_commands']==10 and p['max_physics']==280000 and 10*p['episode_bytes']+p['metadata_bytes']<=p['final_bytes']
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    save(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,original10native_success_reference_hashes_fixed=True,source_policy_hash_verified=p['policy_hash'],global_if_own_full=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']));print('T301 preflight PASS 0native0optimizer')

def main():
    assert not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());start=time.time();meter=None;log=None;rows=[];state='COMPLETE';counts=mp.Array('q',1)
    def status(phase):save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),deadline_epoch=start+p['wall_seconds'],commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),optimizer_calls=0))
    def alarm(signum,frame):raise TimeoutError('T301 absolute deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    try:
        verify(p);meter=Budget(ROOT,OUT,p,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=actor(p);assert S.digest(q.state_dict())==p['policy_hash'];status('STARTED');save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));(OUT/'native').mkdir();log=LimitedLog(OUT/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(OUT/'native')
        for source in p['sources']:
            meter.begin(28000);status(source['id']);job=dict(source['job'],id='recover_'+source['id'],source_reset_id=source['id'],mode='deterministic',seed=0,aux_eta=0.)
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):item=generate(job,q,native,create,meter.deadline-30,counts,0,int(counts[0]))
            meter.physics=int(counts[0]);a=item['arrays'];r=item['row'];payload=pack_full(a);buf=io.BytesIO();np.savez_compressed(buf,**payload);b=buf.getvalue();assert len(b)<=p['episode_bytes'];meter.charge_payload(len(b));f=OUT/(source['id']+'.npz');f.write_bytes(b);r.update(path=str(f),sha256=C.sha(f),policy_hash=p['policy_hash'],source_id=source['id'],source_T299_reference=source['reference']['id'],full120_statechain=True,not_new_independent_navigation_data=True);rows.append(r);save(OUT/'cases.json',rows)
            assert r['complete'] and not r['error'],'Partial retained no restart'
            assert r['array_hashes']==source['reference']['array_hashes'] and r['integration_sha256']==source['reference']['integration_sha256'] and r['physics_steps']==source['reference']['physics_steps']
            assert r['success'] and not r['collision'];meter.checkpoint()
        verify(p);assert S.digest(q.state_dict())==p['policy_hash']
    except BaseException:
        state='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=int(counts[0]),commands=0 if meter is None else meter.commands))
    finally:
        signal.alarm(0)
        if log is not None:log.close()
        if meter is not None:meter.close()
    result=dict(state=state,commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),elapsed_s=time.time()-start,optimizer_calls=0,Actor_updates=0,all_workers_stopped=True,full120_cases=len(rows),source_policy_hash=p['policy_hash'],source_checkpoint_sha256=C.sha(p['source_checkpoint']),own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),limitation=p['contract']);save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result))

def audit():
    dest=DOC/'T301_AUDIT.json';assert not dest.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE';verify(p)
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused',pid))
    rr=json.loads((OUT/'cases.json').read_text());assert len(rr)==10 and len(rr)==s['commands'];total=0
    for row,source in zip(rr,p['sources']):
        assert C.sha(row['path'])==row['sha256']
        with np.load(row['path']) as z:a=unpack_full({k:z[k].copy() for k in z.files})
        assert all(S.digest(v)==row['array_hashes'][k] for k,v in a.items());assert row['array_hashes']==source['reference']['array_hashes'] and row['integration_sha256']==source['reference']['integration_sha256'];assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and a['states'].shape[1]==120 and a['next_states'].shape[1]==120
        assert np.array_equal(a['end_flags'][-1],[0,1,0]) and not a['end_flags'][:-1].any() and row['success'] and not row['collision'];assert row['physics_steps']==int(a['native_counts'].astype(np.int64).sum());total+=row['physics_steps']
    assert total==s['physics_steps']<=p['max_physics'];save(dest,dict(state='PASS',native_added=0,optimizer_calls=0,Actor_updates=0,all10full120_12array_SHA_and_integration_exact_T299=True,all_processes_absent=True,commands=10,physics_steps=total,source_policy_unchanged=True,summary_sha256=C.sha(OUT/'summary.json')));print('T301 audit PASS zero native/optimizer')

if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    elif sys.argv[1:]==['--audit']:audit()
    else:assert not sys.argv[1:];main()
