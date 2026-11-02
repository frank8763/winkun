"""Same full native state, one intervened hold, same frozen continuation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t294_paired_action_value';DOC=ROOT/'stage/docs/distance_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t287_time_aligned_credit'))
from t287_credit import *
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
from t294_native import generate,LimitedLog
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
sys.path.insert(0,str(ROOT/'stage/experiments/t291_hard_guard_ablation'))
from t291_codec import pack_full,unpack_full
import builtins,contextlib,copy,io,json,multiprocessing as mp,os,shutil,signal,symtable,time,traceback


def clean(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(f,v):Path(f).write_text(json.dumps(v,indent=2,default=clean,allow_nan=False)+'\n')
def load_arrays(f):
    with np.load(f) as z:return unpack_full({k:z[k].copy() for k in z.files})
def verify(p):
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert S.used(OUT)<p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def preflight():
    p=json.loads((HERE/'protocol.json').read_text());assert not any((OUT/n).exists() for n in ['status.json','preflight.json']);verify(p);torch.set_num_threads(1)
    q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
    for source in p['sources']:
        a=load_arrays(source['initial_payload']);x=a['states'][0];assert S.digest(x)==source['job']['expected_initial120_sha256']
        with torch.no_grad():_,mean,_,_=q.step(torch.tensor(x[None]))
        expected={name:L.AR.sample(mean[0],np.array([0.,z])).numpy().tolist() for name,z in [('minus',-1),('zero',0),('plus',1)]}
        assert expected==source['first_actions']
        values=[expected[k] for k in ['minus','zero','plus']];assert len({v[0] for v in values})==1 and values[0][1]<values[1][1]<values[2][1]
        assert source['parent']['policy_hash']==p['policy_hash'] and source['parent']['complete']
    tree=symtable.symtable(Path(__file__).read_text(),str(__file__),'exec');missing=set()
    def walk(t):
        for s in t.get_symbols():
            if s.is_referenced() and s.is_global() and s.get_name() not in globals() and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
        for c in t.get_children():walk(c)
    walk(tree);assert not missing,missing
    assert p['max_commands']==16 and p['max_physics']==448000 and p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    save(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,same120_initial_and_policy_SHA_verified=True,first_drive_identical_only_steer_varies_in_three_quantiles=True,all_names_and_budget_verified=True))
    print('T294 preflight PASS zero native/optimizer/Actor')


def main():
    assert not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());start=time.time();counts=mp.Array('q',1);meter=None;raw=[];replays=[];state='COMPLETE';log=None
    def status(phase):save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),absolute_deadline_epoch=start+p['wall_seconds'],commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),optimizer_calls=0,Actor_updates=0))
    def alarm(signum,frame):raise TimeoutError('T294 absolute wall budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    try:
        verify(p);meter=Budget(ROOT,OUT,p,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));status('STARTED');(OUT/'native').mkdir();log=LimitedLog(OUT/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(OUT/'native')
        for source in p['sources']:
            for arm in ['parent','minus','zero','plus']:
                original=None
                for replay in [False,True]:
                    job=dict(source['job'],id=source['job']['id']+'_'+arm+('_replay' if replay else ''),first_action=source['first_actions'].get(arm),mode='deterministic',aux_eta=0.)
                    meter.begin(job['case']['step_cap']);status('NATIVE_'+job['id']);before=int(counts[0])
                    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):item=generate(job,q,native,create,start+p['wall_seconds']-30,counts,0,before)
                    meter.physics=int(counts[0]);row=item['row'];row.update(arm=arm,replay=replay,policy_hash=p['policy_hash'],not_on_policy_training_data=True,fixed_deterministic_continuation=True)
                    complete=row['complete'] and not row['error']
                    if not replay or not complete:
                        a=pack_full(item['arrays']) if not replay else {k:item['arrays'][k] for k in ['actions','native_counts','end_flags','rewards','environment_rewards','aux_costs']}
                        buf=io.BytesIO();np.savez_compressed(buf,**a);b=buf.getvalue();assert len(b)<=p['episode_bytes'] if not replay else len(b)<=p['partial_replay_bytes']
                        f=OUT/(job['id']+'.npz');assert not f.exists();meter.charge_payload(len(b));f.write_bytes(b);row.update(path=str(f),sha256=C.sha(f),saved_array_keys=list(a));raw.append(row);save(OUT/'cases.json',raw)
                    else:row['original_id']=original['row']['id'];replays.append(row);save(OUT/'replays.json',replays)
                    assert complete,'Partial raw retained; no restart'
                    if replay:assert row['array_hashes']==original['row']['array_hashes'] and row['integration_sha256']==original['row']['integration_sha256'] and all(np.array_equal(v,item['arrays'][k]) for k,v in original['arrays'].items())
                    else:
                        original=item
                        if arm=='parent':assert row['array_hashes']==source['parent']['array_hashes'] and row['integration_sha256']==source['parent']['integration_sha256']
                    meter.checkpoint()
        assert len(raw)==len(replays)==8 and sum(r['physics_steps'] for r in raw+replays)==int(counts[0])<=p['max_physics'];verify(p)
    except BaseException:
        state='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=int(counts[0]),commands=0 if meter is None else meter.commands))
    finally:
        signal.alarm(0)
        if log is not None:log.close()
        if meter is not None:meter.close()
    result=dict(state=state,commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),optimizer_calls=0,Actor_updates=0,all_workers_stopped=True,
        unique_cases=len(raw),exact_replays=len(replays),elapsed_s=time.time()-start,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),automatic_next_training=False,seven_or_eight_started=False,
        limitation='Two fixed new1m reset states under original6mgoal. One primitive100nativehold intervention and same frozen safeT255 deterministic continuation, not ARonpolicy or trained navigation. Quantile three actions share identical drive and only steer changes; parentmode control has different drive than truncated distribution median so not puresteer comparison. Exactreplays not independent seeds. No full275/54/fresh16/78, no promotion, no sceneinfeasibility or general actionranking proof.')
    save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result),flush=True)


def audit():
    dest=DOC/'T294_AUDIT.json';assert not dest.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE';verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused PID',pid))
    q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);rows=json.loads((OUT/'cases.json').read_text());rr=json.loads((OUT/'replays.json').read_text());assert len(rows)==len(rr)==8 and len(rows)+len(rr)==s['commands']==16
    assert sum(r['physics_steps'] for r in rows+rr)==s['physics_steps']<=p['max_physics'];by={};details=[]
    for r in rows:
        assert r['complete'] and not r['error'] and C.sha(r['path'])==r['sha256'];a=load_arrays(r['path']);assert len(a)==12 and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items()) and np.array_equal(a['states'][1:],a['next_states'][:-1])
        source=next(v for v in p['sources'] if v['job']['id']==r['source_reset_id']);assert S.digest(a['states'][0])==source['job']['expected_initial120_sha256']
        with torch.no_grad():expected=np.concatenate([q.step(torch.tensor(x[None]))[0].numpy() for x in a['states']])
        if r['arm']=='parent':assert np.array_equal(expected,a['actions']) and r['array_hashes']==source['parent']['array_hashes'] and r['integration_sha256']==source['parent']['integration_sha256']
        else:assert np.array_equal(expected[1:],a['actions'][1:]) and np.array_equal(a['actions'][0],np.array(source['first_actions'][r['arm']],dtype=np.float32)) and int(a['native_counts'][0])==100
        original_G=float(S.monte_carlo(a['environment_rewards'],a['native_counts'])[0]);cost=float(np.dot(R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]],a['aux_costs']))
        desc=dict(task=r['source_reset_id'],arm=r['arm'],first_action=a['actions'][0].tolist(),safe=r['success'] and not r['collision'],collision=r['collision'],timeout=r['timeout'],physics_steps=r['physics_steps'],remaining_m=r['remaining_m'],environment_G0=original_G,bounded_endpoint_cost=cost,eta01_G0=original_G-.1*cost)
        details.append(desc);by[r['id']]=r
    for r in rr:assert r['complete'] and not r['error'] and r['array_hashes']==by[r['original_id']]['array_hashes'] and r['integration_sha256']==by[r['original_id']]['integration_sha256']
    contrasts=[]
    for source in p['sources']:
        vals={v['arm']:v for v in details if v['task']==source['job']['id']};m,z,a=[vals[n] for n in ['minus','zero','plus']]
        assert m['first_action'][0]==z['first_action'][0]==a['first_action'][0]
        contrasts.append(dict(task=source['job']['id'],plus_minus_G0_difference=a['eta01_G0']-m['eta01_G0'],plus_minus_steer_difference=a['first_action'][1]-m['first_action'][1],finite_difference_eta01_return_per_native_steer=(a['eta01_G0']-m['eta01_G0'])/(a['first_action'][1]-m['first_action'][1]),all_three_quantiles_safe=all(vals[n]['safe'] for n in ['minus','zero','plus'])))
    save(DOC/'T294_FIXED_CONTINUATION_VALUES.json',dict(rows=details,contrasts=contrasts,exact_deterministic_interventions_only=True,no_model_or_policy_update=True))
    save(dest,dict(state='PASS',native_added=0,optimizer_calls=0,Actor_updates=0,all_sources_models_payload_SHA_verified=True,all8_full120_12_arrays_exact_replays=True,all_post_first_actions_match_same_frozen_Actor=True,parent_controls_exact_historical_T287=True,all_original_processes_absent=True,total_commands=16,total_physics=s['physics_steps'],summary_sha256=C.sha(OUT/'summary.json')))
    print('T294 independent audit PASS zero native/optimizer, paired deterministic values recorded')


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    elif sys.argv[1:]==['--audit']:audit()
    else:assert not sys.argv[1:];main()
