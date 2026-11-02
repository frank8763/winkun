"""Frozen new-task test of all four already-rejected T287 candidates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t289_rejected_candidates'
sys.path.insert(0,str(ROOT/'stage/experiments/t282_capped_balance'))
from balanced_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t287_time_aligned_credit'))
from t287_credit import *
from t280_worker import generate,LimitedLog
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
import builtins,contextlib,copy,gzip,io,json,multiprocessing as mp,os,shutil,signal,symtable,time,traceback


def blob(path,b,limit,meter):
    assert not path.exists() and len(b)<=limit,(str(path),len(b),limit)
    meter.charge_payload(len(b));path.write_bytes(b)
    return dict(path=str(path),bytes=len(b),sha256=C.sha(path))


def load_arrays(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def reconstruct(plan,guard,opt_event):
    base=json.loads(Path(plan['source_protocol']).read_text());rows=[json.loads(l) for l in Path(plan['source_cases']).read_text().splitlines()]
    updates=[json.loads(l) for l in Path(plan['source_updates']).read_text().splitlines()];cache=json.loads(Path(plan['source_cache']).read_text())['cache']
    parent=policy(base).eval().requires_grad_(False);x,y,offset=anchors(base,parent);captured={};reports=[]
    for old in updates:
        rnd=old['round'];q=policy(base);opt=L.optimizer(q,base);shadow=copy.deepcopy(q);so=L.optimizer(shadow,base)
        def wrap(optimizer,phase):
            original=optimizer.step
            def step(*args,**kwargs):opt_event(dict(round=rnd,phase=phase));return original(*args,**kwargs)
            optimizer.step=step
        wrap(opt,'historical_live');wrap(so,'historical_shadow');items=[]
        for row in rows:
            if row['role']=='training' and row['id'].startswith(f'time_aligned_credit_r{rnd}_'):
                a=load_arrays(row['path']);assert C.sha(row['path'])==row['sha256'] and all(S.digest(v)==row['array_hashes'][k] for k,v in a.items());items.append(dict(row=row,arrays=a))
        assert len(items)==16;data=L.training_batch(items,q)
        def gate(actor):
            ph=S.digest(actor.state_dict());assert ph in cache
            if ph in plan['candidate_hashes']:
                value=copy.deepcopy(actor.state_dict())
                if ph in captured:assert S.digest(captured[ph])==S.digest(value)
                else:captured[ph]=value
            return cache[ph]
        live,audit=paired_balanced_update(q,opt,shadow,so,data,x,y,offset,base,rnd,guard,gate,lambda phase,event:None)
        assert live==audit==old['updates'] and S.digest(q.state_dict())==old['policy_hash'] and S.digest(opt.state_dict())==old['optimizer_hash']
        reports.append(dict(round=rnd,original_updates_exact=True,Actor_Adam_rollback_exact=True,policy_hash=old['policy_hash'],optimizer_hash=old['optimizer_hash']))
    assert set(captured)==set(plan['candidate_hashes'])
    return captured,reports


def main():
    assert not (OUT/'status.json').exists(),'No restart of started/ended directory'
    p=json.loads((HERE/'protocol.json').read_text());base=json.loads(Path(p['source_protocol']).read_text());start=time.time();counts=mp.Array('q',1);meter=None;raw=[];replays=[];exports={};calls=[];log=None;state='COMPLETE'
    def status(phase):C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,absolute_deadline_epoch=start+p['wall_seconds'],updated_epoch=time.time(),commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),historical_clone_optimizer_calls=len(calls),navigation_training_updates=0))
    def guard():
        assert time.time()<start+p['wall_seconds']
        if meter is not None:meter.checkpoint()
    def opt_event(r):
        calls.append(r);assert len(calls)<=p['max_historical_optimizer_calls'];C.save(OUT/'historical_optimizer_calls.json',calls)
    def alarm(signum,frame):raise TimeoutError('T289 absolute wall deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    try:
        assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
        for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
        meter=Budget(ROOT,OUT,p,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));status('RECONSTRUCTING_FOUR_EXISTING_REJECTED_CANDIDATES')
        captured,reports=reconstruct(p,guard,opt_event);assert len(calls)==4;C.save(OUT/'reconstruction.json',dict(original_two_rounds=reports,historical_optimizer_calls=4,navigation_training_updates=0))
        for spec in p['candidates']:
            ph=spec['policy_hash'];buf=io.BytesIO();torch.save(dict(policy=captured[ph],policy_hash=ph,source_round=spec['round'],source_factor=spec['factor'],historically_rejected=True,optimizer_not_resumed=True),buf)
            exports[spec['id']]=blob(OUT/(spec['id']+'.pt.gz'),gzip.compress(buf.getvalue(),compresslevel=6,mtime=0),p['model_bytes'],meter)
            assert S.digest(L.read(exports[spec['id']]['path'])['policy'])==ph
        del captured
        (OUT/'native').mkdir(exist_ok=True);log=LimitedLog(OUT/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(OUT/'native')
        for spec in p['candidates']:
            actor=policy(base);actor.load_state_dict(L.read(exports[spec['id']]['path'])['policy']);actor.eval().requires_grad_(False)
            for source in p['jobs']:
                original=None
                for replay in [False,True]:
                    job=dict(source,id=spec['id']+'_'+source['id']+('_replay' if replay else ''),source_reset_id=source['id'],mode='deterministic',seed=0,aux_eta=0.)
                    meter.begin(job['case']['step_cap']);status('NATIVE_'+job['id']);before=int(counts[0])
                    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):item=generate(job,actor,native,create,start+p['wall_seconds']-30,counts,0,before)
                    meter.physics=int(counts[0]);row=item['row'];row.update(candidate_id=spec['id'],policy_hash=spec['policy_hash'],replay=replay)
                    complete=row['complete'] and not row['error']
                    if not replay or not complete:
                        a=item['arrays'] if not replay else {k:item['arrays'][k] for k in ['actions','native_counts','end_flags','rewards','environment_rewards','aux_costs']}
                        buf=io.BytesIO();np.savez_compressed(buf,**a);row.update(blob(OUT/(job['id']+'.npz'),buf.getvalue(),p['episode_bytes'] if not replay else p['partial_replay_bytes'],meter));row['saved_array_keys']=list(a);raw.append(row);C.save(OUT/'cases.json',raw)
                    else:
                        row['original_id']=original['row']['id'];replays.append(row);C.save(OUT/'replays.json',replays)
                    assert complete,'Partial/error retained; no continuation'
                    if replay:
                        assert row['array_hashes']==original['row']['array_hashes'] and row['integration_sha256']==original['row']['integration_sha256']
                        assert all(np.array_equal(v,item['arrays'][k]) for k,v in original['arrays'].items())
                    else:original=item
                    guard()
        assert len(raw)==len(replays)==16 and sum(z['physics_steps'] for z in raw+replays)==int(counts[0])<=p['max_physics']
        for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    except BaseException:
        state='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),commands=0 if meter is None else meter.commands,physics_steps=int(counts[0])))
    finally:
        signal.alarm(0)
        if log is not None:log.close()
        if meter is not None:meter.close()
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    by={}
    for spec in p['candidates']:
        rr=[r for r in raw if r['candidate_id']==spec['id'] and not r['replay']];by[spec['id']]=dict(cases=len(rr),safe=sum(safe(r) for r in rr),contact=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),rows=[dict(task=r['source_reset_id'],success=r['success'],contact=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],physics_steps=r['physics_steps']) for r in rr])
    s=dict(state=state,commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),historical_clone_optimizer_calls=len(calls),navigation_training_updates=0,models=exports,candidates=by,
        all_workers_stopped=True,elapsed_s=time.time()-start,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),full_275_54_16_78_not_executed=True,model_promoted=False,seven_or_eight_started=False,
        limitation='All four pre-existing T287 numeric-pass old-guard-rejected candidates selected before this test. Reconstructed old optimizer steps are not new learning. Four fixed center/reset tasks and exact replay do not establish full six-metre stability. Historical parent control reused explicitly. No new seed/step-factor/algorithm grid.')
    C.save(OUT/'summary.json',s);C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)


def preflight():
    p=json.loads((HERE/'protocol.json').read_text());assert not any((OUT/f).exists() for f in ['status.json','preflight.json'])
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    t=symtable.symtable(Path(__file__).read_text(),str(__file__),'exec');missing=set()
    def walk(t):
        for s in t.get_symbols():
            if s.is_referenced() and s.is_global() and s.get_name() not in globals() and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
        for c in t.get_children():walk(c)
    walk(t);assert not missing,missing
    assert len(p['candidates'])==len(set(p['candidate_hashes']))==4 and len(p['jobs'])==4
    assert p['max_commands']==32 and sum(j['case']['step_cap'] for j in p['jobs'])*8<=p['max_physics']
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    used=S.used(ROOT/'stage/runs');assert used-S.used(OUT)+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    base=json.loads(Path(p['source_protocol']).read_text());actor=policy(base);assert S.digest(actor.state_dict())==base['policy_hash']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,native_environments_created=0,all_global_names_resolve=True,
        exact_four_original_candidate_set=True,source_model_SHA_verified=True,global_if_full=used-S.used(OUT)+p['final_bytes'],protocol_sha256=C.sha(HERE/'protocol.json')))
    print('T289 PREFLIGHT PASS: zero native/optimizer; four existing candidates and fixed scope locked')


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    else:assert not sys.argv[1:];main()
