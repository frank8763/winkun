"""Bounded post-shutdown audit of already-recorded T422 optimizer events.

This is not a resumed experiment. Native execution and model export are
forbidden. Only previously logged discarded historical Adam steps are rebuilt;
the incomplete fourth shadow is stopped at its last recorded AFTER event.
"""
from pathlib import Path
import sys,json,time,copy,signal,traceback,hashlib,gzip,io,collections
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
T422=ROOT/'stage/experiments/t422_feedback_recovery_night'
sys.path.insert(0,str(T422))
import audit as A
import t422_common as C
import learn as LEARN
E,np,torch=A.E,A.np,A.torch
OUT=C.OUT;FOLDER=OUT/'learning';DOC=C.DOC
REPORT=DOC/'T422_PARTIAL_OPTIMIZER_AUDIT.json'
MAX_ADAM=1567;MAX_SECONDS=1200

def main():
    assert not REPORT.exists()
    start=time.time();deadline=start+MAX_SECONDS
    p=E.read(OUT/'protocol.json');C.verify(p)
    assert not (FOLDER/'summary.json').exists() and not (FOLDER/'audit.json').exists()
    absent=A.check_native_stopped(FOLDER)+A.check_native_stopped(OUT/'search')
    for name in ['launch.json','driver_started.json']:
        value=E.read(OUT/name)
        for key in ['pid','parent']:
            if value.get(key):assert not Path('/proc',str(value[key])).exists()
    originals={str(f):A.sha(f) for f in OUT.rglob('*') if f.is_file()}
    original_events=A.jsonl(FOLDER/'optimizer_calls.jsonl')
    assert len(original_events)==2*MAX_ADAM
    grouped=collections.Counter((r['tag'],r['which'],r['where']) for r in original_events)
    assert grouped[('BC3','shadow','AFTER')]==151 and original_events[-1]['k']==150
    assert all(grouped[tag,which,'BEFORE']==grouped[tag,which,'AFTER'] for tag,which,_ in grouped)
    E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def no_native(*a,**k):raise AssertionError('Morning audit forbids native simulation or policy export')
    C.setup=no_native;E.L.Session=no_native;E.Pool=no_native;E.export=no_native;E.N.write_npz=no_native
    adam=0;cursor=0;reconstructed=[]
    original_step=torch.optim.Adam.step
    def guard():assert time.time()<deadline-5
    def counted_step(opt,*args,**kwargs):
        nonlocal adam
        guard();assert adam<MAX_ADAM
        result=original_step(opt,*args,**kwargs);adam+=1;return result
    torch.optim.Adam.step=counted_step
    def timeout(signum,frame):raise TimeoutError('Independent partial audit 1200s wall budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(MAX_SECONDS)
    base=E.N.science(p);q,opt=E.learner(base,p);fixed=E.original_hash(q)
    rows=A.jsonl(FOLDER/'cases.jsonl');reps=A.jsonl(FOLDER/'replays.jsonl')
    initial_rows=E.read(FOLDER/'initial_arrays.json');assert len(initial_rows)==14
    assert rows[:14]==initial_rows
    initial_values=[]
    for row in initial_rows:
        guard();assert A.sha(row['path'])==row['sha256'] and row['full']
        a=E.N.S.arrays(row['path']);E.audit_array(q,a,p,0,None)
        assert E.digest(a)==row['array_digest'];initial_values.append((a,row))
    initial_ids={r['job_id'] for r in initial_rows if r['success']};selected=initial_ids.copy()
    assert initial_ids==set(p['initial_expected_success_ids']) and q.phash()==p['initial_expected_policy_hash']
    rounds=E.read(FOLDER/'rounds.json');assert len(rounds)==3
    search=E.read(OUT/'search/summary.json');assert E.read(OUT/'search/audit.json')['state']=='PASS'
    ref=A.REFERENCE.load_reference(p['reference_path'],p['reference_sha256'])
    def callback(tag,which,model,optimizer,stop_after=None):
        def event(where,k):
            nonlocal cursor
            guard();actual=dict(tag=tag,which=which,where=where,k=k,hashes=E.L.statehash(model,optimizer))
            assert cursor<len(original_events) and actual==original_events[cursor],('Event mismatch',cursor,tag,which,where,k)
            reconstructed.append(actual);cursor+=1
            if where=='AFTER' and stop_after is not None and k==stop_after:raise RecordedPrefixEnd()
        return event
    class RecordedPrefixEnd(Exception):pass
    def validate_recovery(a,entry,first,model):
        assert A.sha(entry['path'])==entry['sha256']
        assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        E.audit_array(model,a,p,first,None)
        A.check_feedback(a,ref,first,search['result']['best']['gains'],search['result']['best']['mode'])
        if entry.get('historical_search_reuse'):
            assert entry['path']==search['result']['path'] and entry['sha256']==search['result']['sha256'] and entry['new_native']==0
        else:
            row=entry['row'];A.check_episode(a,row)
            assert row['policy_hash']==model.phash()
            exact=E.read(Path(entry['path']).with_name(Path(entry['path']).stem+'_exact.json'))
            assert exact==dict(array_digest=row['array_digest'],physics=row['physics'],exact=True,independent_sample=False)
    def pair(tag,fit,expected):
        before=E.L.state(q,opt);states=[]
        for which in ['live','shadow']:
            E.L.load(q,opt,before);result=fit(callback(tag,which,q,opt));assert result==expected
            states.append(E.L.state(q,opt))
        assert E.digest(states[0])==E.digest(states[1]);E.L.load(q,opt,states[0])
    retained=[]
    for rnd,record in enumerate(rounds):
        guard();assert E.read(FOLDER/'blocks'/str(rnd)/'result.json')==record
        assert record['round']==rnd and record['selected_before']==sorted(selected)
        A.check_model(q,opt,record['before_model']);before=copy.deepcopy(A.read_snapshot(record['before_model'])['state'])
        first=p['recovery_prefixes'][rnd%len(p['recovery_prefixes'])];entry=record['query'];a=E.N.S.arrays(entry['path'])
        validate_recovery(a,entry,first,q)
        entries=[dict(arrays=E.N.S.arrays(x['path']),first=x['first'],source_id=x['source_id']) for x in record['recovery_sources']]
        for source in record['recovery_sources']:assert A.sha(source['path'])==source['sha256']
        g=LEARN.build_groups(q,initial_values,p,entries,p['query_sources']);bc=record['bc'];assert bc is not None
        pair('BC'+str(rnd),lambda event:LEARN.fit_bc(q,opt,g,p,guard,event,steps=p['bc_steps']),bc['fit'])
        A.check_model(q,opt,bc['proposed_model'])
        _,safe=A.joint_rows(rows,p,'BC'+str(rnd)+'_proposal',q,bc['checks'])
        accepted=initial_ids<=safe and selected<=safe and bool(safe-selected);assert accepted==bc['accepted']
        if accepted:selected=safe
        else:E.L.load(q,opt,before)
        assert E.L.statehash(q,opt)==bc['retained_hashes']
        rl=record['rl']
        if rl is not None:
            assert accepted and bc['checks']['old8']['success']==8
            opt=torch.optim.Adam(E.params(q),lr=p['lr']);A.check_model(q,opt,rl['before_model']);rstate=copy.deepcopy(A.read_snapshot(rl['before_model'])['state'])
            items=[]
            for item in rl['items']:
                assert A.sha(item['path'])==item['sha256'];arr=E.N.S.arrays(item['path']);A.check_episode(arr,item)
                E.audit_array(q,arr,p,0,item['seed']);assert not arr['ownership'].any() and arr['stochastic'].all();items.append(arr)
                repeat=next(r for r in reps if r['role']==item['role']+'_exact' and r['seed']==item['seed'])
                for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:assert repeat[key]==item[key]
                assert repeat['full_array_exact'] and repeat['exact'] and not repeat['independent_sample']
            pair('RL'+str(rnd),lambda event:LEARN.fit_ppo(q,opt,items,g,p,rnd,guard,event),rl['fit'])
            A.check_model(q,opt,rl['proposed_model']);_,safe=A.joint_rows(rows,p,'RL'+str(rnd)+'_proposal',q,rl['checks'])
            accepted=rl['fit']['numeric_accept'] and initial_ids<=safe and selected<=safe;assert accepted==rl['accepted']
            if accepted:selected=safe
            else:E.L.load(q,opt,rstate)
            assert E.L.statehash(q,opt)==rl['retained_hashes']
        assert selected==set(record['selected_after']) and E.original_hash(q)==fixed
        retained.append(dict(round=rnd,hashes=E.L.statehash(q,opt),policy_hash=q.phash(),selected_success_ids=sorted(selected)))
        print('T423_COMPLETED_RECONSTRUCTION_ROUND',rnd,'discarded_Adam',adam,flush=True)
    last_file=FOLDER/'blocks/3/before.pt.gz';last=torch.load(io.BytesIO(gzip.decompress(last_file.read_bytes())),weights_only=False,map_location='cpu')
    assert last['hashes']==E.L.statehash(q,opt) and last['policy_hash']==q.phash()
    before3=copy.deepcopy(last['state']);best_policy=q.phash();best_hashes=copy.deepcopy(last['hashes'])
    row=E.read(FOLDER/'recoveries/03.json');a=E.N.S.arrays(row['path'])
    entry=dict(path=row['path'],sha256=row['sha256'],first=20,source_id=row['role'],row=row,exact=True,uid=[q.phash(),20,E.digest(a)])
    validate_recovery(a,entry,20,q)
    recovery_sources=copy.deepcopy(rounds[2]['recovery_sources'])
    if not any(x['uid']==entry['uid'] for x in recovery_sources):recovery_sources.append(entry)
    entries=[dict(arrays=E.N.S.arrays(x['path']),first=x['first'],source_id=x['source_id']) for x in recovery_sources]
    g=LEARN.build_groups(q,initial_values,p,entries,p['query_sources'])
    full=LEARN.fit_bc(q,opt,g,p,guard,callback('BC3','live',q,opt),steps=p['bc_steps'])
    live3=E.L.statehash(q,opt);assert full['steps']==200
    E.L.load(q,opt,before3)
    try:
        LEARN.fit_bc(q,opt,g,p,guard,callback('BC3','shadow',q,opt,stop_after=150),steps=p['bc_steps'])
        raise AssertionError('Partial audit ran beyond original shadow event prefix')
    except RecordedPrefixEnd:pass
    assert cursor==len(original_events) and reconstructed==original_events and adam==MAX_ADAM
    shadow3=E.L.statehash(q,opt)
    assert not (FOLDER/'blocks/3/bc_proposed.pt.gz').exists() and not (FOLDER/'blocks/3/result.json').exists()
    E.L.load(q,opt,before3);assert E.L.statehash(q,opt)==best_hashes and q.phash()==best_policy
    assert originals=={str(f):A.sha(f) for f in OUT.rglob('*') if f.is_file()},'Original T422 evidence changed during independent audit'
    guard();signal.alarm(0)
    findings=dict(state='PASS_FOR_RECORDED_PARTIAL_HISTORY_NOT_FULL_OVERNIGHT_ACCEPTANCE',elapsed_s=time.time()-start,wall_limit_s=MAX_SECONDS,native=0,new_navigation_training=0,new_retained_policy=0,discarded_historical_Adam=dict(live=808,shadow=759,total=adam),recorded_optimizer_event_count=cursor,all_recorded_BEFORE_AFTER_complete_Actor_Adam_hashes_exact=True,completed_rounds_reconstructed=[0,1,2],incomplete_round3=dict(live_completed=200,shadow_completed=151,last_shadow_k=150,shadow_stopped_exactly_at_last_recorded_AFTER=True,no_missing_steps_computed=True,proposed_model_exists=False,acceptance_exists=False,live_end_hashes=live3,shadow_prefix_end_hashes=shadow3),last_complete_retained_model=dict(path=str(last_file),sha256=A.sha(last_file),hashes=best_hashes,policy_hash=best_policy,model_was_serialized_before_interrupted_round=True),retained_completed_rounds=retained,all_source_input_SHA=True,all_original_T422_files_SHA_unchanged=True,original_file_count=len(originals),actual_full_history_hidden_rewards_and_four_fresh_onpolicy_trajectories_checked=True,recovery_feedback_controls_safe_goal_and_existing_exact_readback_records_checked=True,old275=None,wide54=None,fresh16=None,reload78=None,stable_six_not_claimed=True,seven_eight_entered=False,limits='This post-shutdown audit reconstructs only persisted historical optimizer events. Original T422 learning has no summary/final native acceptance, and it remains interrupted. No native simulation or policy model was exported.')
    REPORT.write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    print('T423_PARTIAL_HISTORICAL_OPTIMIZER_AUDIT_PASS',adam,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        error=DOC/'T422_PARTIAL_OPTIMIZER_AUDIT_ERROR.json'
        error.write_text(json.dumps(dict(state='FAILED_INDEPENDENT_POSTHOC_AUDIT',error=traceback.format_exc(),native=0,no_original_T422_files_modified=True),ensure_ascii=False,indent=2)+'\n')
        raise
