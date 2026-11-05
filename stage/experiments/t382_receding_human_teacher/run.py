"""Human-action warm start, receding native teacher; zero student updates.

Five fixed antithetic proposals per replan, native feasibility rejection, native
validation of the weighted proposal, execute one hold then replan. This is a
privileged training-teacher pilot, not an autonomous Actor acceptance test.
"""
from pathlib import Path
import copy
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t382_receding_human_teacher';DOC=ROOT/'stage/docs/narrow_20261105'
spec=importlib.util.spec_from_file_location('t382_frozen_native_contract',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);R.OUT=OUT
S,K,np,torch=R.S,R.K,R.np,R.torch
read,save,digest,sha,equal,hashes,write_npz=R.read,R.save,R.digest,R.sha,R.equal,R.hashes,R.write_npz
SOURCE=R.SOURCE;NAMES=R.NAMES


def budget(p):
    assert K.S.used(ROOT/'stage/runs')<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert S.shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    categories={'mixed.npz':p['full_bytes'],'predictions.npz':p['prediction_bytes'],
        'prefix.npz':p['prefix_bytes'],'first_mismatch.npz':p['mismatch_bytes']}
    meta=logs=0
    for f in OUT.rglob('*'):
        if not f.is_file():continue
        if f.name in categories:assert f.stat().st_size<=categories[f.name]
        elif f.suffix=='.log':logs+=f.stat().st_size
        else:meta+=f.stat().st_size
    assert meta<=p['metadata_bytes'] and logs<=p['log_bytes'],(meta,logs)


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    science=read(p['source_protocol'])
    for f,h in science['input_hashes'].items():assert sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    budget(p);return science


def compact(p):
    n,c,h=p['max_replans'],p['candidates']+1,p['horizon_holds']
    return dict(controls=np.zeros((n,c,h,2),np.float32),next_poses=np.full((n,c,h,3),np.nan),
        counts=np.zeros((n,c,h),np.uint8),flags=np.zeros((n,c,h,3),np.uint8),
        lengths=np.zeros((n,c),np.uint8),costs=np.full((n,c),np.nan),remaining_m=np.full((n,c),np.nan),
        trace_sha256=np.zeros((n,c,32),np.uint8),integration_sha256=np.zeros((n,c,32),np.uint8),
        weights=np.zeros((n,p['candidates'])),snapshot_sha256=np.zeros((n,32),np.uint8))


def proposal(nominal,k,p):
    assert nominal.shape==(20,2) and nominal.dtype==np.float32
    rng=np.random.default_rng(np.random.SeedSequence([p['seed'],k]))
    z=np.repeat(rng.normal(size=(2,4,2)),5,axis=1)*np.asarray(p['noise_sigma'])
    raw=np.stack([nominal,nominal+z[0],nominal-z[0],nominal+z[1],nominal-z[1]])
    return np.clip(raw,-1,1).astype(np.float32),raw


def feasible_weights(costs,flags,lengths,p):
    feasible=np.asarray([bool(n) and not flags[k,int(n)-1,0] and
        (n==p['horizon_holds'] or flags[k,int(n)-1,1]) for k,n in enumerate(lengths)])
    w=np.zeros(p['candidates'])
    if feasible.any():
        z=np.exp(-(costs[feasible]-costs[feasible].min())/p['temperature']);w[feasible]=z/z.sum()
    return feasible,w


def snapshot_hash(s):
    return digest(dict(packet=s.packet(),history=s.w.get(s.env),hidden=s.hidden,
        previous=s.previous,pn=s.pn,physics=K.R.integration(s.env)))


def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t381_native_mppi_teacher';old=read(prior/'summary.json')
    assert old['state']=='COMPLETE' and not old['teacher_assisted_safe_GOAL'] and read(prior/'audit.json')['state']=='PASS'
    R.absent(old['pid']);R.absent(read(prior/'launch.json')['pid'])
    rc=ROOT/'stage/docs/long_distance_20261101/release_receipt_T381.json'
    assert read(rc)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    source=ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json'
    files=[source,SOURCE/'centre.npz',SOURCE/'candidate.pt.gz',SOURCE/'summary.json',SOURCE/'cases.json',prior/'protocol.json',prior/'summary.json',prior/'audit.json',
        ROOT/'stage/experiments/t381_native_mppi_teacher/run.py',rc,DOC/'T382_PLAN.md',DOC/'T381_TEACHER_LITERATURE.md',DOC/'NEXT_AFTER_T381.md',*HERE.glob('*.py')]
    science=read(source);demo=S.arrays(science['demonstration'])
    assert len(demo['actions'])==295 and demo['end_flags'][-1,1] and not demo['end_flags'][:,0].any()
    p=dict(id='T382',source_protocol=str(source),inputs={str(f):sha(f) for f in sorted(files)},human_source=science['demonstration'],
        seed=20261105382,candidates=5,passes=2,max_replans=60,prefix_holds=40,horizon_holds=20,execute_holds=1,
        block_holds=5,noise_sigma=[.20,.40],temperature=1.,Actor_updates=0,optimizer_updates=0,
        final_bytes=944*1024,full_bytes=512*1024,prediction_bytes=320*1024,prefix_bytes=4*1024,
        metadata_bytes=48*1024,log_bytes=16*1024,mismatch_bytes=32*1024,
        global_bytes=1024**3,min_free_bytes=40*1024**3,main_seconds=2400,audit_seconds=300,backup_seconds=300,all_work_seconds=3060,
        max_commands=860,max_physics=1610000,preflight_geometry_commands=1,
        snapshot='Actual T3798s state; native integration, Witness, previous actions, actual history and Actor hidden',
        nominal='Human actions40:60 first; weighted unexecuted shift plus human action[40+k+20] each accepted hold; no state transplant',
        stopping='NO_FEASIBLE_TEACHER or WEIGHTED_NATIVE_REJECTED stops actual teacher branch without Actor continuation; no unsafe expert labels',
        teacher_window='At most60 accepted actual holds8..20s then same frozen Actor until80s or true terminal',
        distinction='Privileged simulation training teacher; no autonomous Actor six acceptance or claimed originality')
    assert p['prefix_holds']+p['max_replans']+p['horizon_holds']<len(demo['actions'])
    used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T382_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_reserved=p['final_bytes'],full_reserved=used+p['final_bytes'],global_limit=p['global_bytes'],
        layout_bytes=sum(p[k] for k in ['full_bytes','prediction_bytes','prefix_bytes','metadata_bytes','log_bytes','mismatch_bytes']),
        prediction_uncompressed_bytes=sum(a.nbytes for a in compact(p).values()),per_pass_worst_commands=424,two_pass_plus_geometry_commands=849,
        per_pass_worst_physics=800500,two_pass_worst_physics=1601000,
        per_pass_breakdown=dict(baseline=40000,prefix=4000,reference=500,candidates=600000,weighted_validation=120000,actual_teacher=6000,Actor_continuation=30000)))
    print('REGISTERED_T382')


def preflight():
    p=read(HERE/'protocol.json');science=verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    native,create,w,job=R.setup(science);s=None
    try:
        s=R.Session(job,native,create,w,lambda **_:None,lambda:(_ for _ in ()).throw(AssertionError('zero physics')),R.actor(science))
        ss=s.capture();s.restore(ss);assert equal(s.initial,K.R.integration(s.env))
        human=S.arrays(p['human_source']);nominal=human['actions'][40:60].copy();c,raw=proposal(nominal,0,p)
        assert equal(c,proposal(nominal,0,p)[0]) and np.abs(c).max()<=1 and c[:,0,1].min()<0<c[:,0,1].max()
        fake=np.zeros((5,20,3),np.uint8);lengths=np.full(5,20,np.uint8);costs=np.arange(5,dtype=np.float64)
        f,wg=feasible_weights(costs,fake,lengths,p);assert f.all() and abs(wg.sum()-1)<1e-14
        fake[:,:,0]=1;f,wg=feasible_weights(costs,fake,lengths,p);assert not f.any() and not wg.any()
        fake[2,:,0]=0;f,wg=feasible_weights(costs,fake,lengths,p);assert f.tolist()==[False,False,True,False,False] and wg[2]==1
        shifted=np.concatenate([c[0,1:],human['actions'][60:61]])
        assert shifted.shape==nominal.shape and equal(shifted[:-1],c[0,1:]) and equal(shifted[-1],human['actions'][60])
        rng=np.random.default_rng(382);data=compact(p)
        for a in data.values():
            if np.issubdtype(a.dtype,np.floating):a[:]=rng.normal(size=a.shape)
            else:a[:]=rng.integers(0,256,size=a.shape,dtype=np.uint8)
        pred_size=len(R.blob(data));assert pred_size<=p['prediction_bytes']
        ref=S.arrays(SOURCE/'centre.npz');full={}
        for name,a in ref.items():
            shape=(400,)+a.shape[1:] if a.ndim and len(a)==len(ref['states']) else (401,)+a.shape[1:] if name.startswith('history_') else a.shape
            full[name]=rng.normal(size=shape).astype(a.dtype) if np.issubdtype(a.dtype,np.floating) else rng.integers(0,256,size=shape,dtype=np.uint8)
        full['next_states']=np.concatenate([full['states'][1:],full['next_states'][-1:]])
        full.update(ownership=rng.integers(0,2,size=400,dtype=np.uint8),hold_sha256=rng.integers(0,256,size=(400,32),dtype=np.uint8))
        full_size=len(R.blob(S.packed(full)));assert full_size<=p['full_bytes']
        save(OUT/'preflight.json',dict(state='PASS',physics=0,commands=1,geometry=1,Actor=0,Adam=0,
            snapshot_native_Witness_history_and_previous_exact=True,prediction_incompressible_bytes=pred_size,full_incompressible_bytes=full_size,
            antithetic_sampler_and_shift_reference_index_exact=True,initial_first_steer_range=[float(c[:,0,1].min()),float(c[:,0,1].max())],
            all_rejected_and_single_feasible_weight_contract=True,scene_sha256=s.scene,policy_hash=s.q.phash()))
    finally:
        if s:s.close()
        w.close()
    budget(p);print('PREFLIGHT_PASS_ZERO_NATIVE_ZERO_ADAM')


def main():
    p=read(HERE/'protocol.json');science=verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No student training'))
    start=time.time();deadline=read(OUT/'launch.json')['main_deadline_epoch'];physical=0;commands=1;last=0.;phase='SETUP';state='FAILED';s=None;w=None;results=[];pred=compact(p);reference=None;active_query=None
    def guard(physical=False):
        nonlocal last
        assert time.time()<deadline-5 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();budget(p);save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,commands=commands,physics_steps=count(),deadline_epoch=deadline))
    def count():return physical
    def charge():
        nonlocal physical
        physical+=1;assert physical<=p['max_physics']
    def command():
        nonlocal commands
        commands+=1;guard()
    def persist():
        if s is not None and s.rows:write_npz(OUT/'mixed.npz',S.packed(s.mixed()),p['full_bytes'])
    def mismatch(kind,index,a,b):
        write_npz(OUT/'first_mismatch.npz',dict(kind=np.asarray(kind),index=np.asarray(index),actual_hash=np.asarray(digest(a)),expected_hash=np.asarray(digest(b)),
            integration=K.R.integration(s.env),packet=s.packet(),history_poses=w.get(s.env)[0],history_goals=w.get(s.env)[1],history_times=w.get(s.env)[2]),p['mismatch_bytes'])
        raise AssertionError(('EXACT_REPLAY_MISMATCH',kind,index))
    def query(k,j,controls,rep):
        nonlocal active_query
        command();s.restore(snapshot);hds=[];first=None;active_query=dict(replan=k,candidate=j,start_rows=len(s.rows),physics_start=physical)
        if rep==0:
            pred['controls'][k,j]=controls;write_npz(OUT/'predictions.npz',pred,p['prediction_bytes'])
        for t in range(p['horizon_holds']):
            row,hd=s.step(controls[t],owner=1,actor_active=False);hds.append(hd)
            if t==0:first=(copy.deepcopy(row),copy.deepcopy(w.get(s.env)),K.R.integration(s.env).copy())
            if rep==0:
                pred['next_poses'][k,j,t]=row[5];pred['counts'][k,j,t]=row[2];pred['flags'][k,j,t]=row[6]
            elif not (equal(pred['next_poses'][k,j,t],row[5]) and pred['counts'][k,j,t]==row[2] and equal(pred['flags'][k,j,t],np.asarray(row[6],np.uint8))):mismatch('prediction_hold',[k,j,t],row,pred['next_poses'][k,j,t])
            if s.done():break
        n=t+1;rem=float(s.env._goal_state()[1]);score=R.cost(controls,n,s.info.get('collision'),rem)
        tr=np.frombuffer(bytes.fromhex(digest(np.asarray(hds,np.uint8))),np.uint8);integ=np.frombuffer(bytes.fromhex(digest(K.R.integration(s.env))),np.uint8)
        values=dict(controls=controls,lengths=np.uint8(n),costs=np.float64(score),remaining_m=np.float64(rem),trace_sha256=tr,integration_sha256=integ)
        if rep==0:
            for key,v in values.items():pred[key][k,j]=v
            write_npz(OUT/'predictions.npz',pred,p['prediction_bytes'])
        else:
            for key,v in values.items():
                if not equal(pred[key][k,j],v):mismatch('prediction_'+key,[k,j],v,pred[key][k,j])
        active_query=None
        return first,dict(contact=bool(s.info.get('collision')),goal=bool(s.info.get('reach_goal')),length=n)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T382 main deadline')));signal.alarm(max(1,int(deadline-time.time())-2))
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        native,create,w,job=R.setup(science);source=S.arrays(SOURCE/'centre.npz');human=S.arrays(p['human_source']);source_hashes=hashes(source)
        for rep in range(2):
            phase=f'PASS{rep}_HISTORICAL_BASELINE';command();s=R.Session(job,native,create,w,guard,charge,R.actor(science))
            for i in range(len(source['states'])):
                s.step()
                for j,name in enumerate(NAMES):assert equal(np.asarray(s.rows[-1][j]),source[name][i]),(name,i)
            assert s.done() and hashes(s.arrays())==source_hashes
            base=s.result();s.close();s=None
            phase=f'PASS{rep}_PREFIX';command();s=R.Session(job,native,create,w,guard,charge,R.actor(science))
            for i in range(40):
                s.step();assert all(equal(np.asarray(s.rows[-1][j]),source[name][i]) for j,name in enumerate(NAMES))
            assert s.steps==4000 and not s.done();prefix=s.capture();prefix_hash=snapshot_hash(s)
            if rep==0:
                write_npz(OUT/'prefix.npz',dict(physics=prefix['native']['physics'],previous=s.previous,hidden=s.hidden.numpy()),p['prefix_bytes']);persist()
            else:assert prefix_hash==results[0]['prefix_sha256']
            phase=f'PASS{rep}_RESTORE_REFERENCE';command()
            for i in range(40,45):
                s.step();assert all(equal(np.asarray(s.rows[-1][j]),source[name][i]) for j,name in enumerate(NAMES))
            s.restore(prefix);assert snapshot_hash(s)==prefix_hash
            nominal=human['actions'][40:60].copy();stop='MAX_REPLANS_HANDOFF';accepted=0;attempts=0;validation_rejects=0;no_feasible=0
            for k in range(p['max_replans']):
                attempts+=1;phase=f'PASS{rep}_REPLAN{k}_CANDIDATES';snapshot=s.capture();ph=np.frombuffer(bytes.fromhex(snapshot_hash(s)),np.uint8)
                if rep==0:pred['snapshot_sha256'][k]=ph
                else:assert equal(ph,pred['snapshot_sha256'][k])
                controls,raw=proposal(nominal,k,p)
                for j in range(p['candidates']):query(k,j,controls[j],rep)
                feasible,wg=feasible_weights(pred['costs'][k,:5],pred['flags'][k,:5],pred['lengths'][k,:5],p)
                if rep==0:pred['weights'][k]=wg;write_npz(OUT/'predictions.npz',pred,p['prediction_bytes'])
                else:assert equal(wg,pred['weights'][k])
                s.restore(snapshot)
                if not feasible.any():stop='NO_FEASIBLE_TEACHER';no_feasible+=1;break
                weighted=np.einsum('i,ijk->jk',wg,controls.astype(np.float64)).astype(np.float32)
                phase=f'PASS{rep}_REPLAN{k}_WEIGHTED_VALIDATION';first,validation=query(k,5,weighted,rep);s.restore(snapshot)
                if validation['contact'] or (validation['length']<20 and not validation['goal']):
                    stop='WEIGHTED_NATIVE_REJECTED';validation_rejects+=1;break
                phase=f'PASS{rep}_REPLAN{k}_EXECUTE_ONE_HOLD';command();row,hd=s.step(weighted[0],owner=1)
                assert all(equal(np.asarray(a),np.asarray(b)) for a,b in zip(row,first[0]))
                assert all(equal(a,b) for a,b in zip(w.get(s.env),first[1])) and equal(K.R.integration(s.env),first[2])
                accepted+=1
                if rep==0:persist()
                elif len(s.rows)>len(reference['states']) or not equal(hd,reference['hold_sha256'][len(s.rows)-1]):mismatch('actual_teacher',k,hd,reference['hold_sha256'][min(len(s.rows)-1,len(reference['states'])-1)])
                if s.done():stop='ACTUAL_TERMINAL';break
                nominal=np.concatenate([weighted[1:],human['actions'][40+k+20:40+k+21]])
            assert accepted<=60
            if stop=='MAX_REPLANS_HANDOFF':
                phase=f'PASS{rep}_FROZEN_ACTOR_CONTINUATION';command()
                while not s.done():
                    _,hd=s.step()
                    if rep==0:
                        if len(s.rows)%10==0 or s.done():persist()
                    elif len(s.rows)>len(reference['states']) or not equal(hd,reference['hold_sha256'][len(s.rows)-1]):mismatch('actual_Actor_continuation',len(s.rows)-1,hd,reference['hold_sha256'][min(len(s.rows)-1,len(reference['states'])-1)])
            a=s.mixed();result=s.result()
            if rep==0:persist();reference=copy.deepcopy(a)
            else:assert hashes(a)==hashes(reference) and a.keys()==reference.keys() and all(equal(v,reference[k]) for k,v in a.items())
            results.append(dict(pass_index=rep,baseline=base,historical_source_exact=True,prefix_sha256=prefix_hash,reference5_exact=True,
                attempts=attempts,accepted_teacher_holds=accepted,no_feasible_stops=no_feasible,weighted_validation_rejects=validation_rejects,
                stop=stop,mixed=result,mixed_array_hashes=hashes(a),prediction_hashes=hashes(pred),independent_sample=rep==0,
                student_updates=0,original_six_stable=False))
            save(OUT/'passes.json',results);s.close();s=None
        state='COMPLETE'
    except BaseException:
        original_error=traceback.format_exc()
        if s is not None and not (OUT/'first_mismatch.npz').exists():
            try:
                partial=dict(kind=np.asarray('ADMINISTRATIVE_OR_ENGINEERING_INTERRUPTION'),integration=K.R.integration(s.env),packet=s.packet(),
                    recorded_steps=np.asarray(s.steps),all_physics=np.asarray(physical),previous=s.previous,pn=np.asarray(s.pn),
                    current_history_poses=w.get(s.env)[0],current_history_goals=w.get(s.env)[1],current_history_times=w.get(s.env)[2])
                if active_query is not None:
                    start_rows=active_query['start_rows'];branch=s.rows[start_rows:]
                    for j,name in enumerate(NAMES):partial['branch_'+name]=np.asarray([row[j] for row in branch])
                    h=s.hist[start_rows:]+[w.get(s.env)]
                    partial.update(branch_history_poses=np.asarray([v[0] for v in h]),branch_history_goals=np.asarray([v[1] for v in h]),branch_history_times=np.asarray([v[2] for v in h]))
                write_npz(OUT/'first_mismatch.npz',partial,p['mismatch_bytes'])
            except BaseException:save(OUT/'partial_failure.json',dict(error=traceback.format_exc()))
        if s is not None and phase.startswith('PASS0_') and not any(tag in phase for tag in ['CANDIDATES','VALIDATION','HISTORICAL','REFERENCE']):
            try:persist()
            except BaseException:save(OUT/'partial_failure.json',dict(error=traceback.format_exc()))
        save(OUT/'failure.json',dict(error=original_error,phase=phase,commands=commands,physics_steps=physical,active_query=active_query))
    finally:
        if s is not None:s.close()
        if w is not None:w.close()
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,commands=commands,physics_steps=physical,Actor_updates=0,optimizer_updates=0,
            passes=results,all_workers_stopped=True,parent_model_sha256=sha(SOURCE/'candidate.pt.gz'),
            teacher_assisted_safe_GOAL=bool(results and results[0]['mixed']['success']),autonomous_six_stable=False,
            old275=None,original_six54=None,fresh16=None,reload78=None,seven_eight=False,
            files={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ['main.log','status.json']})
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),commands=commands,physics_steps=physical));signal.alarm(0);budget(p)
        print(json.dumps(dict(state=state,commands=commands,physics_steps=physical,elapsed_s=summary['elapsed_s'])))
    if state!='COMPLETE':raise RuntimeError('T382 failed; do not restart completed directory')


def audit():
    p=read(HERE/'protocol.json');science=verify(p);summary=read(OUT/'summary.json');assert not (OUT/'audit.json').exists()
    for pid in [summary['pid'],read(OUT/'launch.json')['pid']]:R.absent(pid)
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T382 audit')));signal.alarm(p['audit_seconds'])
    assert summary['state']=='COMPLETE' and len(summary['passes'])==2 and summary['Actor_updates']==summary['optimizer_updates']==0
    with np.load(OUT/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    a=S.arrays(OUT/'mixed.npz');source=S.arrays(SOURCE/'centre.npz');human=S.arrays(p['human_source']);first=summary['passes'][0];second=summary['passes'][1]
    assert all(equal(a[name][:40],source[name][:40]) for name in NAMES)
    assert first['stop']==second['stop'] and first['attempts']==second['attempts'] and first['accepted_teacher_holds']==second['accepted_teacher_holds']
    nominal=human['actions'][40:60].copy();expected_commands=3;accepted=0
    for k in range(first['attempts']):
        controls,_=proposal(nominal,k,p);assert equal(controls,pred['controls'][k,:5]);expected_commands+=5
        for j in range(6):
            n=int(pred['lengths'][k,j])
            if not n:continue
            assert 1<=n<=20 and (pred['counts'][k,j,:n]>0).all() and (pred['counts'][k,j,:n]<=100).all() and not pred['flags'][k,j,:n-1].any()
            assert pred['costs'][k,j]==R.cost(pred['controls'][k,j],n,pred['flags'][k,j,n-1,0],pred['remaining_m'][k,j])
        feasible,wg=feasible_weights(pred['costs'][k,:5],pred['flags'][k,:5],pred['lengths'][k,:5],p);assert equal(wg,pred['weights'][k])
        if not feasible.any():assert first['stop']=='NO_FEASIBLE_TEACHER' and k==first['attempts']-1 and not pred['lengths'][k,5];break
        expected_commands+=1;weighted=np.einsum('i,ijk->jk',wg,controls.astype(np.float64)).astype(np.float32);assert equal(weighted,pred['controls'][k,5])
        n=int(pred['lengths'][k,5]);valid=not pred['flags'][k,5,n-1,0] and (n==20 or pred['flags'][k,5,n-1,1])
        if not valid:assert first['stop']=='WEIGHTED_NATIVE_REJECTED' and k==first['attempts']-1;break
        expected_commands+=1;assert a['ownership'][40+accepted] and equal(a['actions'][40+accepted],weighted[0]);accepted+=1
        nominal=np.concatenate([weighted[1:],human['actions'][40+k+20:40+k+21]])
    assert accepted==first['accepted_teacher_holds'] and int(a['ownership'].sum())==accepted
    expected_owner=np.zeros(len(a['states']),np.uint8);expected_owner[40:40+accepted]=1;assert equal(a['ownership'],expected_owner)
    if first['stop']=='MAX_REPLANS_HANDOFF':assert accepted==60;expected_commands+=1
    else:assert len(a['states'])==40+accepted
    q=R.actor(science);hidden=None;goal=np.asarray(read(SOURCE/'cases.json')[0]['goal_xy'])
    for i,x in enumerate(a['states']):
        assert np.abs(K.H.rebuild(x,a['poses'][i],a['history_poses'][i],goal,a['history_goals'][i])-x).max()<=2e-6
        before=np.zeros(16) if hidden is None else hidden[0,0].numpy().copy()
        with torch.no_grad():mode,_,hidden,_=q.step(torch.tensor(x[None]),hidden)
        assert equal(before,a['hidden_before'][i])
        for key,v in q.tape[-1].items():assert equal(v,a['actor_proposal_'+key][i])
        if not a['ownership'][i]:assert equal(mode[0].numpy(),a['actions'][i])
        d0=np.linalg.norm(a['poses'][i,:2]-goal);d1=np.linalg.norm(a['next_poses'][i,:2]-goal);n=int(a['native_counts'][i]);z=K.R.GAMMA**n;flags=a['end_flags'][i];terminal=bool(flags.any())
        rw=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not terminal)
        if terminal:rw+=K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
        assert abs(a['rewards'][i]-rw)<1e-12
    assert equal(hidden[0,0].numpy(),a['last_hidden_after'])
    physical=2*(int(source['native_counts'].sum())+4000+500+int(pred['counts'].sum())+int(a['native_counts'][40:].sum()))
    assert physical==summary['physics_steps']<=p['max_physics'] and summary['commands']==1+2*expected_commands<=p['max_commands']
    assert hashes(a)==first['mixed_array_hashes']==second['mixed_array_hashes'] and hashes(pred)==first['prediction_hashes']==second['prediction_hashes']
    assert sha(SOURCE/'candidate.pt.gz')==summary['parent_model_sha256']
    for path,h in summary['files'].items():assert sha(OUT/path)==h,path
    report=dict(state='PASS',elapsed_s=time.time()-start,native=0,Actor=0,optimizer=0,all_original_native_pids_absent=True,
        sources_parent_and_complete_raw_hashes_exact=True,noise_nominal_shift_and_feasibility_cost_weights_verified=True,
        actual_ownership_Actor_hidden_causal120_reward_and_all_physics_commands_verified=True,two_pass_exact_evidence=True,
        commands=summary['commands'],physics_steps=physical,replans=first['attempts'],accepted_teacher_holds=accepted,
        stop=first['stop'],teacher_assisted_safe_GOAL=summary['teacher_assisted_safe_GOAL'],autonomous_six_stable=False)
    save(OUT/'audit.json',report);save(DOC/'T382_AUDIT.json',report);signal.alarm(0);budget(p);print(json.dumps(report))


def bounded_call(args,seconds,path=None):
    remaining=read(DOC/'T382_DRIVER_LAUNCH.json')['all_work_deadline_epoch']-time.time()-5
    assert remaining>1
    result=subprocess.run(args,cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(seconds,remaining))
    if path is not None:
        assert len(result.stdout)<=4*1024;path.write_bytes(result.stdout)
    assert result.returncode==0,result.stdout[-3000:].decode(errors='replace')
    return result.stdout.decode(errors='replace')


def driver():
    assert not (OUT/'launch.json').exists();p=read(HERE/'protocol.json');verify(p)
    envelope=read(DOC/'T382_DRIVER_LAUNCH.json');state='START_NATIVE';native_proc=None
    def mark(**kw):save(DOC/'T382_DRIVER.json',dict(state=state,pid=os.getpid(),updated_epoch=time.time(),**kw))
    try:
        epoch=time.time();assert epoch+2400+300+300<=envelope['all_work_deadline_epoch']
        receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
        with (OUT/'main.log').open('xb') as log:
            native_proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=1s','2400s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],
                cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT)
        save(OUT/'launch.json',dict(pid=native_proc.pid,launch_epoch=epoch,main_deadline_epoch=epoch+2400,all_work_deadline_epoch=envelope['all_work_deadline_epoch'],
            driver_pid=os.getpid(),driver_outer_pid=envelope['pid'],source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(HERE/'protocol.json')))
        state='NATIVE_RUNNING';mark(native_outer_pid=native_proc.pid)
        code=native_proc.wait(timeout=2405);assert code==0,code
        R.absent(read(OUT/'summary.json')['pid']);R.absent(native_proc.pid)
        state='INDEPENDENT_AUDIT';mark()
        bounded_call(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'audit'],300,OUT/'audit.log')
        state='REMOTE_BACKUP';mark()
        rc=DOC.parent/'long_distance_20261101/release_receipt_T382.json';assert not rc.exists()
        bounded_call(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T382'],300)
        assert read(rc)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        state='COMPLETE_AUDIT_AND_REMOTE_BACKUP';mark(receipt=str(rc),source_summary_sync_pending=True)
    except BaseException:
        if native_proc is not None and native_proc.poll() is None:
            native_proc.terminate()
            try:native_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:native_proc.kill();native_proc.wait(timeout=3)
        save(DOC/'T382_DRIVER_ERROR.json',dict(state='FAILED',phase=state,pid=os.getpid(),error=traceback.format_exc(),updated_epoch=time.time()))
        raise


def launch():
    p=read(HERE/'protocol.json');verify(p)
    assert not (OUT/'launch.json').exists() and not (DOC/'T382_DRIVER_LAUNCH.json').exists() and read(OUT/'preflight.json')['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage receding human-action training teacher',
        'timeout','--signal=TERM','--kill-after=1s','3060s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],
        cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    save(DOC/'T382_DRIVER_LAUNCH.json',dict(pid=proc.pid,launch_epoch=epoch,all_work_deadline_epoch=epoch+3060,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='DRIVER_LAUNCHED',pid=proc.pid)))


if __name__=='__main__':
    assert sys.argv[1] in ['prepare','preflight','launch','driver','main','audit'];globals()[sys.argv[1]]()
