"""Bounded native-MuJoCo sampling teacher; no Actor or optimizer updates.

Prediction is privileged simulation and is charged as physical work. A weighted
command sequence is validated by execution; a safe sampled trajectory is never
substituted after seeing the weighted result.
"""
from pathlib import Path
import collections
import copy
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t381_native_mppi_teacher'; DOC=ROOT/'stage/docs/narrow_20261105'
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import t377_core as S
np,torch,K=S.np,S.torch,S.K
read,save,digest=S.read,S.save,S.digest
SOURCE=ROOT/'stage/runs/t379_human_only_control'
NAMES=['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards']


def sha(path):return K.C.sha(str(path))
def equal(a,b):return np.array_equal(a,b,equal_nan=True)
def hashes(a):return {k:digest(v) for k,v in a.items()}
def blob(a):
    b=io.BytesIO();np.savez_compressed(b,**a);return b.getvalue()
def write_npz(path,a,limit):
    b=blob(a);assert len(b)<=limit,(path,len(b),limit);Path(path).write_bytes(b)
def absent(pid):assert not Path('/proc',str(pid)).exists(),pid


def budget(p):
    assert K.S.used(OUT)<=p['final_bytes']
    assert K.S.used(ROOT/'stage/runs')<=p['global_bytes']
    assert S.shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    science=read(p['source_protocol'])
    for f,h in science['input_hashes'].items():assert sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    budget(p);return science


def actor(science):
    q=S.NightActor(science,0.).eval().requires_grad_(False)
    q.load_learned(S.read_model(SOURCE/'candidate.pt.gz'));q.sample=False;q.reset_noise()
    assert q.phash()==read(SOURCE/'summary.json')['model']['policy_hash'];return q


def proposals(nominal,p):
    z=np.random.default_rng(p['seed']).normal(size=(32,12,2))
    noise=np.repeat(z,5,axis=1)*np.asarray(p['noise_sigma'])
    raw=np.concatenate([np.broadcast_to(nominal,(1,60,2)),nominal[None,None]+noise],0)
    return np.clip(raw,-1,1).astype(np.float32),raw


def cost(controls,n,collision,remaining):
    return 1000.*bool(collision)+10.*remaining+.01*float(np.square(np.diff(controls[:n].astype(np.float64),axis=0)).sum())


def weights(costs,controls,p):
    w=np.exp(-(costs-costs.min())/p['temperature']);w/=w.sum()
    return w,np.einsum('i,ijk->jk',w,controls.astype(np.float64)).astype(np.float32)


def compact():
    return dict(controls=np.zeros((33,60,2),np.float32),next_poses=np.full((33,60,3),np.nan),
        kinematics=np.full((33,60,7),np.nan,np.float32),counts=np.zeros((33,60),np.uint8),
        flags=np.zeros((33,60,3),np.uint8),hold_sha256=np.zeros((33,60,32),np.uint8),
        final_integration=np.full((33,203),np.nan),lengths=np.zeros(33,np.uint8),
        costs=np.full(33,np.nan),remaining_m=np.full(33,np.nan))


def kinematics(x):return np.asarray([x[53],x[54],2*x[55],.5*x[113],.5*x[114],3*x[115],3*x[116]],np.float32)


class Session:
    def __init__(self,job,native,create,witness,guard,physics,q=None):
        self.env,self.obs,_=K.R.task_env(create,job);self.obs=K.C.perturb_anchor(self.env,self.obs,job)
        self.w=witness;self.native=native;self.q=q;self.guard=guard;self.physics=physics
        self.steps=0;self.previous=np.zeros(2,np.float32);self.pn=0;self.hidden=None
        self.rows=[];self.hist=[];self.hb=[];self.ha=[];self.proposal=[];self.owner=[];self.digests=[]
        self.term=self.trunc=False;self.info={};self.cap=job['case']['step_cap']
        self.goal=self.env.goal_xy.copy();self.scene=K.C.model_hash(self.env);self.initial=K.R.integration(self.env).copy()
        assert not self.env._collision() and self.cap==40000 and len(self.initial)==203
        if q is not None:q.reset_noise()
    def close(self):self.w.events.pop(id(self.env),None);self.env.close()
    def packet(self):return K.T.L.packet(self.env,self.obs,self.steps,self.cap,self.previous,self.pn)
    def done(self):return self.term or self.trunc
    def capture(self):
        return dict(native=K.C.capture(self.env,self.obs),meta=copy.deepcopy(self.w.get(self.env)),
            steps=self.steps,previous=self.previous.copy(),pn=self.pn,hidden=copy.deepcopy(self.hidden),
            term=self.term,trunc=self.trunc,info=copy.deepcopy(self.info),packet=self.packet().copy(),
            rows=copy.deepcopy(self.rows),hist=copy.deepcopy(self.hist),hb=copy.deepcopy(self.hb),ha=copy.deepcopy(self.ha),
            proposal=copy.deepcopy(self.proposal),owner=list(self.owner),digests=list(self.digests),
            qstate=None if self.q is None else (copy.deepcopy(self.q.previous_raw),copy.deepcopy(self.q.previous_location),copy.deepcopy(self.q.tape)))
    def restore(self,s):
        self.obs=K.C.restore(self.env,s['native']);hp,hg,ht=s['meta']
        self.w.events[id(self.env)]=collections.deque([(f,p.copy(),g.copy(),float(t)) for f,p,g,t in zip(self.env.history,hp,hg,ht)],maxlen=48)
        for k in ['steps','previous','pn','hidden','term','trunc','info','rows','hist','hb','ha','proposal','owner','digests']:
            setattr(self,k,copy.deepcopy(s[k]))
        if self.q is not None:self.q.previous_raw,self.q.previous_location,self.q.tape=copy.deepcopy(s['qstate'])
        assert equal(K.R.integration(self.env),s['native']['physics']) and equal(self.packet(),s['packet'])
        assert all(equal(a,b) for a,b in zip(self.w.get(self.env),s['meta']))
        assert self.scene==K.C.model_hash(self.env) and equal(self.goal,self.env.goal_xy)
    def step(self,forced=None,owner=0,actor_active=True):
        self.guard();assert not self.done() and self.steps<self.cap
        x=self.packet();pose=np.asarray(self.env._pose()[:3]);meta=self.w.get(self.env)
        assert np.abs(K.H.rebuild(x,pose,meta[0],self.goal,meta[1])-x).max()<=2e-6
        proposal=None;before_h=after_h=None
        if self.q is not None and actor_active:
            before_h=np.zeros(16) if self.hidden is None else self.hidden[0,0].numpy().copy()
            with torch.no_grad():a,_,self.hidden,_=self.q.step(torch.tensor(x[None]),self.hidden)
            after_h=self.hidden[0,0].numpy().copy();proposal=copy.deepcopy(self.q.tape[-1]);mode=a[0].numpy()
        else:assert forced is not None
        action=mode if forced is None else np.asarray(forced,np.float32)
        assert action.shape==(2,) and np.isfinite(action).all() and np.abs(action).max()<=1
        before=self.steps;d0=float(self.env._goal_state()[1])
        for _ in range(min(100,self.cap-self.steps)):
            self.guard(physical=True)
            self.obs,_,self.term,self.trunc,self.info=self.native.step_checked(self.env,action)
            self.steps+=1;self.physics()
            if self.done():break
        n=self.steps-before;flags=[bool(self.info.get('collision')),bool(self.info.get('reach_goal')),bool(self.trunc)]
        d1=float(self.env._goal_state()[1]);z=K.R.GAMMA**n;terminal=any(flags)
        rw=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not terminal)
        if terminal:rw+=K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
        nxt=K.T.L.packet(self.env,self.obs,self.steps,self.cap,action,n)
        row=(x,action.copy(),n,nxt,pose,np.asarray(self.env._pose()[:3]),flags,rw)
        record={k:np.asarray(v) for k,v in zip(NAMES,row)}
        record.update(history_poses=meta[0],history_goals=meta[1],history_times=meta[2],
            next_history=self.w.get(self.env),integration=K.R.integration(self.env),ownership=owner,
            hidden_before=before_h,hidden_after=after_h,actor_proposal=proposal)
        hd=np.frombuffer(bytes.fromhex(digest(record)),np.uint8).copy()
        self.rows.append(row);self.hist.append(meta);self.hb.append(before_h);self.ha.append(after_h)
        self.proposal.append(proposal);self.owner.append(owner);self.digests.append(hd)
        self.previous=action.copy();self.pn=n
        assert self.scene==K.C.model_hash(self.env) and equal(self.goal,self.env.goal_xy)
        return row,hd
    def arrays(self,actor_fields=True):
        assert self.rows
        a={name:np.asarray([r[k] for r in self.rows]) for k,name in enumerate(NAMES)}
        a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
        h=self.hist+[self.w.get(self.env)]
        a.update(history_poses=np.asarray([v[0] for v in h]),history_goals=np.asarray([v[1] for v in h]),history_times=np.asarray([v[2] for v in h]),
            environment_rewards=a['rewards'].copy(),initial_native_integration=self.initial.copy(),final_native_integration=K.R.integration(self.env).copy())
        if actor_fields:
            a.update(hidden_before=np.asarray(self.hb),last_hidden_after=self.ha[-1].copy())
            a.update({k:np.asarray([v[k] for v in self.proposal]) for k in self.proposal[0]})
        return a
    def mixed(self):
        a=self.arrays()
        for name in self.proposal[0]:a['actor_proposal_'+name]=a.pop(name)
        a['ownership']=np.asarray(self.owner,np.uint8);a['hold_sha256']=np.asarray(self.digests,np.uint8)
        return a
    def result(self):
        return dict(complete=bool(self.done()),collision=bool(self.info.get('collision')),success=bool(self.info.get('reach_goal')) and not bool(self.info.get('collision')),
            timeout=bool(self.trunc),administrative_censor=not bool(self.done()),physics_steps=self.steps,remaining_m=float(self.env._goal_state()[1]))


def setup(science):
    S.P.C.C.OUT=OUT
    m,native,create,w,_=S.P.C.B.setup(OUT/'native',science)
    job=science['narrow_jobs'][0];m.ACTIVE=copy.deepcopy(m.ACTIVE);m.ACTIVE['circle_shift_x_m']=job['circle_shift_x_m']
    return native,create,w,job


def prepare():
    assert not OUT.exists()
    for name in ['t379_human_only_control','t380_teacher_model_audit','t380_evidence_recovery']:
        r=ROOT/'stage/runs'/name;absent(read(r/'summary.json')['pid']);absent(read(r/'launch.json')['pid'])
    assert read(SOURCE/'audit.json')['state']=='PASS'
    prior=ROOT/'stage/docs/long_distance_20261101/release_receipt_T380-RECOVERED.json'
    assert read(prior)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    files=[ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json',SOURCE/'candidate.pt.gz',SOURCE/'centre.npz',SOURCE/'cases.json',SOURCE/'summary.json',SOURCE/'audit.json',
        DOC/'T381_PLAN.md',DOC/'T381_NATIVE_MPPI_DRAFT.md',prior,*HERE.glob('*.py')]
    p=dict(id='T381',source_protocol=str(files[0]),inputs={str(f):sha(f) for f in sorted(files)},seed=20261105381,
        samples=33,horizon_holds=60,prefix_holds=40,block_holds=5,noise_sigma=[.20,.40],temperature=1.,passes=2,
        final_bytes=1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        full_bytes=512*1024,prediction_bytes=256*1024,prefix_bytes=4*1024,metadata_bytes=96*1024,log_bytes=32*1024,mismatch_bytes=32*1024,
        main_seconds=1200,audit_seconds=300,backup_seconds=300,all_work_seconds=1800,
        max_commands=80,max_physics=600000,preflight_geometry_commands=1,Actor_updates=0,optimizer_updates=0,
        model_privilege='Full native MuJoCo state and collision queries, training teacher only; no deployment teacher',
        success='Actual mixed safe GOAL only; never autonomous Actor acceptance; original275/six54/fresh16/reload78 not executed')
    used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T381_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,full_reserved=used+p['final_bytes'],global_limit=p['global_bytes'],
        per_pass_commands=38,per_pass_physics=278500,two_pass_physics=557000,geometry_commands=1,
        layout_reserved_bytes=932*1024,prediction_uncompressed_array_bytes=sum(a.nbytes for a in compact().values()),
        full_trace_worst_holds=400,preflight_uses_incompressible_encoding_to_bound_capacity=True))
    print('REGISTERED_T381')


def preflight():
    p=read(HERE/'protocol.json');science=verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    native,create,w,job=setup(science);s=None
    try:
        s=Session(job,native,create,w,lambda **_:None,lambda:(_ for _ in ()).throw(AssertionError('zero physics preflight')),actor(science))
        snapshot=s.capture();s.restore(snapshot);assert equal(s.initial,K.R.integration(s.env))
        proposals_,raw=proposals(np.array([.99,.3],np.float32),p);assert proposals_.shape==(33,60,2)
        assert equal(proposals_,proposals(np.array([.99,.3],np.float32),p)[0])
        assert np.abs(proposals_).max()<=1 and equal(proposals_[:,::5],proposals_[:,4::5])
        c=np.arange(33,dtype=np.float64);ww,weighted=weights(c,proposals_,p)
        assert np.isfinite(ww).all() and abs(ww.sum()-1)<1e-14 and np.abs(weighted).max()<=1
        cdata=compact();rng=np.random.default_rng(381)
        for a in cdata.values():
            if np.issubdtype(a.dtype,np.floating):a[:]=rng.normal(size=a.shape)
            else:a[:]=rng.integers(0,256,size=a.shape,dtype=np.uint8)
        prediction_size=len(blob(cdata));assert prediction_size<=p['prediction_bytes']
        reference=S.arrays(SOURCE/'centre.npz');synthetic={}
        for name,a in reference.items():
            shape=(400,)+a.shape[1:] if a.ndim and len(a)==len(reference['states']) else (401,)+a.shape[1:] if name.startswith('history_') else a.shape
            if np.issubdtype(a.dtype,np.floating):synthetic[name]=rng.normal(size=shape).astype(a.dtype)
            else:synthetic[name]=rng.integers(0,256,size=shape,dtype=np.uint8)
        synthetic['next_states']=np.concatenate([synthetic['states'][1:],synthetic['next_states'][-1:]])
        synthetic['ownership']=rng.integers(0,2,size=400,dtype=np.uint8);synthetic['hold_sha256']=rng.integers(0,256,size=(400,32),dtype=np.uint8)
        full_size=len(blob(S.packed(synthetic)));assert full_size<=p['full_bytes']
        prefix_size=len(blob(dict(physics=s.initial,previous=s.previous,hidden=np.zeros((1,1,16)))))
        assert prefix_size<=p['prefix_bytes']
        save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,commands=1,geometry=1,Actor=0,Adam=0,
            snapshot_native_and_causal_history_exact=True,integration_slots=len(s.initial),scene_sha256=s.scene,
            prediction_incompressible_bytes=prediction_size,full_incompressible_bytes=full_size,prefix_bytes=prefix_size,
            sampler_weighted_range_and_reproducibility=True,policy_hash=s.q.phash()))
    finally:
        if s:s.close()
        w.close()
    budget(p);print('PREFLIGHT_PASS_ZERO_PHYSICS')


def main():
    p=read(HERE/'protocol.json');science=verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No optimizer in T381'))
    start=time.time();deadline=read(OUT/'launch.json')['main_deadline_epoch'];physical=0;commands=1;last=0.;phase='SETUP';state='FAILED';results=[];w=None;s=None;pred=compact();reference=None
    def guard(physical=False):
        nonlocal last
        assert time.time()<deadline-5 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            budget(p);last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,commands=commands,physics_steps=physical_count(),deadline_epoch=deadline))
    def physical_count():return physical
    def charge():
        nonlocal physical
        physical+=1;assert physical<=p['max_physics']
    def command():
        nonlocal commands
        commands+=1;guard()
    def persist_mixed():
        if s and s.rows:write_npz(OUT/'mixed.npz',S.packed(s.mixed()),p['full_bytes'])
    def mismatch(kind,index,actual,expected):
        a=dict(kind=np.asarray(kind),index=np.asarray(index),actual=np.frombuffer(bytes.fromhex(digest(actual)),np.uint8),expected=np.frombuffer(bytes.fromhex(digest(expected)),np.uint8),
            integration=K.R.integration(s.env),packet=s.packet(),history_poses=s.w.get(s.env)[0],history_goals=s.w.get(s.env)[1],history_times=s.w.get(s.env)[2])
        write_npz(OUT/'first_mismatch.npz',a,p['mismatch_bytes']);raise AssertionError(('EXACT_REPLAY_MISMATCH',kind,index))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T381 main deadline')))
    signal.alarm(max(1,int(deadline-time.time())-2));save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        native,create,w,job=setup(science);source=S.arrays(SOURCE/'centre.npz');basehash=hashes(source)
        for rep in range(2):
            phase=f'PASS{rep}_HISTORICAL_BASELINE';command();s=Session(job,native,create,w,guard,charge,actor(science))
            for i in range(len(source['states'])):
                s.step();a=s.rows[-1]
                for j,name in enumerate(NAMES):
                    if not equal(np.asarray(a[j]),source[name][i]):mismatch('baseline_'+name,i,a[j],source[name][i])
            assert s.done() and hashes(s.arrays())==basehash
            baseline_result=s.result();s.close();s=None
            phase=f'PASS{rep}_PREFIX';command();s=Session(job,native,create,w,guard,charge,actor(science))
            for i in range(40):
                s.step();assert all(equal(np.asarray(s.rows[-1][j]),source[name][i]) for j,name in enumerate(NAMES))
            assert s.steps==4000 and not s.done();snapshot=s.capture();prefix_hash=digest(dict(packet=s.packet(),meta=w.get(s.env),hidden=s.hidden,integration=K.R.integration(s.env)))
            if rep==0:
                write_npz(OUT/'prefix.npz',dict(physics=snapshot['native']['physics'],previous=s.previous,hidden=s.hidden.numpy()),p['prefix_bytes']);persist_mixed()
            else:assert prefix_hash==results[0]['prefix_sha256']
            phase=f'PASS{rep}_RESTORE_REFERENCE';command()
            for i in range(40,45):
                s.step();assert all(equal(np.asarray(s.rows[-1][j]),source[name][i]) for j,name in enumerate(NAMES))
            s.restore(snapshot);assert prefix_hash==digest(dict(packet=s.packet(),meta=w.get(s.env),hidden=s.hidden,integration=K.R.integration(s.env)))
            controls,raw=proposals(source['actions'][40],p)
            phase=f'PASS{rep}_NATIVE_PREDICTION'
            if rep==0:pred['controls'][:]=controls
            else:assert equal(pred['controls'],controls)
            for k in range(33):
                command();s.restore(snapshot)
                for i in range(60):
                    row,hd=s.step(controls[k,i],owner=1,actor_active=False)
                    kine=kinematics(row[3]);flags=np.asarray(row[6],np.uint8)
                    if rep==0:
                        pred['next_poses'][k,i]=row[5];pred['kinematics'][k,i]=kine;pred['counts'][k,i]=row[2];pred['flags'][k,i]=flags;pred['hold_sha256'][k,i]=hd
                    elif not (equal(pred['next_poses'][k,i],row[5]) and equal(pred['kinematics'][k,i],kine) and pred['counts'][k,i]==row[2] and equal(pred['flags'][k,i],flags) and equal(pred['hold_sha256'][k,i],hd)):
                        mismatch('candidate_hold',[k,i],hd,pred['hold_sha256'][k,i])
                    if s.done():break
                length=i+1;remaining=float(s.env._goal_state()[1]);score=cost(controls[k],length,s.info.get('collision'),remaining);integration=K.R.integration(s.env)
                if rep==0:
                    pred['lengths'][k]=length;pred['costs'][k]=score;pred['remaining_m'][k]=remaining;pred['final_integration'][k]=integration
                    write_npz(OUT/'predictions.npz',pred,p['prediction_bytes'])
                else:
                    assert length==pred['lengths'][k] and score==pred['costs'][k] and remaining==pred['remaining_m'][k] and equal(integration,pred['final_integration'][k])
            ww,weighted=weights(pred['costs'],controls,p)
            if rep==0:
                pred.update(weights=ww,weighted_controls=weighted);write_npz(OUT/'predictions.npz',pred,p['prediction_bytes'])
            else:assert equal(ww,pred['weights']) and equal(weighted,pred['weighted_controls'])
            s.restore(snapshot);phase=f'PASS{rep}_ACTUAL_WEIGHTED_TEACHER';command()
            mixed_ref=None if rep==0 else reference
            for i in range(60):
                _,hd=s.step(weighted[i],owner=1)
                if rep==0:
                    if i%10==0 or s.done():persist_mixed()
                elif len(s.rows)>len(mixed_ref['states']) or not equal(hd,mixed_ref['hold_sha256'][len(s.rows)-1]):mismatch('mixed_teacher',len(s.rows)-1,hd,mixed_ref['hold_sha256'][min(len(s.rows)-1,len(mixed_ref['states'])-1)])
                if s.done():break
            weighted_result=s.result();weighted_result['horizon_complete']=i==59 and not s.done()
            phase=f'PASS{rep}_ACTOR_CONTINUATION';command()
            while not s.done():
                _,hd=s.step()
                if rep==0:
                    if len(s.rows)%10==0 or s.done():persist_mixed()
                elif len(s.rows)>len(mixed_ref['states']) or not equal(hd,mixed_ref['hold_sha256'][len(s.rows)-1]):mismatch('mixed_continuation',len(s.rows)-1,hd,mixed_ref['hold_sha256'][min(len(s.rows)-1,len(mixed_ref['states'])-1)])
            a=s.mixed();mixed_result=s.result()
            if rep==0:persist_mixed();reference=copy.deepcopy(a)
            else:assert hashes(a)==hashes(reference) and a.keys()==reference.keys() and all(equal(v,reference[k]) for k,v in a.items())
            results.append(dict(pass_index=rep,baseline=baseline_result,historical_T379_all_arrays_exact=True,prefix_sha256=prefix_hash,
                prefix_reference5_exact=True,weighted_horizon=weighted_result,mixed=mixed_result,mixed_array_hashes=hashes(a),prediction_hashes=hashes(pred),
                independent_sample=rep==0,Actor_updates=0,optimizer_updates=0,teacher_holds=int(a['ownership'].sum()),
                sample_command_clipped_fraction=float(np.mean(np.abs(raw)>1)),sample_action_std=controls[1:].std(axis=(0,1)).tolist(),weight_effective_samples=float(1/np.square(ww).sum())))
            save(OUT/'passes.json',results);s.close();s=None
        state='COMPLETE'
    except BaseException:
        if s is not None and phase.startswith('PASS0_') and ('ACTUAL' in phase or 'CONTINUATION' in phase or phase.endswith('PREFIX')):
            try:persist_mixed()
            except BaseException:save(OUT/'partial_failure.json',dict(error=traceback.format_exc()))
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=physical))
    finally:
        if s is not None:s.close()
        if w is not None:w.close()
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,commands=commands,physics_steps=physical,
            Actor_updates=0,optimizer_updates=0,passes=results,all_workers_stopped=True,
            teacher_assisted_safe_GOAL=bool(results and results[0]['mixed']['success']),Actor_autonomous_six_stable=False,
            old275=None,original_six54=None,fresh16=None,independent_reload78=None,seven_eight_entered=False,
            parent_model_sha256=sha(SOURCE/'candidate.pt.gz'),files={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ['main.log','status.json']})
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),commands=commands,physics_steps=physical))
        signal.alarm(0);budget(p);print(json.dumps(dict(state=state,commands=commands,physics_steps=physical,elapsed_s=summary['elapsed_s'])))
    if state!='COMPLETE':raise RuntimeError('T381 failed; original evidence preserved, no automatic restart')


def audit():
    p=read(HERE/'protocol.json');science=verify(p);summary=read(OUT/'summary.json')
    assert not (OUT/'audit.json').exists()
    for pid in [summary['pid'],read(OUT/'launch.json')['pid']]:absent(pid)
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T381 audit')));signal.alarm(p['audit_seconds'])
    assert summary['state']=='COMPLETE' and len(summary['passes'])==2 and summary['Actor_updates']==summary['optimizer_updates']==0
    with np.load(OUT/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    source=S.arrays(SOURCE/'centre.npz');controls,raw=proposals(source['actions'][40],p)
    assert equal(controls,pred['controls']);scores=[]
    for k,n in enumerate(pred['lengths']):
        n=int(n);assert 1<=n<=60 and (pred['counts'][k,:n]>0).all() and (pred['counts'][k,:n]<=100).all()
        assert not pred['flags'][k,:n-1].any()
        scores.append(cost(controls[k],n,pred['flags'][k,n-1,0],pred['remaining_m'][k]))
    assert equal(np.asarray(scores),pred['costs']);ww,weighted=weights(np.asarray(scores),controls,p)
    assert equal(ww,pred['weights']) and equal(weighted,pred['weighted_controls'])
    a=S.arrays(OUT/'mixed.npz');n=len(a['states']);assert all(equal(a[name][:40],source[name][:40]) for name in NAMES)
    assert equal(a['ownership'],np.asarray([0]*40+[1]*min(60,n-40)+[0]*max(0,n-100),np.uint8))
    assert equal(a['actions'][40:min(n,100)],weighted[:min(60,n-40)])
    q=actor(science);h=None;goal=np.asarray(read(SOURCE/'cases.json')[0]['goal_xy'])
    for i,x in enumerate(a['states']):
        assert np.abs(K.H.rebuild(x,a['poses'][i],a['history_poses'][i],goal,a['history_goals'][i])-x).max()<=2e-6
        before=np.zeros(16) if h is None else h[0,0].numpy().copy()
        with torch.no_grad():mode,_,h,_=q.step(torch.tensor(x[None]),h)
        assert equal(before,a['hidden_before'][i])
        for k,v in q.tape[-1].items():assert equal(v,a['actor_proposal_'+k][i])
        if not a['ownership'][i]:assert equal(mode[0].numpy(),a['actions'][i])
        d0=np.linalg.norm(a['poses'][i,:2]-goal);d1=np.linalg.norm(a['next_poses'][i,:2]-goal);count=int(a['native_counts'][i]);z=K.R.GAMMA**count;flags=a['end_flags'][i];terminal=bool(flags.any())
        reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not terminal)
        if terminal:reward+=K.R.GAMMA**(count-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
        assert abs(float(a['rewards'][i])-reward)<1e-12
    assert equal(h[0,0].numpy(),a['last_hidden_after'])
    total=2*(int(source['native_counts'].sum())+4000+500+int(pred['counts'].sum())+int(a['native_counts'][40:].sum()))
    assert total==summary['physics_steps']<=p['max_physics'] and summary['commands']==77<=p['max_commands']
    assert hashes(a)==summary['passes'][0]['mixed_array_hashes']==summary['passes'][1]['mixed_array_hashes']
    assert hashes(pred)==summary['passes'][0]['prediction_hashes']==summary['passes'][1]['prediction_hashes']
    assert sha(SOURCE/'candidate.pt.gz')==summary['parent_model_sha256']
    for path,h in summary['files'].items():assert sha(OUT/path)==h,path
    report=dict(state='PASS',elapsed_s=time.time()-start,native=0,Actor=0,Adam=0,all_sources_parent_SHA_exact=True,
        all_proposals_costs_weights_recomputed=True,causal120_human_teacher_ownership_hidden_current_observations_and_rewards_verified=True,
        first_pass_and_second_exact_native_replay_evidence_hashes=True,physics_steps=total,commands=summary['commands'],
        unique_mixed_trajectories=1,exact_repeated_mixed_trajectories=1,teacher_assisted_safe_GOAL=summary['teacher_assisted_safe_GOAL'],
        autonomous_Actor_six_stable=False,own_bytes_before_audit=K.S.used(OUT))
    save(OUT/'audit.json',report);save(DOC/'T381_AUDIT.json',report);signal.alarm(0);budget(p);print(json.dumps(report))


def launch():
    p=read(HERE/'protocol.json');verify(p)
    assert not (OUT/'launch.json').exists() and read(OUT/'preflight.json')['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'main.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage native sampling teacher feasibility',
            'timeout','--signal=TERM','--kill-after=1s','1200s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],
            cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,main_deadline_epoch=epoch+1200,all_work_deadline_epoch=epoch+1800,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


if __name__=='__main__':
    assert sys.argv[1] in ['prepare','preflight','launch','main','audit'];globals()[sys.argv[1]]()
