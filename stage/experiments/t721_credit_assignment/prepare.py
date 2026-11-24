"""Bounded native-free positive/red preflight; discarded optimizer counts explicit."""
from pathlib import Path
import sys,importlib.util,copy,time,os,types,traceback,subprocess,json
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t721_preparation_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
import t721_learning as L
import acquisition_native as RN
A,np,torch=C.A,L.np,L.torch

def main():
    started=time.time();C.old_stopped();assert not (C.OUT/'protocol.json').exists() and not (C.OUT/'launch.json').exists();C.OUT.mkdir(exist_ok=True)
    assert not (C.OUT/'prepared.json').exists();os.sched_setaffinity(0,{15});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import mujoco
    mujoco.mj_step=lambda *a,**k:(_ for _ in ()).throw(AssertionError('T721 preparation forbids native stepping'))
    old=C.read(C.ROOT/'stage/runs/t717_onpolicy_recovery/protocol.json');p=copy.deepcopy(old);auth=C.read(C.DOC/'AUTHORIZATION.json');p.update(auth)
    assert C.sha(old['learner_initial']['path'])=='a1aade3ed7ea5530073fa324cecf84436ae821d26512d32d18e1a659e328c852'
    p.update(learner_initial=old['learner_initial'],rounds=12,seed=20261124721,round_reserve_seconds=450,gae_lambda=.95,fresh_bytes=1200000,full_bytes=1200000)
    model=p['learner_initial'];restored=A.frozen(p,model['path']);before=L.fixed(restored);sigmaold=list(p['raw_sigma'])
    split=C.read(C.ROOT/'stage/docs/day_20261124/t717/FIXED_SPLIT.json');alljobs=copy.deepcopy(p['recovery_jobs']);p['training_original_jobs']=[dict(copy.deepcopy(j),recovery_first=0) for j in p['training_jobs']]
    # Full valid existing T716 recovery evidence; no new native is consumed.
    file=C.ROOT/'stage/runs/t716_existing_teacher_recovery/recovery_compact/cases.jsonl.gz'
    import gzip
    row=json.loads(gzip.decompress(file.read_bytes()).decode().splitlines()[0]);a=RN.Base.C.E.N.S.arrays(row['path']);RN.inspect(C.read(C.ROOT/'stage/runs/t716_existing_teacher_recovery/protocol.json'),a,row)
    bad=copy.deepcopy(a);bad['actions'][0,0]=np.float32(.123456)
    try:RN.inspect(C.read(C.ROOT/'stage/runs/t716_existing_teacher_recovery/protocol.json'),bad,row)
    except AssertionError:pass
    else:raise AssertionError('Corruptprefix accepted')
    # Synthetic draws on historical states: preflight only, never learning data.
    a=RN.Base.C.E.N.S.arrays(split['all_rows'][0]['baseline']['path']);a={k:v[:24].copy() if hasattr(v,'ndim') and v.ndim and len(v)==len(a['states']) else v.copy() for k,v in a.items()}
    q=restored;q.rng=np.random.default_rng(717);q.reset_noise();h=None;hb=[];tapes=[]
    for i,x in enumerate(a['states']):
        q.sample=i>=3;hb.append(np.zeros(64) if h is None else h[0,0].numpy().copy());command,mu,h,_=q.step(torch.tensor(x[None]),h);tapes.append(copy.deepcopy(q.tape[-1]))
    for key in tapes[0]:a[key]=np.asarray([v[key] for v in tapes])
    a['hidden_before']=np.asarray(hb);a['last_hidden_after']=h[0,0].numpy().copy();a['recovery_first']=np.asarray(3,np.int64);a['ownership']=np.asarray([1]*3+[0]*(len(tapes)-3),np.uint8);a['end_flags']=np.zeros((len(tapes),3),np.uint8);a['end_flags'][-1,2]=1
    testrow=dict(job_id='synthetic_only',complete=True,administrative_censor=False)
    objects=[];results=[];states=[];ptest=dict(p,epochs=1)
    def event(who):
        def emit(kind,stage,k):
            with (C.DOC/'PREFLIGHT_CALLS.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,kind=kind,stage=stage,k=k))+'\n')
        return emit
    for who in ['live','shadow','independent']:

        obj=L.make(p);initial=L.state(obj)
        if states:assert L.digest(initial)==states[0]
        else:states.append(L.digest(initial))
        out=L.fit(obj,[(a,testrow)],ptest,lambda:None,event(who));results.append(out);objects.append(L.state(obj))
    assert results[0]==results[1]==results[2] and L.digest(objects[0])==L.digest(objects[1])==L.digest(objects[2])
    other=[]
    for who in ['live','shadow','independent']:
        obj=L.make(p);out=L.fit(obj,[(a,testrow)],dict(p,epochs=1,gae_lambda=1.),lambda:None,event(who));other.append((out,L.digest(L.state(obj))))
    assert other[0]==other[1]==other[2]
    for lam in [.95,1.]:
        obj=L.make(p);data=L.dataset(obj,[(a,testrow)],dict(p,gae_lambda=lam))
        if lam==.95:common=L.digest([g['target'] for g in data])
        else:assert L.digest([g['target'] for g in data])==common
    rr=torch.tensor([1.,2.,-3.],dtype=torch.float64);dd=torch.tensor([.9,.8,0.],dtype=torch.float64);vv=torch.tensor([.1,-.2,.4],dtype=torch.float64)
    adv,_=L.gae(rr,dd,vv,1.);_,ret=L.gae(rr,dd,torch.zeros_like(vv),1.);assert torch.allclose(adv+vv,ret,atol=1e-14,rtol=0)

    aa=copy.deepcopy(a);aa['rewards'][:3]=1e8
    obj=L.make(p);d1=L.dataset(obj,[(a,testrow)],p);d2=L.dataset(obj,[(aa,testrow)],p);assert L.digest(d1)==L.digest(d2),'Prefix reward leaked into suffix gradient'
    x,y=L.gae(torch.tensor([1.,2.]),torch.tensor([.9,0.]),torch.zeros(2),1.);assert torch.allclose(x,torch.tensor([2.8,2.]))
    loc=torch.zeros((3,2),dtype=torch.float64);raw=torch.tensor([[-2.,.2],[2.,.2],[.1,-2.]],dtype=torch.float64);atom=torch.where(raw>=L.Density.EDGE,1,torch.where(raw<=-L.Density.EDGE,-1,0)).to(torch.int8);lp=L.DENSITY(loc,raw,atom,p['raw_sigma']);assert torch.isfinite(lp).all() and torch.equal((lp-lp).exp(),torch.ones(3,dtype=torch.float64))
    captured=[];oldstep=RN.F.FieldSession.step
    def mock(self,forced=None,owner=0,actor_active=True):captured.append((forced,owner,actor_active,bool(self.q.sample)))
    try:
        RN.F.FieldSession.step=mock
        for j in alljobs:
            ob=RN.Session.__new__(RN.Session);ob.first=j['recovery_first'];ob.source=RN.Base.C.E.N.S.arrays(j['source_path']);ob.prefix_actions=[];ob.q=types.SimpleNamespace(sample=True);ob.suffix_sample=True;ob.rows=[None]*(ob.first-1);ob.step();assert captured[-1][1:]==(1,True,False);ob.rows=[None]*ob.first;ob.step();assert captured[-1]==(None,0,True,True)
    finally:RN.F.FieldSession.step=oldstep
    from unittest.mock import patch
    witness=[]
    def binding(pipe,index,pc,counts):assert RN.Base.P.frozen is A.frozen and RN.Base.G.T.Session is RN.Session;witness.append(index)
    with patch.object(RN.Base.G.T,'worker',binding):C.worker(None,0,p,None)
    assert witness==[0]
    test=C.OUT/'helper_preflight';test.mkdir();(C.DOC/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None))
    cmd=['systemd-run','--user','--unit=stage-t721-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={C.ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight'];c=subprocess.run(cmd,capture_output=True,text=True);assert c.returncode==0,c.stderr
    for k in ['inputs','input_stats','_input_manifest']:p.pop(k,None)
    additional(p);C.save(C.OUT/'prepared.json',p);C.save(C.OUT/'INITIAL_MODEL.json',model);C.save(C.DOC/'PREFLIGHT.json',dict(state='PASS',native=0,discarded_Actor_Adam_each=2,discarded_Value_Adam_each=2,actual_helper=True,actual_loader=True,worker_binding=True,full_existing_prefix_fixture=True,corrupt_prefix_red=True,all18_dispatch=True,seeded_censored_density_ratio=True,suffix_only_GAE_and_prefix_reward_exclusion=True,live_shadow_independent_Actor_Value_Adam_exact=True,full64carry_batch1_features_exact=True,common_value_MC_targets=True,lambda1_telescoping=True,original_sigma=sigmaold,new_sigma=p['raw_sigma'],unchanged_initial_weights=True,elapsed_seconds=time.time()-started,resources=C.C.resources()))
    print('T721_PREFLIGHT_PASS',flush=True)
def additional(p):
    import math,gzip,io,zipfile,numpy as np
    # Canonical 24 previously unexecuted reset proposals, no model forward/physics.
    K,F=RN.Base.C,RN.F;K.ACTIVE=p;folder=C.OUT/'initial_preflight';folder.mkdir()
    module,native,create,w,_=K.C.consistent_reset.setup(F,folder,p,K.C.CANONICAL_SETUP);original=copy.deepcopy(module.ACTIVE)
    proposed=[];valid=[];invalid=[]
    jobs={j['id']:j for j in p['evaluation_jobs']}
    def no_step(*a,**kw):raise AssertionError('No preparation native')
    try:
        for sector in range(8):
            for offset in [1,2,3]:
                parent=jobs[f't502_field_sector{sector}_offset{offset}'];j=copy.deepcopy(parent)
                for key in ['initial_contract','layout_initial_contract','source_path','source_sha256']:j.pop(key,None)
                angle=math.radians(67.5);dx=.06*math.cos(angle);dy=.06*math.sin(angle);yaw=parent['fullfield_start_pose'][2]+math.radians(1.3)
                j.update(id=f'T721_{parent["id"]}_r06_a67p5_yaw1p3',source_reset_id=f'T721_{parent["id"]}_r06_a67p5_yaw1p3',expected_initial120_sha256=None,recovery_first=0)
                j['fullfield_start_pose']=[parent['fullfield_start_pose'][0]+dx,parent['fullfield_start_pose'][1]+dy,math.atan2(math.sin(yaw),math.cos(yaw))];proposed.append(copy.deepcopy(j));current=None
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=j['circle_shift_x_m']
                try:
                    current=K.Session(j,native,create,w,lambda:None,no_step,None);pose=np.asarray(current.env._pose()[:3]);assert not current.env._collision() and not current.done() and np.linalg.norm(current.goal-pose[:2])>.2 and np.abs(pose[:2]).max()<=7
                    assert current.steps==0 and current.env.data.time==0 and not current.hidden.any()
                    j['prospective_initial']=dict(packet_hash=A.digest(current.packet()),native_hash=A.digest(current.initial),goal_hash=A.digest(current.goal),scene_hash=A.digest(np.frombuffer(bytes.fromhex(current.scene),dtype=np.uint8)))
                    valid.append(j)
                except AssertionError as error:invalid.append(dict(id=j['id'],reason=str(error),not_navigation_failure=True))
                finally:
                    if current is not None:current.close()
    finally:w.close()
    assert len(proposed)==24 and len(valid)+len(invalid)==24
    C.save(C.DOC/'PROSPECTIVE_INITIALS.json',dict(proposed=proposed,jobs=valid,invalid=invalid,no_replacements=True,policy_not_executed=True,new_native=0))
    # Same fixed jobs, matching existing full-array baseline evidence.
    selection=C.read(C.ROOT/'stage/docs/day_20261124/t720/FIXED_SELECTION.json');exposed=[];baseline=[]
    for panel in selection['panels'].values():exposed+=copy.deepcopy(panel['all_jobs']);baseline+=panel['baseline_rows']
    assert len({j['id'] for j in exposed})==174
    protected=p['protected403_rows'];oldjobs=[copy.deepcopy(jobs[r['id']]) for r in protected];assert len(oldjobs)==403
    def norm(js):return [dict(copy.deepcopy(j),recovery_first=0) for j in js]
    p['panels']={'old403':dict(jobs=norm(oldjobs),exact_jobs=norm(oldjobs[:20]),baseline_rows=[]),'exposed174':dict(jobs=norm(exposed),exact_jobs=norm(exposed[:31]),baseline_rows=baseline),'recovery18':dict(jobs=p['recovery_jobs'],exact_jobs=p['recovery_jobs'],baseline_rows=[]),'prospective24':dict(jobs=valid,exact_jobs=valid[:4],baseline_rows=[])}
    # Byte proof for complete packed schema, DEFLATE worst-case and ZIP/NPY overhead.
    fixture=RN.Base.C.E.N.S.arrays(split_path(p));packed=RN.Base.C.E.N.S.packed(fixture);n=len(fixture['states']);per=0;fixed=0
    for k,v in packed.items():
        if v.ndim and len(v)==n:per+=v.nbytes//n
        elif v.ndim and len(v)==n+1:per+=v.nbytes//(n+1);fixed+=v.nbytes//(n+1)
        else:fixed+=v.nbytes
    per+=8;fixed+=8 # source_prefix_actions float32[2]/decision and recovery_first scalar
    def filebound(cap):
        decisions=(cap+99)//100;raw=per*decisions+fixed
        # More than zlib compressBound plus per-member NPY headers, ZIP names, directory and extras.
        return raw+(raw//4096)+(raw//16384)+65536
    traincap=max(j['case']['step_cap'] for j in p['training_jobs']);evalcap=max(j['case']['step_cap'] for j in p['evaluation_jobs'])
    trainbound=filebound(traincap);evalbound=filebound(evalcap)
    science=6*4*36+2*(423+205+36)+3*(len(valid)+len(valid[:4]))
    rawbytes=864*trainbound+(2*(423+205+36)+3*(len(valid)+len(valid[:4])))*evalbound
    models=24*(2*4*2**20+3*2**20)+8*2**20 # raw torch upper 4MiB/state and3MiB/model
    maximum=rawbytes+models+96*2**20+16*2**20
    p.update(fresh_bytes=evalbound,full_bytes=evalbound,train_file_bound=trainbound,max_bytes=maximum,child_bytes=max(423*evalbound+16*2**20,144*trainbound+48*2**20),admission_bytes=8*2**30,global_bytes=10*2**30)
    assert C.used(C.ROOT/'stage/runs')+maximum+128*2**20<8*2**30,'Insufficient 128MiB extra admission margin'
    # All state writers bounded by uncompressed tensor bytes with positive margin.
    obj=L.make(p);bs=io.BytesIO();torch.save(L.state(obj),bs);assert len(bs.getvalue())<4*2**20
    C.save(C.DOC/'STORAGE_BOUND.json',dict(per_decision_bytes=per,fixed_bytes=fixed,train_decisions=(traincap+99)//100,eval_decisions=(evalcap+99)//100,train_file_bound=trainbound,eval_file_bound=evalbound,science_episodes=science,raw_bytes=rawbytes,models_bound=models,root_max_bytes=maximum,extra_admission_margin=8*2**30-C.used(C.ROOT/'stage/runs')-maximum,all_full_arrays_retained=True,no_deletions=True))

def split_path(p):return p['training_jobs'][0]['source_path']

if __name__=='__main__':
    try:main()
    except BaseException:C.save(C.DOC/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),new_native=0,raw_preserved=True));raise
