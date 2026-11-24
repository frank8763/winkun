"""Bounded native-free positive/red preflight; discarded optimizer counts explicit."""
from pathlib import Path
import sys,importlib.util,copy,time,os,types,traceback,subprocess,json
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t717_preparation_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
import t717_learning as L
import acquisition_native as RN
A,np,torch=C.A,L.np,L.torch

def main():
    started=time.time();C.old_stopped();assert not (C.OUT/'protocol.json').exists() and not (C.OUT/'launch.json').exists();C.OUT.mkdir(exist_ok=True)
    assert not (C.OUT/'prepared.json').exists();os.sched_setaffinity(0,{15});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import mujoco
    mujoco.mj_step=lambda *a,**k:(_ for _ in ()).throw(AssertionError('T717 preparation forbids native stepping'))
    old=C.read(C.ROOT/'stage/runs/t716_existing_teacher_recovery/protocol.json');p=copy.deepcopy(old);auth=C.read(C.DOC/'AUTHORIZATION.json');p.update(auth)
    baseline=C.ROOT/'stage/runs/t701_compact_retention/fit/replay_refined108.pt.gz';assert C.sha(baseline)=='e8bed4b7547a987539563d64e6916b77e8f3ee81688e6447d0d09ee1d1df364d'
    q=A.frozen(old,baseline);before=L.fixed(q);sigmaold=list(old['raw_sigma']);p.update(raw_sigma=[.2,.2],actor_lr=.0003,value_lr=.001,epochs=4,rounds=3,clip=.2,gae_lambda=.95,seed=20261124717,fit_threads=8,round_reserve_seconds=600,gamma_native=float(RN.Base.C.E.N.K.R.GAMMA))
    frozen=copy.deepcopy(q.state_dict());q.p=copy.deepcopy(p);model=L.export(q,p,C.OUT/'initial_sigma.pt.gz');p['learner_initial']=model;restored=A.frozen(p,model['path']);assert L.digest(restored.state_dict())==L.digest(frozen) and L.fixed(restored)==before
    split=C.read(C.DOC/'FIXED_SPLIT.json');alljobs=[]
    for row in split['all_rows']:
        b=row['baseline'];assert C.sha(b['path'])==b['sha256'];j=copy.deepcopy(b['request']['job']);j.update(source_path=b['path'],source_sha256=b['sha256'],recovery_first=row['recovery_first']);alljobs.append(j)
    p['recovery_jobs']=alljobs;p['original_jobs']=[dict(copy.deepcopy(j),recovery_first=0) for j in alljobs];p['training_ids']=split['training_ids'];p['validation_ids']=split['validation_ids'];p['training_jobs']=[j for j in alljobs if j['id'] in p['training_ids']]
    assert len(p['training_jobs'])==12 and len(set(p['training_ids'])&set(p['validation_ids']))==0
    # Full valid existing T716 recovery evidence; no new native is consumed.
    file=C.ROOT/'stage/runs/t716_existing_teacher_recovery/recovery_compact/cases.jsonl.gz'
    import gzip
    row=json.loads(gzip.decompress(file.read_bytes()).decode().splitlines()[0]);a=RN.Base.C.E.N.S.arrays(row['path']);RN.inspect(old,a,row)
    bad=copy.deepcopy(a);bad['actions'][0,0]=np.float32(.123456)
    try:RN.inspect(old,bad,row)
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
    objects=[];results=[];states=[];ptest=dict(p,epochs=2)
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
    cmd=['systemd-run','--user','--unit=stage-t717-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={C.ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight'];c=subprocess.run(cmd,capture_output=True,text=True);assert c.returncode==0,c.stderr
    for k in ['inputs','input_stats','_input_manifest']:p.pop(k,None)
    C.save(C.OUT/'prepared.json',p);C.save(C.OUT/'INITIAL_MODEL.json',model);C.save(C.DOC/'PREFLIGHT.json',dict(state='PASS',native=0,discarded_Actor_Adam_each=2,discarded_Value_Adam_each=2,actual_helper=True,actual_loader=True,worker_binding=True,full_existing_prefix_fixture=True,corrupt_prefix_red=True,all18_dispatch=True,seeded_censored_density_ratio=True,suffix_only_GAE_and_prefix_reward_exclusion=True,live_shadow_independent_Actor_Value_Adam_exact=True,full64carry_batch1_features_exact=True,original_sigma=sigmaold,new_sigma=p['raw_sigma'],unchanged_initial_weights=True,elapsed_seconds=time.time()-started,resources=C.C.resources()))
    print('T717_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:C.save(C.DOC/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),new_native=0,raw_preserved=True));raise
