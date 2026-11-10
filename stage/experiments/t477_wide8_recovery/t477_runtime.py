"""Wide8 native task: same base6 pose/geometry, radial final goal before reset."""
from pathlib import Path
import copy,contextlib,fcntl,importlib.util,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t477_learning as L
M,P,E,R,torch,np=L.M,L.P,L.E,L.R,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
OUT=ROOT/'stage/runs/t477_wide8_recovery';ACTIVE=OUT
BASE_SESSION=E.L.Session;BASE_SETUP=E.N.S.P.C.B.setup

def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def native_setup(folder,p):
    module,native,create,w,reward=BASE_SETUP(folder,E.N.science(p));original=E.N.K.R.task_env
    def task_env(create,job):
        env,obs,actual=original(create,job)
        if 'radial_distance_m' in job:
            before=E.N.K.R.integration(env).copy();pose=np.asarray(env._pose()[:3]);base=env.goal_xy.copy();distance=float(np.linalg.norm(base-pose[:2]));assert abs(distance-6)<2e-5
            target=pose[:2]+(base-pose[:2])*(job['radial_distance_m']/distance)
            obs=w.alternate(env,target,w.get(env));assert np.array_equal(before,E.N.K.R.integration(env))
            hp,hg,ht=w.get(env);assert np.all(hg==target) and np.max(np.abs(E.N.K.H.rebuild(E.N.K.T.L.packet(env,obs,0,job['case']['step_cap'],np.zeros(2,np.float32),0),pose,hp,target,hg)-E.N.K.T.L.packet(env,obs,0,job['case']['step_cap'],np.zeros(2,np.float32),0)))<=2e-6
        return env,obs,actual
    E.N.K.R.task_env=task_env
    return module,native,create,w,reward

class Session(BASE_SESSION):
    def __init__(self,job,native,create,witness,guard,physics,q=None):
        self.env,self.obs,_=E.N.K.R.task_env(create,job);self.obs=E.N.K.C.perturb_anchor(self.env,self.obs,job)
        self.w=witness;self.native=native;self.q=q;self.guard=guard;self.physics=physics
        self.steps=0;self.previous=np.zeros(2,np.float32);self.pn=0;self.hidden=torch.zeros((1,1,64),dtype=torch.float64)
        self.rows=[];self.hist=[];self.hb=[];self.ha=[];self.proposal=[];self.owner=[];self.digests=[]
        self.term=self.trunc=False;self.info={};self.cap=job['case']['step_cap'];self.goal=self.env.goal_xy.copy();self.scene=E.N.K.C.model_hash(self.env);self.initial=E.N.K.R.integration(self.env).copy()
        assert not self.env._collision() and 0<self.cap<=53334 and len(self.initial)==203
        if q is not None:q.reset_noise()

def worker(pipe,index,p,counts):
    E.N.S.P.C.B.setup=lambda folder,unused:native_setup(folder,p)
    E.L.Session=Session
    M.worker(pipe,index,p,counts)
class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard,self.children,self.pipes=guard,[],[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';g=read(f) if f.exists() else [];g.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,g)
        except BaseException:self.close();raise

class Runtime:
    def __init__(self,p):
        self.p=p;self.start=time.time();self.deadline=min(self.start+p['main_seconds'],p['absolute_native_stop']);self.learnstop=min(self.start+p['learning_seconds'],p['absolute_learning_stop']);self.last=0.;self.role='SETUP';self.commands=2;self.rows=[];self.replays=[];self.pool=None
        self.counts=R.M.mp.get_context('spawn').RawArray('q',3);self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard(self,learning=False):
        E.L.check_stop();assert time.time()<(self.learnstop if learning else self.deadline)-5 and sum(self.counts)<=self.p['max_physics'] and self.commands+2<=self.p['max_commands']
        if time.time()-self.last>10:self.last=time.time();verify(self.p);save(ACTIVE/'status.json',dict(state=self.role,epoch=self.last,pid=os.getpid(),physics=sum(self.counts),commands=self.commands))
    def request(self,jobs,model,role,seeds=None,full=False,references=None):
        self.role=role;result=[]
        for off in range(0,len(jobs),2):
            js=jobs[off:off+2];self.guard();assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics'];assert self.commands+len(js)+2<=self.p['max_commands'];self.commands+=len(js)
            reqs=[]
            for i,j in enumerate(js):
                path=references[off+i]['path'] if references else str(ACTIVE/'episodes'/f'{len(self.rows)+i:04d}.npz') if full else None
                reqs.append(dict(model=model['path'],job=j,seed=None if seeds is None else seeds[off+i],full_path=path,query=False))
            for local,((a,r),j,req) in enumerate(zip(self.pool.request(reqs),js,reqs)):
                r.update(job_id=j['id'],role=role,seed=req['seed'],request=req,loaded_model_sha256=sha(model['path']));assert r['policy_hash']==model['policy_hash'] and int(a['native_counts'].sum())==r['physics']
                if references:
                    prev=references[off+local];assert r['array_digest']==prev['array_digest'] and r['integration_sha256']==prev['integration_sha256'];r.update(reference_path=prev['path'],exact=True,independent_new_sample=False);self.replays.append(r);log='replays.jsonl'
                else:
                    path=Path(req['full_path']) if full else ACTIVE/'episodes'/f'{len(self.rows):04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    if full:assert digest(E.N.S.arrays(path))==r['array_digest']
                    else:E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},self.p['compact_bytes'])
                    r.update(path=str(path),sha256=sha(path),full=full);self.rows.append(r);log='cases.jsonl'
                with (ACTIVE/log).open('a') as f:f.write(json.dumps(r)+'\n')
                result.append((a,r))
        return result

def setup(folder):
    global ACTIVE
    ACTIVE=Path(folder);M.OUT=ACTIVE;M.verify=verify;M.setup_runtime()

def event(arm,which):
    def f(stage,k):
        with (ACTIVE/'optimizer_calls.jsonl').open('a') as h:h.write(json.dumps([arm,which,stage,k])+'\n')
    return f

def task(p,distance,side):return next(j for j in p['courses'][str(distance)] if j['case']['side']==side)
def safe(values):return sorted(r['job_id'] for a,r in values if r['success'] and not r['collision'])
def assess(rt,p,model,role,alljobs=False):
    jobs=p['wide8_jobs']+p['six_jobs']+p['old275_jobs'] if alljobs else p['panel_jobs']
    values=rt.request(jobs,model,role);return dict(success_ids=safe(values),counts=R.M.tally([r for a,r in values]),model=model),values

def main(mode):
    folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Runtime(p);state='FAILED';error=None;rounds=[];candidate=None;obj=None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        if mode=='baseline':
            c,values=assess(rt,p,p['initial'],'FROZEN_WIDE8',False);candidate=c
        else:
            obj=L.make(p);shadow=L.make(p);current=p['initial'];baseline=read(OUT/'baseline/summary.json')['candidate'];initialpanel,_=assess(rt,p,current,'INITIAL_RETENTION_PANEL');keep=initialpanel['success_ids'];level=0;rejects=0
            for rnd in range(p['rounds']):
                rt.guard(True)
                # Reserve complete final383 raw+logs, remainingprotocol/audit/partial before any nextupdate.
                if M.used(OUT)+14*2**20>read(OUT/'protocol.json')['max_bytes']:
                    save(folder/'capacity_stop.json',dict(reason='COMPLETE_NEXTROUND_PLUS_FINAL383_RESERVATION',round=rnd,used=M.used(OUT),native=0,new_Adam=0));break
                distance=[6.5,7.0,7.5,8.0][level] if mode=='curriculum' else 8.0;jobs=[task(p,distance,side) for side in ['LEFT','RIGHT']];seeds=[[p['seed'],rnd,i] for i in range(2)]
                samples=rt.request(jobs,current,f'TRAIN_{rnd}',seeds=seeds,full=True);refs=[r for a,r in samples];rt.request(jobs,current,f'EXACT_{rnd}',seeds=seeds,full=True,references=refs)
                before=P.state(*obj);assert digest(before)==digest(P.state(*shadow));a=L.fit(obj,[a for a,r in samples],p,lambda:rt.guard(True),event(mode,'live'));b=L.fit(shadow,[a for a,r in samples],p,lambda:rt.guard(True),event(mode,'shadow'))
                assert a==b and digest(P.state(*obj))==digest(P.state(*shadow));proposal=P.export(obj,folder/f'proposal_{rnd}.pt.gz',p['model_bytes']);check,_=assess(rt,p,proposal,f'PROPOSAL_PANEL_{rnd}');accepted=set(keep)<=set(check['success_ids'])
                if accepted:current=proposal;keep=check['success_ids'];rejects=0
                else:P.load(obj,before);P.load(shadow,before);assert digest(P.state(*obj))==digest(before)==digest(P.state(*shadow));rejects+=1
                centers=rt.request(jobs,current,f'COURSE_CENTERS_{rnd}');coursepass=len(safe(centers))==2
                promoted=mode=='curriculum' and accepted and len(safe(samples))>=1 and coursepass and level<3
                if promoted:level+=1
                row=dict(round=rnd,distance_m=distance,samples=refs,model_before=refs[0]['request']['model'],update=a,proposal=proposal,accepted=accepted,retained=current,panel=check,training=R.M.tally([r for a,r in samples]),course_center=R.M.tally([r for a,r in centers]),promoted=promoted,next_level=level,rollback_hash=digest(P.state(*obj)))
                rounds.append(row);save(folder/'rounds.json',rounds)
                if rejects>=2:break
            candidate,values=assess(rt,p,current,'FINAL_ALL383',True)
            candidate.update(wide8=R.M.tally([r for a,r in values[:54]]),wide6=R.M.tally([r for a,r in values[54:108]]),old275=R.M.tally([r for a,r in values[108:]]))
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),mode=mode,error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,candidate=candidate,rounds=rounds,fresh=None,reload=None,full_stability_gate=False));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(mode):
    folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists()
    ids=[s['pid']]+[v for g in read(folder/'worker_generations.json') for v in [g['parent']]+g['workers']];assert all(not Path('/proc',str(v)).exists() for v in ids)
    rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()];replays=[json.loads(x) for x in (folder/'replays.jsonl').read_text().splitlines()] if (folder/'replays.jsonl').exists() else []
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items()) and int(a['native_counts'].sum())==r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        if r['full']:M.audit_full_history(p,r,a)
    bypath={r['path']:r for r in rows}
    for r in replays:
        v=bypath[r['reference_path']];assert r['exact'] and r['array_digest']==v['array_digest'] and r['array_hashes']==v['array_hashes'] and r['integration_sha256']==v['integration_sha256']
    rebuilt=0
    if mode!='baseline':
        obj=L.make(p);current=p['initial'];keep=None
        for row in s['rounds']:
            before=P.state(*obj);aa=[E.N.S.arrays(r['path']) for r in row['samples']];result=L.fit(obj,aa,p,lambda:None,lambda *_:None)
            assert result==row['update'] and digest(P.state(*obj))==row['proposal']['joint_hash'];assert sha(row['proposal']['path'])==row['proposal']['sha256'];rebuilt+=2*result['Actor_Adam']
            if not row['accepted']:P.load(obj,before)
            assert digest(P.state(*obj))==row['rollback_hash']
        assert digest(P.state(*obj))==s['candidate']['model']['joint_hash'] or not s['rounds']
    assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==len(rows)+len(replays)+4
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,historical_discarded_Actor_and_V_Adam=rebuilt,full_training_arrays_hidden_density_reward_reconstructed=True,compact_evaluation_NOT_full120_reconstructed=True,physics=s['physics'],commands=s['commands'],candidate=s['candidate'],raw_sha={str(f):sha(f) for f in folder.rglob('*.npz')}))

if __name__=='__main__':
    try:globals()[sys.argv[1]](*sys.argv[2:])
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
