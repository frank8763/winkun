"""Offline native verification followed by separately audited finite repairs."""
from pathlib import Path
import copy,contextlib,fcntl,importlib.util,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t475_direct_observation_night';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import t475_learning as L
J,H,M,P,E,R,torch,np=L.J,L.H,L.M,L.P,L.E,L.R,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
sp=importlib.util.spec_from_file_location('stage474_pinned473',ROOT/'stage/experiments/t473_full_horizon_teacher/science.py')
S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S)
ACTIVE=OUT

def rootplan():return read(OUT/'protocol.json')
def verify(p,full=False):
    root=rootplan();assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    if ACTIVE!=OUT:assert M.used(ACTIVE)<=p['max_bytes']
    assert M.used(OUT)-(M.used(OUT/'gate') if (OUT/'gate').exists() else 0)<=root['main_partition_bytes']
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def worker(pipe,index,p,counts):
    P.frozen=L.frozen;M.worker(pipe,index,p,counts)
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
        self.p=p;self.start=time.time();self.deadline=min(self.start+p['main_seconds'],p['absolute_native_stop']);self.learnstop=min(self.start+p['learning_seconds'],p['absolute_learning_stop'])
        self.last=0.;self.role='SETUP';self.counts=R.M.mp.get_context('spawn').RawArray('q',3);self.commands=2;self.pool=None;self.rows=[];self.replays=[]
        self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard(self,learning=False):
        E.L.check_stop();assert time.time()<(self.learnstop if learning else self.deadline)-5 and sum(self.counts)<=self.p['max_physics'] and self.commands+2<=self.p['max_commands']
        if time.time()-self.last>10:
            self.last=time.time();verify(self.p);save(ACTIVE/'status.json',dict(state=self.role,epoch=self.last,pid=os.getpid(),physics=sum(self.counts),commands=self.commands))
    def request(self,jobs,model,role,seeds=None,full=False,references=None,query=False):
        assert not query;self.role=role;result=[]
        for off in range(0,len(jobs),2):
            js=jobs[off:off+2];self.guard();assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics'];assert self.commands+len(js)+2<=self.p['max_commands'];self.commands+=len(js)
            requests=[]
            for i,j in enumerate(js):
                path=references[off+i]['path'] if references is not None else str(ACTIVE/'episodes'/f'{len(self.rows)+i:04d}.npz') if full else None
                requests.append(dict(model=model['path'],job=j,seed=None,full_path=path,query=False))
            for local,((a,r),j,req) in enumerate(zip(self.pool.request(requests),js,requests)):
                r.update(job_id=j['id'],role=role,seed=None,request=req,loaded_model_sha256=sha(model['path']));assert r['policy_hash']==model['policy_hash'] and int(a['native_counts'].sum())==r['physics']
                if references is not None:
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
    ACTIVE=Path(folder);M.OUT,M.verify,M.Pool=ACTIVE,verify,Pool;M.setup_runtime()

def main():
    mode="direct"
    assert mode in L.MODES;folder=OUT;p=read(folder/'protocol.json');setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Runtime(p)
    baseguard=rt.guard
    def guard(learning=False):
        assert time.time()<(p['absolute_learning_stop'] if learning else p['absolute_native_stop'])-5;baseguard(learning)
    rt.guard=guard;state='FAILED';error=None;candidate=proposal=None;completed=0;obj=shadow=initial=None
    try:
        obj,shadow=L.make(p),L.make(p);initial=L.state(*obj);batch=L.data(p);trace=[]
        save(folder/'support.json',dict(reference_count=batch['reference_count'],own_count=batch['own_count'],correction_count=4,extra_function_histories=4,counterexample_provenance=batch['counterexample_provenance'],raw15_suffix_only=True))
        for k in range(512):
            rt.role=mode.upper()+'_FIT';rt.guard(True);values=[]
            for which,objects in [('live',obj),('shadow',shadow)]:
                M.events(mode,which)('BEFORE',k);values.append(L.bc_step(objects,batch));M.events(mode,which)('AFTER',k)
            assert values[0]==values[1] and digest(L.state(*obj))==digest(L.state(*shadow));completed+=1;trace.append(values[0])
            if k%16==0:save(folder/'fit_progress.json',dict(steps=completed,latest=values[0],epoch=time.time()))
        save(folder/'repair_trace.json',trace);proposal=L.export(obj,folder/'proposal.pt.gz',p['model_bytes'])
        rt.pool=Pool(dict(p,native_stop_epoch=min(rt.deadline,p['absolute_native_stop'])),rt.counts,rt.guard)
        candidate,_=M.evaluate(rt,p,proposal,'FINAL329');success=set(candidate['success_ids']);candidate.update(preserves_current=set(p['current_success_ids'])<=success,development_admitted=len(success)==329)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and initial is not None:L.load(obj,initial);L.load(shadow,initial);assert digest(L.state(*obj))==digest(initial)
        save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),mode=mode,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,candidate=candidate,proposal=proposal,completed_paired_BC=completed,new_PPO=0,new_Value=0,fresh16=None,reload78=None,eight_started=False))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    mode='direct'
    folder=OUT;p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();ids=[s['pid']]+[v for g in read(folder/'worker_generations.json') for v in [g['parent']]+g['workers']];assert all(not Path('/proc',str(v)).exists() for v in ids)
    obj=L.make(p);batch=L.data(p);trace=[L.bc_step(obj,batch) for _ in range(512)]
    assert trace==read(folder/'repair_trace.json') and digest(L.state(*obj))==s['proposal']['joint_hash'] and L.base_digest(obj[0])==p['frozen327_actor_digest']
    assert sha(s['proposal']['path'])==s['proposal']['sha256'] and L.frozen(p,s['proposal']['path']).phash()==s['proposal']['policy_hash']
    events=[json.loads(x) for x in (folder/'optimizer_calls.jsonl').read_text().splitlines()]
    for which in ['live','shadow']:assert [x[2:] for x in events if x[:2]==[mode,which]]==[[stage,k] for k in range(512) for stage in ['BEFORE','AFTER']]
    rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()];assert len(rows)==len({r['job_id'] for r in rows})==329
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        assert not r['administrative_censor'] and r['policy_hash']==s['proposal']['policy_hash'] and r['loaded_model_sha256']==s['proposal']['sha256']
    safe={r['job_id'] for r in rows if r['success'] and not r['collision']};assert safe==set(s['candidate']['success_ids'])
    assert R.M.tally(rows[:54])==s['candidate']['wide54'] and R.M.tally(rows[54:])==s['candidate']['old275']
    assert s['candidate']['preserves_current']==(set(p['current_success_ids'])<=safe) and s['candidate']['development_admitted']==(len(safe)==329)
    assert s['physics']==sum(r['physics'] for r in rows) and s['commands']==4+len(rows)==333
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Adam=512,all_live_shadow_jointActorAdam_rebuilt=True,all329_actual_success_ids_checked=True,compact329_NOT_full120_reconstructed=True,physics=s['physics'],commands=s['commands']))

def gatemain():
    folder=OUT/'gate';p=read(folder/'protocol.json');setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Runtime(p)
    state='FAILED';error=None;fresh=reload=None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard);model=p['gate_model']
        values=rt.request(p['fresh_jobs'],model,'FRESH16');fresh=R.M.tally([r for a,r in values])
        if fresh['success']==16:
            baseline={r['job_id']:r for r in p['baseline329']};baseline.update({r['job_id']:r for a,r in values})
            rt.pool.close();rt.commands+=2;path=folder/'independent_model.pt.gz';shutil.copyfile(model['path'],path);assert sha(path)==model['sha256'];model=dict(model,path=str(path))
            rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
            values=rt.request(p['old8_jobs']+p['main_jobs']+p['fresh_jobs'],model,'INDEPENDENT_RELOAD78',full=True)
            for a,r in values:
                old=baseline[r['job_id']];assert r['success'] and not r['collision'] and r['array_digest']==old['array_digest'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
            reload=dict(valid=78,exact=78,all_safe=True)
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands,error=error,fresh16=fresh,reload78=reload,native_gate=bool(reload)))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def gateaudit():
    folder=OUT/'gate';p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists()
    ids=[s['pid']]+[v for g in read(folder/'worker_generations.json') for v in [g['parent']]+g['workers']];assert all(not Path('/proc',str(v)).exists() for v in ids)
    rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()];baseline={r['job_id']:r for r in p['baseline329']}
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics']
        if r['role']=='FRESH16':baseline[r['job_id']]=r
        else:
            b=baseline[r['job_id']];assert r['success'] and r['array_digest']==b['array_digest'] and r['array_hashes']==b['array_hashes'] and r['integration_sha256']==b['integration_sha256']
            P.frozen=L.frozen
            try:M.audit_full_history(p,r,a)
            finally:P.frozen=H.ORIGINAL_FROZEN
    assert s['physics']==sum(r['physics'] for r in rows) and s['commands']==4*len(read(folder/'worker_generations.json'))+len(rows)<=102
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,optimizer=0,full78_history_density_hidden_rewards_and_integration_exact=bool(s['reload78']),fresh16=s['fresh16'],reload78=s['reload78'],native_gate=s['native_gate']))

if __name__=='__main__':
    try:globals()[sys.argv[1]](*sys.argv[2:])
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
