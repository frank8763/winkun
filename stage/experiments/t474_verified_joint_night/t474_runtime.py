"""Offline native verification followed by separately audited finite repairs."""
from pathlib import Path
import copy,contextlib,fcntl,importlib.util,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t474_verified_joint_night';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import t474_learning as L
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
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def records_equal(a,b):
    # Fields describing which pass produced a row are expected to differ.
    ignored={'pass_index','first_search_pass','exact_replay_not_independent','independent_new_sample','path','sha256'}
    return {k:v for k,v in a.items() if k not in ignored}=={k:v for k,v in b.items() if k not in ignored}

def worker(pipe,index,p,counts):
    P.frozen=L.frozen;M.worker(pipe,index,p,counts)
class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard,self.children,self.pipes=guard,[],[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(8)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';g=read(f) if f.exists() else [];g.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,g)
        except BaseException:self.close();raise

def setup(folder):
    global ACTIVE
    ACTIVE=Path(folder);M.OUT,M.verify,M.Pool=ACTIVE,verify,Pool;M.setup_runtime()

def verify_main():
    folder=OUT/'verified';p=read(folder/'protocol.json');verify(p,True);assert not (folder/'summary.json').exists()
    S.OUT=folder;S.readonly_cpu();E.L.install_stop_handlers();start=time.time();deadline=min(start+600,p['absolute_native_stop']);physics=commands=0;s=w=None;rows=[];state='FAILED';error=None
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=S.A.arrays(p['source_row']);expected=E.N.S.arrays(p['raw15_reference']['path']);controls=expected['actions'][20:].copy()
    def guard(physical=False):
        E.L.check_stop();assert time.time()<deadline-5 and physics<=p['max_physics']
    def charge():
        nonlocal physics
        physics+=1;assert physics<=p['max_physics']
    try:
        with (folder/'native_console.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            _,native,create,w=S.setup(p,folder/'native');commands=1
            for rep in range(2):
                guard();s=S.session(p,native,create,w,guard,charge);before=physics;commands+=1
                for _ in range(20):s.step(owner=0)
                prefix=S.payload(s);S.prefix_check(prefix,source,True)
                for u in controls:
                    s.step(u,owner=1,actor_active=True)
                    if s.done():break
                assert s.done();a=S.payload(s);assert set(a)==set(expected) and all(np.array_equal(a[k],expected[k]) for k in a)
                assert s.result()['success'] and not s.result()['collision']
                path=folder/f'replay{rep}.npz';S.write_full(path,a,p)
                row=S.make_row(s,a,p,rep,p['raw15_reference']['index'],'selected',physics-before-2000,path)
                row['sha256']=sha(path);row['raw_teacher_control']=True;rows.append(row)
                s.close();s=None;save(folder/'progress.json',dict(replays_completed=rep+1,physics=physics,epoch=time.time()))
            assert records_equal(rows[0],rows[1]);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(S.payload(s)),p['full_bytes'])
            except BaseException:pass
    finally:
        if s:s.close()
        if w:w.close()
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=physics,commands=commands,elapsed_s=time.time()-start,rows=rows,error=error,optimizer=0,no_new_search=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE',error

def verify_audit():
    folder=OUT/'verified';p=read(folder/'protocol.json');verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    assert not (folder/'audit.json').exists();S.readonly_cpu();source=S.A.arrays(p['source_row']);arrays=[]
    for row in s['rows']:
        assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path']);S.audit_payload(p,row,a,source,lambda:verify(p));arrays.append(a)
    old=E.N.S.arrays(p['raw15_reference']['path'])
    assert len(arrays)==2 and all(set(a)==set(old) and all(np.array_equal(a[k],old[k]) for k in a) for a in arrays)
    assert records_equal(*s['rows']) and s['physics']==sum(r['physics'] for r in s['rows'])<=p['max_physics'] and s['commands']==3
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,optimizer=0,actual_two_independent_original_start_replays_all_arrays_exact=True,all_history_density_reward_hidden_and_integration_rebuilt=True,source_T473_failed_preserved=True,physics=s['physics'],commands=s['commands'],verified_raw15=s['rows'][0]))

def train_main(mode):
    assert mode in L.MODES;folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    baseguard=rt.guard
    def guard(learning=False):
        assert time.time()<(p['absolute_learning_stop'] if learning else p['absolute_native_stop'])-5;baseguard(learning)
    rt.guard=guard;state='FAILED';error=None;candidate=proposal=None;completed=0;obj=shadow=initial=None
    try:
        obj,shadow=L.make(p),L.make(p);initial=L.state(*obj);batch=L.data(p);trace=[]
        save(folder/'support.json',dict(reference_count=batch['reference_count'],own_count=batch['own_count'],correction_count=4,extra_function_histories=4,counterexample_provenance=batch['counterexample_provenance'],raw15_suffix_only=True))
        for k in range(128):
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
        if rt.pool:rt.pool.close();rt.commands+=8
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),mode=mode,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,candidate=candidate,proposal=proposal,completed_paired_BC=completed,new_PPO=0,new_Value=0,fresh16=None,reload78=None,eight_started=False))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def train_audit(mode):
    folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();ids=[s['pid']]+[v for g in read(folder/'worker_generations.json') for v in [g['parent']]+g['workers']];assert all(not Path('/proc',str(v)).exists() for v in ids)
    obj=L.make(p);batch=L.data(p);trace=[L.bc_step(obj,batch) for _ in range(128)]
    assert trace==read(folder/'repair_trace.json') and digest(L.state(*obj))==s['proposal']['joint_hash'] and L.base_digest(obj[0])==p['base_actor_digest']
    assert sha(s['proposal']['path'])==s['proposal']['sha256'] and L.frozen(p,s['proposal']['path']).phash()==s['proposal']['policy_hash']
    events=[json.loads(x) for x in (folder/'optimizer_calls.jsonl').read_text().splitlines()]
    for which in ['live','shadow']:assert [x[2:] for x in events if x[:2]==[mode,which]]==[[stage,k] for k in range(128) for stage in ['BEFORE','AFTER']]
    rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()];assert len(rows)==len({r['job_id'] for r in rows})==329
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        assert not r['administrative_censor'] and r['policy_hash']==s['proposal']['policy_hash'] and r['loaded_model_sha256']==s['proposal']['sha256']
    safe={r['job_id'] for r in rows if r['success'] and not r['collision']};assert safe==set(s['candidate']['success_ids'])
    assert R.M.tally(rows[:54])==s['candidate']['wide54'] and R.M.tally(rows[54:])==s['candidate']['old275']
    assert s['candidate']['preserves_current']==(set(p['current_success_ids'])<=safe) and s['candidate']['development_admitted']==(len(safe)==329)
    assert s['physics']==sum(r['physics'] for r in rows) and s['commands']==16+len(rows)==345
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Adam=128,all_live_shadow_jointActorAdam_rebuilt=True,all329_actual_success_ids_checked=True,compact329_NOT_full120_reconstructed=True,physics=s['physics'],commands=s['commands']))

def gates():
    X=L.X;X.OUT,X.HERE,X.DOC=OUT,HERE,DOC;return X

def gatemain():gates().gatemain()
def gateaudit():gates().gateaudit()

if __name__=='__main__':
    try:globals()[sys.argv[1]](*sys.argv[2:])
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
