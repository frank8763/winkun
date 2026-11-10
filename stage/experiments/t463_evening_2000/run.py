"""Finite offline evening relay through20:00, with fixed scientific branches."""
from pathlib import Path
import copy, hashlib, json, os, shutil, subprocess, sys, time, traceback
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t463_evening_2000';DOC=ROOT/'stage/docs/review_20261110'
sys.path.insert(0,str(HERE))
import t463_learning as L
J,H,M,P,E,torch,np=L.J,L.H,L.M,L.P,L.E,L.torch,L.np
R=J.R
read,save,sha,digest=J.read,J.save,J.sha,J.digest
ACTIVE=OUT
MODES=['aggregate','bounded','ppo']


def rootplan():return read(OUT/'protocol.json')
def verify(p,full=False):
    root=rootplan()
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()<=2**30
    if (OUT/'cache_summary.json').exists():assert M.used()-M.used(OUT)+root['max_bytes']<=2**30
    if ACTIVE!=OUT:assert M.used(ACTIVE)<=p['max_bytes']
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f


def worker(pipe,index,p,counts):
    P.frozen=L.frozen
    M.worker(pipe,index,p,counts)


class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard,self.children,self.pipes=guard,[],[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(p.get('worker_count',8)):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close()
                self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(len(self.children))];assert all(r['state']=='READY' for r in self.ready)
            f=Path(p['out'])/'worker_generations.json';gg=read(f) if f.exists() else []
            gg.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,gg)
        except BaseException:self.close();raise


def setup(folder):
    global ACTIVE
    ACTIVE=Path(folder);M.OUT,M.verify,M.Pool=ACTIVE,verify,Pool;M.setup_runtime()


def child_config(root,mode,model,success):
    p=copy.deepcopy(root['science'])
    limit={'aggregate':10,'bounded':10,'ppo':16}[mode]*2**20
    p.update(id='T463_'+mode.upper(),out=str(OUT/mode),mode=mode,learning_initial=model,current_success_ids=success,
        max_bytes=limit,max_physics=8000000,max_commands=450,main_seconds=1800,learning_seconds=900,
        model_bytes=2*2**20 if mode=='ppo' else 768*1024,full_bytes=448*1024,compact_bytes=4096,
        preflight_commands=0,ppo_epochs=4,ppo_aux_weight=.1,target_kl=.02,
        worker_count=8,protected_files=root['protected_files'])
    return p


def validate_bundle(p):
    assert len(p['aggregate_recovery10'])==10 and len(p['current_own6'])==6
    for r in p['aggregate_recovery10']+p['current_own6']:
        assert r['success'] and not r['collision'] and sha(r['path'])==r['sha256']
        a=E.N.S.arrays(r['path']);assert digest(a)==r['array_digest']
        assert {k:digest(v) for k,v in a.items()}==r['array_hashes']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].tolist()==[0,1,0]


def prepare():
    assert not OUT.exists(),'Never restart'
    prior=ROOT/'stage/runs/t462_recurrent_parent_correction';s=read(prior/'summary.json')
    assert s['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS'
    assert read(DOC/'T462_FINAL_REPORT.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'driver_started.json')['pid'])).exists()
    base=read(prior/'protocol.json');init=read(ROOT/'stage/runs/t461_validated_recovery_bundle/summary.json')['candidate']
    keys=['source_protocol','initial_model','student_start_model','student_start_sha256','raw_sigma','phase','relay_stage',
        'old8_jobs','old275_jobs','main_jobs','fresh_jobs','gamma_native','lambda_native','value_lr','actor_seed','value_seed','seed',
        'old_states','old_rows','input_mean','input_scale','teacher_rows','research_initial','initial_success_ids','failed_job','repair_lr',
        'residual_seed','base_actor_digest','own_rows','new_original_own5']
    science={k:copy.deepcopy(base[k]) for k in keys}
    science.update(learning_initial=init['model'],current_success_ids=init['success_ids'],
        current_own6=s['own_rows'],aggregate_recovery10=base['validated_recovery6']+s['query_rows'],
        teacher_first=20,ppo_aux_weight=.1,target_kl=.02)
    validate_bundle(science)
    jobs={j['id']:j for j in science['main_jobs']+science['old275_jobs']}
    trainids=['dev_task6_00','dev_task6_17','dev_task6_22','dev_task6_26',
        'wide117_80_dev6_LEFT_06','wide117_80_dev6_LEFT_13','wide117_80_dev6_RIGHT_13','dev_task2_13']
    science['ppo_training_jobs']=[jobs[k] for k in trainids]
    science['ppo_training_seeds']=[202611104630+i for i in range(8)]
    # Frozen8m centers are only eligible after full6m gates, audit and backup.
    absolute=datetime(2026,11,10,20,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p=dict(id='T463_FINITE_EVENING_RELAY_2000',out=str(OUT),science=science,
        initial_candidate=init,source_parent_T462=s['proposal'],modes=MODES,max_bytes=87*2**20,
        root_partition_bytes=8*2**20,gate_partition_bytes=43*2**20,
        child_partition_bytes=dict(aggregate=10*2**20,bounded=10*2**20,ppo=16*2**20),
        max_physics=30000000,max_commands=1600,max_main_seconds=12000,
        absolute_learning_stop=absolute-4500,absolute_native_stop=absolute-2700,absolute_all_stop=absolute,
        child_main_seconds=1800,child_audit_seconds=600,gate_main_seconds=1800,gate_audit_seconds=600,
        backup_seconds=900,main_reserve_after_each_child=900,
        transfer='Only audited candidate preserving every current success may replace current; all329 triggersgate immediately. Rejected model cannot transfer.',
        gates='same329safe thenone16fresh then78newprocess/modelpathfull exact. Holdoutfailure stops scientificchain, no adaptingtofresh.',
        eight='Full6m nativegate+audit+binarybackup first.8m needs a separate actualregistered geometry andbudget; no8code launched bythis queue.',
        protected_files={**base['protected_files'],init['model']['path']:init['model']['sha256']},
        user_authorization='Nov10 explicitly request inspect and start unattended Stage experiments until20:00; fixed finite relay, no unlimitedbudget/Phase/runtimeplanner.',
        scope='Known methods jointengineeringprototype, not singlefactor/novelty/safety/truewholemapnavigationproof.',
        all_original_errors_models_and_failures_preserved=True)
    # No unimplemented8m geometry is silently invented; eligibility is explicit.
    p['eight_diagnostic_implemented']=False
    files=set(base['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['protected_files'])
    files|={r['path'] for r in science['aggregate_recovery10']+science['current_own6']}
    files|={str(DOC/'T463_PLAN_2000.md'),str(DOC/'T463_PRIMARY_SOURCES.md'),str(DOC/'T462_FINDINGS_AND_EVENING.md'),
        str(prior/'protocol.json'),str(prior/'summary.json'),str(prior/'audit.json'),str(prior/'cases.jsonl'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T462.json')}
    import t463_cache as C
    need=max(1,M.used()+p['max_bytes']+512*1024-2**30)
    cache=C.plan(files,need)
    cache_path=HERE/'cache_plan.json';assert not cache_path.exists()
    cache_path.write_bytes(C.encoded(cache));assert cache_path.stat().st_size<2*2**20
    p['remote_cache_path']=str(cache_path);p['remote_cache_sha256']=sha(cache_path)
    p['cache_total_reserved_bytes']=p['max_bytes'];p['cache_hard_stop_epoch']=time.time()+1200
    files.add(str(cache_path))
    for group in cache['groups']:
        for k in ['receipt','manifest']:files.add(str(ROOT/group[k]))
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    assert M.used()-cache['conservative_released_bytes']+p['max_bytes']+512*1024<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T463_BUDGET.json',dict(state='REGISTERED_BEFORE_CACHE_PREFLIGHT_LAUNCH',
        total_reserved_bytes=p['max_bytes'],current_bytes=M.used(),required_eviction=need,
        cache_net_release=cache['conservative_released_bytes'],partitions=p['child_partition_bytes'],gate=p['gate_partition_bytes'],
        max_physics=30000000,max_commands=1600,max_main_seconds=12000,
        conditional_fresh16_reload78_fully_reserved=True,absolute_all_stop=absolute,
        fixed_modes=MODES,expected_science_minutes=[40,100],hard_stop_NOT_guaranteed_fullwindow=True))
    print('T463_REGISTERED',need,flush=True)


def cache_evict():
    import t463_cache as C
    p=rootplan();assert sha(p['remote_cache_path'])==p['remote_cache_sha256']
    p['remote_cache']=read(p['remote_cache_path']);C.evict(p)


def cache_audit():
    import t463_cache as C
    p=rootplan();assert sha(p['remote_cache_path'])==p['remote_cache_sha256']
    p['remote_cache']=read(p['remote_cache_path']);C.audit(p);verify(rootplan(),True)


def synthetic(obj,p):
    q=obj[0];rng=np.random.default_rng(202611104639);result=[]
    for i in range(8):
        x=(np.asarray(p['input_mean'])[None]+rng.normal(size=(9,120))*.01).astype(np.float32)
        with torch.no_grad():
            loc=q.sequence(torch.tensor(x,device=next(q.parameters()).device));raw=loc+loc.new_tensor(rng.normal(size=(9,2)))*.2
            atom=torch.where(raw>=P.L.EDGE,1,torch.where(raw<=-P.L.EDGE,-1,0)).to(torch.int8)
            lp=P.L.density(loc,raw,atom,p['raw_sigma'])
        flags=np.zeros((9,3),np.uint8);flags[-1,1]=1
        result.append(dict(states=x,next_states=np.concatenate([x[1:],x[-1:]]),locations=loc.cpu().numpy(),
            raw_actions=raw.cpu().numpy(),atom=atom.cpu().numpy(),logp=lp.cpu().numpy(),rewards=rng.normal(size=9)*.01,
            native_counts=np.full(9,100,np.uint8),end_flags=flags,ownership=np.zeros(9,np.uint8),stochastic=np.ones(9,np.uint8)))
    return result


def actual_bound(q,data):
    error=0.
    with torch.no_grad():
        for g in data['groups']+data['own']:
            error=max(error,float((q.sequence(g['x'])-g['target']).abs().max()))
    assert error<=.02+1e-12
    return error


def preflight():
    root=rootplan();verify(root,True)
    assert not (OUT/'preflight.json').exists() and read(OUT/'cache_audit.json')['state'].startswith('PASS')
    results={};models=[]
    for mode in MODES:
        p=child_config(root,mode,root['initial_candidate']['model'],root['initial_candidate']['success_ids'])
        a,b=L.make(p,mode),L.make(p,mode);initial=L.state(*a);data=L.build_data(p)
        if mode=='ppo':
            arrays=synthetic(a,p)
            va=L.ppo_fit(a,arrays,p,data,lambda:None,lambda *_:None)
            vb=L.ppo_fit(b,arrays,p,data,lambda:None,lambda *_:None)
        else:
            va=L.bc_step(a,data,mode);vb=L.bc_step(b,data,mode)
        assert va==vb and digest(L.state(*a))==digest(L.state(*b))
        if mode!='aggregate':
            actual_bound(a[0],data)
            accepted=L.state(*a)
            with torch.no_grad():a[0].head.residual[-1].bias.add_(1.)
            assert L.constraint_snapshot(a[0],data['constraint'])['logit_abs_max']>.016
            clipped=L.project(a[0],data['constraint']);assert clipped['radial_scale']<1
            actual_bound(a[0],data)
            L.load(a,accepted)
        model=L.export(a,OUT/f'preflight_{mode}.pt.gz',p['model_bytes']);models.append(model)
        assert L.frozen(p,model['path']).phash()==a[0].phash()
        L.load(a,initial);L.load(b,initial);assert digest(L.state(*a))==digest(initial)==digest(L.state(*b))
        results[mode]=dict(result=va,model=model,support_references=data['reference_count'],support_own=data['own_count'])
        del a,b,data;torch.cuda.empty_cache()
    p=child_config(root,'ppo',root['initial_candidate']['model'],root['initial_candidate']['success_ids'])
    folder=OUT/'preflight_native';folder.mkdir();p.update(out=str(folder),native_stop_epoch=time.time()+180,max_physics=1)
    counts=R.M.mp.get_context('spawn').RawArray('q',9);pool=Pool(p,counts,lambda:None)
    try:
        reply=pool.request([dict(model=models[-1]['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']])
        assert all(r['state']=='CANCEL_PASS' for r in reply) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,results=results,
        discarded_BC_Adam=4,discarded_PPO_Actor_Adam=8,discarded_Value_Adam=8,
        live_shadow_projection_full_history_actual_bound_and_rollback_exact=True,eight_workers_cancelled=True,
        fresh16_NOT_consumed=True,reload78_NOT_executed=True))
    print('T463_PREFLIGHT_PASS',flush=True)


def childmain(mode):
    root=rootplan();folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True)
    assert not (folder/'summary.json').exists();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    status,error,candidate,model='FAILED',None,None,None;obj=shadow=before=None;fit=None;training=[]
    try:
        obj,shadow=L.make(p,mode),L.make(p,mode);before=L.state(*obj);data=L.build_data(p)
        if mode=='ppo':
            rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
            values=rt.request(p['ppo_training_jobs'],p['learning_initial'],'PURE_STUDENT_TRAIN8',p['ppo_training_seeds'],full=True)
            training=[r for a,r in values]
            rt.request(p['ppo_training_jobs'],p['learning_initial'],'PURE_STUDENT_EXACT8',p['ppo_training_seeds'],full=True,references=training)
            aa=L.ppo_fit(obj,[a for a,r in values],p,data,lambda:rt.guard(True),M.events(mode,'live'))
            bb=L.ppo_fit(shadow,[a for a,r in values],p,data,lambda:rt.guard(True),M.events(mode,'shadow'))
            assert aa==bb and digest(L.state(*obj))==digest(L.state(*shadow));fit=aa
        else:
            trace=[]
            for k in range(128):
                rt.role=mode.upper()+'_FIT';rt.guard(True);values=[]
                for which,objects in [('live',obj),('shadow',shadow)]:
                    M.events(mode,which)('BEFORE',k);values.append(L.bc_step(objects,data,mode));M.events(mode,which)('AFTER',k)
                assert values[0]==values[1] and digest(L.state(*obj))==digest(L.state(*shadow))
                trace.append(values[0])
                if k%16==0:save(folder/'fit_progress.json',dict(steps=k+1,latest=values[0],epoch=time.time()))
            fit=dict(trace=trace,Actor_Adam=128,V_Adam=0)
        if mode!='aggregate':fit['actual_max_support_location_change']=actual_bound(obj[0],data)
        save(folder/'fit.json',fit);model=L.export(obj,folder/'proposal.pt.gz',p['model_bytes'])
        if rt.pool is None:rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        candidate,_=M.evaluate(rt,p,model,'FINAL329')
        success=set(candidate['success_ids'])
        candidate.update(preserves_current=set(p['current_success_ids'])<=success,
            preserves_original328=set(p['initial_success_ids'])<=success,repairs00=p['failed_job']['id'] in success,
            development_admitted=len(success)==329)
        status='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and before is not None:
            L.load(obj,before);L.load(shadow,before);assert digest(L.state(*obj))==digest(before)
        save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(folder/'summary.json',dict(state=status,pid=os.getpid(),mode=mode,elapsed_s=time.time()-rt.start,
            physics=sum(rt.counts),commands=rt.commands,error=error,candidate=candidate,model=model,fit=fit,training_rows=training,
            all_native_workers_stopped=True,fresh16=None,reload78=None,six_gate=False,eight_started=False))
        save(folder/'status.json',dict(state=status,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error
    print('T463_CHILD_COMPLETE',mode,flush=True)


def childaudit(mode):
    folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists()
    ids=[s['pid']]+[pid for g in read(folder/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    obj=L.make(p,mode);data=L.build_data(p)
    rows=list(map(json.loads,(folder/'cases.jsonl').read_text().splitlines()))
    replays=list(map(json.loads,(folder/'replays.jsonl').read_text().splitlines())) if (folder/'replays.jsonl').exists() else []
    if mode=='ppo':
        train=[r for r in rows if r['role']=='PURE_STUDENT_TRAIN8'];assert len(train)==len(replays)==8
        fit=L.ppo_fit(obj,[E.N.S.arrays(r['path']) for r in train],p,data,lambda:None,lambda *_:None)
        calls=fit['Actor_Adam']+fit['V_Adam']
    else:
        fit=dict(trace=[L.bc_step(obj,data,mode) for _ in range(128)],Actor_Adam=128,V_Adam=0);calls=128
    if mode!='aggregate':fit['actual_max_support_location_change']=actual_bound(obj[0],data)
    assert fit==read(folder/'fit.json') and digest(L.state(*obj))==s['model']['joint_hash']
    assert L.base_digest(obj[0])==p['base_actor_digest'] and sha(s['model']['path'])==s['model']['sha256']
    assert L.frozen(p,s['model']['path']).phash()==obj[0].phash()
    ev=list(map(json.loads,(folder/'optimizer_calls.jsonl').read_text().splitlines()))
    n=fit['Actor_Adam']
    assert len(ev)==4*n
    for w in ['live','shadow']:assert [r[2:] for r in ev if r[:2]==[mode,w]]==[[where,k] for k in range(n) for where in ['BEFORE','AFTER']]
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert {k:digest(v) for k,v in a.items()}=={k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum())==r['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        assert not r['teacher'] and not r['administrative_censor']
        if r['full']:
            assert digest(a)==r['array_digest'];P.frozen=L.frozen
            try:M.audit_full_history(p,r,a)
            finally:P.frozen=H.ORIGINAL_FROZEN
    bypath={r['path']:r for r in rows}
    for r in replays:
        old=bypath[r['reference_path']]
        assert r['array_digest']==old['array_digest'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    final=[r for r in rows if r['role']=='FINAL329'];assert len(final)==len({r['job_id'] for r in final})==329
    assert M.R.M.tally(final[:54])==s['candidate']['wide54'] and M.R.M.tally(final[54:])==s['candidate']['old275']
    assert all(r['loaded_model_sha256']==s['model']['sha256'] and r['policy_hash']==s['model']['policy_hash'] for r in final)
    assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==16+len(rows)+len(replays)
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Adam=calls,
        all_live_shadow_historical_Actor_Value_Adam_exact=True,frozen_original_parameters_exact=True,
        exact_support_bound_actual_full_history_verified=mode!='aggregate',
        compact329_NOT_full120_reconstructed=True,full_training_arrays_density_reward_history_exact=mode=='ppo',
        physics=s['physics'],commands=s['commands'],six_gate=False))
    print('T463_CHILD_AUDIT_PASS',mode,flush=True)


def gatemain():
    folder=OUT/'gate';p=read(folder/'protocol.json');setup(folder);verify(p,True)
    assert not (folder/'summary.json').exists();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    status,error,fresh,reload='FAILED',None,None,None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        values=rt.request(p['fresh_jobs'],p['learning_initial'],'FRESH16');fresh=M.R.M.tally([r for a,r in values])
        if fresh['success']==16:
            baseline={r['job_id']:r for r in p['baseline329']};baseline.update({r['job_id']:r for a,r in values})
            rt.pool.close();rt.commands+=8
            copyfile=folder/'independent_model.pt.gz';shutil.copyfile(p['learning_initial']['path'],copyfile)
            assert sha(copyfile)==p['learning_initial']['sha256'];model=dict(p['learning_initial'],path=str(copyfile))
            rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=8
            values=rt.request(p['old8_jobs']+p['main_jobs']+p['fresh_jobs'],model,'INDEPENDENT_RELOAD78',full=True)
            for a,r in values:
                prev=baseline[r['job_id']]
                assert r['success'] and not r['collision'] and r['array_digest']==prev['array_digest']
                assert r['array_hashes']==prev['array_hashes'] and r['integration_sha256']==prev['integration_sha256']
            reload=dict(valid=78,exact=78,all_safe=True)
        status='COMPLETE'
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(folder/'summary.json',dict(state=status,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands,
            error=error,fresh16=fresh,reload78=reload,native_gate=bool(reload),six_gate=False,all_native_workers_stopped=True))
        save(folder/'status.json',dict(state=status,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error


def gateaudit():
    folder=OUT/'gate';p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists()
    ids=[s['pid']]+[pid for g in read(folder/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    rows=list(map(json.loads,(folder/'cases.jsonl').read_text().splitlines()))
    assert s['physics']==sum(r['physics'] for r in rows) and s['commands']==16*len(read(folder/'worker_generations.json'))+len(rows)
    baseline={r['job_id']:r for r in p['baseline329']}
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert {k:digest(v) for k,v in a.items()}=={k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum())==r['physics']
        if r['role']=='FRESH16':baseline[r['job_id']]=r
        else:
            prev=baseline[r['job_id']];assert r['success'] and r['array_hashes']==prev['array_hashes'] and r['array_digest']==prev['array_digest']
            assert r['integration_sha256']==prev['integration_sha256'] and digest(a)==r['array_digest']
            P.frozen=L.frozen
            try:M.audit_full_history(p,r,a)
            finally:P.frozen=H.ORIGINAL_FROZEN
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,optimizer=0,
        every_full78_history_hidden_action_density_reward_integration_exact=bool(s['reload78']),
        fresh16=s['fresh16'],reload78=s['reload78'],native_gate=s['native_gate'],six_gate=False))


def launch():
    p=rootplan();verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert time.time()+1800+600+900<p['absolute_all_stop']
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T463','--why=Finite offline Stage relay until20:00',
            'timeout','--signal=TERM','--kill-after=2',str(int(p['absolute_all_stop']-now)),
            'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=p['absolute_all_stop'],learning_stop=p['absolute_learning_stop'],
        native_stop=p['absolute_native_stop'],protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']))
    print('T463_ACTUALLY_LAUNCHED',c.pid,flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]](*sys.argv[2:])
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
