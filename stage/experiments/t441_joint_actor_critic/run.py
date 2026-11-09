"""Bounded T441 integration: joint native sampling, independent V/GAE/PPO."""
from pathlib import Path
import ast, copy, fcntl, gzip, io, json, multiprocessing as mp, os, shutil
import subprocess, sys, time, traceback
import t441_learning as L
C,E,torch,np=L.C,L.E,L.torch,L.np
ROOT=L.ROOT; HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t441_joint_actor_critic'; DOC=ROOT/'stage/docs/research_rescue_20261109'
read,save,sha,digest=L.read,L.save,L.sha,L.digest


def verify(p, sources=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    if sources:
        for path,h in p['inputs'].items(): assert sha(path)==h,path
    assert E.N.K.S.used(OUT)<=p['max_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30


class Pool(C.Pool):
    def __init__(self,p,counts,guard,n=8):
        self.guard=guard;self.children=[];self.pipes=[];ctx=mp.get_context('spawn')
        try:
            for i in range(n):
                a,b=ctx.Pipe();child=ctx.Process(target=C.worker,args=(b,i,p,counts));child.start();b.close()
                self.children.append(child);self.pipes.append(a)
            save(Path(p['out'])/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]))
            self.ready=[self.receive(i) for i in range(n)]
            assert all(x['state']=='READY' for x in self.ready)
        except BaseException:self.close();raise
    def request(self,items):
        assert 1<=len(items)<=len(self.children)
        for i,x in enumerate(items):self.pipes[i].send(x)
        return [self.receive(i) for i in range(len(items))]


def prepare():
    assert not OUT.exists(),'Never restart an existing directory'
    prior=read(ROOT/'stage/experiments/t437_joint_closed_loop_rl/protocol.json')
    p={k:copy.deepcopy(prior[k]) for k in ['source_protocol','initial_model','student_start_model',
        'student_start_sha256','expected_policy_hash','raw_sigma','phase','old8_jobs','old275_jobs','main_jobs']}
    training=[]
    for r in range(4):
        old=[]
        for task in range(8):
            candidates=sorted([j for j in p['old275_jobs'] if j['group']==f'task{task}' and j['id'].startswith('dev_')],key=lambda j:j['id'])
            old.append(p['old8_jobs'][task] if r==0 else candidates[[0,6,13,20][r]])
        wide=[]
        for side in ['LEFT','RIGHT']:
            for index in [r,r+7,r+14,r+20]:
                wide.append(next(j for j in p['main_jobs'] if j['id']==f'wide117_80_dev6_{side}_{index:02d}'))
        training.append(old+wide)
    centres=p['old8_jobs']+[next(j for j in p['main_jobs'] if j['id']==f'wide117_80_dev6_{s}_13') for s in ['LEFT','RIGHT']]
    train_cap=sum(j['case']['step_cap'] for jobs in training for j in jobs)
    worst=2*train_cap+13*40000+5*sum(j['case']['step_cap'] for j in centres)+sum(j['case']['step_cap'] for j in p['old275_jobs']+p['main_jobs'])
    p.update(id='T441_JOINT_RECURRENT_VALUE_GAE_INTEGRATION',out=str(OUT),relay_stage='representation_rl',
        actor_lr=3e-5,value_lr=1e-3,value_seed=20261109441,seed=202611094410,
        epochs=4,ppo_clip=.2,target_kl=.02,gamma_native=float(np.exp(-.002/60)),lambda_native=float(np.exp(-.002/4)),
        training_jobs=training,centre_jobs=centres,rounds=4,max_physics=14000000,max_commands=600,
        max_bytes=38*2**20,full_bytes=384*1024,compact_bytes=5*1024,
        main_seconds=5400,learning_seconds=3600,audit_seconds=1200,backup_seconds=600,all_seconds=8100,
        max_live_Actor_Adam=16,max_live_V_Adam=16,max_shadow_Actor_Adam=16,max_shadow_V_Adam=16,
        preflight_max_discarded_Actor_Adam=8,preflight_max_discarded_V_Adam=8,preflight_native=0,
        no_teacher_no_auxiliary_no_PCGrad=True,no_old_success_per_round_rollback=True,
        initialization='T427 research policy, fresh Actor Adam and independent zero-output V; safeT255 frozen',
        scope='Warm-start learning-loop integration; existing Actor retained to isolate framework changes. Not the future clean-network/random-init control.',
        final_selection='Fixed final round, not best panel. Full329 at final if all four rounds complete; fresh16/reload78 not admitted in this integration budget.',
        fresh16=None,reload78=None,six_gate=False,eight_started=False,
        worst_physics=worst,maximum_training_decisions=sum((j['case']['step_cap']+99)//100 for jobs in training for j in jobs),
        deployment='one causal120-input recurrent Actor; observation112 already contains remaining-time fraction; no planner/waypoints/runtime teacher/router',
        user_authorization='Nov9 explicit: continue according to reviewed Actor-Critic plan; new finite integration window, no old-run restart or deadline extension')
    assert worst<p['max_physics']
    files={str(f) for folder in [ROOT/'stage/experiments',ROOT/'stage/runtime'] for f in folder.rglob('*.py')}
    files|={p['source_protocol'],p['initial_model'],p['student_start_model']}
    for j in p['old275_jobs']+p['main_jobs']:
        files.update(j[k] for k in ['exact_snapshot','near_snapshot'] if j.get(k))
    base=read(p['source_protocol']); files.update(base['input_hashes'])
    p['inputs']={f:sha(f) for f in sorted(files)}
    before=E.N.K.S.used(ROOT/'stage/runs');assert before+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T441_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=before,
        global_full_reserved=before+p['max_bytes'],max_bytes=p['max_bytes'],worst_physics=worst,
        max_physics=p['max_physics'],max_commands=p['max_commands'],maximum_training_decisions=p['maximum_training_decisions'],
        storage=dict(full65=65*384*1024,models5=5*1024**2,compact392=392*5*1024,metadata=6*1024**2),
        complete_gate_NOT_reserved=True,fresh16=None,reload78=None,eight_started=False))
    assert 65*384*1024+5*1024**2+392*5*1024+6*1024**2<=p['max_bytes']
    verify(p,True);print('T441_REGISTERED',worst,before+p['max_bytes'],flush=True)


def synthetic(q,p):
    rng=np.random.default_rng(p['seed']);arrays=[]
    for k in range(16):
        x=(q.mean.cpu().numpy()[None]+rng.normal(size=(7,120))*.01).astype(np.float32)
        with torch.no_grad():loc=q.sequence(torch.tensor(x),q.base_actions(torch.tensor(x)))
        raw=loc.numpy()+rng.normal(size=(7,2))*.2
        atom=np.where(raw<=-L.EDGE,-1,np.where(raw>=L.EDGE,1,0)).astype(np.int8)
        lp=L.density(loc,torch.tensor(raw),torch.tensor(atom),p['raw_sigma']).numpy()
        f=np.zeros((7,3),np.uint8);f[-1,1]=1
        arrays.append(dict(states=x,next_states=np.concatenate([x[1:],x[-1:]],0),locations=loc.numpy(),
            raw_actions=raw,atom=atom,logp=lp,rewards=rng.normal(size=7)*.01,native_counts=np.full(7,100,np.uint8),
            end_flags=f,stochastic=np.ones(7,np.uint8),ownership=np.zeros(7,np.uint8),_group=k//8))
    return arrays


def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists()
    assert torch.cuda.is_available(),'GPU required for this registered learner'
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
    assert q.phash()==p['expected_policy_hash']
    data=synthetic(q,p)
    # Real CPU step path versus sequence, including persistent hidden carry.
    with torch.no_grad():
        x=torch.tensor(data[0]['states']);q.sample=False;q.reset_noise();h=None;steps=[]
        for row in x:
            _,loc,h,_=q.step(row[None],h);steps.append(loc[0])
        assert float((torch.stack(steps)-q.sequence(x,q.base_actions(x))).abs().max())<1e-10
    a=L.make(p,'cuda');b=L.make(p,'cuda');assert digest(L.state(*a))==digest(L.state(*b))
    before=L.state(*a); events=[]
    ra=L.fit(a,data,p,lambda:None,lambda *x:events.append(['A',*x]))
    rb=L.fit(b,data,p,lambda:None,lambda *x:events.append(['B',*x]))
    torch.cuda.synchronize();assert ra==rb and digest(L.state(*a))==digest(L.state(*b))
    mature=L.state(*a);L.load(a,before);assert digest(L.state(*a))==digest(before)
    L.load(a,mature);assert digest(L.state(*a))==digest(mature)
    save(OUT/'preflight_learning.json',dict(state='PASS',native=0,synthetic_only=True,
        discarded_Actor_Adam=ra['Actor_Adam']+rb['Actor_Adam'],discarded_V_Adam=ra['V_Adam']+rb['V_Adam'],
        cpu_step_sequence_gpu_density_exact_within_locked_tolerances=True,Actor_V_Adam_shadow_and_mature_restore_exact=True,
        GPU=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),fit=ra))
    del a,b,q;torch.cuda.empty_cache()
    counts=mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir()
    cfg=dict(p,out=str(folder),native_stop_epoch=time.time()+180)
    pool=Pool(cfg,counts,lambda:None)
    try:
        req=[dict(model=p['student_start_model'],job=j,mode='student',seed=None,cancel_probe=True) for j in (p['old8_jobs'][:4]+p['centre_jobs'][-2:]*2)]
        result=pool.request(req);assert all(x['state']=='CANCEL_PASS' for x in result) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,zero_physics_commands=16,
        learning=read(OUT/'preflight_learning.json'),eight_workers_cancelled_and_stopped=True))
    print('T441_PREFLIGHT_PASS',flush=True)


class Runtime:
    def __init__(self,p):
        self.p=p;self.start=time.time();self.deadline=self.start+p['main_seconds'];self.learnstop=self.start+p['learning_seconds']
        self.counts=mp.get_context('spawn').RawArray('q',9);self.commands=8;self.rows=[];self.replays=[];self.last=0.;self.role='SETUP';self.pool=None
        self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=self.start,native_stop=self.deadline,learning_stop=self.learnstop))
    def guard(self,learning=False):
        E.L.check_stop();assert time.time()<(self.learnstop if learning else self.deadline)-5
        assert sum(self.counts)<=self.p['max_physics'] and self.commands+16<=self.p['max_commands']
        if time.time()-self.last>10:
            self.last=time.time();verify(self.p)
            save(OUT/'status.json',dict(state=self.role,epoch=self.last,pid=os.getpid(),physics=sum(self.counts),commands=self.commands))
    def start_pool(self):
        self.pool=Pool(dict(self.p,native_stop_epoch=self.deadline),self.counts,self.guard)
    def request(self,jobs,model,role,seeds=None,full=False,references=None,width=8):
        self.role=role;out=[]
        for offset in range(0,len(jobs),width):
            js=jobs[offset:offset+width];self.guard()
            assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics']
            assert self.commands+len(js)+16<=self.p['max_commands'];self.commands+=len(js)
            req=[dict(model=model['path'],job=j,mode='student',seed=None if seeds is None else seeds[offset+i]) for i,j in enumerate(js)]
            for i,((a,r),j,request) in enumerate(zip(self.pool.request(req),js,req)):
                r=dict(r,job_id=j['id'],role=role,seed=request['seed'],request=request,loaded_model_sha256=sha(model['path']))
                assert r['policy_hash']==model['policy_hash'] and int(a['native_counts'].sum())==r['physics']
                if references is not None:
                    previous=references[offset+i]
                    assert r['array_digest']==previous['array_digest'] and r['integration_sha256']==previous['integration_sha256']
                    r.update(exact=True,independent_new_sample=False,reference_path=previous['path']);self.replays.append(r);log='replays.jsonl'
                else:
                    path=OUT/'episodes'/f'{len(self.rows):04d}.npz';path.parent.mkdir(exist_ok=True)
                    E.N.write_npz(path,E.N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},self.p['full_bytes'] if full else self.p['compact_bytes'])
                    r.update(path=str(path),sha256=sha(path),full=full);self.rows.append(r);log='cases.jsonl'
                with (OUT/log).open('a') as f:f.write(json.dumps(r)+'\n')
                out.append((a,r))
        return out


def tally(rows):
    return dict(valid=len(rows),success=sum(bool(r['success']) for r in rows),contact=sum(bool(r['collision']) for r in rows),
                timeout=sum(bool(r['timeout']) for r in rows),invalid=0,censored=0)


def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    E.N.offline();E.L.install_stop_handlers();rt=Runtime(p);objects=None;rounds=[];error=None;state='FAILED';development=None
    try:
        objects=L.make(p,'cuda');model=L.export(objects,OUT/'initial.pt.gz');assert model['policy_hash']==p['expected_policy_hash']
        rt.start_pool();bench=[];reference=None
        job=p['centre_jobs'][-2]
        for n in (1,4,8):
            before=time.monotonic();items=rt.request([job]*n,model,f'THROUGHPUT_{n}',[p['seed']]*n,full=reference is None,
                references=None if reference is None else [reference]*n,width=n)
            elapsed=time.monotonic()-before
            if reference is None:reference=items[0][1]
            bench.append(dict(workers=n,seconds=elapsed,physics=sum(r['physics'] for a,r in items),
                              physics_per_second=sum(r['physics'] for a,r in items)/elapsed))
        selected=max(bench,key=lambda x:(x['physics_per_second'],-x['workers']))['workers']
        save(OUT/'throughput.json',dict(rows=bench,selected_workers=selected,
            one_unique_policy_job_seed_13_physical_repeats_not13_independent_samples=True))
        rt.request(p['centre_jobs'],model,'INITIAL_CENTRES',width=selected)
        for rnd,jobs in enumerate(p['training_jobs']):
            rt.guard(True);before=L.state(*objects);seeds=[p['seed']+1000+rnd*16+i for i in range(16)]
            items=rt.request(jobs,model,f'TRAIN_{rnd}',seeds,True,width=selected)
            rt.request(jobs,model,f'EXACT_{rnd}',seeds,references=[r for a,r in items],width=selected)
            arrays=[]
            for i,(a,r) in enumerate(items):
                a['_group']=i//8;arrays.append(a)
            shadow=L.make(p,'cuda');L.load(shadow,before)
            def event(which):
                def emit(where,k):
                    with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([rnd,which,where,k])+'\n')
                return emit
            try:
                live=L.fit(objects,arrays,p,lambda:rt.guard(True),event('live'))
                other=L.fit(shadow,arrays,p,lambda:rt.guard(True),event('shadow'))
                assert live==other and digest(L.state(*objects))==digest(L.state(*shadow))
            except BaseException:
                L.load(objects,before);save(OUT/f'interrupted_round_{rnd}.json',dict(Actor_V_both_optimizers_rollback_exact=digest(L.state(*objects))==digest(before)));raise
            del shadow;torch.cuda.empty_cache()
            model=L.export(objects,OUT/f'round_{rnd}.pt.gz')
            panel=rt.request(p['centre_jobs'],model,f'CENTRES_{rnd}',width=selected)
            record=dict(round=rnd,training_paths=[r['path'] for a,r in items],sampling_model_sha256=items[0][1]['loaded_model_sha256'],
                fit=live,model=model,old8=tally([r for a,r in panel[:8]]),wide2=tally([r for a,r in panel[8:]]),
                training_old=tally([r for a,r in items[:8]]),training_wide=tally([r for a,r in items[8:]]),
                research_updates_retained=True,no_success_set_rollback=True)
            rounds.append(record);save(OUT/'rounds.json',rounds)
        final=rt.request(p['main_jobs']+p['old275_jobs'],model,'FINAL329',width=selected)
        development=dict(wide54=tally([r for a,r in final[:54]]),old275=tally([r for a,r in final[54:]]),
            all329_safe=all(r['success'] and not r['collision'] for a,r in final),model=model)
        save(OUT/'development.json',development);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,physics=sum(rt.counts),commands=rt.commands,role=rt.role))
    finally:
        if rt.pool:rt.pool.close()
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands,
            rounds=rounds,development=development,error=error,all_workers_stopped=True,fresh16=None,reload78=None,six_gate=False,eight_started=False,
            live_Actor_Adam=sum(r['fit']['Actor_Adam'] for r in rounds),live_V_Adam=sum(r['fit']['V_Adam'] for r in rounds),
            scientific_configuration='joint recurrent warm-start integration, not clean Actor capacity or estimator-only causal ablation')
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,epoch=time.time(),physics=sum(rt.counts),commands=rt.commands))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    ids=[s['pid']]+read(OUT/'workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    assert not (OUT/'audit.json').exists();objects=L.make(p,'cuda');L.load(objects,L.read_state(OUT/'initial.pt.gz'))
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()]
    replays=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()]
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        if r['full']:
            assert digest(a)==r['array_digest']
            for k,h in r['array_hashes'].items():assert digest(a[k])==h
        assert int(a['native_counts'].sum())==r['physics']
    calls=0
    for rr in s['rounds']:
        arrays=[]
        mirror=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
        mirror.load_learned(L.state(*objects)['actor'])
        for i,path in enumerate(rr['training_paths']):
            a=E.N.S.arrays(path)
            row=next(r for r in rows if r.get('path')==path)
            E.NATIVE_AUDIT.audit_array(mirror,a,p,seed=row['seed'])
            a['_group']=i//8;arrays.append(a)
        fit=L.fit(objects,arrays,p,lambda:None,lambda *a:None)
        assert fit==rr['fit'] and digest(L.state(*objects))==rr['model']['joint_hash']
        assert sha(rr['model']['path'])==rr['model']['sha256'];calls+=fit['Actor_Adam']+fit['V_Adam']
    events=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    assert len(events)==4*s['live_Actor_Adam']
    for rr in s['rounds']:
        for which in ['live','shadow']:
            actual=[e[2:] for e in events if e[:2]==[rr['round'],which]]
            assert actual==[[where,k] for k in range(rr['fit']['Actor_Adam']) for where in ['BEFORE','AFTER']]
    assert s['physics']==sum(r['physics'] for r in rows+replays)
    assert s['commands']==8+len(rows)+len(replays)
    assert s['commands']+16<=p['max_commands'] and s['physics']<=p['max_physics']
    for r in replays:
        ref=next(x for x in rows if x['path']==r['reference_path'])
        assert r['array_digest']==ref['array_digest'] and r['integration_sha256']==ref['integration_sha256']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_RL=0,
        discarded_historical_Actor_plus_V_Adam=calls,all_saved_Actor_V_optimizers_exact=True,physics=s['physics'],commands=s['commands'],
        all_complete_training_raw_arrays_exact=True,fresh16=None,reload78=None,six_gate=False))
    print('T441_AUDIT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T441',
            '--why=Bounded joint Actor Critic integration','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),
            'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,all_stop_epoch=now+p['all_seconds'],source_remote=receipt['remote_commit'],
        protocol_sha256=sha(OUT/'protocol.json'),native_stop_latest=now+p['main_seconds'],new_independent_window=True))
    print('T441_ACTUALLY_LAUNCHED',child.pid,flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
