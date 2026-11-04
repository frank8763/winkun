from night_core import *
import contextlib
OUT=ROOT/'stage/runs/t353_offline_night_chain'


def numerical_worker(pipe,p,rho,x):
    torch.set_num_threads(1);q=NightActor(p,rho).requires_grad_(False);q.sample=True;q.rng=np.random.default_rng(p['seed']);h=None;acts=[]
    with torch.no_grad():
        for row in x:a,_,h,_=q.step(torch.tensor(row[None]),h);acts.append(a[0].numpy())
    pipe.send(dict(actions=np.asarray(acts),tape=q.tape,phash=q.phash()));pipe.close()


def main():
    if OUT.exists():assert (OUT/'preflight_failed_001/failure.json').exists() and not (OUT/'protocol.json').exists() and not (OUT/'preflight').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert read(ROOT/'stage/docs/review_20261104/T352_AUDIT.json')['state']=='PASS'
    p=copy.deepcopy(read(ROOT/'stage/runs/t351_recurrent_recovery/protocol.json'))
    p.update(initial_model=str(ROOT/'stage/runs/t349_consistent_demo/memory_final.pt.gz'),reference_cases=str(ROOT/'stage/runs/t351_recurrent_recovery/cases.jsonl'),
        raw_sigma=[.2,.2],actor_lr=.0003,epochs=4,seed=20261104353,prior_initial_old275=read(ROOT/'stage/runs/t351_recurrent_recovery/summary.json')['result']['initial']['old275'],
        model_bytes=256*1024,full_bytes=384*1024,compact_bytes=4096,max_compact_cases=730,max_train_episodes=24,
        final_bytes=18*1024**2,max_commands=1000,max_physics=18_000_000,wall_seconds=5400,curriculum_shifts=[.6,.3,0.],
        original_stop_epoch=1793836800.,learning_stop_epoch=1793831400.,native_stop_epoch=1793834100.)
    # No new network dependencies. All three mechanisms and conditional choices
    # are fixed now, before any overnight result is observed.
    stages=[dict(id='T353',folder='t353_temporal_exploration',arms={'iid':0.,'ar':math.exp(-.2)},rounds=2,credit='full_G0',curriculum=False),
            dict(id='T354',folder='t354_causal_sequence_credit',arms={'rtg':math.exp(-.2)},rounds=6,credit='reward_to_go',curriculum=False),
            dict(id='T355',folder='t355_gap_curriculum',arms={'course':math.exp(-.2)},rounds=6,credit='reward_to_go',curriculum=True)]
    OUT.mkdir(exist_ok=True);folder=OUT/'preflight';folder.mkdir();tests=[];discarded=0
    for rho in [0.,math.exp(-.2)]:
        q=NightActor(p,rho);a=arrays(p['preflight_sequence']);x=torch.tensor(a['states'][:40]);q.sample=True;q.rng=np.random.default_rng(p['seed']);h=None;loc=[];acts=[]
        with torch.no_grad():
            for row in x:
                ac,l,h,_=q.step(row[None],h);loc.append(l[0]);acts.append(ac[0].numpy())
            serial=torch.stack(loc);seq=q.sequence(x,q.base_actions(x));assert float((serial-seq).abs().max())<1e-10
            raw=torch.tensor(np.asarray([v['raw_actions'] for v in q.tape]));lp=raw_logp(seq,raw,rho,p['raw_sigma'])
            assert float((lp-torch.tensor(np.asarray([v['logp'] for v in q.tape]))).abs().max())<1e-9
        ctx=mp.get_context('spawn');pa,pb=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']='1';child=ctx.Process(target=numerical_worker,args=(pb,p,rho,x.numpy()));child.start();pb.close()
        assert pa.poll(60);got=pa.recv();child.join(10);assert child.exitcode==0 and not child.is_alive()
        assert got['phash']==q.phash() and np.array_equal(got['actions'],np.asarray(acts)) and digest(got['tape'])==digest(q.tape)
        os.environ['STAGE_PROCESS_CPU']='0'
        probe=seq.detach().clone().requires_grad_(True);grad=torch.autograd.grad(raw_logp(probe,raw,rho,p['raw_sigma']).sum(),probe)[0]
        for k,j in [(0,0),(1,1),(38,0),(39,1)]:
            plus=probe.detach().clone();minus=plus.clone();plus[k,j]+=1e-6;minus[k,j]-=1e-6
            fd=(raw_logp(plus,raw,rho,p['raw_sigma']).sum()-raw_logp(minus,raw,rho,p['raw_sigma']).sum())/(2e-6)
            assert abs(float(fd-grad[k,j]))<1e-6
        refs=references(p,q);items=[]
        for j in range(4):
            aa=copy.deepcopy(a)
            for key in list(aa):
                if key not in ['initial_native_integration','final_native_integration','last_hidden_after']:aa[key]=aa[key][:40] if len(aa[key])==len(a['states']) else aa[key][:41]
            aa['states']=a['states'][:40];aa['actions']=np.asarray(acts);aa['end_flags']=np.zeros((40,3),np.uint8);aa['end_flags'][-1,2]=1
            aa['rewards']=np.zeros(40);aa['rewards'][-1]=-.5+.1*j
            aa.update({name:np.asarray([v[name] for v in q.tape]) for name in q.tape[0]});items.append(dict(arrays=aa,row=dict(stochastic=True)))
        for credit in ['full_G0','reward_to_go']:
            pp=dict(p,epochs=2,credit=credit);live=NightLearner(pp,rho);shadow=NightLearner(pp,rho);before=live.state();imm=digest(before);shadow.load(before)
            l=optimize(pp,live,items,refs,0,lambda:None,lambda r:None);r=optimize(pp,shadow,items,refs,0,lambda:None,lambda r:None);discarded+=4
            assert l==r and live.hashes()==shadow.hashes() and digest(before)==imm
            mature=live.state();original=digest(mature);clone=NightLearner(pp,rho);clone.load(mature)
            with torch.no_grad():next(v for state in clone.opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)).add_(1.)
            assert digest(mature)==original;clone.load(mature);assert clone.hashes()==live.hashes()
        tests.append(dict(rho=rho,worker_and_sequence_density_exact=True,raw_AR_gradient_previous_location_finite_difference=True,synthetic_clone_shadow_mature_Adam_isolation=True))
    geometries=[]
    with (folder/'geometry.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);original=copy.deepcopy(module.ACTIVE)
        try:
            for shift in [.6,.3,0.]:
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=shift
                j=copy.deepcopy(p['course_dev_jobs'][0]);env,obs,_=K.R.task_env(create,j)
                try:
                    assert env.step_count==0 and env.max_wheel==5.;g=P.C.B.geometry(env,obs,w,40000);geometries.append(dict(shift=shift,scene_sha256=g['scene_sha256'],valid=g['valid']))
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    assert discarded==16 and len({g['scene_sha256'] for g in geometries})==3
    # Stage worst size reserve uses uncompressed upper bounds already measured
    # by T351 (331188 for400 rows). This representation removes the atom array.
    assert 24*p['full_bytes']+2*p['full_bytes']+4*p['model_bytes']+730*4096+2*1024**2+128*1024<=p['final_bytes']
    source={str(f):K.C.sha(f) for f in HERE.glob('*.py')}
    locked={**p['input_hashes'],**source}
    locked.update({p['initial_model']:K.C.sha(p['initial_model']),p['reference_cases']:K.C.sha(p['reference_cases'])})
    for row in map(json.loads,Path(p['reference_cases']).read_text().splitlines()):
        if row.get('full_arrays_saved') and row['role'] in ['initial_old275','initial_wide5']:locked[row['path']]=row['sha256']
    p['input_hashes']=locked
    protocol=dict(experiment='T353_T355_PREPROGRAMMED_NETWORK_INDEPENDENT_NIGHT',base=p,stages=stages,source_hashes=source,
        native_deadline=1793834100.,learning_deadline=1793831400.,all_work_deadline=1793836800.,
        night_limits=dict(physics=100_000_000,commands=4500,main_wall_s=25200),
        prior_T351_actual=dict(physics=6121459,commands=415,main_wall_s=1944.62797665596),
        fixed_selection='Among retained final arms: wide5 successes descending, old20 descending, registered arm order. No intermediate or fresh selection.',
        conditional_parent='Next mechanism starts from previous selected retained model only if independent audit PASS; otherwise stop scientific chain. Never use rejected candidate.',
        offline_contract='No network/API/LLM in native runner or dispatcher. Code/local clean commit/SHA and preflight locked now. Per-phase input/protocol lineage hashes signed locally before launch. Remote raw/source pending until connectivity returns; never label verified. No parallel Stage writer. Each phase<=18M/1000/5400 and global1GiB/free40GiB, completeworstcapacity/timeadmission. Fullsix gate required before7/8; chain does not preimplement7/8 and will report unexecuted rather than invent it.',
        preflight=dict(state='PASS',native=0,geometry_commands=6,discarded_Adam=34,fresh_discarded_Adam=16,prior_failed_discarded_Adam=18,synthetic_not_navigation=True,tests=tests,geometries=geometries),
        geometry_commands_counted_once=6)
    save(HERE/'protocol.json',protocol);save(OUT/'protocol.json',protocol);save(OUT/'preflight.json',protocol['preflight'])
    print(json.dumps(dict(state='NIGHT_REGISTERED_PREFLIGHT_PASS',native=0,discarded_Adam=34,geometries=6)))


if __name__=='__main__':main()
