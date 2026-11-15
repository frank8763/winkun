"""Bounded same-teacher consolidation. Register only after zero-step checks."""
from pathlib import Path
import copy,gzip,io,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t566_runtime as R
ROOT,OUT,DOC,L,torch,np,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.torch,R.np,R.read,R.save,R.sha,R.digest

def build():
    old=ROOT/'stage/runs/t555_adam_direction_control';p=read(old/'protocol.json');by={j['id']:j for j in p['evaluation_jobs']};evaluation=p['evaluation_jobs'][:8]+p['evaluation_jobs'][14:];perturb=[]
    for ident in ['t502_field_sector1_offset2','t502_field_sector4_offset2']:
        for sign in [-1,1]:
            j=copy.deepcopy(by[ident]);j['id']=ident+'_T566_yaw_'+str(sign);j['fullfield_start_pose'][2]+=sign*np.deg2rad(.5);j['source_task_id']=ident;j['heldout_yaw_deg']=sign*.5;perturb.append(j)
    short=copy.deepcopy(evaluation[:2])
    for j in short:j['case']['step_cap']=500
    previous=read(ROOT/'stage/runs/t562_full_history_representation/summary.json')['result'];initial=previous['model'];qualified=[v['id'] for v in previous['fit']['after']['rows'] if v['command_RMS']<=.02 and v['command_max']<=.1];assert len(qualified)==24
    p['inputs'][initial['path']]=initial['sha256']
    p.update(student_initial=initial,protected_fit_ids=qualified,projection_sweeps=256,projection_tolerance=1e-8,constraint_slack_fraction=.5,protected_coordinate_threshold=.07,maximum_protected_coordinates=128,parameter_radius_cap=.005,joint_internal_target=.075,minimum_active_coordinates=32,maximum_active_coordinates=128,relative_damping=1e-4,minimum_damping=1e-12,step_factors=[1.,.5,.25,.125,.0625,.03125,.015625],minimum_peak_decrease=1e-6,consecutive_reject_stop=1,id='T566_LINEARIZED_HISTORY_CONSTRAINED_JOINT_STEP',teacher=p['frozen_base'],student_seed=559,batch_seed=559001,fit_steps=8,fit_lr=1.0,rows_per_trajectory=16,fit_gate_RMS=.02,fit_gate_max=.1,evaluation_jobs=evaluation,perturb_jobs=perturb,probe_jobs=short,protected_success_ids=p['before_check']['success_ids'],max_bytes=int(6*2**20),child_bytes=int(6*2**20),model_bytes=int(1.5*2**20),main_seconds=2400,learning_seconds=1500,audit_seconds=900,transport_seconds=600,source_seconds=180,max_main_seconds=2400,max_commands=72,scope='InitializeREJECTED_T562researchstudent_notpromoted;unfreezeencoder_core_student93186parameters;full27currentprefixBPTTmax8linearizedHistoryConstrainedJointsteps+sameShadow/audit;maskedforcedpulses;frozenoldhead/adapters/normalization;originalteacherT550successlabels.No6restoreexperts/noRL/noAdam;signedcommandResidualJacobian;projectdirectiononto27RMSandprotected24nearcoordinateLinearizedInequalities;dualCyclic256sweepsKKT1e-8;allbadplusnearby32max128coords;damping1e-4meanGramdiag;internalResidualtarget.075;paramnormcap.005;SGDonlyappliescomputedDirection;eachproposalprotect24fit/all27RMS/strictpeakdecrease;factors1,.5,.25,.125,.0625,.03125,.015625;wholemodelSGDrollback;firstrejectstop;earlyofflinegatestop.OnlyifperhistoryRMS<=.02/max<=.1test2short+exactthen32+32;keepall23required.Noadditionalyawprobes. Same12064carry/action/physics/reward;representationandcapacityjointengineeringpilotnotoriginality. NoautoRL/full407/fresh/reload/promotion. Noepoch/grid/restart.')
    p['perturb_jobs']=[]
    p['max_physics']=2000+2*sum(j['case']['step_cap'] for j in evaluation)
    for key in ['arms','training_jobs','final_jobs','restore_rows','pair_rows','before_check','training_seeds_fixed']:p.pop(key,None)
    return p

def preflight():
    DOC.mkdir(parents=True,exist_ok=True);assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p0=build();p=read(p0['actor_configuration']);p.update(p0)
    # Orthogonal known projection and deliberately inconsistent inequalities, no policy optimizer calls.
    A=torch.eye(2,dtype=torch.float64);bound=torch.tensor([1.,1.],dtype=torch.float64);d,c=L.project_direction(torch.tensor([2.,-1.],dtype=torch.float64),A,bound,p);assert c['pass_tolerance'] and torch.equal(d,torch.tensor([1.,-1.],dtype=torch.float64))
    A=torch.tensor([[1.],[-1.]],dtype=torch.float64);H=A@A.T;bound=torch.tensor([-1.,-1.],dtype=torch.float64);lam=torch.zeros(2,dtype=torch.float64);c=-bound
    for _ in range(4):
        for i in range(2):lam[i]=torch.clamp(lam[i]+(c[i]-(H@lam)[i])/H[i,i],min=0.)
    assert float(torch.relu(A@(-A.T@lam)-bound).max())>=1., 'INCONSISTENT_SYSTEM_MUST_NOT_PASS'
    data=L.dataset(p);q,opt=L.make(p);shadow=L.make(p);before=L.state((q,opt));body=L.fixed_hash(q);start=time.time();events=[];event_times=[]
    def event(w):
        def append(a,k):
            events.append([w,a,k]);event_times.append((w,a,k,time.perf_counter()));save(DOC/'PREFLIGHT_PROGRESS.json',dict(events=events,completed_discarded_SGD=sum(v[1]=='AFTER' for v in events),native=0,epoch=time.time()))
        return append
    a=L.fit((q,opt),data,p,lambda:None,event('live'),steps=2);b=L.fit(shadow,data,p,lambda:None,event('shadow'),steps=2);elapsed=time.time()-start
    save(DOC/'PREFLIGHT_CLONE_CHECK.json',dict(fit_equal=digest(a)==digest(b),state_equal=digest(L.state((q,opt)))==digest(L.state(shadow)),fit_a=a['retained_proposals'],fit_b=b['retained_proposals'],opt_states=len(opt.state)))
    assert digest(a)==digest(b), 'FIT_SHADOW_MISMATCH'
    assert digest(L.state((q,opt)))==digest(L.state(shadow)), 'JOINT_SHADOW_MISMATCH'
    mature=L.state((q,opt));assert L.fixed_hash(q)==body
    assert not opt.state, 'SGD_NO_MOMENTUM_EXPECTS_NO_MOMENT_STATE'
    # Independent persistence in RAM; no dropped temporary checkpoint.
    blob=io.BytesIO();torch.save(dict(actor=q.state_dict(),opt=opt.state_dict()),blob);saved=torch.load(io.BytesIO(blob.getvalue()),map_location='cpu',weights_only=False);again=L.Actor(p);again.load_state_dict(saved['actor']);assert again.phash()==q.phash()
    x=torch.tensor(L.G.E.N.S.arrays(p['keep_rows'][0]['path'])['states'][:16]);seq=q.sequence(x);future=x.clone();future[8:]+=1;assert torch.equal(q.sequence(future)[:8],seq[:8]);h=None;q.reset_noise();q.sample=False
    with torch.no_grad():
        actual=[]
        for row in x:u,loc,h,_=q.step(row[None],h);actual.append(loc)
    assert float((torch.cat(actual)-seq).abs().max())<=1e-10
    q.reset_noise();q.sample=True;q.rng=np.random.default_rng(5597);h=None
    with torch.no_grad():
        for row in x[:4]:u,loc,h,_=q.step(row[None],h)
    tape=q.tape;loc=torch.tensor(np.asarray([v['locations'] for v in tape]));raw=torch.tensor(np.asarray([v['raw_actions'] for v in tape]));atom=torch.tensor(np.asarray([v['atom'] for v in tape]));old=torch.tensor(np.asarray([v['logp'] for v in tape]));assert float((L.P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max())<1e-9
    L.load((q,opt),before);assert digest(L.state((q,opt)))==digest(before);L.load((q,opt),mature);assert digest(L.state((q,opt)))==digest(mature)
    full=L.dataset(p);ranges={}
    for k in range(2):ranges[str(k)]=dict(target_min=min(float(g['target'][:,k].min()) for g in full),target_max=max(float(g['target'][:,k].max()) for g in full),required_residual_absmax=max(float((g['target'][:,k]-g['base'][:,k]).abs().max()) for g in full))
    assert not any(k.startswith(('repair_','added_','layers.')) for k in q.state_dict());assert sum(z.numel() for z in L.variables(q))==93186
    # Full Actor plus stateless SGD metadata fits registered1.5MiB.
    nb=sum(v.numel()*v.element_size() for v in q.state_dict().values());model_upper=nb+0+131072;assert model_upper<p['model_bytes']
    used=R.used(ROOT/'stage/runs');assert used+p['max_bytes']+131072<2**30
    cadence=[]
    for who in ['live','shadow']:
        after=[v[3] for v in event_times if v[0]==who and v[1]=='AFTER'];cadence.append(after[1]-after[0])
    sample_seconds=max(cadence);fit_upper_seconds=sample_seconds*(8*2)*3+120
    assert fit_upper_seconds<p['learning_seconds']-60
    source={str(f):sha(f) for f in HERE.glob('*.py')}
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),epoch=time.time(),synthetic_discarded_Adam=0,synthetic_discarded_SGD=4,new_native=0,new_RL=0,source=source,trainable_parameters=93186,actual27_trajectories=sum(len(g['f']) for g in full),source_targets=len(full),target_ranges=ranges,compact_banks_absent=True,initial_state_exact_T562_REJECTED_RESEARCH_STUDENT=True,frozen_head_and_adapters_exact=True,sequence_step_density_contract=True,SGD_shadow_and_parameter_rollback=True,model_raw_upper_bytes=model_upper,synthetic_profile_seconds=elapsed,estimated_two_full_fit_upper_seconds=fit_upper_seconds,global_worst=used+p['max_bytes']+131072,not_navigation_success=True));print('T566_ZERO_PREFLIGHT_PASS',p['max_physics'],model_upper,fit_upper_seconds,flush=True)

def prepare():
    assert not OUT.exists();pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE' and all(sha(f)==h for f,h in pref['source'].items());p=build();a=read(DOC/'AUTHORIZATION.json');assert a['scope_approved']
    p.update(absolute_all_stop=a['all_stop_epoch'],absolute_native_stop=a['native_stop_epoch'],absolute_learning_stop=a['learning_stop_epoch'])
    prior=[read(ROOT/'stage/docs/day_20261115/t555/FINAL_REPORT.json')['ledger'],read(ROOT/'stage/runs/t557_bc_stage_isolation/summary.json'),read(ROOT/'stage/runs/t559_smooth_policy_consolidation/summary.json'),read(ROOT/'stage/runs/t561_sine_policy_consolidation/summary.json'),read(ROOT/'stage/runs/t562_full_history_representation/summary.json'),read(ROOT/'stage/runs/t563_guarded_tail_fit/summary.json'),read(ROOT/'stage/runs/t564_local_peak_descent/summary.json'),read(ROOT/'stage/runs/t565_joint_residual_step/summary.json')];usedphys=sum(x['physics'] for x in prior);usedcmd=sum(x['commands'] for x in prior);usedmain=sum(x.get('main_seconds',x.get('elapsed_s',0)) for x in prior)
    assert usedphys+p['max_physics']<=19_000_000 and usedcmd+p['max_commands']<=600 and usedmain+p['max_main_seconds']<=12300
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'AUTHORIZATION.json',DOC/'PLAN.md',DOC/'SOURCES.md',DOC/'PREFLIGHT.json',ROOT/'stage/docs/day_20261115/t564/ACTIVE_PEAK_DIAGNOSTIC.json']:p['inputs'][str(f)]=sha(f)
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30 and time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+60<p['absolute_all_stop']
    OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',physics=p['max_physics'],commands=p['max_commands'],main=p['max_main_seconds'],day_physics_worst=usedphys+p['max_physics'],day_commands_worst=usedcmd+p['max_commands'],day_main_worst=usedmain+p['max_main_seconds'],own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,max_fit_direction_applications=8,max_shadow_direction_applications=8,max_audit_discarded_direction_applications=8,max_fit_linear_solves=8,max_shadow_linear_solves=8,max_audit_linear_solves=8,max_projection_solves_each=8,preflight_discarded_SGD=4,new_Adam=0,new_RL=0));print('T566_REGISTERED',flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        DOC.mkdir(parents=True,exist_ok=True);save(DOC/('ERROR_'+sys.argv[1]+'_'+str(int(time.time()))+'.json'),dict(error=traceback.format_exc(),epoch=time.time(),new_native_launch=False));raise
