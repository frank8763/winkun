"""Register and launch exactly one offline compatibility experiment."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t517_runtime as R
L,G,P,torch,read,save,sha,digest=R.L,R.G,R.P,R.torch,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t516_nonlinear_compatibility'
    assert read(prior/'fit/audit.json')['state']=='PASS' and not read(prior/'fit/summary.json')['result']['offline_joint_gate']
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T516.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(prior/'protocol.json');parent=p['parent_model'];now=time.time()
    p.update(id='T517_OFFLINE_NONLINEAR_RECOVERY_COMPATIBILITY',out=str(OUT),initial=parent,parent_model=parent,initial_joint_locked=False,
             dataset_protocol=p['dataset_protocol'],dataset_protocol_sha256=p['dataset_protocol_sha256'],residual_seed=20261113516,residual_lr=.001,
             fit_steps=2048,backtracking_factors=[2.**(-i) for i in range(8)],minimum_repair_decrease=1e-12,teacher_RMSE_limit=.05,keep_RMSE_limit=.002,keep_max_abs_limit=.01,
             max_bytes=10*2**20,model_bytes=2*2**20,main_seconds=1800,audit_seconds=1800,absolute_science_stop=now+4800,absolute_all_stop=now+6600,
             Actor_Adam=2048,shadow_Actor_Adam=2048,audit_discarded_Actor_Adam=2048,preflight_discarded_Actor_Adam=5,Value_Adam=0,new_policy_gradient_updates=0,
             max_physics=0,max_commands=0,preflight_commands=0,
             scientific_scope='Frozen sameT516zero parent/capacity/seed/data/split/loss/lr, new explicit hardstoredoutputacceptance for actualAdam displacement. Atmost2048 proposals perlive/shadow/audit. Eight fixedbacktrack1..1/128, choosefirst19keep+prefix eachRMSE<=.002/maxabs<=.01 andtrainingrepairlossdecrease>1e-12. Heldteacherlabels NEVER stepselection. Rejectedproposalwholeweights+Adamrollbackandstop, not runclockfiller. Zero native/Value/PG, no promotion.',
             optimizer_contract='EachactualAdam proposal momentscomputedonce; backtrackonlyactualparameterdisplacement, acceptedmomentsretained andcounteradvances1. Rejectedwholeparameter+Adamstate exactrollback. Not rawgradientprojection/TRPO/unbiasedRL/safetycertificate.',
             gate='Unchanged final teachertrain+held peraxisexecutedRMSE<=.05, each19keep+prefix peraxislocationRMSE<=.002/maxabs<=.01. All necessary for later newlyregistered native check. This run0native/no automatic conditional launch.',
             authorization='Nov13user nextstep moresteps: new hardoutputconstraint mechanism withatmost2048 proposals, samefixedarchitecture/data/seed. Not continuing endedT516weightedonlyoptimizer or silently relaxing its thresholds.')
    p['parent_Actor_Adam_hash']=digest(P.checkpoint(parent['path'])['joint_state']['actor_opt'])
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),DOC/'T517_PLAN.md',DOC/'T517_PRIMARY_SOURCES.md',prior/'protocol.json',prior/'fit/summary.json',prior/'fit/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T516.json']:
        p['inputs'][str(f)]=sha(f)
    src=read(p['dataset_protocol']);p['protected_files']={parent['path']:parent['sha256']}
    for row in src['keep_rows']+src['repair_rows']:p['inputs'][row['path']]=row['sha256']
    p['inputs'][src['initial']['path']]=src['initial']['sha256'];p['inputs'].update(p['protected_files'])
    assert G.M.used()+p['max_bytes']+256*1024<2**30
    OUT.mkdir();obj=L.make(p);assert not obj[2].state;parentq=L.A.frozen(p,parent['path']);maxdiff=0.
    with torch.no_grad():
        for row in src['keep_rows']+src['repair_rows']:
            x=torch.tensor(G.E.N.S.arrays(row['path'])['states']);new=obj[0].sequence(x);old=parentq.sequence(x)
            assert torch.equal(new,old);maxdiff=max(maxdiff,float((new-old).abs().max()))
    model=L.export(obj,OUT/'zero_residual_initial.pt.gz',p['model_bytes']);p['initial']=model;p['initial_joint_locked']=True
    p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256']
    assert digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(OUT/'zero_function.json',dict(state='ALL20_FULL_HISTORY_PARENT_EXACT',max_location_error=maxdiff,parent_Adam_hash=p['parent_Actor_Adam_hash'],native=0,Adam=0,new_parameters=13986))
    save(DOC/'T517_BUDGET.json',dict(max_physics=0,max_commands=0,main_seconds=1800,audit_seconds=1800,backup_reserve_seconds=1200,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],live_new_Adam_max=2048,shadow_Adam_max=2048,audit_historical_Adam_max=2048,backtracking_evaluations_max=16384,preflight_discarded_Adam=5,ValueAdam=0,newPG=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T517_OFFLINE_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();R.ban_native()
    obj=L.make(p);shadow=L.make(p);initial=P.state(*obj);rows=L.data(p)
    assert sum(z.numel() for z in obj[0].parameters() if z.requires_grad)==13986
    assert all(z['max_abs']<1e-10 for z in L.metrics(obj[0],rows,p)['keep19'])
    repair=next(g for g in rows if g['kind']=='repair');original=repair['target'].clone()
    # Held-label perturbation must leave the complete training gradient exact.
    def gradient():
        obj[2].zero_grad(set_to_none=True);L.loss(obj[0],repair,p).backward()
        return digest([z.grad for z in L.variables(obj[0])])
    before=gradient();repair['target'][repair['held_mask']]=-.123
    assert gradient()==before;repair['target']=original;obj[2].zero_grad(set_to_none=True)
    a=L.fit(obj,rows,p,lambda:None,lambda *_:None,steps=2);b=L.fit(shadow,rows,p,lambda:None,lambda *_:None,steps=2)
    assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
    accepted_state=P.state(*obj)
    assert a['Actor_Adam']==b['Actor_Adam']==2 and a['accepted_parameter_updates']==2
    rejected=L.fit(obj,rows,dict(p,minimum_repair_decrease=1e9),lambda:None,lambda *_:None,steps=1)
    assert rejected['Actor_Adam']==1 and rejected['accepted_parameter_updates']==0 and digest(P.state(*obj))==digest(accepted_state)
    model=L.export(obj,OUT/'preflight_model.pt.gz' ,p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    q=L.frozen(p,model['path'])
    src=read(p['dataset_protocol']);rr=src['repair_rows'][0];full=G.E.N.S.arrays(rr['path']);x=torch.tensor(full['states'])
    mask=torch.tensor(full['ownership'],dtype=torch.bool)
    with torch.no_grad():assert float((q.sequence(x)[mask]-L.location(q,repair)).abs().max())<1e-10
    for o in [obj,shadow]:P.load(o,initial);assert digest(P.state(*o))==p['initial']['joint_hash']
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=0,discarded_new_Adam=5,forced_rejection_whole_ActorAdam_rollback_PASS=True,held_label_gradient_leak_test_PASS=True,true_full_history_features_and_sequence_output_consistent=True,parent_function_exact_at_zero=True,new_trainable_parameters=13986,live_shadow_serialization_and_complete_rollback_exact=True))
    print('T517_NONLINEAR_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not Path('/proc',str(read(OUT/'preflight.json')['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>5000
    args=['systemd-run','--user','--unit=stage-t517-hard-output-constraint','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T517','--why=Bounded offline nonlinear recovery compatibility','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T517_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
