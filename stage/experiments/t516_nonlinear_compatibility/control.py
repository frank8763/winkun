"""Register and launch exactly one offline compatibility experiment."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t516_runtime as R
L,G,P,torch,read,save,sha,digest=R.L,R.G,R.P,R.torch,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t515_decoder_recovery_transfer'
    assert read(prior/'pilot/audit.json')['state']=='PASS' and not read(prior/'pilot/summary.json')['result']['accepted']
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T515.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(prior/'protocol.json');parent=p['parent_model'];now=time.time()
    p.update(id='T516_OFFLINE_NONLINEAR_RECOVERY_COMPATIBILITY',out=str(OUT),initial=parent,parent_model=parent,initial_joint_locked=False,
             dataset_protocol=str(prior/'protocol.json'),dataset_protocol_sha256=sha(prior/'protocol.json'),residual_seed=20261113516,residual_lr=.001,
             fit_steps=512,teacher_RMSE_limit=.05,keep_RMSE_limit=.002,keep_max_abs_limit=.01,
             max_bytes=8*2**20,model_bytes=2*2**20,main_seconds=1200,audit_seconds=1200,absolute_science_stop=now+3600,absolute_all_stop=now+5400,
             Actor_Adam=512,shadow_Actor_Adam=512,audit_discarded_Actor_Adam=512,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_updates=0,
             max_physics=0,max_commands=0,preflight_commands=0,
             scientific_scope='Offline compatibility only,13986param184->64tanh->32tanh->2global nonlinear logit residual,zero final layer,whole acceptedparent frozen. Fixed512Adam+512shadow+512independent historicalreconstruction; no native setup/physics or navigation promotion. Teacher59suffix labels alternating5decisionblocks train30/held29,hold labels never loss/step selection;19keep+prefix166fullhistory. Original120causalobs/64carry, no taskID/planner/map/teacher at deployment.',
             gate='Final only: teacher train AND held executedaction RMSE<=.05 eachaxis; eachold19/prefix unclippedlocation RMSE<=.002 eachaxis AND maxabs<=.01. All conditions required. Offline pass permits only a new independently registered native test, does not authorize auto launch or prove safe navigation.',
             authorization='Nov13user begin next step after T515 output-layer interference; bounded offline nonlinear compatibility first, not repetition of native20 or increasing failed T515 steps. No grid or same-directory restart.')
    p['parent_Actor_Adam_hash']=digest(P.checkpoint(parent['path'])['joint_state']['actor_opt'])
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),DOC/'T516_PLAN.md',DOC/'T516_PRIMARY_SOURCES.md',prior/'protocol.json',prior/'pilot/summary.json',prior/'pilot/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T515.json']:
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
    save(DOC/'T516_BUDGET.json',dict(max_physics=0,max_commands=0,main_seconds=1200,audit_seconds=1200,backup_reserve_seconds=1200,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],live_new_Adam=512,shadow_Adam=512,audit_historical_Adam=512,preflight_discarded_Adam=4,ValueAdam=0,newPG=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T516_OFFLINE_REGISTERED_NOT_LAUNCHED',flush=True)
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
    model=L.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    q=L.frozen(p,model['path'])
    src=read(p['dataset_protocol']);rr=src['repair_rows'][0];full=G.E.N.S.arrays(rr['path']);x=torch.tensor(full['states'])
    mask=torch.tensor(full['ownership'],dtype=torch.bool)
    with torch.no_grad():assert float((q.sequence(x)[mask]-L.location(q,repair)).abs().max())<1e-10
    for o in [obj,shadow]:P.load(o,initial);assert digest(P.state(*o))==p['initial']['joint_hash']
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=0,discarded_new_Adam=4,held_label_gradient_leak_test_PASS=True,true_full_history_features_and_sequence_output_consistent=True,parent_function_exact_at_zero=True,new_trainable_parameters=13986,live_shadow_serialization_and_complete_rollback_exact=True))
    print('T516_NONLINEAR_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not Path('/proc',str(read(OUT/'preflight.json')['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>3800
    args=['systemd-run','--user','--unit=stage-t516-offline-compatibility','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T516','--why=Bounded offline nonlinear recovery compatibility','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T516_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
