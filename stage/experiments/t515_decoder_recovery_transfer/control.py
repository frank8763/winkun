"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t515_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t514_strict_teacher_recovery'
    assert read(prior/'probe/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']) and not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T514.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(ROOT/'stage/runs/t509_two_goal_joint/protocol.json');repair=read(prior/'probe/summary.json')['result']['teacher_rows'];assert len(repair)==1 and repair[0]['success']
    target=repair[0]['job_id'];guard=p['guard_jobs'][:-2]+[next(j for j in p['field_jobs'] if j['id']==target)]
    rest=[j for j in p['field_jobs'] if j['id'] not in {j['id'] for j in guard}];assert len(guard)==20 and len(rest)==12
    physics=2*sum(j['case']['step_cap'] for j in guard)+sum(j['case']['step_cap'] for j in rest+p['field_jobs']);commands=2*len(guard)+len(rest)+len(p['field_jobs'])+4
    shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in guard+rest);now=time.time();parent=p['parent_model']
    p.update(id='T515_SINGLE_DECODER_REAL_RECOVERY_TRANSFER',out=str(OUT),initial=parent,parent_model=parent,initial_joint_locked=False,
        repair_rows=repair,teacher_first=read(prior/'protocol.json')['teacher_first'],guard_jobs=guard,remaining_field_jobs=rest,new_task_ids=[target],
        max_blocks=1,block_steps=128,decoder_lr=.001,max_bytes=30*2**20,preflight_commands=6,pilot_commands=commands,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=commands+6,root_max_main_seconds=1800,
        absolute_learning_stop=now+1200,absolute_native_stop=now+2700,absolute_all_stop=now+5400,
        Actor_Adam=128,shadow_Actor_Adam=128,audit_discarded_Actor_Adam=128,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_Adam=0,
        scientific_scope='One actual T514 GOAL teacher suffix ownership only, same-task actual prefix selfkeep and own19historykeep. Only existing130param final decoder trainable, parentadapters11906/encoder/GRU frozen. One128supervised+shadow,20native+20exact hardprotect19; newtaskpassed thenremaining12+24exact; no newPG/Value or teacher at deployment.',
        optimizer_contract='Preserveoriginal8adapterparameterAdamstates1025/385 exact unchanged, gradientsNone; addhead2parametersfreshAdamgroup0 lr.001, all128stepsonlyhead. Initialpolicyfunctionequalsparent but jointoptimizerchangesexplicit. EntireActor+bothAOgroupsandValueAdamrollback.',
        acceptance='Protect19 initialsuccessIDs/0invalidcensor/repairlossdecrease; targetoriginalstartsingleActorGOAL required beforeconditionalfull24. Finalold8safe/all11fieldIDsretained/qualified>11/contact<=9, otherwisewholeexperimentrollback. All old275/six54/eight54/fresh16/reload78null.',
        authorization='Nov13user next round after actual student-prefix teacher recovery verified; action-support audit motivates freeing existing130decoder. Independent bounded new supervised mechanism, no restart or same-noise grid.')
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'};p['protected_files']={parent['path']:parent['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'T515_PLAN.md',DOC/'T515_PRIMARY_SOURCES.md',prior/'probe/summary.json',prior/'probe/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T514.json']:
        p['inputs'][str(f)]=sha(f)
    for r in repair+p['keep_rows']:p['inputs'][r['path']]=r['sha256']
    p['inputs'].update(p['protected_files'])
    assert shape+6*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+256*1024<2**30
    OUT.mkdir();obj=L.make(p);q=obj[0];assert q.phash()==parent['policy_hash'] and all(z not in obj[2].state for z in q.head.parameters())
    a=G.E.N.S.arrays(repair[0]['path']);g=L.B.cached(q,a);mask=torch.tensor(a['ownership'],dtype=torch.bool);y=torch.tensor(a['actions'],dtype=torch.float64)
    lo=1.25*torch.tanh(g['logits']-.3);hi=1.25*torch.tanh(g['logits']+.3);outside=((y<lo)|(y>hi))[mask]
    save(OUT/'action_support_audit.json',dict(state='FIXED_HEAD_BOUNDED_ADAPTER_TARGETS_OUTSIDE_RANGE',owned_decisions=int(mask.sum()),outside_count_per_axis=outside.sum(0).tolist(),first_target=y[mask][0].tolist(),first_lower=lo[mask][0].tolist(),first_upper=hi[mask][0].tolist(),native=0,Adam=0,not_proof_navigation_impossible=True))
    assert outside.all()
    model=L.A.export(obj,OUT/'decoder_initial.pt.gz',p['model_bytes']);p['initial']=model;p['initial_joint_locked']=True;p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256']
    assert model['policy_hash']==parent['policy_hash'] and digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T515_BUDGET.json',dict(max_physics=physics,max_commands=commands+6,main_seconds=1800,learning_seconds=600,audit_seconds=600,backup_reserve=1200,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],full_shape_bound=shape,models_metadata_partial_reserve=6*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],live_headAdam=128,shadow_headAdam=128,audit_historical_headAdam=128,preflight_discarded_headAdam=4,newPG=0,ValueAdam=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T515_DECODER_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);data=L.groups(p)
    assert sum(z.numel() for z in obj[0].parameters() if z.requires_grad)==130
    with torch.no_grad():
        err=max(float((L.location(obj[0],g)-g['target']).abs().max()) for g in data if g['kind']!='repair');assert err<1e-10
    aa=L.fit(obj,data,p,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,data,p,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    assert aa['final']['repair_0']<aa['initial']['repair_0'] and aa['old_adapter_Adam_unchanged']
    model=L.A.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        values=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['guard_jobs'][-1],p['old8_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_headAdam=4,ValueAdam=0,full_history_teacher_suffix_only=True,prefix_original_location_keep=True,initial_keep_max_error=err,trainable_parameters=130,old_adapter_Adam_and_backbone_unchanged=True,live_shadow_fullrollback_save_readback_exact=True,actual_workers_load_cancel_PASS=True))
    print('T515_DECODER_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>3800
    args=['systemd-run','--user','--unit=stage-t515-decoder-recovery-transfer','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T515','--why=Bounded actual recovery transfer to one neural decoder','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T515_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
