"""Zero-native and zero-optimizer actual-data preparation."""
from pathlib import Path
import ast,copy,io,json,os,subprocess,sys,time
import t637_data as D
import t637_learning as M
import t637_native as N
ROOT=D.ROOT;HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261118/t637'
OUT=ROOT/'stage/runs/t637_joint_coverage_recovery'
read,save,sha,digest,torch,np=D.read,D.save,D.sha,D.digest,D.torch,D.np


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(read(ROOT/'stage/experiments/t636_joint_best_reproduction/protocol.json'))
    for key in ['inputs','reference_rows','authorization_approved','user_authorization','registration_epoch','absolute_native_stop','absolute_all_stop']:
        p.pop(key,None)
    old,field=D.original_rows();refs=D.selected();target=next(r for r in field if r['job_id']=='t502_field_sector1_offset1')
    assert not target['success'] and target['timeout'] and not target['collision']
    source=D.E.N.S.arrays(target['path']);t=np.r_[0,np.cumsum(source['native_counts'])[:-1]]*.002
    # Closest approach minus one decision, determined from frozen T501 history
    # before new outcomes, without a takeover-time search.
    distance=np.linalg.norm(source['poses'][:,:2]-source['task_goal_xy'],axis=1)
    first=max(1,int(distance.argmin())-1)
    sentinel=read(ROOT/'stage/experiments/t636_joint_best_reproduction/protocol.json')['reference_rows']
    evaluation=[r['request']['job'] for r in old+field]
    keys=[r['job_id'] for r in sentinel]+[target['job_id']]
    physics=sum(j['case']['step_cap'] for j in evaluation)+4*target['request']['job']['case']['step_cap']+2*sum(r['request']['job']['case']['step_cap'] for r in sentinel)+2*target['request']['job']['case']['step_cap']
    p.update(id='T637_FULL384_JOINT_RETENTION_AND_VERIFIED_RECOVERY',out=str(OUT),
        before_success_ids=sorted(r['job_id'] for r in refs),target_job=target['request']['job'],target_reference=target,
        teacher_first=first,teacher_kind='GAP',sentinel_jobs=[r['request']['job'] for r in sentinel],key_replay_ids=keys,evaluation_jobs=evaluation,
        field_limit_m=7.,fit_steps=128,fit_lr=.001,keep_weight=10.,model_bytes=2*2**20,compact_bytes=12288,
        max_bytes=52*2**20,max_physics=physics,max_commands=431,main_seconds=3600,audit_seconds=1800,
        learning_seconds=600,full_evaluation_reserve_seconds=900,transport_seconds=300,source_seconds=180,all_seconds=7200,
        new_Actor_Adam=128,shadow_Actor_Adam=128,independent_historical_Actor_Adam=128,preflight_optimizer=0,
        new_RL=0,new_Critic=0,authorization_approved=False,inputs={})
    D.B.setup(p);q=D.L.frozen(p,p['initial']['path']);before=digest(q.state_dict())
    # Verify all local field success memories and at least one entire archive
    # through the real network-only subprocess, without restoring NPZ caches.
    started=time.time();data=[];sources=[]
    def guard():assert time.time()-started<600,'Preparation network/history bound'
    grouped=D.groups(refs)
    first_remote=next(key for key in grouped if key!='LOCAL')
    fixture_rows=grouped[first_remote]+grouped.get('LOCAL',[])
    for a,r in D.arrays(fixture_rows,guard):
        data.append(D.features(q,a,r));sources.append(dict(id=r['job_id'],sha256=r['sha256'],history_sha256=digest(a)))
    assert len(data)>5 and len(data)<384
    correction=M.correction(p,source,first)
    # Pure GPU forward/backward; no optimizer instance or step is created.
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not jobs
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    q.cuda();q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    packed=M.pack(data,'cuda');new={k:v.cuda() for k,v in correction.items()}
    # Fixture labels here are T501's own failed actions, not successful recovery
    # training labels; no learning and no admission occurs in preparation.
    loss,repair,keep=M.objective(q,packed,new,p);loss.backward()
    assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in D.L.variables(q))
    assert float(keep.detach())<1e-18 and digest(q.state_dict())==before
    cpu=q.cpu();cpu.zero_grad(set_to_none=True)
    x=torch.tensor(source['states'][:32]);cpu.sample=False;cpu.reset_noise();h=None;uu=[]
    with torch.no_grad():
        loc=cpu.sequence(x)
        for z in x:
            u,_,h,_=cpu.step(z[None],h);uu.append(u[0])
    assert torch.equal(torch.stack(uu),loc.clamp(-1,1).float())
    # Schema/size account includes the four teacher-specific arrays.
    n=900;k=len(source['states'])
    extra=4*n*(48+2)+8*5+8
    full_upper=sum(v.dtype.itemsize*(n*int(np.prod(v.shape[1:])) if v.ndim and v.shape[0]==k else v.size) for v in source.values())+extra+131072
    assert full_upper<p['full_bytes'],full_upper
    own_upper=16*p['full_bytes']+p['model_bytes']+407*p['compact_bytes']+10*2**20
    worst=D.B.used(ROOT/'stage/runs')+p['max_bytes']+131072
    assert own_upper<=p['max_bytes'] and worst<2**30
    resource=subprocess.check_output([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t636_joint_best_reproduction/control.py'),'resources'],text=True)
    save(DOC/'RESOURCE_PREFLIGHT.json',json.loads(resource))
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files={Path(p['initial']['path']),Path(target['path']),ROOT/'stage/docs/day_20261118/t636/FINAL_REPORT.json',DOC/'PLAN.md',N.TEACHER_SOURCE,
           ROOT/'stage/runs/t501_joint_command_repair/joint/cases.jsonl.t595.xz',next((ROOT/'stage/runs/t502_fullfield_probe/probe').glob('cases.jsonl*'))}
    files|={Path(j['exact_snapshot']) for j in evaluation}
    files|={Path(g) for g in grouped if g!='LOCAL'}
    files|={Path(r['path']) for r in refs if Path(r['path']).exists()}
    files|={Path(p['source_protocol'])}
    science=D.E.N.science(p);files|={Path(f) for f in science['input_hashes']}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    files|=set(HERE.glob('*.py'))
    p['inputs']={str(f):sha(f) for f in sorted(files)}
    save(HERE/'protocol.json',p)
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_histories=len(data),sources=sources,
         remote_archives_readback=1,remote_every_member_SHA=True,restored_disk_cache_bytes=0,
         full384_available_from_verified_local_or_remote=True,full384_not_yet_all_readback_this_round=True,
         real_GPU_forward_backward_finite=True,initial_keep_error=float(keep.detach()),initial_model_unchanged=True,
         actual_carry_step_sequence_exact=True,teacher_first=first,teacher_first_time=float(t[first]),
         teacher_qualification_not_yet_run=True,zero_optimizer=True,full_file_upper=full_upper,
         own_storage_upper=own_upper,global_worst=worst,free_bytes=D.B.shutil.disk_usage(ROOT).free,
         preparation_seconds=time.time()-started))
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_bytes','max_physics','max_commands','main_seconds','audit_seconds','learning_seconds','transport_seconds','source_seconds','all_seconds','new_Actor_Adam','shadow_Actor_Adam','independent_historical_Actor_Adam','preflight_optimizer']})
    print('T637_PREPARED_NO_REGISTRATION_NO_LAUNCH',len(data),physics,worst,flush=True)


if __name__=='__main__':main()
