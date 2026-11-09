"""One frozen-data error/gradient audit, no optimizer or native integration."""
from pathlib import Path
import ast,copy,gzip,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t447_fixed_credit_audit';DOC=ROOT/'stage/docs/research_rescue_20261109'
sys.path.insert(0,str(ROOT/'stage/experiments/t446_plain_joint'))
import t446_learning as P
torch,np,E=P.torch,P.np,P.E
read,save,sha,digest=P.read,P.save,P.sha,P.digest

def used(folder=None):return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert used(OUT)<=p['max_bytes'] and used()-used(OUT)+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30

def feedback():
    """Deterministic red loop backed by already recorded real native failures."""
    folder=ROOT/'stage/runs/t446_plain_joint';rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()]
    row=next(r for r in rows if r['role']=='FINAL329' and r['job_id']=='dev_task6_00');assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path'])
    assert bool(a['end_flags'][-1,0])==row['collision'] and int(a['native_counts'].sum())==row['physics']
    print(json.dumps(dict(known_native_task=row['job_id'],success=row['success'],collision=row['collision'],new_native=0,new_optimizer=0)),flush=True)
    assert row['success'] and not row['collision'],'Recorded exact closed-loop regression: oldLEFT5 dev00 still collides'

def prepare():
    assert not OUT.exists(),'Never restart ended run'
    old=ROOT/'stage/runs/t446_plain_joint';s=read(old/'summary.json');assert s['state']=='COMPLETE' and read(old/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T446.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(old/'SEALED.json')['scientific_pids'])
    base=read(old/'protocol.json');p={k:copy.deepcopy(base[k]) for k in ['actor_seed','value_seed','bc_lr','actor_lr','value_lr','raw_sigma','input_mean','input_scale','old_rows','old_states']}
    rows=[json.loads(x) for x in (old/'cases.jsonl').read_text().splitlines()];p.update(id='T447_FIXED_SUCCESS_HISTORY_ERROR_AND_GRADIENT_AUDIT',model=s['bc_model'],
        teacher_rows=[r for r in rows if r['role']=='TEACHER_RECONSTRUCT'],max_bytes=1024**2,main_seconds=900,audit_seconds=900,backup_seconds=300,all_seconds=2400,
        max_native=0,max_optimizer=0,exact_history_gradients=10,preflight_discarded_Adam=0,
        phases=['first2s','2to20s','after20s'],edge_threshold=2e-6,linearized_loss_tests_NOT_causal_or_policy_gain=True,
        user_authorization='Nov9 next step after failed T446: bounded existing-history analysis, no extension of old training budget or native experiment.')
    files={str(f) for f in HERE.glob('*.py')}|{str(ROOT/'stage/experiments/t446_plain_joint/t446_learning.py'),p['old_rows'],p['old_states'],p['model']['path'],
        str(old/'summary.json'),str(old/'audit.json'),str(old/'cases.jsonl'),str(old/'bc_trace.json'),str(DOC/'T447_PLAN.md'),str(DOC/'T447_PRIMARY_SOURCES.md')}
    original=[json.loads(x) for x in gzip.decompress(Path(p['old_rows']).read_bytes()).decode().splitlines()];files|={r['path'] for r in original}|{r['path'] for r in p['teacher_rows']}
    p['inputs']={f:sha(f) for f in sorted(files)};assert used()+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p)
    save(DOC/'T447_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_bytes=p['max_bytes'],global_before=used()-used(OUT),global_full=used()-used(OUT)+p['max_bytes'],
        main_seconds=900,audit_seconds=900,all_seconds=2400,max_native=0,max_Adam=0,gradients_per_execution=10,original_model_frozen=True))
    print('T447_REGISTERED_ZERO_NATIVE_ZERO_OPTIMIZER',flush=True)

def objects(p):
    obj=P.make(p);P.load(obj,P.checkpoint(p['model']['path'])['joint_state']);return obj

def calculate(p):
    """Pure frozen-model derivatives; never change weights or Adam state."""
    obj=objects(p);q=obj[0];before=digest(P.state(*obj));data=P.dataset(p,p['teacher_rows']);groups=P.packs(data,'cuda');stats=[];grad=[];taskloss=[]
    parameters=list(q.parameters())
    for task in range(10):
        selected=[g for g in groups if g['task']==task];q.zero_grad(set_to_none=True);loss_value=0.;per=[];phases={k:[] for k in p['phases']}
        for g in selected:
            x,y=g['x'],g['y'];loc=q.sequence(x);loss=P.action_loss(loc,y).mean()/len(selected);loss.backward();loss_value+=float(loss.detach())
            raw=loc.detach();action=raw.clamp(-1,1).float().double();err=(action-y).cpu().numpy();time_s=np.arange(len(x),dtype=np.float64)*.2
            for phase,mask in zip(p['phases'],[time_s<2,(time_s>=2)&(time_s<20),time_s>=20]):
                if mask.any():phases[phase].append(dict(mse=np.mean(err[mask]**2,axis=0),bias=np.mean(err[mask],axis=0),maximum=np.max(np.abs(err[mask]),axis=0),count=int(mask.sum())))
            per.append(dict(id=g['id'],decisions=len(x),mse_drive_steer=np.mean(err**2,axis=0).tolist(),first10_bias_drive_steer=np.mean(err[:10],axis=0).tolist(),
                first10_max_drive_steer=np.max(np.abs(err[:10]),axis=0).tolist(),peak_error_index_drive_steer=np.argmax(np.abs(err),axis=0).tolist(),
                boundary_label_fraction=float((y.abs()>=1).double().mean()),raw_outside_execution_fraction=float((raw.abs()>1).double().mean())))
        vec=torch.cat([torch.zeros_like(v).flatten() if v.grad is None else v.grad.detach().flatten() for v in parameters]).cpu().numpy();grad.append(vec);taskloss.append(loss_value)
        phase_summary={}
        for name,values in phases.items():
            if values:phase_summary[name]=dict(episodes=len(values),mean_episode_RMSE_drive_steer=np.sqrt(np.mean([v['mse'] for v in values],axis=0)).tolist(),
                mean_episode_bias_drive_steer=np.mean([v['bias'] for v in values],axis=0).tolist(),maximum_error_drive_steer=np.max([v['maximum'] for v in values],axis=0).tolist(),decisions=sum(v['count'] for v in values))
        stats.append(dict(task=task,episodes=len(selected),mean_surrogate_loss=loss_value,gradient_norm=float(np.linalg.norm(vec)),phases=phase_summary,episodes_detail=per))
    G=np.stack(grad);gram=G@G.T;norm=np.sqrt(np.diag(gram));cos=gram/np.maximum(norm[:,None]*norm[None,:],1e-30)
    weights=np.r_[np.full(8,1/16),[1/4,1/4]];total=weights@G
    # Compare only analytic directions, not a trained alternate method.
    focused=weights.copy();focused[6]*=4;focused/=focused.sum();focus=focused@G
    aopt=P.state(*obj)['actor_opt'];pred=[]
    # The mature Adam preconditioned first-order direction is computed directly
    # without step() or touching optimizer state; this is NOT a retained update.
    offset=0;adam=[]
    clip_coefficient=min(1.,1./(float(np.linalg.norm(total))+1e-6))
    for group in aopt['param_groups']:
        beta1,beta2=group['betas'];lr=group['lr'];eps=group['eps']
        for pid in group['params']:
            moment=aopt['state'][pid];size=moment['exp_avg'].numel();gg=clip_coefficient*total[offset:offset+size].reshape(moment['exp_avg'].shape)
            mm=moment['exp_avg'].numpy();vv=moment['exp_avg_sq'].numpy();step=int(moment['step'].item())+1
            m=beta1*mm+(1-beta1)*gg;v=beta2*vv+(1-beta2)*gg**2
            direction=-lr*(m/(1-beta1**step))/(np.sqrt(v/(1-beta2**step))+eps);adam.append(direction.ravel());offset+=size
    assert offset==G.shape[1];ad=np.concatenate(adam)
    for task in range(10):
        pred.append(dict(task=task,old_weight=float(weights[task]),analytic_uniform_descent_dot=float(-G[task]@total),
            analytic_left4x_descent_dot=float(-G[task]@focus),mature_Adam_one_step_linear_delta=float(G[task]@ad),
            gradient_cosine_to_joint=float(G[task]@total/max(1e-30,np.linalg.norm(G[task])*np.linalg.norm(total)))))
    assert digest(P.state(*obj))==before
    return dict(state='COMPLETE_ZERO_UPDATE_DIAGNOSIS',task_stats=stats,group_grad_gram=gram.tolist(),group_grad_cosine=cos.tolist(),
        group_gradient_SHA=digest(G),weights=weights.tolist(),first_order_predictions=pred,
        left_gradient_cosine_to_wide_left=float(cos[6,8]),left_gradient_cosine_to_wide_right=float(cos[6,9]),
        analytic_Adam_gradient_clip_coefficient=clip_coefficient,
        no_finite_update_or_closed_loop_gain_measured=True,model_and_Actor_V_Adam_hash_unchanged=before,
        phases_use_decision_time_0_2sec_last_hold_only_may_be_shorter=True,
        original_models_unchanged=True,optimizer_updates=0,native=0,model_gradient_evaluations=10)

def prohibit():
    def no(*a,**k):raise AssertionError('T447 forbids optimizer/native work')
    for cls in [torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=no
    import mujoco as mj
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mj,name,no)

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();prohibit()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=objects(p);s=P.state(*obj);q=obj[0];g=P.old_data(p)[0];x=torch.tensor(g['x'][:12],device='cuda');loss=q.sequence(x).square().mean();gg=torch.autograd.grad(loss,q.parameters());assert all(torch.isfinite(v).all() for v in gg)
    assert digest(P.state(*obj))==digest(s)
    save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,one_partial_history_gradient_only_engineering=True,model_Actor_V_Adam_unchanged=True))
    print('T447_PREFLIGHT_PASS',flush=True)

def main():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists();start=time.time()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T447 bound')));signal.alarm(p['main_seconds']);prohibit()
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start))
    try:result=calculate(p);save(OUT/'diagnosis.json',result);save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,native=0,optimizer=0,gradients=10,model_hash=result['model_and_Actor_V_Adam_hash_unchanged']));print('T447_COMPLETE',flush=True)
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),native=0,optimizer=0));raise

def audit():
    p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json');assert not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists();prohibit()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T447 audit bound')));signal.alarm(p['audit_seconds']);result=calculate(p);assert result==read(OUT/'diagnosis.json')
    save(OUT/'audit.json',dict(state='PASS_FROZEN_DIAGNOSIS_ONLY',pid=os.getpid(),new_native=0,optimizer_updates=0,independent10gradients_and_phase_errors_exact=True,model_Adam_and_original_input_hashes_unchanged=True))
    print('T447_AUDIT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip();now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=now+p['all_seconds'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')));print('T447_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
