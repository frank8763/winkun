"""Fixed same-history action and integer-physical-time path diagnosis.

No new simulator work, optimizer, policy update, or handoff trial. Inference on
a fixed parent history is not a candidate closed-loop navigation experiment.
"""
from pathlib import Path
import sys,json,time,hashlib,io,fcntl,copy,shutil,traceback,subprocess
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t419_same_history_diagnosis';DOC=ROOT/'stage/docs/day_20261107'
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E
torch,np=E.torch,E.np
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,v):Path(f).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def used(d):return sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file())
def prepare():
    assert not OUT.exists();origin=ROOT/'stage/runs/t418_evening_repair_chain'
    source=read(origin/'repair/protocol.json');rows=[json.loads(v) for v in (origin/'repair/cases.jsonl').read_text().splitlines()]
    initial=[r for r in rows if r['role'].startswith('initial_')];assert len(initial)==14 and all(r['full'] for r in initial)
    chosen=['control_safe_reset_dev_task6_13','control_left_carry_wide_dev_3','teacher_0_safe_reset_dev_task6_13','teacher_0_left_carry_wide_dev_3']
    teachers=[next(r for r in rows if r['role']==name) for name in chosen]
    models=[dict(name='parent',path=source['student_start_model'],sha256=source['student_start_sha256'],policy_hash=source['initial_expected_policy_hash'])]
    for phase,rnd,kind in [('repair',0,'bc'),('protect',0,'bc'),('rl',0,'rl'),('rl',1,'rl')]:
        s=read(origin/phase/'summary.json');assert s['state']=='COMPLETE' and read(origin/phase/'audit.json')['state']=='PASS'
        m=s['rounds'][rnd][kind]['proposed_model'];models.append(dict(name=phase+'_'+str(rnd),path=m['path'],sha256=m['sha256'],policy_hash=m['policy_hash']))
    assert len({m['policy_hash'] for m in models})==5
    files=set(source['inputs'])|{str(HERE/'run.py'),str(HERE/'transport.py'),str(origin/'protocol.json'),str(origin/'repair/cases.jsonl'),str(DOC/'T418_CLOSEOUT_RECEIPT.json')}
    files.update(m['path'] for m in models);files.update(r['path'] for r in initial+teachers)
    for phase in ['repair','protect','rl']:files.update(str(origin/phase/name) for name in ['summary.json','audit.json'])
    p=dict(id='T419_FIXED_SAME_HISTORY_AND_GEOMETRY_DIAGNOSTIC',source_config=source,models=models,initial_rows=initial,teacher_rows=teachers,
        inputs={f:sha(f) for f in sorted(files)},own_bytes=2*2**20,main_seconds=600,audit_seconds=600,backup_seconds=180,all_seconds=1500,
        absolute_stop=1794056400,native=0,optimizer_steps=0,Actor_updates=0,
        action_threshold=.05,action_sustained_holds=5,position_threshold_m=.10,yaw_threshold_deg=5.,geometry_sustained_holds=3,handoff_search_max_hold=20,
        handoff_rule='Common exact integernative elapsed times only, before first3-sustained position>.10m OR yaw>5deg; preceding studenthold within0..19. No match=>NO_REGISTERED_HANDOFF; hold0=>historical t0control only. Proposed heuristic, not navigation/safety certificate.',
        scope='Fixed historical models and actual complete parent histories only; no counterfactual native rollout, no perturbed-state training labels, no global navigation claim.')
    assert time.time()+p['all_seconds']<p['absolute_stop'] and used(ROOT/'stage/runs')+p['own_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print('T419_REGISTERED_ZERO_NATIVE_ZERO_ADAM',flush=True)
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert used(OUT)<=p['own_bytes'] and used(ROOT/'stage/runs')<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
def sustained(mask,count):
    for k in range(len(mask)-count+1):
        if bool(np.all(mask[k:k+count])):return k
    return None
def no_mutations():
    E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def prohibited(*a,**k):raise AssertionError('T419 prohibits optimizer/native execution')
    torch.optim.Adam.step=prohibited;torch.optim.AdamW.step=prohibited;torch.optim.SGD.step=prohibited
    import mujoco
    mujoco.mj_step=prohibited;mujoco.mj_step1=prohibited;mujoco.mj_step2=prohibited
def compute(p,seconds):
    no_mutations();start=time.time();deadline=min(start+seconds-5,p['absolute_stop']-10);cfg=p['source_config'];base=E.N.science(cfg);results=[];traces={};parent_own=[]
    arrays=[E.N.S.arrays(r['path']) for r in p['initial_rows']]
    for mi,m in enumerate(p['models']):
        q=E.safe_initial(base,cfg).eval().requires_grad_(False);q.load_learned(E.N.S.read_model(m['path']));assert sha(m['path'])==m['sha256'] and q.phash()==m['policy_hash'];assert q.core.batch_first is False
        before=E.digest(q.state_dict());model_rows=[]
        for ri,(a,row) in enumerate(zip(arrays,p['initial_rows'])):
            assert time.time()<deadline;assert sha(row['path'])==row['sha256'];assert E.digest(a)==row['array_digest']
            assert np.array_equal(a['states'][1:],a['next_states'][:-1]);x=torch.tensor(a['states']);b=q.base_actions(x)
            with torch.no_grad():
                location=q.sequence(x,b);whole,last=q.core(q.encoder(q.normalized(x))[:,None]);hidden=None;locs=[];hs=[]
                q.reset_noise();q.sample=False
                for v in x:
                    _,loc,hidden,_=q.step(v[None],hidden);locs.append(loc[0]);hs.append(hidden[0,0])
                loc_error=float((location-torch.stack(locs)).abs().max());hidden_error=float((whole[:,0]-torch.stack(hs)).abs().max())
                assert loc_error<1e-10 and hidden_error<1e-10 and float((last-hidden).abs().max())<1e-10
                pred=location.numpy();action=location.clamp(-1,1).float().numpy()
            if mi==0:
                E.audit_array(q,a,cfg,0,None);assert np.max(np.abs(pred-a['locations']))<1e-10 and np.array_equal(action,a['actions']);parent_own.append(True)
            delta=np.abs(action.astype(np.float64)-a['actions']);latent=((pred-a['locations'])**2/(2*np.asarray(cfg['raw_sigma'])**2)).mean(1)
            elapsed=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))][:-1];first=[]
            for dim in range(2):
                k=sustained(delta[:,dim]>p['action_threshold'],p['action_sustained_holds']);first.append(None if k is None else dict(hold=k,native_elapsed=int(elapsed[k]),seconds=float(elapsed[k]/500)))
            item=dict(job_id=row['job_id'],parent_closed_loop_success=row['success'],history_path=row['path'],history_sha256=row['sha256'],decisions=len(x),physical_steps=int(a['native_counts'].sum()),mean_abs_executed_error=delta.mean(0).tolist(),max_abs_executed_error=delta.max(0).tolist(),first_sustained_error_drive_steer=first,latent_KL_mean=float(latent.mean()),latent_KL_max=float(latent.max()),sequence_step_location_error=loc_error,sequence_step_hidden_error=hidden_error,
                not_candidate_closed_loop_result=True)
            model_rows.append(item);traces[f'{mi}_{ri}_native_elapsed']=elapsed;traces[f'{mi}_{ri}_action_delta']=delta;traces[f'{mi}_{ri}_latent_KL']=latent
            assert E.digest(q.state_dict())==before
        results.append(dict(model=m,histories=model_rows))
    geo=[]
    for job_id,teacher_kind in [('dev_task6_13','safe_reset'),('wide_dev_3','left_carry')]:
        row=next(r for r in p['initial_rows'] if r['job_id']==job_id);control=next(r for r in p['teacher_rows'] if r['job_id']==job_id and r['source_control']);recovery=next(r for r in p['teacher_rows'] if r['job_id']==job_id and not r['source_control'])
        student=E.N.S.arrays(row['path']);expert=E.N.S.arrays(control['path']);failed_or_safe=E.N.S.arrays(recovery['path'])
        q=E.safe_initial(base,cfg).eval().requires_grad_(False);case=dict(id=control['case_id'],job=next(j for j in cfg['old8_jobs']+cfg['left5_jobs'] if j['id']==job_id),first=0,teacher_kind=teacher_kind,source_control=True)
        E.audit_query(q,expert,case,cfg);case=dict(case,id=recovery['case_id'],first=20,source_control=False);E.audit_query(q,failed_or_safe,case,cfg)
        assert np.array_equal(student['states'][0],expert['states'][0]) and np.array_equal(student['poses'][0],expert['poses'][0])
        ts=np.r_[0,np.cumsum(student['native_counts'].astype(np.int64))][:-1];te=np.r_[0,np.cumsum(expert['native_counts'].astype(np.int64))][:-1]
        common,si,ei=np.intersect1d(ts,te,return_indices=True);position=np.linalg.norm(student['poses'][si,:2]-expert['poses'][ei,:2],axis=1)
        yaw=np.abs((student['poses'][si,2]-expert['poses'][ei,2]+np.pi)%(2*np.pi)-np.pi)
        mask=(position>p['position_threshold_m'])|(yaw>np.deg2rad(p['yaw_threshold_deg']));k=sustained(mask,p['geometry_sustained_holds']);proposed=None;status='NO_REGISTERED_HANDOFF'
        if k is not None and 0<int(si[k])<=p['handoff_search_max_hold']:
            proposed=int(si[k])-1;status='HISTORICAL_T0_CONTROL_REPEAT' if proposed==0 else 'UNTESTED_SINGLE_HANDOFF_DRAFT'
        # Exact time gaps are disclosed; sustained points must also be adjacent
        # parent holds rather than three unrelated matched timestamps.
        if k is not None:assert np.array_equal(si[k:k+3],np.arange(si[k],si[k]+3))
        result=dict(job_id=job_id,teacher_kind=teacher_kind,parent_success=row['success'],teacher_origin_success=control['success'],known4s_handoff_success=recovery['success'],known4s_handoff_remaining_m=recovery.get('remaining_m'),matched_timepoints=len(common),first_sustained_geometry=None if k is None else dict(parent_hold=int(si[k]),native_elapsed=int(common[k]),seconds=float(common[k]/500),position_m=float(position[k]),yaw_deg=float(np.rad2deg(yaw[k]))),proposed_handoff_hold=proposed,proposal_status=status,proposal_not_executed=True,limitation='Path difference is between two different closed-loop histories, not causal failure localization or unrecoverability proof.')
        geo.append(result);traces[job_id+'_common_native']=common;traces[job_id+'_position_m']=position;traces[job_id+'_yaw_rad']=yaw
    assert time.time()<deadline and len(parent_own)==14
    buffer=io.BytesIO();np.savez_compressed(buffer,**traces);blob=buffer.getvalue();assert len(blob)<1024*1024
    return dict(models=results,geometry=geo,native=0,optimizer_steps=0,Actor_updates=0,read_only_parent14_exact=True,trace_sha256=hashlib.sha256(blob).hexdigest(),trace_bytes=len(blob),not_six_metre_acceptance=True,seven_eight_entered=False),blob
def run():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists();start=time.time()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==tree
    assert start+p['main_seconds']+p['audit_seconds']+p['backup_seconds']+120<p['absolute_stop']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'main_started.json',dict(pid=__import__('os').getpid(),epoch=start,native=0,optimizer_steps=0,source_remote_commit=receipt['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')));state='FAILED'
    try:
        result,blob=compute(p,p['main_seconds']);(OUT/'traces.npz').write_bytes(blob);save(OUT/'metrics.json',result);state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),native=0,optimizer_steps=0))
    finally:
        save(OUT/'summary.json',dict(state=state,pid=__import__('os').getpid(),elapsed_s=time.time()-start,native=0,optimizer_steps=0,Actor_updates=0));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE';print('T419_COMPLETE_0NATIVE_0ADAM',flush=True)
def audit():
    p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    start=time.time();result,blob=compute(p,p['audit_seconds']);assert result==read(OUT/'metrics.json') and blob==(OUT/'traces.npz').read_bytes();verify(p)
    save(OUT/'audit.json',dict(state='PASS',elapsed_s=time.time()-start,native=0,optimizer_steps=0,Actor_updates=0,all_model_SHA_history_hidden_action_physical_time_and_serialized_trace_exact=True));print('T419_AUDIT_PASS_0NATIVE_0ADAM',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
