"""One verified actual-state correction with own-function retention; no grid."""
from pathlib import Path
import copy,gzip,importlib.util,io,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t451_single_failure_recovery';DOC=ROOT/'stage/docs/review_20261110'
spec=importlib.util.spec_from_file_location('stage451_pinned449',ROOT/'stage/experiments/t449_offline_joint_recovery/run.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
import t451_storage as Z
P,R,torch,np,E=M.P,M.R,M.torch,M.np,M.E
read,save,sha,digest=P.read,P.save,P.sha,P.digest
def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert Z.used(OUT)<=p['max_bytes'] and Z.used()-Z.used(OUT)+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f
def setup_runtime():M.OUT=OUT;M.verify=verify;M.Pool=Pool;M.setup_runtime()
def worker(pipe,index,p,counts):M.worker(pipe,index,p,counts)
class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(8)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';gg=read(f) if f.exists() else [];gg.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,gg)
        except BaseException:self.close();raise
def objects(p):
    obj=P.make(p);P.load(obj,P.checkpoint(p['research_initial']['path'])['joint_state']);assert digest(P.state(*obj))==p['research_initial']['joint_hash']
    for g in obj[2].param_groups:g['lr']=p['repair_lr']
    return obj
def reference_data(p):
    # Never preserve the known failed00 function as if it were a success.
    keep=set(p['initial_success_ids']);data=[g for g in M.dataset(p) if g['id'] in keep]
    assert len(data)==218 and all(g['id']!=p['failed_job']['id'] for g in data);return data
def anchors(q,data):
    result=[]
    with torch.no_grad():
        for g in P.packs(data,'cuda'):result.append(dict(g,target=q.sequence(g['x']).detach().clone()))
    return result
def repair_step(obj,groups,own,correction):
    q,v,ao,vo=obj;ao.zero_grad(set_to_none=True);loss=P.action_loss(q.sequence(correction['x']),correction['y']).mean();rec=float(loss.detach());loss.backward();global_anchor=own_anchor=0.
    for task in range(10):
        selected=[g for g in groups if g['task']==task];weight=1/16 if task<8 else 1/4
        for off in range(0,len(selected),8):
            chunk=selected[off:off+8];x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk]);target=torch.nn.utils.rnn.pad_sequence([g['target'] for g in chunk]);loc=q.sequence(x)
            loss=sum((loc[:len(g['x']),i]-target[:len(g['x']),i]).square().mean() for i,g in enumerate(chunk))*weight/len(selected);loss.backward();global_anchor+=float(loss.detach())
    for off in range(0,len(own),5):
        chunk=own[off:off+5];x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk]);target=torch.nn.utils.rnn.pad_sequence([g['target'] for g in chunk]);loc=q.sequence(x)
        loss=sum((loc[:len(g['x']),i]-target[:len(g['x']),i]).square().mean() for i,g in enumerate(chunk))/len(own);loss.backward();own_anchor+=float(loss.detach())
    norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));ao.step();return dict(recovery=rec,reference218=global_anchor,own10=own_anchor,gradient_norm=norm)
def mature_size(obj,p):
    clone=P.make(p);P.load(clone,P.state(*obj))
    for opt,net in [(clone[2],clone[0]),(clone[3],clone[1])]:
        opt.zero_grad(set_to_none=True)
        for v in net.parameters():v.grad=torch.ones_like(v)
        opt.step()
    mem=io.BytesIO();s=P.state(*clone);torch.save(dict(joint_state=s,policy_hash=clone[0].phash(),joint_hash=digest(s),architecture='plain120_128_64GRU_2',single_actor=True),mem)
    raw=len(mem.getvalue());assert raw+65536<=p['model_bytes'];return raw
def check_parent_reproduction(values,previous):
    for a,r in values:
        old=previous[r['job_id']];assert r['array_digest']==old['array_digest'] and r['integration_sha256']==old['integration_sha256'] and r['array_hashes']==old['array_hashes']
def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists();setup_runtime();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p);state='FAILED';error=None;gate=None;candidate=None;training=None;proposal=None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard);parent={r['job_id']:r for r in map(json.loads,Path(p['parent_cases']).read_text().splitlines()) if r['role']=='FINAL_recovery'}
        failed=rt.request([p['failed_job']],p['research_initial'],'INITIAL_FAILED00',full=True);check_parent_reproduction(failed,parent);assert not failed[0][1]['success'];rt.request([p['failed_job']],p['research_initial'],'INITIAL_FAILED00_EXACT',full=True,references=[r for a,r in failed])
        preservation=rt.request(p['preserve_jobs'],p['research_initial'],'INITIAL_OWN10',full=True);check_parent_reproduction(preservation,parent);assert all(r['success'] and not r['collision'] for a,r in preservation);rt.request(p['preserve_jobs'],p['research_initial'],'INITIAL_OWN10_EXACT',full=True,references=[r for a,r in preservation])
        query=rt.request([p['failed_job']],p['research_initial'],'CURRENT_RECOVERY00',full=True,query=True);rt.request([p['failed_job']],p['research_initial'],'CURRENT_RECOVERY00_EXACT',full=True,references=[r for a,r in query],query=True)
        if query[0][1]['success'] and not query[0][1]['collision']:
            obj=objects(p);shadow=P.make(p);P.load(shadow,P.state(*obj));groups=anchors(obj[0],reference_data(p));own=anchors(obj[0],[dict(id=r['job_id'],task=10,x=a['states'],y=a['actions']) for a,r in preservation]);a=query[0][0];correction=P.packs([dict(x=a['states'],y=a['teacher_actions'])],'cuda')[0];trace=[]
            for k in range(p['steps']):
                rt.role='TARGETED_00_REPAIR';rt.guard(True)
                def step(which,objects):
                    with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([which,'BEFORE',k])+'\n')
                    v=repair_step(objects,groups,own,correction)
                    with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([which,'AFTER',k])+'\n')
                    return v
                aa=step('live',obj);bb=step('shadow',shadow);assert aa==bb;trace.append(aa)
                if k%32==0:assert digest(P.state(*obj))==digest(P.state(*shadow));save(OUT/'fit_progress.json',dict(steps=k+1,latest=aa,epoch=time.time()))
            assert digest(P.state(*obj))==digest(P.state(*shadow));save(OUT/'repair_trace.json',trace);proposal=P.export(obj,OUT/'proposal.pt.gz',p['model_bytes']);training=dict(steps=p['steps'],Actor_Adam=p['steps'],shadow_Actor_Adam=p['steps'],V_Adam=0,correction=query[0][1],own_rows=[r for a,r in preservation])
            candidate,baseline=M.evaluate(rt,p,proposal,'DEVELOPMENT329');candidate['preserves_initial328']=set(p['initial_success_ids'])<=set(candidate['success_ids']);gate=M.try_gate(rt,p,candidate,baseline)
        else:gate=dict(state='STOPPED_TEACHER_CANNOT_RECOVER_CURRENT_STUDENT_STATE',fresh16=None,reload78=None,six_gate=False)
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,physics=sum(rt.counts),commands=rt.commands))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+p['preflight_commands'],candidate=candidate,training=training,gate=gate,error=error,all_native_workers_stopped=True,six_gate=False,eight_started=False));save(OUT/'status.json',dict(state=state,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error;print('T451_MAIN_COMPLETE',flush=True)
def prepare():
    assert not OUT.exists();oldout=ROOT/'stage/runs/t449_offline_joint_recovery';base=read(oldout/'protocol.json');reconstruction=ROOT/'stage/runs/t450_t449_history_reconstruction';assert read(reconstruction/'summary.json')['state']=='COMPLETE' and read(reconstruction/'audit.json')['state']=='PASS_COMPLETED_BC_HISTORY_AND_RECONSTRUCTED_UNSAVED_PPO'
    assert not Path('/proc',str(read(reconstruction/'summary.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T450.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    parent=read(oldout/'phases.json')[1]['candidate'];assert parent['old275']['success']==274 and parent['wide54']['success']==54
    p=copy.deepcopy(base)
    for key in ['cold_storage','byte_references','night_initial_result','night_initial_cases_path','night_initial_role','inputs','constructor_reads','night_training_jobs','registration_recovery','absolute_learning_stop','absolute_native_stop','absolute_all_stop']:p.pop(key,None)
    for key in list(p):
        if key.startswith(('maximum_','preflight_','max_live_','max_shadow_')):p.pop(key)
    for key in ['night_initial','initial_plain_model','continuation_cursor','arms','tail_steps','recovery_steps','training_jobs','continuation_is_matched_control_NOT_unbounded_T446_resume','metadata_finalize_no_new_native_or_Adam','source_model_uncompressed_bytes','parent_T447_conclusion']:p.pop(key,None)
    p.update(id='T451_SINGLE_OLD00_ACTUAL_STATE_REPAIR',out=str(OUT),research_initial=parent['model'],initial_success_ids=parent['success_ids'],parent_cases=str(oldout/'cases.jsonl'),steps=400,repair_lr=3e-5,
        teacher_first=5,max_bytes=51*2**20,model_bytes=2*2**20,full_bytes=448*1024,compact_bytes=4*1024,max_physics=14000000,max_commands=600,
        main_seconds=5400,learning_seconds=2700,audit_seconds=1200,backup_seconds=600,all_seconds=7800,preflight_commands=24,
        bc_steps=400,maximum_bc_live_Adam=400,maximum_bc_shadow_Adam=400,maximum_actual_BC_live_Adam=400,maximum_BC_shadow_Adam=400,
        maximum_audit_discarded_Adam=400,maximum_ppo_live_Actor_Adam=0,maximum_ppo_live_V_Adam=0,PPO_updates=0,V_updates=0,
        maximum_new_training_episodes=0,maximum_exact_training_replays=0,maximum_teacher_trajectories=1,maximum_teacher_exact_replays=1,
        mechanism='Current328/329 model actual-state verified teacher recovery00,218 known-success CURRENT-policy zero-loss location anchors(exclude failed00 from original219) plus10actual own successful histories, matureActorAdam lr3e-5,400BC+400shadow;0PPO/V. Joint pilot, not isolated causal control or originality.',
        training='ONE400step supervised Actor proposal plus400shadow, no Value/PPO update.218 known-success initial-policy function anchors,10actual-own-success anchors,1new verified current-state recovery.',
        new_mechanism='Replace old mixed-expert219 action targets with current-policy function retention on218 known-success task histories, excluding the known failed00, while correcting that sole failure on newly induced states.',
        comparison='No matched causal control; targeted joint engineering repair. Previous T449 branches and rejection rules unchanged.',
        parent_comparison='Research parent328/329, sole00 fails. All328 initial successIDs must remain safe; no phase selection from the345 or78 navigation holdout.',
        original_T449_acceptance_and_FAILED_unchanged=True,preflight_discarded_Actor_Adam=5,preflight_discarded_V_Adam=1,
        user_authorization='Nov10 explicitly start next step after328/329 morning evidence. Separate bounded targeted00 mechanism; no T449 restart or seed/epoch grid.',
        final_selection='Fixed400th proposal, full329. Preserve original328 and repair00 before fresh16/78; no intermediate best selection. Research candidate not safeT255 replacement before whole gate.')
    p['failed_job']=next(j for j in p['old275_jobs'] if j['id']=='dev_task6_00');p['preserve_jobs']=p['old8_jobs']+[next(j for j in p['main_jobs'] if j['id']==f'wide117_80_dev6_{s}_13') for s in ['LEFT','RIGHT']]
    p['protected_files']={**base['protected_files'],parent['model']['path']:parent['model']['sha256']}
    protected=set(base['inputs'])|set(base['constructor_reads'])|set(p['protected_files'])|set(read(reconstruction/'protocol.json')['inputs'])|{parent['model']['path'],str(oldout/'cases.jsonl'),str(reconstruction/'audit.json'),str(reconstruction/'summary.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T450.json'),str(ROOT/'stage/docs/long_distance_20261101/source_manifest_T450.json')}
    references,saving=Z.candidates(protected,max(20*2**20,Z.used()+p['max_bytes']+1024**2-2**30));p['byte_references']=references
    protected|={str(ROOT/r['canonical']) for r in references};p['constructor_reads']=sorted(protected)
    files=protected|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T451_PLAN.md')}
    for r in references:
        for k in ['receipt','manifest','canonical_receipt','canonical_manifest']:files.add(str(ROOT/r[k]))
    assert not {str(ROOT/r['original']) for r in references}&files;p['inputs']={f:sha(f) for f in sorted(files)}
    p['worst_physics']=2*sum(j['case']['step_cap'] for j in [p['failed_job']]+p['preserve_jobs']+[p['failed_job']])+sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs']+p['fresh_jobs']+p['old8_jobs']+p['main_jobs']+p['fresh_jobs']);assert p['worst_physics']<p['max_physics']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    storage=90*p['full_bytes']+2*p['model_bytes']+345*p['compact_bytes']+5*2**20;assert storage<=p['max_bytes']
    save(DOC/'T451_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',own_max=p['max_bytes'],storage_worst=storage,global_full_after_reference=Z.used()-Z.used(OUT)-saving+p['max_bytes'],reference_files=len(references),reference_saving=saving,unique_bytes_deleted=0,
        max_physics=p['max_physics'],worst_physics=p['worst_physics'],max_commands=600,maximum_commands=503,main_seconds=5400,learning_seconds=2700,audit_seconds=1200,all_seconds=7800,
        Actor_live=400,Actor_shadow=400,V=0,PPO=0,independent_historical_Adam=400,all16fresh78fullgate_reserved=True))
    print('T451_REGISTERED',len(references),saving,p['worst_physics'],flush=True)
def encode():Z.encode(read(OUT/'protocol.json'))
def storage_audit():p=read(OUT/'protocol.json');Z.audit(p);verify(p,True)
def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'storage_audit.json')['state']=='PASS' and not (OUT/'preflight.json').exists();setup_runtime();obj=objects(p);shadow=objects(p);before=P.state(*obj);data=reference_data(p);groups=anchors(obj[0],data)
    own=anchors(obj[0],[dict(g,task=10) for g in data[:10]]);row=next(r for r in map(json.loads,Path(p['parent_cases']).read_text().splitlines()) if r['role']=='RECOVERY_QUERY' and r['job_id']=='dev_task6_00');a=E.N.S.arrays(row['path']);assert row['success'] and not row['collision'];correction=P.packs([dict(x=a['states'],y=a['teacher_actions'])],'cuda')[0]
    trace=[];start=time.monotonic()
    for k in range(2):
        aa=repair_step(obj,groups,own,correction);bb=repair_step(shadow,groups,own,correction);assert aa==bb;trace.append(aa)
    torch.cuda.synchronize();estimate=(time.monotonic()-start)/2*p['steps'];worst_estimate=estimate*max(1.,400/min(max(len(g['x']) for g in own),len(correction['x'])));assert worst_estimate<2200 and digest(P.state(*obj))==digest(P.state(*shadow));assert trace[0]['reference218']<1e-24 and trace[0]['own10']<1e-24
    P.load(obj,before);assert digest(P.state(*obj))==digest(before);raw=mature_size(obj,p)
    counts=R.M.mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir();pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:replies=pool.request([dict(model=p['research_initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']]);assert all(r['state']=='CANCEL_PASS' for r in replies) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,discarded_Actor_Adam=5,discarded_V_Adam=1,zero_initial_anchors=trace[0],live_shadow_and_mature_rollback_exact=True,all_dense_twoAdam_uncompressed_checkpoint_bytes=raw,raw_plus64KiB_within2MiB=True,estimated400live_shadow_seconds=estimate,estimated_with_conservative400decision_length_factor_seconds=worst_estimate,correction_is_original_parent_engineering_fixture_NOT_new_current_recovery=True,eight_workers_cancelled_and_stopped=True))
    print('T451_PREFLIGHT_PASS',estimate,raw,flush=True)
def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();ids=[s['pid']]+[v for g in read(OUT/'worker_generations.json') for v in g['workers']];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()));replays=list(map(json.loads,(OUT/'replays.jsonl').read_text().splitlines()));assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==24+16*len(read(OUT/'worker_generations.json'))+len(rows)+len(replays)
    calls=0
    if s['training']:
        obj=objects(p);groups=anchors(obj[0],reference_data(p));own=anchors(obj[0],[dict(id=r['job_id'],task=10,x=(a:=E.N.S.arrays(r['path']))['states'],y=a['actions']) for r in s['training']['own_rows']]);a=E.N.S.arrays(s['training']['correction']['path']);correction=P.packs([dict(x=a['states'],y=a['teacher_actions'])],'cuda')[0];trace=[]
        for k in range(p['steps']):trace.append(repair_step(obj,groups,own,correction));calls+=1
        assert trace==read(OUT/'repair_trace.json') and digest(P.state(*obj))==s['candidate']['model']['joint_hash'];assert sha(s['candidate']['model']['path'])==s['candidate']['model']['sha256']
        ev=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines()))
        for which in ['live','shadow']:assert [r[1:] for r in ev if r[0]==which]==[[w,k] for k in range(p['steps']) for w in ['BEFORE','AFTER']]
    expert=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        if r['full']:
            assert digest(a)==r['array_digest'] and {k:digest(v) for k,v in a.items()}==r['array_hashes'];M.audit_full_history(p,r,a)
            if r['teacher']:
                with torch.no_grad():lab=expert.base_actions(torch.tensor(a['states'])).float().numpy()
                assert np.array_equal(lab,a['teacher_actions'])
    bypath={r['path']:r for r in rows}
    for r in replays:assert r['array_digest']==bypath[r['reference_path']]['array_digest'] and r['integration_sha256']==bypath[r['reference_path']]['integration_sha256']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Actor_Adam=calls,full_training_history_hidden_reward_teacher_and_array_SHA_verified=True,final_compact329_full120_NOT_reconstructed=True,physics=s['physics'],commands=s['commands'],gate=s['gate']))
    print('T451_AUDIT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T451','--why=One bounded current-state correction and whole navigation gates','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,learning_stop=now+p['learning_seconds'],native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']));print('T451_ACTUALLY_LAUNCHED',c.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
