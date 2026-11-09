"""Three bounded closed-loop relay mechanisms; strict joint navigation gate."""
from t434_core import *
class Runtime:
    def __init__(self,name,p):
        self.name=name;self.p=p;self.folder=OUT/ALIASES[name];self.folder.mkdir();self.limit=p['stage_budgets'][name];self.start=time.time();self.deadline=min(self.start+self.limit['main_seconds'],p['deadline_native']);self.learnstop=min(self.deadline-p['validation_reserve_seconds'],p['deadline_learning']);self.physics=mp.get_context('spawn').RawArray('q',5);self.commands=0;self.rows=[];self.replays=[];self.last=0.;self.role='SETUP';self.pool=None
        self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(self.folder/'main_started.json',dict(pid=os.getpid(),epoch=self.start,deadline=self.deadline,stage=name));E.set_out(self.folder)
    def start_pool(self):
        cfg=dict(self.p,out=str(self.folder),native_stop_epoch=self.deadline,max_physics=self.limit['max_physics'],relay_stage=self.name if self.name in ['fresh_rl','iterative_recovery','representation_rl'] else 'fresh_rl');self.pool=Pool(cfg,self.physics,self.guard);self.commands=4
        for cid,row in enumerate(self.pool.ready,1):self.log(cid,'SETUP',0,row['pid'])
    def guard(self):
        E.L.check_stop();assert time.time()<self.deadline-5 and sum(self.physics)<=self.limit['max_physics'] and self.commands<=self.limit['max_commands']
        if time.time()-self.last>5:
            self.last=time.time();verify(self.p);assert E.N.K.S.used(self.folder)<=self.limit['max_bytes'];save(self.folder/'status.json',dict(state=self.role,epoch=self.last,pid=os.getpid(),physics=sum(self.physics),commands=self.commands,rows=len(self.rows)))
    def learning_guard(self):self.guard();assert time.time()<self.learnstop-5
    def log(self,cid,role,physics,pid):
        with (self.folder/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=role,physics=physics,pid=pid))+'\n')
    def request(self,requests,role,full=False,refs=None):
        self.role=role;output=[]
        for offset in range(0,len(requests),4):
            req=requests[offset:offset+4];self.guard();assert self.commands+len(req)<=self.limit['max_commands'];assert sum(self.physics)+sum(x['job']['case']['step_cap'] for x in req)<=self.limit['max_physics'];ids=list(range(self.commands+1,self.commands+len(req)+1));self.commands+=len(req)
            for i,((a,row),request,cid) in enumerate(zip(self.pool.request(req),req,ids)):
                row=dict(row,role=role,job_id=request['job']['id'],physics_steps=int(a['native_counts'].sum()),loaded_model_sha256=sha(request['model']),seed=request.get('seed'),request=copy.deepcopy(request))
                assert row['physics_steps']==row['physics']
                if refs is not None:
                    old,previous=refs[offset+i]
                    assert set(a)==set(old)
                    for key in a:assert E.equal(a[key],old[key]),(role,key)
                    assert row['array_digest']==previous['array_digest'];row.update(exact=True,all_arrays_exact=True,independent_sample=False,reference_path=previous['path']);self.replays.append(row);name='replays.jsonl'
                else:
                    keep_full=full(request) if callable(full) else full;path=self.folder/'episodes'/f'{len(self.rows):04}.npz';path.parent.mkdir(exist_ok=True);E.N.write_npz(path,E.N.S.packed(a) if keep_full else {k:a[k] for k in ['actions','native_counts','end_flags']},self.p['full_bytes'] if keep_full else 5*1024);row.update(path=str(path),sha256=sha(path),full=bool(keep_full));self.rows.append(row);name='cases.jsonl'
                with (self.folder/name).open('a') as f:f.write(json.dumps(row)+'\n')
                self.log(cid,role,row['physics'],row['pid']);output.append((a,row))
        return output
    def student(self,jobs,model,role,full=False,seeds=None,refs=None):
        return self.request([dict(model=model['path'],job=j,mode='student',seed=None if seeds is None else seeds[i]) for i,j in enumerate(jobs)],role,full,refs)
    def development(self,model,role):
        values=self.student([*self.p['main_jobs'],*self.p['old275_jobs']],model,role,full=False)
        rows=[row for a,row in values];ids=sorted(r['job_id'] for r in rows if r['success'] and not r['collision'])
        return dict(success_ids=ids,all_safe=len(ids)==329,wide54=E.tally(rows[:54]),old275=E.tally(rows[54:]),policy_hash=model['policy_hash'],rows=rows,compact_arrays_only_full_hashes=True)
    def close(self,state,result,error=None):
        if self.pool:self.pool.close()
        save(self.folder/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-self.start,physics=sum(self.physics),commands=self.commands,result=result,error=error,all_workers_stopped=True,all_sessions_closed=True));save(self.folder/'status.json',dict(state=state,epoch=time.time(),physics=sum(self.physics),commands=self.commands));E.N.manifest(self.folder);fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()



def authenticated(row):
    assert row['full'] and sha(row['path'])==row['sha256']
    a=E.N.S.arrays(row['path']);assert set(a)==set(row['array_hashes'])
    assert digest(a)==row['array_digest'] and digest(a['final_native_integration'])==row['integration_sha256']
    for key,value in a.items():assert digest(value)==row['array_hashes'][key],(row['job_id'],key)
    return a

def exact_historical_row(actual,previous):
    for key in ['job_id','array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','remaining_m','policy_hash','loaded_model_sha256']:
        assert actual[key]==previous[key],(actual['job_id'],key)


def dataset(p,q,stage,dynamic=()):
    entries=read(p['relay_dataset_path'])['result'];anchors=entries['anchors_c'];targets=entries['targets']
    assert len(anchors)==62 and len(targets)==7
    for x in anchors:
        assert sha(x['path'])==x['sha256'];authenticated(x['row'])
    pairs=[(E.N.S.arrays(x['path']),x['row']) for x in anchors]
    labels=[dict(arrays=E.N.S.arrays(x['path']),first=x['first'],id=x['id']) for x in targets]
    return LEARNING.groups(q,dict(p,relay_stage=stage),pairs,labels,additional_targets=list(dynamic))

def stage_source(p,name):
    options=[]
    for order,previous in enumerate(['fresh_rl','iterative_recovery','representation_rl']):
        if previous==name:break
        f=OUT/previous/'summary.json'
        if not f.exists():continue
        s=read(f)
        if s['state']=='COMPLETE' and s['result']['accepted']:
            r=s['result'];assert read(OUT/previous/'audit.json')['state']=='PASS'
            options.append((len(r['selected_success_ids']),order,r))
    if options:
        r=max(options,key=lambda v:v[:2])[2]
        return r['model'],read(r['development_file'])['rows'],r['selected_success_ids']
    return initial_record(p),[json.loads(line) for line in Path(p['current_cases_path']).read_text().splitlines()],p['initial_success_ids']

def score(row):
    return (int(row['success'] and not row['collision']),int(not row['collision']),-float(row['remaining_m']))

def exact_prefix(a,b,first):
    assert first>0
    for key in ['states','poses','hidden_before']:
        assert np.array_equal(a[key][:first+1],b[key][:first+1]),key
    for key in ['actions','native_counts','rewards','locations','raw_actions','innovations','logp','atom','stochastic']:
        assert np.array_equal(a[key][:first],b[key][:first]),key
    for key in ['history_poses','history_goals','history_times']:
        assert np.array_equal(a[key][:first+1],b[key][:first+1]),key
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])

def dynamic_first(arrays,reference,prior_first,p):
    n=min(len(arrays['poses']),len(reference['poses']))
    error=np.linalg.norm(arrays['poses'][:n,:2]-reference['poses'][:n,:2],axis=1)
    yaw=np.abs(R.wrap(arrays['poses'][:n,2]-reference['poses'][:n,2]))
    bad=(error>p['deviation_position_m'])|(yaw>p['deviation_yaw_rad'])
    hits=[k for k in range(10,max(10,n-2)) if bool(bad[k:k+3].all())]
    first=max(1,min(prior_first,hits[0] if hits else prior_first)-p['intervene_earlier_holds'])
    return int(min(first,len(arrays['states'])-1)),dict(first_persistent_deviation=hits[0] if hits else None,position_threshold_m=p['deviation_position_m'],yaw_threshold_rad=p['deviation_yaw_rad'],earlier_holds=p['intervene_earlier_holds'],clock_and_history_not_reset=True)

def learn(rt,p,name):
    pp=dict(p,relay_stage=name);source,source_rows,parent_success=stage_source(p,name)
    assert sha(source['path'])==source['sha256'];q=frozen(pp,source['path']);assert q.phash()==source['policy_hash'];opt=LEARNING.make_optimizer(q,pp);assert not opt.state
    initial=LEARNING.export_model(q,opt,rt.folder/'initial.pt.gz');whole=E.L.state(q,opt);current=initial;fixed=LEARNING.fixedhash(q)
    jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']};panel_jobs=[jobs[i] for i in p['panel_ids']]
    before_panel={r['job_id']:r for r in source_rows if r['job_id'] in p['panel_ids']};assert len(before_panel)==len(panel_jobs)
    group=dataset(p,q,name);dynamic=[];rounds=[];events_file=rt.folder/'optimizer_calls.jsonl';updates=dict(live=0,shadow=0);rejects=0;kept=0
    def ev(tag,which,model,optimizer):
        def emit(where,k):
            if where=='AFTER':updates[which]+=1
            with events_file.open('a') as f:f.write(json.dumps(dict(tag=tag,which=which,where=where,k=k,hashes=E.L.statehash(model,optimizer),gradient_sha256=gradient_sha(model)))+'\n')
        return emit
    target_map={t['id']:t for t in read(p['verified_target_file'])}
    def pair_fit(tag,kind,items=None,rnd=0):
        before=E.L.state(q,opt);shadow=frozen(pp,current['path']);so=LEARNING.make_optimizer(shadow,pp);E.L.load(shadow,so,before)
        fn=lambda model,optimizer,which: (LEARNING.fit_bc(model,optimizer,group,pp,rt.learning_guard,ev(tag,which,model,optimizer)) if kind=='bc' else LEARNING.fit_ppo(model,optimizer,items,group,pp,rnd,rt.learning_guard,ev(tag,which,model,optimizer)))
        live=fn(q,opt,'live');other=fn(shadow,so,'shadow');assert live==other and E.L.statehash(q,opt)==E.L.statehash(shadow,so)
        assert LEARNING.fixedhash(q)==fixed
        return live
    for rnd in range(p['rounds'][name]):
        rt.learning_guard();folder=rt.folder/'blocks'/str(rnd);folder.mkdir(parents=True);before=E.L.state(q,opt);before_model=current;round_targets=[];stages=[]
        focus_id=p['focus_order'][rnd%len(p['focus_order'])]
        failed=[i for i in p['focus_order'] if not before_panel[i]['success']]
        if name=='iterative_recovery' and failed:focus_id=failed[rnd%len(failed)]
        job=jobs[focus_id];teacher_record=None
        if name=='iterative_recovery':
            diagnostic=rt.student([job],current,f'DIAG_{rnd}',True)[0]
            if not diagnostic[1]['success']:
                old=target_map[focus_id];ref=R.load_reference(old['path'],old['sha256']);first,why=dynamic_first(diagnostic[0],ref,old['first'],p)
                request=dict(model=current['path'],job=job,mode='front_feedback',seed=None,first=first,reference_path=old['path'],reference_sha256=old['sha256'],gains=p['gains'])
                recovery=rt.request([request],f'RECOVERY_{rnd}',True)[0]
                exact_prefix(recovery[0],diagnostic[0],first)
                rt.request([request],f'RECOVERY_EXACT_{rnd}',refs=[recovery])
                teacher_record=dict(row=recovery[1],first=first,selection=why,diagnostic=diagnostic[1],safe_verified=bool(recovery[1]['success'] and not recovery[1]['collision']))
                if teacher_record['safe_verified']:
                    dynamic.append(dict(arrays=recovery[0],first=first,id=f'{focus_id}:iteration{rnd}',path=recovery[1]['path'],sha256=recovery[1]['sha256']))
                    group=dataset(p,q,name,dynamic)
                    fit=pair_fit(f'round{rnd}_bc','bc');record=LEARNING.export_model(q,opt,folder/'after_bc.pt.gz')
                    stages.append(dict(tag=f'round{rnd}_bc',kind='bc',fit=fit,after_model=record));current=record
        seeds=[p['seed_base']+rnd*4+i for i in range(4)]
        training=rt.student([job]*4,current,f'TRAIN_{rnd}',True,seeds=seeds)
        rt.student([job]*4,current,f'TRAIN_EXACT_{rnd}',seeds=seeds,refs=training)
        for a,row in training:E.audit_array(frozen(pp,current['path']).requires_grad_(False),a,pp,0,row['seed'])
        fit=pair_fit(f'round{rnd}_ppo','ppo',[a for a,r in training],rnd)
        proposed=LEARNING.export_model(q,opt,folder/'proposed.pt.gz');stages.append(dict(tag=f'round{rnd}_ppo',kind='ppo',fit=fit,after_model=proposed,training_paths=[r['path'] for a,r in training],sampling_model=current,rnd=rnd))
        panel=rt.student(panel_jobs,proposed,f'PANEL_{rnd}');after_panel={r['job_id']:r for a,r in panel}
        previous_ids={i for i,r in before_panel.items() if r['success'] and not r['collision']};new_ids={i for i,r in after_panel.items() if r['success'] and not r['collision']}
        progress=bool(new_ids-previous_ids) or score(after_panel[focus_id])>score(before_panel[focus_id])
        accepted=bool(fit['numeric_accept'] and previous_ids<=new_ids and progress)
        if accepted:current=proposed;before_panel=after_panel;rejects=0;kept+=1
        else:E.L.load(q,opt,before);current=before_model;rejects+=1
        record=dict(round=rnd,focus_id=focus_id,before_model=before_model,teacher=teacher_record,dynamic_target_paths=[d['path'] for d in dynamic],stages=stages,accepted=accepted,previous_panel_success_ids=sorted(previous_ids),proposed_panel_success_ids=sorted(new_ids),focus_before_score=score({r['job_id']:r for r in source_rows}[focus_id]) if rnd==0 else None,focus_after_score=score(after_panel[focus_id]),numeric_accept=fit['numeric_accept'],whole_Actor_Adam_rollback=not accepted,kept_model=current,consecutive_rejects=rejects)
        save(folder/'result.json',record);rounds.append(record)
        if rejects>=2:break
    # A completed scientific negative needs no duplicate 329 measurement of
    # the unchanged source Actor. Every kept changed Actor receives full329.
    check=None;accepted=False
    if kept:
        check=rt.development(current,'FINAL329_'+name);save(rt.folder/'development.json',check);ids=set(check['success_ids']);accepted=set(parent_success)<=ids and bool(ids-set(parent_success))
    if not accepted:E.L.load(q,opt,whole)
    model=LEARNING.export_model(q,opt,rt.folder/'candidate.pt.gz')
    return dict(stage=name,initial_model=initial,source_model=source,parent_success_ids=parent_success,model=model,accepted=accepted,selected_success_ids=check['success_ids'] if accepted else parent_success,development_all_safe=bool(accepted and check['all_safe']),development_file=str(rt.folder/'development.json') if accepted else None,check_file=str(rt.folder/'development.json') if check else None,full329_executed=check is not None,full329_unexecuted_reason=None if check else 'NO_RESEARCH_UPDATE_KEPT_SOURCE_UNCHANGED',rounds=rounds,updates=updates,retained_research_rounds=kept,trainable_parameters=sum(v.numel() for v in LEARNING.params(q)),source_Adam_not_inherited=True,whole_stage_Actor_Adam_rollback=not accepted,strict_parent_success_ID_retention=True,shared_mean_density_thresholds_unchanged=True)


def gate(rt,p):
    choice=selected_arm(p);assert choice is not None
    name,prior=choice;assert prior['development_all_safe'] and read(OUT/name/'audit.json')['state']=='PASS'
    model=prior['model'];development=read(prior['development_file']);assert development['all_safe']
    fresh_ids={j['id'] for j in p['fresh_jobs']};jobs=[j for j in p['reload_jobs'] if j['id'] not in fresh_ids];assert len(jobs)==62
    original={r['job_id']:r for r in development['rows']};rebuilt=rt.student(jobs,model,'REBUILD62',True);refs={}
    for a,row in rebuilt:
        # Model bytes include different Adam headers, but physics depends on
        # exactly the same learned policy. Explicitly compare policy+arrays.
        previous=original[row['job_id']]
        for k in ['job_id','array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','remaining_m','policy_hash']:assert row[k]==previous[k],(row['job_id'],k)
        assert row['success'] and not row['collision'];refs[row['job_id']]=(a,row)
    for path,h in p['fresh_ledger'].items():assert sha(path)==h
    # Re-check all previous stages: fresh IDs must still be unconsumed.
    for file in OUT.rglob('cases.jsonl'):
        for line in file.read_text().splitlines():assert json.loads(line)['job_id'] not in fresh_ids
    fresh=rt.student(p['fresh_jobs'],model,'FRESH16',True)
    result=dict(model=model,selected_arm=name,six_gate=False,fresh16=E.tally([r for a,r in fresh]),reload78=None,development_file=prior['development_file'])
    if all(r['success'] and not r['collision'] for a,r in fresh):
        refs.update({r['job_id']:(a,r) for a,r in fresh});rt.student(p['reload_jobs'],model,'RELOAD78',refs=[refs[j['id']] for j in p['reload_jobs']]);result.update(six_gate=True,reload78=dict(exact=78,all_arrays_exact=True,independent_sample=False))
    return result

def eight(rt,p):
    assert read(OUT/'gate/summary.json')['result']['six_gate'] and read(OUT/'gate/audit.json')['state']=='PASS'
    model=selected_model(p);geometry=read(p['eight_geometry_path']);valid={g['id'] for g in geometry if g['valid']}
    values=rt.student([j for j in p['eight_jobs'] if j['id'] in valid],model,'EIGHT_DIAGNOSTIC')
    return dict(model=model,diagnostic=E.tally([r for a,r in values]),eight_started=True,eight_training=False,not_eight_stable_acceptance=True,seven_skipped=True)

def main(name):
    assert name in ALIASES;E.N.offline();p=read(OUT/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers();rt=None;state='FAILED';result={};error=None
    try:
        rt=Runtime(name,p);rt.start_pool()
        if name in ['gate','eight']:ban_learning()
        result=learn(rt,p,name) if name in ['fresh_rl','iterative_recovery','representation_rl'] else globals()[name](rt,p);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if rt:save(rt.folder/'failure.json',dict(error=error,role=rt.role,physics=sum(rt.physics),commands=rt.commands,no_restart=True))
    finally:
        if rt:rt.close(state,result,error)
    assert state=='COMPLETE';print('T434_STAGE_COMPLETE',name,flush=True)
if __name__=='__main__':main(sys.argv[1])
