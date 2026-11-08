"""Once-only pre-registered data, BC, PPO, heldout and eight-metre stages."""
from t425_core import *
class Runtime:
    def __init__(self,name,p):
        self.name=name;self.p=p;self.folder=OUT/ALIASES[name];self.folder.mkdir();self.limit=p['stage_budgets'][name];self.start=time.time();self.deadline=min(self.start+self.limit['main_seconds'],p['deadline_native']);self.learnstop=min(self.deadline-p['validation_reserve_seconds'],p['deadline_learning']);self.physics=mp.get_context('spawn').RawArray('q',5);self.commands=0;self.rows=[];self.replays=[];self.last=0.;self.role='SETUP';self.pool=None
        self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(self.folder/'main_started.json',dict(pid=os.getpid(),epoch=self.start,deadline=self.deadline,stage=name));E.set_out(self.folder)
        cfg=dict(p,out=str(self.folder),native_stop_epoch=self.deadline,max_physics=self.limit['max_physics']);self.pool=Pool(cfg,self.physics,self.guard);self.commands=4
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
            req=requests[offset:offset+4];self.guard();assert sum(self.physics)+sum(x['job']['case']['step_cap'] for x in req)<=self.limit['max_physics'];ids=list(range(self.commands+1,self.commands+len(req)+1));self.commands+=len(req)
            for i,((a,row),request,cid) in enumerate(zip(self.pool.request(req),req,ids)):
                row=dict(row,role=role,job_id=request['job']['id'],physics_steps=int(a['native_counts'].sum()),loaded_model_sha256=sha(request['model']),seed=request.get('seed'),request=copy.deepcopy(request))
                assert row['physics_steps']==row['physics']
                if refs is not None:
                    old,previous=refs[offset+i]
                    for key in a:assert E.equal(a[key],old[key]),(role,key)
                    assert row['array_digest']==previous['array_digest'];row.update(exact=True,all_arrays_exact=True,independent_sample=False,reference_path=previous['path']);self.replays.append(row);name='replays.jsonl'
                else:
                    path=self.folder/'episodes'/f'{len(self.rows):04}.npz';path.parent.mkdir(exist_ok=True);E.N.write_npz(path,E.N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},self.p['full_bytes'] if full else 5*1024);row.update(path=str(path),sha256=sha(path),full=full);self.rows.append(row);name='cases.jsonl'
                with (self.folder/name).open('a') as f:f.write(json.dumps(row)+'\n')
                self.log(cid,role,row['physics'],row['pid']);output.append((a,row))
        return output
    def student(self,jobs,model,role,full=False,seeds=None,refs=None):
        return self.request([dict(model=model['path'],job=j,mode='student',seed=None if seeds is None else seeds[i]) for i,j in enumerate(jobs)],role,full,refs)
    def development(self,model,role):
        values=self.student([*self.p['main_jobs'],*self.p['old275_jobs']],model,role);rows=[row for a,row in values];ids=sorted(r['job_id'] for r in rows if r['success'] and not r['collision']);return dict(success_ids=ids,all_safe=len(ids)==329,wide54=E.tally(rows[:54]),old275=E.tally(rows[54:]),policy_hash=model['policy_hash'],rows=rows)
    def close(self,state,result,error=None):
        if self.pool:self.pool.close()
        save(self.folder/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-self.start,physics=sum(self.physics),commands=self.commands,result=result,error=error,all_workers_stopped=True,all_sessions_closed=True));save(self.folder/'status.json',dict(state=state,epoch=time.time(),physics=sum(self.physics),commands=self.commands));E.N.manifest(self.folder);fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
def teacher_requests(p,model,targets,first,gains_by_id=None):
    result=[]
    for target in targets:
        req=dict(model=model['path'],job=target['job'],mode=target['kind'],first=0 if target['kind']=='safe_reset' else first,seed=None)
        if target['kind']=='feedback':req.update(reference_path=target['reference_path'],reference_sha256=target['reference_sha256'],gains=(gains_by_id or {}).get(target['job']['id'],p['feedback_gains'][0]))
        result.append(req)
    return result
def data(rt,p):
    model=selected_model(p);targets=[];configs={}
    # Old teacher is queried at the actual state from t0, not labels copied
    # from compact historical files. Wide teacher uses a nearby reference.
    for target in p['targets']:
        candidates=[];gains=p['feedback_gains'] if target['kind']=='feedback' else [None]
        for k,gain in enumerate(gains):
            requests=teacher_requests(p,model,[target],p['teacher_first']);
            if gain is not None:requests[0]['gains']=gain
            pair=rt.request(requests,'QUERY_'+target['job']['id']+'_'+str(k))[0];a,row=pair
            if row['success'] and not row['collision']:candidates.append((a,row,requests[0]))
        if not candidates:continue
        a,row,req=min(candidates,key=lambda item:(item[1]['physics'],item[1]['remaining_m']))
        # The selected successful arrays alone become a full training label.
        path=rt.folder/'targets'/f'{len(targets):02}.npz';path.parent.mkdir(exist_ok=True);E.N.write_npz(path,E.N.S.packed(a),p['full_bytes']);chosen=dict(row,path=str(path),sha256=sha(path),full=True)
        rt.request([req],'EXACT_'+target['job']['id'],refs=[(a,chosen)])
        entry=dict(id=target['job']['id'],path=str(path),sha256=sha(path),first=req['first'],row=chosen,request=req,exact=True);targets.append(entry)
        if target['kind']=='feedback':configs[target['job']['id']]=req['gains']
    save(rt.folder/'targets.json',targets)
    return dict(model=model,learning_data_available=bool(targets),targets=targets,feedback_configs=configs,complete_target_coverage=len(targets)==19,unique_safe_recovery_targets=len(targets),development_all_safe=False)
def arm(rt,p,kind):
    d=read(OUT/'data/summary.json')['result'];source=selected_model(p);q=frozen(p,source['path']);opt=LEARNING.make_optimizer(q,p);initial_ids=set(p['initial_success_ids']);selected=set(initial_ids)
    if kind=='b' and (OUT/'a/summary.json').exists():selected=set(read(OUT/'a/summary.json')['result']['selected_success_ids'])
    source_model=LEARNING.export_model(q,opt,rt.folder/'initial.pt.gz');target_entries=copy.deepcopy(d['targets']);rounds=[];updates=dict(live=0,shadow=0);consecutive=0;development_file=None;all_safe=False
    if kind=='b' and (OUT/'a/summary.json').exists():
        prev=read(OUT/'a/summary.json')['result'];all_safe=prev['development_all_safe'];development_file=prev.get('development_file')
    def event(tag,which,model,optimizer):
        def emit(where,k):
            if where=='AFTER':updates[which]+=1
            with (rt.folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(tag=tag,which=which,where=where,k=k,hashes=E.L.statehash(model,optimizer),gradient_sha256=gradient_sha(model)))+'\n')
        return emit
    for rnd in range(p['maximum_rounds'][kind]):
        if (time.time()+p['full_cycle_reserve_seconds'][kind]>=rt.deadline or time.time()+p['update_reserve_seconds'][kind]>=rt.learnstop):break
        block=rt.folder/'blocks'/str(rnd);block.mkdir(parents=True);before=E.L.state(q,opt);before_model=LEARNING.export_model(q,opt,block/'before.pt.gz');refresh=[]
        # Refresh the four difficult LEFT states after a policy change. Old
        # t0 safe-teacher labels have the same true plant and state stream.
        if rnd or kind=='b':
            for target in p['targets']:
                identity=target['job']['id']
                if target['kind']!='feedback' or identity not in d['feedback_configs']:continue
                req=teacher_requests(p,before_model,[target],p['teacher_first'],d['feedback_configs']);a,row=rt.request(req,'REFRESH_'+str(rnd)+'_'+identity,True)[0]
                if row['success'] and not row['collision']:
                    rt.request(req,'REFRESH_EXACT_'+str(rnd)+'_'+identity,refs=[(a,row)])
                    entry=dict(id=identity,path=row['path'],sha256=row['sha256'],first=req[0]['first'],row=row,request=req[0],exact=True);target_entries=[x for x in target_entries if x['id']!=identity]+[entry];refresh.append(entry)
        current=[dict(arrays=E.N.S.arrays(x['path']),first=x['first'],id=x['id']) for x in target_entries];g=LEARNING.make_groups(q,p,anchors(p),current);items=None;training=[]
        if kind=='b':
            seeds=[[p['seed'],rnd,k] for k in range(4)];values=rt.student(p['wide_failed_jobs'],before_model,'TRAIN_'+str(rnd),True,seeds);rt.student(p['wide_failed_jobs'],before_model,'TRAIN_EXACT_'+str(rnd),seeds=seeds,refs=values);items=[a for a,row in values];training=[row for a,row in values]
        shadow=frozen(p,before_model['path']);so=LEARNING.make_optimizer(shadow,p);E.L.load(shadow,so,before)
        tag=kind+str(rnd)
        if kind=='a':
            fit=LEARNING.fit_a(q,opt,g,p,rt.learning_guard,event(tag,'live',q,opt));sf=LEARNING.fit_a(shadow,so,g,p,rt.learning_guard,event(tag,'shadow',shadow,so))
        else:
            fit=LEARNING.fit_b(q,opt,items,g,p,rnd,rt.learning_guard,event(tag,'live',q,opt));sf=LEARNING.fit_b(shadow,so,items,g,p,rnd,rt.learning_guard,event(tag,'shadow',shadow,so))
        assert fit==sf and E.L.statehash(q,opt)==E.L.statehash(shadow,so)
        proposed=LEARNING.export_model(q,opt,block/'proposed.pt.gz');check=rt.development(proposed,'PROPOSAL_'+str(rnd));success=set(check['success_ids']);accepted=selected<=success and initial_ids<=success and bool(success-selected) and (kind=='a' or fit['numeric_accept'])
        record=dict(round=rnd,kind=kind,before_model=before_model,targets=copy.deepcopy(target_entries),refresh=refresh,training=training,fit=fit,proposed_model=proposed,check=check,accepted=accepted,selected_before=sorted(selected))
        if accepted:
            selected=success;all_safe=check['all_safe'];development_file=str(block/'development.json');save(development_file,check)
        else:E.L.load(q,opt,before)
        consecutive=0 if accepted else consecutive+1;record.update(selected_after=sorted(selected),retained_hashes=E.L.statehash(q,opt),consecutive_rejections=consecutive);rounds.append(record);save(block/'result.json',record);save(rt.folder/'rounds.json',rounds)
        if consecutive>=2 or all_safe:break
    result=dict(model=LEARNING.export_model(q,opt,rt.folder/'candidate.pt.gz'),selected_success_ids=sorted(selected),development_all_safe=all_safe,development_file=development_file,initial_model=source_model,rounds=rounds,updates=updates,old_training_audit_not_repaired=True)
    return result
def gate(rt,p):
    model=selected_model(p);prior=read(OUT/('b' if (OUT/'b/summary.json').exists() else 'a')/'summary.json')['result'];assert prior['development_all_safe'];development=read(prior['development_file']);assert len(development['success_ids'])==329 and development['all_safe']
    lookup={r['job_id']:r for r in development['rows']};jobs=[j for j in [*p['main_jobs'],*p['old275_jobs']] if j['id'] in {j['id'] for j in p['reload_jobs']}];assert len(jobs)==62
    values=rt.student(jobs,model,'REBUILD62',True)
    for a,row in values:
        old=lookup[row['job_id']]
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==old[key],key
    for f,h in p['fresh_ledger'].items():assert sha(f)==h
    fresh=rt.student(p['fresh_jobs'],model,'FRESH16',True);result=dict(model=model,six_gate=False,fresh16=E.tally([r for a,r in fresh]),reload78=None,development_file=prior['development_file'])
    if all(r['success'] and not r['collision'] for a,r in fresh):
        refs={r['job_id']:(a,r) for a,r in [*values,*fresh]};rt.student(p['reload_jobs'],model,'RELOAD78',refs=[refs[j['id']] for j in p['reload_jobs']]);result.update(six_gate=True,reload78=dict(exact=78,all_arrays_exact=True,independent_sample=False))
    return result
def eight(rt,p):
    assert read(OUT/'gate/summary.json')['result']['six_gate'];model=selected_model(p);geometry=read(OUT/'eight_geometry.json');valid={g['id'] for g in geometry if g['valid']};jobs=[j for j in p['eight_jobs'] if j['id'] in valid];values=rt.student(jobs,model,'EIGHT_DIAGNOSTIC',False)
    return dict(model=model,diagnostic=E.tally([r for a,r in values]),invalid_geometry=[g for g in geometry if not g['valid']],eight_started=True,eight_training=False,seven_skipped=True,six_gate_previously_passed=True,not_eight_stable_acceptance=True)
def main(name):
    assert name in ALIASES;E.N.offline();p=read(OUT/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers();rt=None;state='FAILED';result={};error=None
    try:
        rt=Runtime(name,p)
        result=data(rt,p) if name=='data' else arm(rt,p,'a' if name=='arm_a' else 'b') if name in ['arm_a','arm_b'] else gate(rt,p) if name=='gate' else eight(rt,p);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if rt:save(rt.folder/'failure.json',dict(error=error,role=rt.role,physics=sum(rt.physics),commands=rt.commands,no_restart=True))
    finally:
        if rt:rt.close(state,result,error)
    assert state=='COMPLETE';print('T425_STAGE_COMPLETE',name,flush=True)
if __name__=='__main__':main(sys.argv[1])
