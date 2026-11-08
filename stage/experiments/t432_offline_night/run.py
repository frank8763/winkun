"""Frozen-representation66-parameter repair, one candidate and strict joint gate."""
from t432_core import *
class Runtime:
    def __init__(self,name,p):
        self.name=name;self.p=p;self.folder=OUT/ALIASES[name];self.folder.mkdir();self.limit=p['stage_budgets'][name];self.start=time.time();self.deadline=min(self.start+self.limit['main_seconds'],p['deadline_native']);self.learnstop=min(self.deadline-p['validation_reserve_seconds'],p['deadline_learning']);self.physics=mp.get_context('spawn').RawArray('q',5);self.commands=0;self.rows=[];self.replays=[];self.last=0.;self.role='SETUP';self.pool=None
        self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(self.folder/'main_started.json',dict(pid=os.getpid(),epoch=self.start,deadline=self.deadline,stage=name));E.set_out(self.folder)
    def start_pool(self):
        cfg=dict(self.p,out=str(self.folder),native_stop_epoch=self.deadline,max_physics=self.limit['max_physics']);self.pool=Pool(cfg,self.physics,self.guard);self.commands=4
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

def data(rt,p):
    model=initial_record(p);assert model['policy_hash']==p['expected_policy_hash']
    proofs={r['job_id']:r for r in p['data_proofs']}
    assert len(proofs)==3 and {j['id'] for j in p['data_jobs']}==set(proofs)
    values=rt.student(p['data_jobs'],model,'ADDITIONAL_OLD3',True)
    added=[]
    for a,row in values:
        previous=proofs[row['job_id']];exact_historical_row(row,previous)
        assert row['success'] and not row['collision'] and not a['ownership'].any()
        added.append(dict(path=row['path'],sha256=row['sha256'],row=row))
    rt.student(p['data_jobs'],model,'ADDITIONAL_OLD3_EXACT',refs=values)
    assert len(added)==3
    anchors=[]
    for row in p['current_anchor_rows']:
        a=authenticated(row);assert row['policy_hash']==model['policy_hash'] and row['success'] and not row['collision'] and not a['ownership'].any()
        anchors.append(dict(path=row['path'],sha256=row['sha256'],row=row))
    assert len(anchors)==59 and len({e['row']['job_id'] for e in anchors+added})==62
    assert sha(p['verified_target_file'])==p['inputs'][p['verified_target_file']]
    assert read(p['verified_target_audit'])['state']=='PASS'
    targets=read(p['verified_target_file']);assert len(targets)==7
    for target in targets:
        a=authenticated(target['row']);first=target['first']
        assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        assert not a['ownership'][:first].any() and a['ownership'][first:].all()
        assert target['row']['policy_hash']==model['policy_hash']
    save(rt.folder/'anchors_b.json',anchors);save(rt.folder/'anchors_c.json',anchors+added);save(rt.folder/'targets.json',targets)
    return dict(source_model=model,anchors_b=anchors,anchors_c=anchors+added,additional_old3=added,targets=targets,learning_data_available=True,verified_recovery_targets=7,additional_old3_original_arrays_exact=True,additional_old3_native_exact=True,existing59_and7_reused_not_new_samples=True)

def learn(rt,p,name):
    assert name in ['b','c']
    dataset=read(OUT/'data/summary.json')['result'];assert read(OUT/'data/audit.json')['state']=='PASS' and dataset['learning_data_available']
    q=frozen(p);opt=LEARNING.make_optimizer(q,p);assert not opt.state
    initial=LEARNING.export_model(q,opt,rt.folder/'initial.pt.gz');before=E.L.state(q,opt);fixed=LEARNING.fixedhash(q)
    entries=dataset['anchors_'+name]
    anchors=[(E.N.S.arrays(x['path']),x['row']) for x in entries]
    targets=[dict(arrays=E.N.S.arrays(x['path']),first=x['first'],id=x['id']) for x in dataset['targets']]
    jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    group_p=dict(p,arm=name,phase_arm=name,anchor_jobs=[jobs[x['row']['job_id']] for x in entries],recovery_target_ids=[t['id'] for t in targets])
    g=LEARNING.groups(q,group_p,anchors,targets);updates=dict(live=0,shadow=0)
    def event(which,model,optimizer):
        def emit(where,k):
            if where=='AFTER':updates[which]+=1
            with (rt.folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(tag='fit120_'+name,which=which,where=where,k=k,hashes=E.L.statehash(model,optimizer),gradient_sha256=gradient_sha(model)))+'\n')
        return emit
    shadow=frozen(p);so=LEARNING.make_optimizer(shadow,p);E.L.load(shadow,so,before);rt.role='NONLINEAR4450_'+name
    fit=LEARNING.fit(q,opt,g,group_p,rt.learning_guard,event('live',q,opt));sf=LEARNING.fit(shadow,so,g,group_p,rt.learning_guard,event('shadow',shadow,so))
    assert fit==sf and E.L.statehash(q,opt)==E.L.statehash(shadow,so) and LEARNING.fixedhash(q)==fixed
    proposed=LEARNING.export_model(q,opt,rt.folder/'proposed.pt.gz');check=rt.development(proposed,'PROPOSAL120_'+name)
    path=rt.folder/'development.json';save(path,check);ids=set(check['success_ids']);initial_ids=set(p['initial_success_ids'])
    accepted=initial_ids<=ids and bool(ids-initial_ids)
    if not accepted:E.L.load(q,opt,before)
    model=LEARNING.export_model(q,opt,rt.folder/'candidate.pt.gz');assert LEARNING.fixedhash(q)==fixed
    return dict(arm=name,initial_model=initial,before_model=initial,proposed_model=proposed,fit=fit,check_file=str(path),accepted=accepted,model=model,selected_success_ids=sorted(ids if accepted else initial_ids),development_all_safe=bool(accepted and check['all_safe']),development_file=str(path) if accepted else None,updates=updates,all_except_existing_extra_frozen=True,trainable_parameters=4450,full62_references_saved_during_development=False,compact_development_only=True,fresh_Adam_initialization=True,recovery_target_ids=[t['id'] for t in targets],current_anchor_count=len(entries),initial_source_policy_hash=p['expected_policy_hash'],same_start_for_b_c=True,old_T422_audit_NOT_repaired=True)

def gate(rt,p):
    choice=selected_arm(p);assert choice is not None
    name,prior=choice;assert prior['development_all_safe']
    assert read(OUT/name/'audit.json')['state']=='PASS'
    model=prior['model'];development=read(prior['development_file'])
    assert development['all_safe'] and len(development['success_ids'])==329
    fresh_ids={j['id'] for j in p['fresh_jobs']};ref_jobs=[j for j in p['reload_jobs'] if j['id'] not in fresh_ids]
    assert len(ref_jobs)==62
    originals={r['job_id']:r for r in development['rows']};rebuilt=rt.student(ref_jobs,model,'REBUILD62',True);refs={}
    for a,row in rebuilt:
        exact_historical_row(row,originals[row['job_id']]);assert row['success'] and not row['collision']
        refs[row['job_id']]=(a,row)
    for path,h in p['fresh_ledger'].items():assert sha(path)==h
    fresh=rt.student(p['fresh_jobs'],model,'FRESH16',True)
    result=dict(model=model,selected_arm=name,six_gate=False,fresh16=E.tally([r for a,r in fresh]),reload78=None,development_file=prior['development_file'],reference62_paths=[r['path'] for a,r in rebuilt],reference62_saved_in_original329=False,reference62_rebuilt_native_exact=True)
    if all(r['success'] and not r['collision'] for a,r in fresh):
        refs.update({r['job_id']:(a,r) for a,r in fresh})
        rt.student(p['reload_jobs'],model,'RELOAD78',refs=[refs[j['id']] for j in p['reload_jobs']])
        result.update(six_gate=True,reload78=dict(exact=78,all_arrays_exact=True,independent_sample=False))
    return result

def eight(rt,p):
    assert read(OUT/'gate/summary.json')['result']['six_gate'] and read(OUT/'gate/audit.json')['state']=='PASS'
    model=selected_model(p);geometry=read(p['eight_geometry_path']);valid={g['id'] for g in geometry if g['valid']}
    values=rt.student([j for j in p['eight_jobs'] if j['id'] in valid],model,'EIGHT_DIAGNOSTIC')
    return dict(model=model,diagnostic=E.tally([r for a,r in values]),invalid_geometry=[g for g in geometry if not g['valid']],eight_started=True,eight_training=False,seven_skipped=True,six_gate_previously_passed=True,not_eight_stable_acceptance=True)

def main(name):
    assert name in ALIASES;E.N.offline();p=read(OUT/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers();rt=None;state='FAILED';result={};error=None
    try:
        rt=Runtime(name,p);rt.start_pool()
        if name not in ['b','c']:ban_learning()
        result=learn(rt,p,name) if name in ['b','c'] else globals()[name](rt,p)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if rt:save(rt.folder/'failure.json',dict(error=error,role=rt.role,physics=sum(rt.physics),commands=rt.commands,no_restart=True))
    finally:
        if rt:rt.close(state,result,error)
    assert state=='COMPLETE';print('T432_STAGE_COMPLETE',name,flush=True)
if __name__=='__main__':main(sys.argv[1])
