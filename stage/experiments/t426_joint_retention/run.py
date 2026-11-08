"""One fixed supervised research trajectory; full closed-loop acceptance."""
from t426_core import *
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

def data(rt,p):
    model=initial_record(p);proofs={x['job_id']:x for x in p['anchor_proofs']};anchors=[];targets=[]
    for a,row in rt.student(p['anchor_jobs'],model,'ANCHOR61',True):
        old=proofs[row['job_id']]
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==old[key],(row['job_id'],key)
        assert row['success'] and not row['collision'];anchors.append(dict(path=row['path'],sha256=row['sha256'],row=row))
    assert len(anchors)==61
    for job in p['recovery_jobs']:
        for first in p['allowed_teacher_first']:
            request=dict(model=model['path'],job=job,mode='safe_reset',first=first,seed=None)
            a,row=rt.request([request],f'QUERY_{job["id"]}_{first}')[0]
            if not row['success'] or row['collision']:continue
            path=rt.folder/'targets'/f'{len(targets):02}.npz';path.parent.mkdir(exist_ok=True);E.N.write_npz(path,E.N.S.packed(a),p['full_bytes']);chosen=dict(row,path=str(path),sha256=sha(path),full=True)
            rt.request([request],'EXACT_'+job['id'],refs=[(a,chosen)])
            targets.append(dict(id=job['id'],path=str(path),sha256=sha(path),first=first,row=chosen,request=request,exact=True));break
    save(rt.folder/'anchors.json',anchors);save(rt.folder/'targets.json',targets)
    return dict(source_model=model,anchors=anchors,targets=targets,teacher_targets_complete=len(targets)==11,learning_data_available=len(targets)==11,current_A_anchors_rebuilt_exact=True,historical_anchor_executions_reconstructed=61,independent_new_anchor_samples=0)

def learn(rt,p):
    data=read(OUT/'data/summary.json')['result'];assert data['teacher_targets_complete'];q=frozen(p);opt=LEARNING.make_optimizer(q,p)
    initial=LEARNING.export_model(q,opt,rt.folder/'initial.pt.gz');selected=initial;success=set(p['initial_success_ids']);records=[];updates=dict(live=0,shadow=0);bestfile=None;all_safe=False
    anchors=[(E.N.S.arrays(x['path']),x['row']) for x in data['anchors']];targets=[dict(arrays=E.N.S.arrays(x['path']),first=x['first'],id=x['id']) for x in data['targets']];g=LEARNING.groups(q,p,anchors,targets)
    probe=frozen(p);variables=E.params(probe);weights=digest(probe.state_dict());started=time.monotonic();losses=LEARNING.support_losses(probe,g,p,0);reference=E.vector_gradient(losses[0]+losses[1],variables);broad=E.protect_vector(E.vector_gradient(losses[2],variables),reference);correction=E.protect_vector(E.vector_gradient(losses[3],variables),reference);E.install_vector(variables,reference+broad+correction);torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True);elapsed=time.monotonic()-started;assert weights==digest(probe.state_dict())
    performance=dict(elapsed_s=elapsed,before_weights=weights,after_weights=digest(probe.state_dict()),gradient_sha256=gradient_sha(probe),projected_240_updates_seconds=elapsed*240,audit_estimate_s=elapsed*240+120,main_estimate_s=elapsed*240+2150,new_Adam=0,native=0)
    performance['admitted']=performance['audit_estimate_s']<=p['stage_budgets']['learn']['audit_seconds'] and performance['main_estimate_s']<=p['stage_budgets']['learn']['main_seconds'];save(rt.folder/'performance.json',performance);del probe
    if not performance['admitted']:
        return dict(initial_model=initial,model=LEARNING.export_model(q,opt,rt.folder/'candidate.pt.gz'),records=[],selected_success_ids=sorted(success),development_all_safe=False,development_file=None,updates=updates,performance=performance,continuous_research_trajectory_not_checkpoint_rollback=True,final_exact_Actor_Adam_selected_snapshot=True,old_T422_audit_NOT_repaired=True)
    def event(tag,which,model,optimizer):
        def emit(where,k):
            if where=='AFTER':updates[which]+=1
            with (rt.folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(tag=tag,which=which,where=where,k=k,hashes=E.L.statehash(model,optimizer),gradient_sha256=gradient_sha(model)))+'\n')
        return emit
    for block_index,start in enumerate([0,60]):
        if time.time()+p['update_window_seconds']>=rt.learnstop or time.time()+p['cycle_window_seconds']>=rt.deadline:break
        folder=rt.folder/'blocks'/str(block_index);folder.mkdir(parents=True);before=E.L.state(q,opt);before_model=LEARNING.export_model(q,opt,folder/'before.pt.gz');shadow=frozen(p,before_model['path']);so=LEARNING.make_optimizer(shadow,p);E.L.load(shadow,so,before)
        rt.role='LEARN_'+str(start+60);tag='block'+str(block_index);fit=LEARNING.stepfit(q,opt,g,p,start,60,rt.learning_guard,event(tag,'live',q,opt));sf=LEARNING.stepfit(shadow,so,g,p,start,60,rt.learning_guard,event(tag,'shadow',shadow,so));assert fit==sf and E.L.statehash(q,opt)==E.L.statehash(shadow,so)
        proposed=LEARNING.export_model(q,opt,folder/'proposed.pt.gz');check=rt.development(proposed,'CHECK_'+str(start+60));checkfile=folder/'development.json';save(checkfile,check);ids=set(check['success_ids']);eligible=set(p['initial_success_ids'])<=ids and bool(ids-set(p['initial_success_ids']))
        record=dict(start=start,count=60,before_model=before_model,fit=fit,proposed_model=proposed,development_file=str(checkfile),development={k:v for k,v in check.items() if k!='rows'},accepted=eligible,selected_for_retention=eligible and len(ids)>len(success))
        records.append(record);save(folder/'result.json',record);save(rt.folder/'records.json',records)
        if eligible and len(ids)>len(success):selected=proposed;success=ids;bestfile=str(checkfile);all_safe=check['all_safe']
        if all_safe:break
        # This is one pre-registered120-step research trajectory. A failed60
        # checkpoint is never deployed, but the fixed trajectory may continue.
    E.L.load(q,opt,model_decode(selected['path'])['state']);assert E.L.statehash(q,opt)==selected['hashes'];final=LEARNING.export_model(q,opt,rt.folder/'candidate.pt.gz')
    return dict(initial_model=initial,model=final,records=records,selected_success_ids=sorted(success),development_all_safe=all_safe,development_file=bestfile,updates=updates,performance=performance,continuous_research_trajectory_not_checkpoint_rollback=True,final_exact_Actor_Adam_selected_snapshot=True,old_T422_audit_NOT_repaired=True)

def gate(rt,p):
    prior=read(OUT/'learn/summary.json')['result'];assert prior['development_all_safe'];model=selected_model(p);development=read(prior['development_file']);assert development['all_safe'] and len(development['success_ids'])==329
    lookup={r['job_id']:r for r in development['rows']};ids={j['id'] for j in p['reload_jobs']};jobs=[j for j in [*p['main_jobs'],*p['old275_jobs']] if j['id'] in ids];assert len(jobs)==62
    values=rt.student(jobs,model,'REBUILD62',True)
    for a,row in values:
        previous=lookup[row['job_id']]
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==previous[key],key
    for f,h in p['fresh_ledger'].items():assert sha(f)==h
    fresh=rt.student(p['fresh_jobs'],model,'FRESH16',True);result=dict(model=model,six_gate=False,fresh16=E.tally([r for a,r in fresh]),reload78=None,development_file=prior['development_file'])
    if all(r['success'] and not r['collision'] for a,r in fresh):
        refs={r['job_id']:(a,r) for a,r in [*values,*fresh]};rt.student(p['reload_jobs'],model,'RELOAD78',refs=[refs[j['id']] for j in p['reload_jobs']]);result.update(six_gate=True,reload78=dict(exact=78,all_arrays_exact=True,independent_sample=False))
    return result

def eight(rt,p):
    assert read(OUT/'gate/summary.json')['result']['six_gate'];model=selected_model(p);geometry=read(OUT/'eight_geometry.json');valid={g['id'] for g in geometry if g['valid']};jobs=[j for j in p['eight_jobs'] if j['id'] in valid];values=rt.student(jobs,model,'EIGHT_DIAGNOSTIC')
    return dict(model=model,diagnostic=E.tally([r for a,r in values]),invalid_geometry=[g for g in geometry if not g['valid']],eight_started=True,eight_training=False,seven_skipped=True,six_gate_previously_passed=True,not_eight_stable_acceptance=True)

def main(name):
    assert name in ALIASES;E.N.offline();p=read(OUT/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers();rt=None;state='FAILED';result={};error=None
    try:
        rt=Runtime(name,p);result=globals()[name](rt,p);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if rt:save(rt.folder/'failure.json',dict(error=error,role=rt.role,physics=sum(rt.physics),commands=rt.commands,no_restart=True))
    finally:
        if rt:rt.close(state,result,error)
    assert state=='COMPLETE';print('T426_STAGE_COMPLETE',name,flush=True)
if __name__=='__main__':main(sys.argv[1])
