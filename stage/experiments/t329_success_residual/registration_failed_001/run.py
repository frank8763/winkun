from learning import *
from native_worker import worker
import multiprocessing as mp,signal

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=model(p);opt=optimizer(q,p,'bc');h=K.S.digest(q.state_dict());oh=K.S.digest(opt.state_dict());data=bc_data(p,q)
    with torch.no_grad():
        pred=mode(q,data['x']);ref=mode(q.frozen,data['x']);assert torch.equal(pred,ref)
    losses=bc_backward(q,data,p);assert q.correction[-1].weight.grad.norm()>0
    q.zero_grad(set_to_none=True);assert K.S.digest(q.state_dict())==h and K.S.digest(opt.state_dict())==oh
    # Four discarded synthetic Adam calls and exact all-state rollback.
    torch.manual_seed(p['seed']);a=torch.nn.Linear(3,2);b=copy.deepcopy(a);oa=torch.optim.Adam(a.parameters());ob=torch.optim.Adam(b.parameters())
    initial=copy.deepcopy(a.state_dict());empty=copy.deepcopy(oa.state_dict())
    for _ in range(2):
        for m,o in [(a,oa),(b,ob)]:o.zero_grad();m(torch.ones(2,3)).square().mean().backward();o.step()
    assert K.S.digest(a.state_dict())==K.S.digest(b.state_dict()) and K.S.digest(oa.state_dict())==K.S.digest(ob.state_dict())
    a.load_state_dict(initial);oa.load_state_dict(empty);assert K.S.digest(a.state_dict())==K.S.digest(initial) and K.S.digest(oa.state_dict())==K.S.digest(empty)
    # Actual new worker entry + strict composite state load, zero native setup.
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe();proc=ctx.Process(target=worker,args=(child,0,p,None,True));proc.start();child.close()
    parent.send(dict(state=K.S.state_np(q),x=data['x'][:1].numpy()));assert parent.poll(30);answer=parent.recv();proc.join(10);assert not proc.is_alive() and proc.exitcode==0
    with torch.no_grad():m,mu,_,_=q.step(data['x'][:1])
    assert answer['policy_hash']==h and np.array_equal(m.numpy(),answer['mode']) and np.array_equal(mu.numpy(),answer['mean'])
    save(OUT/'preflight.json',dict(state='PASS',native_commands=0,physics_steps=0,Actor_navigation_updates=0,
        actual_composite_worker_load_and_actions_exact=True,worker_stopped=True,real_full_BC_backward_calls=1,
        synthetic_discarded_Adam_calls=4,zero_initialization_exact=True,frozen_base_has_no_gradient=True,
        Adam_policy_full_rollback_exact=True,objective=losses,protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print('T329 preflight PASS0native/0ActorAdam;4syntheticAdam')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');verify(p);start=time.time();deadline=start+p['wall_seconds'];train_end=start+p['training_seconds']
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];meter=None;q=None;opt=None;raw=[];replays=[];phase='STARTING';state_end='FAILED';blocks={};patches={};calls=[];active=[];last=0
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,training_deadline_epoch=train_end,
            commands=0 if meter is None else meter.commands,physics_steps=sum(physical[:]),cases=len(raw),replays=len(replays),optimizer_calls=len(calls),inflight=active,blocks=blocks))
    def guard():assert time.time()<train_end;meter.hold();status()
    def event(replica,e):
        row=dict(e);row['replica']=replica;calls.append(row)
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-15
        answer=pipe.recv();assert 'error' not in answer,answer;return answer
    def save_patch(name,actor,optimizer_):
        blob=patch(actor,optimizer_);meter.charge_payload(len(blob));f=OUT/(name+'.pt.gz');assert not f.exists();f.write_bytes(blob)
        patches[name]=dict(path=str(f),sha256=K.C.sha(f),policy_hash=K.S.digest(actor.state_dict()),optimizer_hash=K.S.digest(optimizer_.state_dict()));save(OUT/'patches.json',patches)
    def roll(jobs,prefix,role):
        nonlocal phase,active
        phase=prefix;result=[]
        for off in range(0,len(jobs),4):
            group=[dict(j,id=prefix+'_'+j['id'],source_reset_id=j['source_reset_id'] if j.get('source_reset_id') else j['id'],role=role,
                policy_np=K.S.state_np(q),policy_hash=K.S.digest(q.state_dict()),deadline_epoch=(train_end if role=='training' else deadline-30),aux_eta=0.,mode='iid' if role=='training' else 'deterministic') for j in jobs[off:off+4]]
            reference=None
            for replay in [False,True]:
                meter.checkpoint();assert time.time()<deadline-30 and meter.commands+len(group)<=p['max_commands']
                assert sum(physical[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                meter.commands+=len(group);active=[j['id'] for j in group];status(True)
                for pipe,j in zip(pipes,group):pipe.send(j)
                batch=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(physical[:]);active=[]
                for k,item in enumerate(batch):
                    row,a=item['row'],item['arrays'];row['job']={k:v for k,v in group[k].items() if k!='policy_np'}
                    if replay:
                        ref=reference[k];assert row['array_hashes']==ref['row']['array_hashes'] and row['integration_sha256']==ref['row']['integration_sha256']
                        assert all(np.array_equal(v,ref['arrays'][n]) for n,v in a.items())
                        rep=dict(id=row['id'],physics_steps=row['physics_steps'],all_arrays_exact=True,integration_sha256=row['integration_sha256'],not_independent_sample=True)
                        replays.append(rep);save(OUT/'replays.json',replays)
                    else:
                        data=packed(a) if role=='training' else {k:a[k] for k in ['actions','native_counts','end_flags','environment_rewards']}
                        buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue();assert len(blob)<=(p['training_episode_bytes'] if role=='training' else p['compact_bytes'])
                        meter.charge_payload(len(blob));f=OUT/'episodes'/(row['id']+'.npz');assert not f.exists();f.write_bytes(blob)
                        row.update(path=str(f),sha256=K.C.sha(f),saved_array_keys=list(data));raw.append(row)
                        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
                        result.append(item)
                    assert row['complete'] and not row.get('error'),'Partial evidence kept; no training'
                if not replay:reference=batch
                status(True)
        return result
    def safe(items):return all(i['row']['complete'] and i['row']['success'] and not i['row']['collision'] for i in items)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T329 absolute budget')));signal.alarm(p['wall_seconds']-5)
    try:
        assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);q=model(p);opt=optimizer(q,p,'bc');initial=copy.deepcopy(state(q));initial_opt=copy.deepcopy(opt.state_dict());save_patch('initial',q,opt)
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);w=ctx.Process(target=worker,args=(b,k,p,physical));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        roll([p['wide_job'],*p['original_jobs']],'baseline','baseline')
        phase='BC100';status(True);shadow=copy.deepcopy(q);so=optimizer(shadow,p,'bc')
        live=fit(p,q,opt,guard,lambda e:event('live',e));other=fit(p,shadow,so,guard,lambda e:event('shadow',e));assert live==other
        assert K.S.digest(q.state_dict())==K.S.digest(shadow.state_dict()) and K.S.digest(opt.state_dict())==K.S.digest(so.state_dict());save_patch('bc_proposal',q,opt)
        accepted=safe(roll(p['old_jobs'],'bc_guard','old_guard'));roll([p['wide_job']],'bc_course','course_candidate')
        blocks['bc']=dict(accepted=accepted,live_Adam=len(live),shadow_Adam=len(other))
        if not accepted:load(q,initial);opt.load_state_dict(initial_opt)
        save_patch('selected_after_bc',q,opt);save(OUT/'blocks.json',blocks)
        items=roll(p['training_jobs'],'onpolicy','training')
        opt=optimizer(q,p,'pg');before=copy.deepcopy(state(q));empty=copy.deepcopy(opt.state_dict());shadow=copy.deepcopy(q);so=optimizer(shadow,p,'pg')
        phase='ONE_REAL_RL_PROPOSAL';status(True);proposal=pg(p,q,opt,items,guard,lambda e:event('live',e));shadow_result=pg(p,shadow,so,items,guard,lambda e:event('shadow',e))
        assert proposal==shadow_result and K.S.digest(q.state_dict())==K.S.digest(shadow.state_dict()) and K.S.digest(opt.state_dict())==K.S.digest(so.state_dict())
        save_patch('pg_proposal',q,opt);accepted=proposal['numeric_pass']
        if accepted:accepted=safe(roll(p['old_jobs'],'pg_guard','old_guard'))
        roll([p['wide_job']],'pg_course','course_candidate')
        blocks['pg']=dict(accepted=accepted,live_Adam=1,shadow_Adam=1,numeric_pass=proposal['numeric_pass'],k3=proposal['k3'],max_abs_logratio=proposal['max_abs_logratio'])
        if not accepted:load(q,before);opt.load_state_dict(empty)
        save(OUT/'blocks.json',blocks);save_patch('final',q,opt)
        roll(p['old_jobs'],'final_old','final_old');roll(p['course_dev_jobs'],'final_course','final_course');roll(p['original_jobs'],'final_original','final_original')
        assert K.S.digest(q.frozen.state_dict())==p['frozen_policy_hash'];meter.checkpoint();state_end='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,inflight=active,physics_steps=sum(physical[:]),commands=0 if meter is None else meter.commands))
        if q is not None and opt is not None and 'partial' not in patches:
            try:save_patch('partial',q,opt)
            except BaseException:save(OUT/'partial_save_failure.json',dict(error=traceback.format_exc()))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(5)
            if w.is_alive():w.kill();w.join(3)
        signal.alarm(0)
        if meter:meter.close()
    summary=dict(state=state_end,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=deadline,commands=0 if meter is None else meter.commands,
        physics_steps=sum(physical[:]),cases=len(raw),exact_replays=len(replays),optimizer_calls=len(calls),blocks=blocks,patches=patches,
        all_workers_stopped=all(not w.is_alive() for w in workers),own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'),
        original_275_54_fresh16_reload78_executed=False,six_metre_stable=False,seven_eight_entered=False,
        counts={role:{kind:sum(r['role']==role and (r['collision'] if kind=='CONTACT' else r['timeout'] if kind=='TIMEOUT' else r['success'] and not r['collision']) for r in raw) for kind in ['GOAL','CONTACT','TIMEOUT']} for role in sorted({r['role'] for r in raw})})
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary))

if __name__=='__main__':globals()[sys.argv[1]]()
