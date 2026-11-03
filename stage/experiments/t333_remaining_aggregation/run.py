from t333_learning import *
from t333_native_worker import worker
import multiprocessing as mp,signal

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    original=read(Path(p['previous_run'])/'preflight.json');assert original['state']=='PASS'
    assert read(DOC/'T332_PARTIAL_AUDIT.json')['state']=='PASS_PARTIAL_EVIDENCE_ONLY'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for job in [*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']]:
        assert isinstance(job.get('source_reset_id') or job['id'],str)
    q=model(p);opt=optimizer(q,p,'bc');v=read_patch(p['cached_control_patch']);load(q,v['residual']);opt.load_state_dict(v['optimizer'])
    assert K.S.digest(q.state_dict())==v['policy_hash'] and K.S.digest(opt.state_dict())==v['optimizer_hash']
    d=bc_data(p,q,'recovery');h=K.S.digest(q.state_dict());oh=K.S.digest(opt.state_dict())
    objective=bc_backward(q,d,p);q.zero_grad(set_to_none=True)
    assert K.S.digest(q.state_dict())==h and K.S.digest(opt.state_dict())==oh
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe();proc=ctx.Process(target=worker,args=(child,0,p,None,True));proc.start();child.close()
    parent.send(dict(state=K.S.state_np(q),x=d['x'][:1].numpy()));assert parent.poll(30);answer=parent.recv();proc.join(10)
    assert not proc.is_alive() and proc.exitcode==0 and answer['policy_hash']==h
    with torch.no_grad():actual=q.step(d['x'][:1])
    assert np.array_equal(answer['mode'],actual[0].numpy()) and np.array_equal(answer['mean'],actual[1].numpy())
    save(OUT/'preflight.json',dict(state='PASS',native_commands=0,physics_steps=0,Actor_Adam_calls=0,
        new_synthetic_Adam_calls=0,previous_4_synthetic_Adam_proof_reused=True,
        full_actual_backward_calls=1,all54_identifiers_validated=True,cached_control_full_Actor_Adam_loaded_exact=True,
        actual_new_worker_load_actions_exact=True,absolute_T332_deadline_unchanged=True,objective=objective,
        protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print('T333 remaining-only preflight PASS0native0newAdam;cached control and all job IDs exact')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');verify(p)
    start=time.time();deadline=p['absolute_deadline_epoch'];train_end=p['absolute_training_deadline_epoch']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];meter=None;q=None;opt=None
    raw=[];replays=[];calls=[];patches={};counts={};phase='STARTING';active=[];last=0;end='FAILED'
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,
            deadline_epoch=deadline,training_deadline_epoch=train_end,physics_steps=sum(physical[:]),
            commands=0 if meter is None else meter.commands,cases=len(raw),exact_replays=len(replays),
            supervised_Adam_calls=len(calls),real_RL_updates=0,inflight=active))
    def guard():assert time.time()<train_end;meter.hold();status()
    def event(replica,e):
        row=dict(e,replica=replica);calls.append(row)
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-15
        result=pipe.recv();assert 'error' not in result,result;return result
    def export(name):
        blob=patch(q,opt);meter.charge_payload(len(blob));f=OUT/(name+'.pt.gz');assert not f.exists();f.write_bytes(blob)
        patches[name]=dict(path=str(f),sha256=K.C.sha(f),policy_hash=K.S.digest(q.state_dict()),optimizer_hash=K.S.digest(opt.state_dict()));save(OUT/'patches.json',patches)
    def roll(jobs,arm,role):
        nonlocal phase,active
        phase=arm+'_'+role;out=[]
        for off in range(0,len(jobs),4):
            group=[dict(j,id=arm+'_'+role+'_'+j['id'],source_reset_id=j.get('source_reset_id') or j['id'],role=role,
                policy_np=K.S.state_np(q),policy_hash=K.S.digest(q.state_dict()),deadline_epoch=deadline-30,
                aux_eta=0.,mode='deterministic') for j in jobs[off:off+4]]
            reference=None
            for replay in [False,True]:
                meter.checkpoint();assert time.time()<deadline-30 and meter.commands+len(group)<=p['max_commands']
                assert sum(physical[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                meter.commands+=len(group);active=[j['id'] for j in group];status(True)
                for pipe,j in zip(pipes,group):pipe.send(j)
                batch=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(physical[:]);active=[]
                for k,item in enumerate(batch):
                    row,a=item['row'],item['arrays'];row.update(arm=arm,job={key:v for key,v in group[k].items() if key!='policy_np'})
                    if not row['complete'] or row.get('error'):
                        data={key:a[key] for key in ['actions','native_counts','end_flags','environment_rewards']}
                        buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue();assert len(blob)<=p['compact_bytes']
                        meter.charge_payload(len(blob));f=OUT/('partial_'+row['id']+('_replay' if replay else '')+'.npz');f.write_bytes(blob)
                        save(OUT/('partial_'+row['id']+('_replay' if replay else '')+'.json'),dict(row,path=str(f),sha256=K.C.sha(f),censored_or_invalid=True))
                        raise AssertionError('Partial evidence retained; no success claim')
                    if replay:
                        r=reference[k];assert row['array_hashes']==r['row']['array_hashes'] and row['integration_sha256']==r['row']['integration_sha256']
                        assert all(np.array_equal(v,r['arrays'][key]) for key,v in a.items())
                        replays.append(dict(id=row['id'],physics_steps=row['physics_steps'],all_arrays_exact=True,
                            integration_sha256=row['integration_sha256'],not_independent_sample=True));save(OUT/'replays.json',replays)
                    else:
                        data={key:a[key] for key in ['actions','native_counts','end_flags','environment_rewards']}
                        buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue();assert len(blob)<=p['compact_bytes']
                        meter.charge_payload(len(blob));f=OUT/'episodes'/(row['id']+'.npz');assert not f.exists();f.write_bytes(blob)
                        row.update(path=str(f),sha256=K.C.sha(f),saved_array_keys=list(data));raw.append(row);out.append(row)
                        with (OUT/'cases.jsonl').open('a') as log:log.write(json.dumps(row,separators=(',',':'))+'\n')
                if not replay:reference=batch
                status(True)
        return {name:sum(bool(r[name]) for r in out) for name in ['success','collision','timeout']}
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T333 absolute budget')));signal.alarm(max(1,int(deadline-time.time())-5))
    try:
        assert read(OUT/'preflight.json')['state']=='PASS';meter=K.Budget(ROOT,OUT,p,start_epoch=start)
        q=model(p);opt=optimizer(q,p,'bc');export('initial')
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);w=ctx.Process(target=worker,args=(b,k,p,physical));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        for arm in p['arms']:
            phase=arm+'_BC300';status(True);q=model(p);opt=optimizer(q,p,'bc');d=bc_data(p,q,arm)
            if arm=='control':
                v=read_patch(p['cached_control_patch']);load(q,v['residual']);opt.load_state_dict(v['optimizer'])
                assert K.S.digest(q.state_dict())==v['policy_hash'] and K.S.digest(opt.state_dict())==v['optimizer_hash']
                patches['control_candidate']=dict(path=p['cached_control_patch'],sha256=K.C.sha(p['cached_control_patch']),policy_hash=v['policy_hash'],optimizer_hash=v['optimizer_hash'],historical_training_reused=True)
                save(OUT/'patches.json',patches)
            else:
                shadow=copy.deepcopy(q);so=optimizer(shadow,p,'bc')
                live=fit(p,q,opt,guard,lambda e:event('live',e),arm,d)
                other=fit(p,shadow,so,guard,lambda e:event('shadow',e),arm,d)
                assert live==other and K.S.digest(q.state_dict())==K.S.digest(shadow.state_dict()) and K.S.digest(opt.state_dict())==K.S.digest(so.state_dict())
                export(arm+'_candidate')
            counts[arm]={}
            counts[arm]['old20']=roll(p['old_jobs'],arm,'old20')
            counts[arm]['wide5']=roll(p['course_dev_jobs'],arm,'wide5')
            counts[arm]['original2']=roll(p['original_jobs'],arm,'original2');save(OUT/'counts.json',counts)
        meter.checkpoint();end='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,inflight=active,
            physics_steps=sum(physical[:]),commands=0 if meter is None else meter.commands))
        if q is not None and opt is not None:
            try:export('partial')
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
    summary=dict(state=end,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=deadline,
        commands=0 if meter is None else meter.commands,physics_steps=sum(physical[:]),cases=len(raw),exact_replays=len(replays),
        supervised_Adam_calls=len(calls),prior_control_supervised_Adam_calls=600,combined_supervised_Adam_calls=len(calls)+600,real_RL_updates=0,patches=patches,counts=counts,
        all_workers_stopped=all(not w.is_alive() for w in workers),own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'),
        no_promotion=True,old275_six54_fresh16_reload78_executed=False,six_stable=False,seven_eight_entered=False)
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary))

if __name__=='__main__':globals()[sys.argv[1]]()
