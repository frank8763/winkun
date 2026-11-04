from t348_core import *
import multiprocessing as mp, signal, subprocess


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'main.log').open('xb') as f:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage matched causal sequence warmstart',
            'timeout','--signal=TERM','--kill-after=1s',str(p['wall_seconds'])+'s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],
            cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+p['wall_seconds'],all_work_deadline_epoch=epoch+p['all_work_seconds'],
         source_remote_commit=receipt['remote_commit'],source_root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid,deadline_epoch=epoch+p['wall_seconds'])))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();deadline=min(start+p['wall_seconds'],read(OUT/'launch.json')['deadline_epoch']);learn_end=deadline-p['final_reserve_seconds']
    ctx=mp.get_context('spawn');counts=ctx.Array('q',4);processes=[];pipes=[];rows=[];replays=[];models={};blocks=[];result={};state='FAILED';phase='STARTING';commands=7;last=0.;updates={'live':0,'shadow':0};active=[]
    objs={arm:Learner(p,arm) for arm in p['arms']}
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,learning_deadline_epoch=learn_end,
             commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active,completed_blocks=len(blocks)))
    def budget():
        assert K.S.used(OUT)<=p['final_bytes'] and K.S.used(ROOT/'stage/runs')<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    def fit_guard():assert time.time()<learn_end;budget();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-10
        return pipe.recv()
    def jobs(js,role,obj,model=None,reference=None):
        nonlocal phase,commands,active
        phase=role;out=[];q=obj.actor;arm='memory' if q.memory_on else 'reset'
        for off in range(0,len(js),4):
            group=js[off:off+4];assert time.time()+p['per_batch_admission_seconds']<deadline;budget()
            assert commands+len(group)<=p['max_commands'] and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            commands+=len(group);active=[j['id'] for j in group];status(True)
            for pipe,j in zip(pipes,group):
                pipe.send(dict(j,id=arm+'_'+role+'_'+j['id'],source_reset_id=j['id'],role=role,arm=arm,learned=q.learned(),policy_hash=q.phash(),deadline_epoch=deadline-15,readback_model=model))
            batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[];bad=False
            for k,item in enumerate(batch):
                if 'error' in item:bad=True;continue
                r,a=item['row'],item['arrays']
                if not r['complete']:
                    path=OUT/'workers'/str(k)/'partial.npz';b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];assert not path.exists();path.write_bytes(blob)
                    with (OUT/'partial_cases.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                    bad=True;continue
                if reference is not None:
                    ref=reference[r['job_id']];assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                    assert a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][name]) for name,v in a.items())
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);replays.append(r)
                    with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                else:
                    full=role=='final27' and r['job_id'] in ['wide_dev_0','wide_dev_1']
                    data=packed(a) if full else {name:a[name] for name in ['actions','native_counts','end_flags']}
                    b=io.BytesIO();np.savez_compressed(b,**data);blob=b.getvalue();assert len(blob)<=p['full_bytes' if full else 'compact_bytes']
                    assert sum(x['full_arrays_saved']==full for x in rows)<p['max_full_cases' if full else 'max_compact_cases']
                    path=OUT/'episodes'/(r['id']+'.npz');assert not path.exists();path.write_bytes(blob);r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=full)
                    rows.append(r)
                    with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                out.append(item)
            status(True)
            assert not bad,'Failed/administratively partial commands retained; no replacement launch'
        return out
    def count(items):return {key:sum(bool(v['row'][key]) for v in items) for key in ['success','collision','timeout']}
    def event(arm,replica,row):
        if row['phase']=='AFTER':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(arm=arm,replica=replica,**row),separators=(',',':'))+'\n')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T348 bounded native deadline')));signal.alarm(max(1,int(deadline-time.time())-5))
    try:
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(x['ready'] and x['affinity']==[k+1] for k,x in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[q.pid for q in processes],ready=ready))
        initial=jobs(p['course_dev_jobs'],'initial_wide5',objs['reset']);result['shared_initial_wide5']=count(initial)
        phase='DATASET_AND_SEQUENCE_CHECK';status(True);data=dataset(p,objs['reset'].actor,fit_guard)
        save(OUT/'data_contract.json',dict(old_episodes=211,old_states=28629,new_sources=[dict(name=g['id'],rows=len(g['x']),label_start=g['start']) for g in data[1]],
             schedule_sha256=digest(schedule(p,data[0])),trainable_parameters=sum(v.numel() for v in objs['reset'].actor.trainable()),new_RL_updates=0))
        for arm,obj in objs.items():
            phase='SUPERVISED_'+arm;status(True);before=obj.state();immutable=digest(before);input_hashes=obj.hashes();shadow=Learner(p,arm);shadow.load(before)
            try:
                live=fit(p,obj,data,fit_guard,lambda row:event(arm,'live',row))
                other=fit(p,shadow,data,fit_guard,lambda row:event(arm,'shadow',row))
                assert live==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
            except BaseException:
                obj.load(before);save(OUT/('interrupted_'+arm+'.json'),dict(input_hashes=input_hashes,restored_hashes=obj.hashes(),retained_updates=0));raise
            tracepath=OUT/('trace_'+arm+'.json');tracepath.write_text(json.dumps(live,separators=(',',':'))+'\n')
            block=dict(arm=arm,input_hashes=input_hashes,output_hashes=obj.hashes(),live_Adam=600,shadow_Adam=600,exact_shadow=True,retained=True,trace_sha256=digest(live),trace_file_sha256=K.C.sha(tracepath))
            blocks.append(block)
            with (OUT/'blocks.jsonl').open('a') as f:f.write(json.dumps(block,separators=(',',':'))+'\n')
            models[arm]=export(p,obj,OUT/(arm+'_final.pt.gz'));save(OUT/'models.json',models)
        for arm,obj in objs.items():
            js=[*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']]
            final=jobs(js,'final27',obj);mapping={v['row']['job_id']:v for v in final}
            reload=jobs(js,'model_readback27',obj,model=models[arm]['path'],reference=mapping)
            result[arm]=dict(old20=count(final[:20]),wide5=count(final[20:25]),original2=count(final[25:]),model_readback27_exact=len(reload)==27,
                 old275=None,six54=None,fresh16=None,reload78=None,stable_six=False)
            save(OUT/'result.json',result)
        state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError,BrokenPipeError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    summary=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),updates=updates,
         models=models,result=result,primary_cases=len(rows),model_readback_cases=len(replays),blocks=len(blocks),new_RL_updates=0,new_training_native_collection=0,
         all_workers_stopped=all(not q.is_alive() for q in processes),stable_six=False,entered7or8=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary))


if __name__=='__main__':{'launch':launch,'main':main}[sys.argv[1]]()
