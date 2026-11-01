from frontier_runtime import *
import multiprocessing as mp,fcntl,io,shutil
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t256_distance_frontier'
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No restart T256'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();deadline=start+plan['wall_seconds'];physics=commands=0;workers=[];pipes=[];recovered=[];longrows=[];invalid=[];contexts=[];offsets=[0];ids=[];tasks=[];reason='COMPLETE';datasets={}
    def save_status(state):C.save(OUT/'status.json',dict(state=state,pid=os.getpid(),physics_steps=physics,commands=commands,recovered=len(recovered),new_distance_cases=len(longrows),invalid_initial_states=len(invalid),updated_epoch=time.time(),elapsed_s=time.time()-start))
    def guard(n=0,steps=0,pending=0):
        if time.time()>=deadline:raise Stop('WALL')
        if commands+n>plan['max_commands'] or physics+steps>plan['max_physics']:raise Stop('COUNTS')
        if S.used(OUT)+pending>=plan['final_bytes'] or S.used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<plan['min_free_bytes']:raise Stop('STORAGE')
    def append(name,r):
        with (OUT/name).open('a') as f:f.write(json.dumps(r)+'\n')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.time()>=deadline:raise Stop('RESPONSE_WALL')
        r=pipe.recv();assert 'error' not in r,r;return r
    def batch(kind,jobs,name):
        nonlocal commands,physics
        result=[]
        for off in range(0,len(jobs),4):
            jj=jobs[off:off+4];guard(len(jj),sum(j['case']['step_cap'] for j in jj) if kind=='rollout' else 0,512*1024)
            for pipe,j in zip(pipes,jj):pipe.send((kind,j))
            rr=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(rr);physics+=sum(r['row']['physics_steps'] if kind=='rollout' else 0 for r in rr)
            for j,item in zip(jj,rr):
                r=item['row'] if kind=='rollout' else item
                if kind=='rollout':
                    a=item['arrays'];full=j.get('scope') in ['distance','eight'];buf=io.BytesIO();np.savez_compressed(buf,**(a if full else {k:v for k,v in a.items() if k not in ['states','next_states']}));data=buf.getvalue();guard(pending=len(data));path=OUT/'episodes'/(j['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();path.write_bytes(data);r.update(path=str(path),sha256=C.sha(path));assert r['complete'],'Censored episode'
                append(name,r);result.append(item)
            save_status(name)
        return result
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);save_status('STARTING');ctx=mp.get_context('spawn')
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);w=ctx.Process(target=worker,args=(b,k,str(OUT),plan,deadline));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        made=batch('make',plan['basecases'],'base_geometry.jsonl');jobs=[]
        for template,base in zip(plan['basecases'],made):
            for ix,dx in enumerate([-.03,0,.03]):
                for iy,dy in enumerate([-.03,0,.03]):
                    j=dict(template,id=f'{template["id"]}_{ix}{iy}',source_reset_id=f'{template["id"]}_{ix}{iy}',exact_snapshot=base['snapshot'],delta_x_m=dx,delta_y_m=dy,scope='distance');jobs.append(j)
        geometry=batch('inspect',jobs,'initial_geometry.jsonl');valid=[]
        for j,g in zip(jobs,geometry):
            if g['valid']:valid.append(j)
            else:invalid.append(g)
        old={r['source_reset_id']:r for r in map(json.loads,Path(plan['reference_rows']).read_text().splitlines())}
        oldjobs=[dict(j,id='recover_'+j['id'],source_reset_id=j['id'],scope='recovery',seed=0,mode='deterministic') for j in plan['old_development']]
        for off in range(0,len(oldjobs),4):
            items=batch('rollout',oldjobs[off:off+4],'recovery.jsonl')
            for item in items:
                r=item['row'];ref=old[r['source_reset_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'];assert r['success'] and not r['collision'];x=item['arrays']['states'];contexts.append(x.astype(np.float32));offsets.append(offsets[-1]+len(x));ids.append(r['source_reset_id']);tasks.append(int(r['group'][4:]));recovered.append(r)
        buf=io.BytesIO();np.savez_compressed(buf,states=np.concatenate(contexts),offsets=np.asarray(offsets,dtype=np.int64),case_ids=np.asarray(ids),task=np.asarray(tasks,dtype=np.uint8));data=buf.getvalue();guard(pending=len(data));path=OUT/'stable_5m_current_states.npz';path.write_bytes(data);datasets['retention']=dict(path=str(path),sha256=C.sha(path),bytes=len(data),episodes=len(ids),rows=offsets[-1],all_original10_arrays_and_integration_exact=True)
        longrows=[i['row'] for i in batch('rollout',valid,'distance.jsonl')]
        refs=list(map(json.loads,Path(plan['eight_reference']).read_text().splitlines()));ej=[dict(j,id='audit8_'+j['source_reset_id'],scope='eight') for j in plan['eight_jobs']];eight=batch('rollout',ej,'eight_recovery.jsonl');queries=[]
        base=M.read(plan['initial_base'])['policy'];p=Actor(base,False,2026103501);p.load_state_dict(M.read(plan['checkpoint'])['policy']);p.eval().requires_grad_(False)
        for item,ref in zip(eight,refs):
            r=item['row'];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'];x=item['arrays']['states'][0].copy();raw=[]
            for distance in [5.,6.,7.,8.]:
                q=x.copy();q[[48,51,52,72,84,96,108]]*=distance/8.
                with torch.no_grad():action=p.step(torch.tensor(q[None]))[0][0].numpy()
                raw.append(dict(radial_goal_distance_m=distance,action=action.tolist()))
            queries.append(dict(source=r['source_reset_id'],physical_steps=0,offline_query_only=True,queries=raw))
        C.save(OUT/'goal_magnitude_queries.json',dict(queries=queries,not_deployed_or_success_evidence=True))
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=4*37334))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    safe=lambda r:r['complete'] and r['success'] and not r['collision'];scores={g:dict(valid=sum(r['group']==g for r in longrows),safe=sum(r['group']==g and safe(r) for r in longrows),collision=sum(r['group']==g and r['collision'] for r in longrows),timeout=sum(r['group']==g and r['timeout'] for r in longrows)) for g in ['6m_LEFT','6m_RIGHT','7m_LEFT','7m_RIGHT']}
    result=dict(state=reason,physics_steps=physics,commands=commands,optimizer_calls=0,recovered_old_cases=len(recovered),recovery_not_new_independent_data=True,scores=scores,invalid_initial_states=invalid,datasets=datasets,all_workers_stopped=all(not w.is_alive() for w in workers),elapsed_s=time.time()-start,own_bytes=S.used(OUT),runs_MiB=S.used(ROOT/'stage/runs')/1024**2,automatic_next_training=False)
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
