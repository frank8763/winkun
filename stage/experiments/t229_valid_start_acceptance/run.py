from valid_support import *
import multiprocessing as mp,fcntl,shutil,io,gzip
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT229'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds'];physics=commands=rollouts=controls=0;workers=[];pipes=[];validity=[];fresh=[];merged=[];reason=None
    def status(state):C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,fresh_rollouts=rollouts,validity_cases=len(validity),fresh_cases=len(fresh),controls_verified=controls,optimizer_calls=0,elapsed_s=time.monotonic()-start))
    def guard(n=0,count=0,pending=0):
        if time.monotonic()+5>=deadline:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def append(name,obj):
        with (OUT/name).open('a') as f:f.write(json.dumps(obj)+'\n')
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        state=torch.load(io.BytesIO(gzip.decompress(Path(plan['checkpoint']).read_bytes())),map_location='cpu',weights_only=False)['policy'];assert digest(state)==plan['policy_hash'];ps={k:v.numpy().copy() for k,v in state.items()};ph=digest(state)
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(p) for p in pipes];assert all(r.get('ready') for r in ready);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        for off in range(0,216,4):
            batch=plan['jobs'][off:off+4];guard(len(batch),0,8192)
            for p,j in zip(pipes,batch):p.send(('validate',j))
            rows=[receive(p) for p in pipes[:len(batch)]];commands+=len(rows)
            for row in rows:validity.append(row);append('validity.jsonl',row)
            status('ZERO_STEP_VALIDITY')
        byid={v['id']:v for v in validity};prior=list(map(json.loads,Path(plan['prior_cases']).read_text().splitlines()));old={r['id']:r for r in prior};jobs={j['id']:j for j in plan['jobs']}
        for r in prior:
            assert r['valid_start']==byid[r['id']]['valid_start'],'Physicalvaliditymismatch'
            if r['valid_start']:
                assert r['complete'] and r['policy_hash']==ph and r['scene_sha256']==byid[r['id']]['scene_sha256'];merged.append(dict(r,evidence_source='T228unchanged',original_trace_file=plan['prior_cases']))
        control_jobs=[j for j in plan['jobs'] if int(j['group'][4:])<4 and j['delta_x_m']==j['delta_y_m']==j['delta_yaw_deg']==0.]
        missing=[j for j in plan['jobs'] if byid[j['id']]['valid_start'] and j['id'] not in old]
        requests=[dict(j,control_only=True) for j in control_jobs]+[dict(j,control_only=False) for j in missing];assert len(requests)<=plan['max_rollouts']
        for off in range(0,len(requests),4):
            batch=requests[off:off+4];guard(len(batch),sum(j['case']['step_cap'] for j in batch),len(batch)*192*1024+16384)
            for pipe,j in zip(pipes,batch):pipe.send(('evaluate',dict(j,policy_np=ps,policy_hash=ph,arm='frozenT228',seed=0,mode='deterministic')))
            items=[receive(p) for p in pipes[:len(batch)]];commands+=len(items);rollouts+=len(items);physics+=sum(i['row']['physics_steps'] for i in items)
            for j,item in zip(batch,items):
                row=item['row'];a=item['arrays'];assert row['complete']
                if j['control_only']:
                    ref=old[j['id']];assert ref['physics_steps']==row['physics_steps'] and ref['states_sha256']==digest(a['states']) and ref['actions_sha256']==digest(a['actions']) and ref['integration_sha256']==row['integration_sha256'];controls+=1;append('controls.jsonl',row)
                else:
                    assert np.array_equal(a['next_states'][:-1],a['states'][1:]);data={k:v for k,v in a.items() if k!='next_states'};data['final_next_state']=a['next_states'][-1]
                    buf=io.BytesIO();np.savez_compressed(buf,**data);encoded=buf.getvalue();assert len(encoded)<192*1024;guard(pending=len(encoded)+4096);path=OUT/(j['id']+'.npz');assert not path.exists();path.write_bytes(encoded)
                    row.update(path=str(path),sha256=C.sha(path),evidence_source='T229newfrozen',case=j['case'],delta_x_m=j['delta_x_m'],delta_y_m=j['delta_y_m'],delta_yaw_deg=j['delta_yaw_deg']);fresh.append(row);merged.append(row);append('fresh_cases.jsonl',row)
            status('FROZEN_MISSING_VALID_CASES')
        assert controls==4;C.save(OUT/'merged_valid_cases.json',merged);reason='VALIDITY_AND_MISSING_CASES_COMPLETE'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    finally:
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    scores={};safe=lambda r:r['complete'] and r['success'] and not r['collision']
    for i in range(8):
        g=f'task{i}';vv=[v for v in validity if jobs.get(v['id'],{}).get('group')==g] if 'jobs' in locals() else [];rr=[r for r in merged if r['group']==g];scores[g]=dict(grid_total=len(vv),valid_starts=sum(v['valid_start'] for v in vv),invalid_starts=sum(not v['valid_start'] for v in vv),measured_valid=len(rr),safeGOAL=sum(safe(r) for r in rr),CONTACT=sum(bool(r['collision']) for r in rr),timeout=sum(bool(r['timeout']) for r in rr))
    gate=reason=='VALIDITY_AND_MISSING_CASES_COMPLETE' and controls==4 and len(merged)==sum(v['valid_start'] for v in validity) and all(safe(r) for r in merged)
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,scores=scores,validity_cases=len(validity),invalid_start_ids=[v['id'] for v in validity if not v['valid_start']],merged_measured_valid=len(merged),merged_safeGOAL=sum(safe(r) for r in merged),new_rollouts=rollouts,new_physics_steps=physics,new_commands=commands,prior_T228_physics_separately_counted=True,controls_verified=controls,optimizer_calls=0,valid_domain_local_gate=gate,original_strict_T228_gate=False,heldout_n=0,automatic_promotion=False,all_workers_stopped=True,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and rollouts<=plan['max_rollouts'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
