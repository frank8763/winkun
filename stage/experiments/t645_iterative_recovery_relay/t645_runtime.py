"""Up to three distinct data-aggregation fits, one conditional joint407 gate."""
from pathlib import Path
import copy,fcntl,gzip,io,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t645_iterative_recovery_relay';DOC=ROOT/'stage/docs/day_20261118/t645'
sys.path.insert(0,str(ROOT/'stage/experiments/t641_verified_recovery_learning'))
import t641_runtime as Parent
R=Parent.R;Mem=Parent.Mem
R.HERE=HERE;R.OUT=OUT;R.DOC=DOC;Mem.OUT=OUT;Mem.HERE=HERE;Mem.DOC=DOC
import t645_learning as M
import t645_native as N
R.M=M
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np


def plan():return read(HERE/'protocol.json')
def setup(p):
    R.setup(p);R.N.worker=N.worker;R.N.inspect=N.inspect;R.D.load=Mem.load


def correction(p,item):
    row=item['row'];assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
    a=R.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
    assert int(a['student_takeover_first'])==item['first']
    assert a['ownership'][:item['first']].all() and not a['ownership'][item['first']:].any()
    v=M.correction(p,a,item['first']);v.update(id=row['job_id'],source_sha256=row['sha256']);return v


def event(who,rnd):
    def emit(stage,k):
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,rnd,stage,k])+'\n')
    return emit


def input_check(p):
    original=Parent.teacher_check(p)
    R.P.frozen=R.L.frozen
    second=p['initial_corrections'][1];row=second['row'];assert sha(row['path'])==row['sha256']
    N.inspect(p,R.E.N.S.arrays(row['path']),row)
    assert read(p['second_input_audit'])['state']=='PASS'
    return dict(state='PASS',first=original,second_sha256=row['sha256'],second_stochastic_success_checked=True,new_native=0,new_optimizer=0)


def verify_inputs():
    p=plan();R.verify(p);save(OUT/'INPUT_AUDIT.json',input_check(p))


def preflight_inputs():
    assert not OUT.exists();print(json.dumps(input_check(plan())))


def control_fingerprint(item):
    row=item['row']
    return digest(dict(job_id=row['job_id'],first=item['first'],arrays={k:row['array_hashes'][k] for k in ['states','actions','native_counts','task_goal_xy']}))


class Runtime(R.Runtime):
    def request(self,jobs,model,role,full=False,references=None,seeds=None):
        values=[]
        for off in range(0,len(jobs),3):
            self.guard();js=jobs[off:off+3]
            assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics']
            assert self.commands+len(js)+3<=self.p['max_commands'];self.commands+=len(js)
            req=[]
            for i,j in enumerate(js):
                name=f'{len(self.rows)+i:04d}_{role}'
                req.append(dict(model=model['path'],job=copy.deepcopy(j),seed=None if seeds is None else seeds[off+i],
                   query=False,record_key=name,full_path=str(OUT/'episodes'/(name+'.npz')) if full else None))
            for i,((a,row),request) in enumerate(zip(self.pool.request(req),req)):
                path=Path(request['full_path']) if full else OUT/'evaluation'/(request['record_key']+'.npz')
                if not full:
                    path.parent.mkdir(exist_ok=True);assert not path.exists()
                    R.E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},self.p['compact_bytes'])
                else:assert path.stat().st_size<=self.p['fresh_bytes']
                row.update(job_id=request['job']['id'],request=request,role=role,seed=request['seed'],full=full,path=str(path),sha256=sha(path),
                    loaded_model_sha256=model['sha256'],field_boundary=bool((np.abs(a['poses'][:,:2])>7).any()))
                if references is not None:row.update(reference_path=references[off+i]['path'],exact=True)
                self.rows.append(row)
                with gzip.open(OUT/'cases.jsonl.gz','ab') as f:f.write((json.dumps(row)+'\n').encode())
                assert row['policy_hash']==model['policy_hash']
                try:
                    N.inspect(self.p,a,row)
                    if references is not None:
                        previous=references[off+i]
                        assert row['array_hashes']==previous['array_hashes'] and row['integration_sha256']==previous['integration_sha256']
                except BaseException:
                    R.E.N.write_npz(OUT/'contract_failure.npz',R.E.N.S.packed(a),self.p['full_bytes']);raise
                values.append((a,row))
        return values


def select_departure(a,b):
    ta=np.r_[0,np.cumsum(a['native_counts'])[:-1]];tb=np.r_[0,np.cumsum(b['native_counts'])[:-1]]
    n=min(len(ta),len(tb));assert np.array_equal(ta[:n],tb[:n])
    xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1)
    dy=a['poses'][:n,2]-b['poses'][:n,2];yaw=np.abs(np.arctan2(np.sin(dy),np.cos(dy)))*180/np.pi
    indices=np.flatnonzero((xy>=.02)|(yaw>=1.))
    if not len(indices):return None
    k=int(indices[0]);assert k>0
    return dict(decision=k,seconds=float(ta[k]*.002),xy_m=float(xy[k]),yaw_degrees=float(yaw[k]),rule='first XY>=.02m OR yaw>=1degree on actual aligned trajectories')


def persist_source(p,rnd,a,row):
    path=OUT/'sources'/f'round_{rnd:02d}.npz';path.parent.mkdir(exist_ok=True);assert not path.exists()
    R.E.N.write_npz(path,R.E.N.S.packed(a),p['fresh_bytes'])
    return dict(row,path=str(path),sha256=sha(path),full=True,derived_from_compact_path=row['path'],no_additional_native=True)


def main():
    p=plan();R.verify(p);setup(p);R.E.N.offline();R.E.L.install_stop_handlers()
    assert read(OUT/'INPUT_AUDIT.json')['state']=='PASS'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=Runtime(p);error=None;rounds=[];corrections=copy.deepcopy(p['initial_corrections']);final407=None;retained=p['initial'];reason='MAX_ROUNDS'
    try:
        data=R.D.load(p,rt.guard)
        save(OUT/'FULL384_MEMORY_MANIFEST.json',dict(state='ALL384_FULL_HISTORY_VERIFIED_IN_RAM',histories=Parent.manifest(data),restored_disk_cache_bytes=0))
        rt.pool=R.N.Pool(dict(p,out=str(OUT),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for rnd in range(p['max_rounds']):
            assert time.time()+p['learning_seconds']<p['absolute_learning_stop']
            current=copy.deepcopy(corrections);cpu=[correction(p,item) for item in current]
            R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            keep=M.pack(data,'cuda');new=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in g.items()} for g in cpu]
            # Every candidate starts from the same safe original, with fresh
            # Adam. Only validated accumulated recovery data changes rounds.
            obj=M.make(p);shadow=M.make(p);before=M.state(obj);assert digest(before)==digest(M.state(shadow))
            start=time.time();last=[0.]
            def learn_guard():
                rt.guard();assert time.time()<min(start+p['learning_seconds'],p['absolute_learning_stop'])-5
                if time.time()-last[0]>5:last[0]=time.time();R.gpu_guard()
            u=M.fit(obj,keep,new,p,learn_guard,event('live',rnd));v=M.fit(shadow,keep,new,p,learn_guard,event('shadow',rnd))
            assert u==v and digest(M.state(obj))==digest(M.state(shadow));mature=M.state(obj);mature_hash=digest(mature)
            M.load(obj,before);assert digest(M.state(obj))==digest(before);M.load(obj,mature);assert digest(M.state(obj))==mature_hash
            proposal=M.export(obj,p,OUT/f'proposal_{rnd:02d}.pt.gz')
            record=dict(round=rnd,corrections=current,update=u,proposal=proposal,trained_state_hash=mature_hash,
                precheck=None,selected_source=None,departure=None,exploration=[],added_corrections=[],accepted=False)
            save(OUT/f'LEARNING_{rnd:02d}.json',record)
            obj[0].cpu();shadow[0].cpu();del keep,new;torch.cuda.empty_cache();fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
            vv=rt.request(p['precheck_jobs'],proposal,f'pre_{rnd}')
            rows=[r for a,r in vv];record['precheck']=rows;rounds.append(record)
            save(OUT/'ROUNDS.json',rounds)
            if all(R.safe(r) for r in rows):
                assert time.time()+p['full_evaluation_reserve_seconds']<rt.deadline
                by={r['job_id']:r for r in rows};assert len(by)==7
                rest=[j for j in p['evaluation_jobs'] if j['id'] not in by];assert len(rest)==400
                following=rt.request(rest,proposal,'joint_rest400');by.update({r['job_id']:r for a,r in following})
                ordered=[by[j['id']] for j in p['evaluation_jobs']];assert len(ordered)==407
                keys=[by[key] for key in p['key_replay_ids']]
                rt.request([r['request']['job'] for r in keys],proposal,'exact_final7',True,keys)
                ids={r['job_id'] for r in ordered if R.safe(r)};lost=sorted(set(p['before_success_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
                accepted=not lost and p['target_job']['id'] in ids and bool(gained)
                final407=dict(round=rnd,success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=gained,accepted=accepted)
                record['accepted']=accepted;retained=proposal if accepted else p['initial']
                with gzip.open(OUT/'JOINT407.json.gz','xb') as f:f.write(json.dumps(dict(rows=ordered,model=proposal,prechecks_reused=7)).encode())
                M.load(obj,before);M.load(shadow,before);record['rollback_state_hash']=digest(M.state(obj))
                reason='COMPLETE407_ACCEPTED' if accepted else 'COMPLETE407_REJECTED';break
            M.load(obj,before);M.load(shadow,before);assert digest(M.state(obj))==digest(before)==digest(M.state(shadow))
            record['rollback_state_hash']=digest(M.state(obj))
            if rnd+1==p['max_rounds']:reason='MAX_THREE_DISTINCT_DATA_ROUNDS';break
            failed={r['job_id'] for r in rows if not R.safe(r)}
            chosen=next(key for key in p['failure_priority_ids'] if key in failed)
            a,row=next(v for v in vv if v[1]['job_id']==chosen);ref=p['reference_by_id'][chosen]
            assert sha(ref['path'])==ref['sha256'];b=R.E.N.S.arrays(ref['path'])
            departure=select_departure(a,b)
            if departure is None:reason='NO_PREDECLARED_DEPARTURE_IN_COMMON_HISTORY';break
            source=persist_source(p,rnd,a,row);record.update(selected_source=source,departure=departure)
            job=dict(row['request']['job'],source_path=source['path'],student_takeover_first=departure['decision'])
            seeds=[[p['exploration_seed'],rnd,i] for i in range(4)]
            samples=rt.request([job]*4,proposal,f'recover_{rnd}',True,seeds=seeds)
            sampled=[r for a,r in samples];rt.request([job]*4,proposal,f'exact_recover_{rnd}',references=sampled,seeds=seeds)
            record['exploration']=sampled
            qualified=[r for r in sampled if R.safe(r)]
            seen={control_fingerprint(item) for item in corrections}
            for r in qualified:
                item=dict(row=r,first=departure['decision'],origin='new_actual_candidate_state',round=rnd)
                # Every accepted label is the actual sampled executed command,
                # not the failed deterministic mean of its producing policy.
                fingerprint=control_fingerprint(item)
                if fingerprint in seen:continue
                seen.add(fingerprint);corrections.append(item);record['added_corrections'].append(item)
            save(OUT/'ROUNDS.json',rounds)
            if not record['added_corrections']:reason='NO_NEW_VERIFIED_RECOVERY_NO_REPEAT_IDENTICAL_FIT';break
        save(OUT/'ROUNDS.json',rounds)
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,
          physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,stop_reason=reason,
          rounds=rounds,final407=final407,retained=retained,new_RL=0,new_Critic=0,
          actual_live_Actor_started=sum(json.loads(v)[0]=='live' and json.loads(v)[2]=='BEFORE' for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()) if (OUT/'optimizer_calls.jsonl').exists() else 0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();R.verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(v)).exists() for v in [g['parent'],*g['workers']])
    rows=[json.loads(z) for z in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert len({r['path'] for r in rows})==len(rows);by={r['path']:r for r in rows}
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:N.inspect(p,a,row)
        else:assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        if row.get('exact'):
            original=by[row['reference_path']];assert row['seed']==original['seed'] and row['array_hashes']==original['array_hashes'] and row['integration_sha256']==original['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    data=R.D.load(p,lambda:None);assert Parent.manifest(data)==read(OUT/'FULL384_MEMORY_MANIFEST.json')['histories']
    calls=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    corrections=copy.deepcopy(p['initial_corrections']);historical=0;deadline=min(time.time()+p['audit_seconds']-30,p['absolute_all_stop']-60)
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for index,record in enumerate(s['rounds']):
        rnd=record['round'];assert record['corrections']==corrections
        assert rnd==index
        if index:assert len(record['corrections'])>len(s['rounds'][index-1]['corrections'])
        cpu=[correction(p,item) for item in corrections];keep=M.pack(data,'cuda')
        new=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in g.items()} for g in cpu]
        obj=M.make(p);before=M.state(obj)
        def guard():assert time.time()<deadline;R.gpu_guard()
        u=M.fit(obj,keep,new,p,guard,event('independent_history',rnd));historical+=u['Actor_Adam']
        assert u==record['update'] and digest(M.state(obj))==record['trained_state_hash'] and obj[0].phash()==record['proposal']['policy_hash']
        assert sha(record['proposal']['path'])==record['proposal']['sha256']
        M.load(obj,before);assert digest(M.state(obj))==record['rollback_state_hash']
        expected=[[stage,k] for k in range(p['fit_steps']) for stage in ['BEFORE','AFTER']]
        assert all([[v[2],v[3]] for v in calls if v[:2]==[who,rnd]]==expected for who in ['live','shadow'])
        if record['selected_source']:
            src=record['selected_source'];assert sha(src['path'])==src['sha256'];a=R.E.N.S.arrays(src['path']);N.inspect(p,a,src)
            assert src['array_hashes']==by[src['derived_from_compact_path']]['array_hashes']
            ref=p['reference_by_id'][src['job_id']];assert select_departure(a,R.E.N.S.arrays(ref['path']))==record['departure']
            assert len(record['exploration'])==4
            expected=[];seen={control_fingerprint(item) for item in corrections}
            for row in record['exploration']:
                if not R.safe(row):continue
                item=dict(row=row,first=record['departure']['decision'],origin='new_actual_candidate_state',round=rnd)
                fingerprint=control_fingerprint(item)
                if fingerprint not in seen:expected.append(item);seen.add(fingerprint)
            assert record['added_corrections']==expected
        corrections+=record['added_corrections'];obj[0].cpu();del keep,new,obj;torch.cuda.empty_cache()
    fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    final=s['final407']
    if final is not None:
        data407=json.loads(gzip.decompress((OUT/'JOINT407.json.gz').read_bytes()))
        actual=data407['rows'];assert len(actual)==407 and len({r['job_id'] for r in actual})==407
        assert [r['job_id'] for r in actual]==[j['id'] for j in p['evaluation_jobs']]
        assert all(r['loaded_model_sha256']==data407['model']['sha256'] for r in actual)
        ids={r['job_id'] for r in actual if R.safe(r)};lost=sorted(set(p['before_success_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
        assert final['success_ids']==sorted(ids) and final['lost_prior']==lost and final['gained_prior']==gained
        assert final['accepted']==(not lost and p['target_job']['id'] in ids and bool(gained))
    assert sha(p['initial']['path'])==p['initial']['sha256']
    allcalls=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    assert sum(v[0]=='independent_history' and v[2]=='AFTER' for v in allcalls)==historical<=p['max_rounds']*p['fit_steps']
    save(OUT/'audit.json',dict(state='PASS',new_native=0,historical_Actor_Adam=historical,
       independent_full384_and_all_rounds_Actor_Adam_rebuilt=True,all_success_label_provenance_and_exact_replays=True,
       no_identical_dataset_repeat=True,unique_files_and_conditional407_checked=True,original_T501_unchanged=True))


if __name__=='__main__':
    def no_network(*args,**kwargs):raise RuntimeError('Verified RAM only during science')
    R.D.remote_bytes=no_network;R.E.N.offline()
    globals()[sys.argv[1]]()
