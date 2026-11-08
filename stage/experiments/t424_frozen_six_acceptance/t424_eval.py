"""Frozen serialized single-Actor six-metre acceptance; zero learning."""
from pathlib import Path
import sys,os,time,json,copy,shutil,traceback,contextlib,fcntl,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t424_frozen_six_acceptance';DOC=ROOT/'stage/docs/acceptance_20261108'
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E
torch,np=E.torch,E.np
read,save,sha,digest=E.read,E.save,E.sha,E.digest
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert E.N.K.S.used(OUT)<=p['final_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
def ban_learning():
    def no(*a,**k):raise AssertionError('T424 frozen acceptance forbids all learning/backward')
    torch.optim.Adam.step=no;torch.optim.AdamW.step=no;torch.optim.SGD.step=no;torch.Tensor.backward=no
def frozen(p):
    q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
    assert q.phash()==p['expected_policy_hash'];q.sample=False;q.reset_noise();return q
def worker(pipe,index,p,counts):
    E.N.offline();ban_learning();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.set_out(p['out']);base=E.N.science(p)
    q=frozen(p);s=w=None;folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'console.log').open('w');stdout,stderr=sys.stdout,sys.stderr
    def guard(physical=False):E.L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,_=E.N.S.P.C.B.setup(folder/'native',base);original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers();pipe.send(dict(state='READY',pid=os.getpid(),cpu=index,native=0))
            while True:
                request=pipe.recv()
                if request is None:break
                assert request['model']==p['student_start_model'] and request.get('seed') is None and not request.get('query',False)
                assert sha(request['model'])==p['student_start_sha256'];q.load_learned(E.N.S.read_model(request['model']));q.reset_noise();q.sample=False;q.rng=None;assert q.phash()==p['expected_policy_hash']
                job=request['job'];module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m']);before=counts[index+1]
                s=E.L.Session(job,native,create,w,guard,charge,q);E.L.install_stop_handlers()
                if request.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise E.L.StageStop('T424 cancel before first native')
                    s.guard=cancel
                    try:s.step(owner=0);raise AssertionError('Cancellation absent')
                    except E.L.StageStop:pass
                    assert s.steps==0 and not s.rows;s.close();s=None;pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                while not s.done():s.step(owner=0)
                a=E.payload(s);row=s.result();row.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=digest(a['final_native_integration']),pid=os.getpid(),cpu=index,teacher=False)
                assert not a['ownership'].any() and not a['stochastic'].any();s.close();s=None;pipe.send((a,row))
    except BaseException:
        save(folder/'failure.json',dict(error=traceback.format_exc(),recorded_physics=counts[index+1]))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(E.payload(s)),p['full_bytes'])
            except BaseException:pass
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if s:s.close()
        if w:w.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()
class Pool(E.NATIVE_AUDIT.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];context=mp.get_context('spawn')
        try:
            for index in range(4):
                parent,child=context.Pipe();process=context.Process(target=worker,args=(child,index,p,counts));process.start();child.close();self.children.append(process);self.pipes.append(parent)
            save(Path(p['out'])/'workers.json',dict(parent=os.getpid(),workers=[x.pid for x in self.children]));self.ready=[self.receive(i) for i in range(4)];assert all(v['state']=='READY' for v in self.ready)
        except BaseException:self.close();raise
def main():
    E.N.offline();ban_learning();p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists();E.set_out(OUT)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers();start=time.time();deadline=start+p['main_seconds'];config=dict(p,native_stop_epoch=deadline)
    counts=mp.get_context('spawn').RawArray('q',5);commands=0;rows=[];replays=[];pool=None;last=0.;phase='SETUP';state='FAILED';result=dict(precheck14=None,old275=None,wide54=None,fresh16=None,reload78=None,navigation_gate_pass=False,old_training_audit='FAIL_RECONSTRUCTION_CAUSE_UNRESOLVED',research_promotion=False)
    full_ids=set(j['id'] for j in p['reload_jobs']);refs={}
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline,Actor_updates=0,optimizer_steps=0))
    def guard():
        nonlocal last
        E.L.check_stop();assert time.time()<deadline-5 and sum(counts)<=p['max_physics'] and commands<=p['max_main_commands']
        if time.time()-last>5:
            last=time.time();verify(p);save(OUT/'status.json',dict(state=phase,pid=os.getpid(),epoch=last,physics=sum(counts),commands=commands,completed_cases=len(rows),replays=len(replays),Actor_updates=0,optimizer_steps=0))
    def evaluate(jobs,role,precheck=False,reload=False):
        nonlocal phase,commands
        phase=role;values=[]
        for offset in range(0,len(jobs),4):
            js=jobs[offset:offset+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in js)<=p['max_physics'];ids=list(range(commands+1,commands+len(js)+1));commands+=len(js)
            requests=[dict(model=p['student_start_model'],job=job,seed=None) for job in js]
            for i,((a,row),job,cid) in enumerate(zip(pool.request(requests),js,ids)):
                assert row['policy_hash']==p['expected_policy_hash'] and not row['teacher'] and int(a['native_counts'].sum())==row['physics']
                row.update(role=role,job_id=job['id'],physics_steps=int(a['native_counts'].sum()),loaded_model_sha256=p['student_start_sha256'],seed=None)
                if reload:
                    prior=refs[job['id']];old=E.N.S.arrays(prior['path'])
                    assert set(old)==set(a)
                    for key in a:assert E.equal(old[key],a[key]),('Model independent native readback differs',job['id'],key)
                    for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==prior[key],key
                    row.update(exact=True,all_arrays_exact=True,independent_native_execution=True,independent_sample=False,reference_path=prior['path']);replays.append(row);name='replays.jsonl'
                else:
                    full=precheck or job['id'] in full_ids;path=OUT/'episodes'/f'{len(rows):04}.npz';path.parent.mkdir(exist_ok=True)
                    E.N.write_npz(path,E.N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},p['full_bytes'] if full else 5*1024)
                    row.update(path=str(path),sha256=sha(path),full=full);rows.append(row);name='cases.jsonl'
                    if not precheck and full:refs[job['id']]=row
                    if precheck:
                        prior=p['precheck_history_rows'][offset+i]
                        for key in ['job_id','policy_hash','array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==prior[key],('T422BC2 historical exact differs',job['id'],key)
                with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n')
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=role,physics=row['physics'],pid=row['pid']))+'\n')
                values.append(row)
        save(OUT/'completed_groups.json',dict(result=result,rows=len(rows),replays=len(replays),physics=sum(counts),commands=commands));return values
    try:
        pool=Pool(config,counts,guard);commands=4
        for cid,row in enumerate(pool.ready,1):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role='worker_setup',physics=0,pid=row['pid']))+'\n')
        pre=evaluate([*p['old8_jobs'],*p['left5_jobs'],p['right_job']],'precheck14',precheck=True);result['precheck14']=E.tally(pre);assert all(x['success'] for x in pre)
        # Full locked development and old regression are completed even after
        # an observed failure, so neither group is reported from a subset.
        wide=evaluate(p['main_jobs'],'wide54');result['wide54']=E.tally(wide)
        old=evaluate(p['old275_jobs'],'old275');result['old275']=E.tally(old)
        if all(row['success'] for row in [*wide,*old]):
            for f,h in p['fresh_ledger'].items():assert sha(f)==h
            fresh=evaluate(p['fresh_jobs'],'fresh16');result['fresh16']=E.tally(fresh)
            if all(row['success'] for row in fresh):
                reload=evaluate(p['reload_jobs'],'reload78',reload=True);result['reload78']=dict(exact=len(reload),independent_native_execution=True,independent_sample=False,all_arrays_exact=True);result['navigation_gate_pass']=len(reload)==78
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,physics=sum(counts),commands=commands,no_restart=True))
    finally:
        if pool:pool.close()
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),main_commands=commands,preflight_zero_commands=8,all_commands=commands+8,Actor_updates=0,optimizer_steps=0,result=result,model_path=p['student_start_model'],model_sha256=sha(p['student_start_model']),policy_hash=p['expected_policy_hash'],all_workers_stopped=True,all_sessions_closed=True,seven_eight_entered=False,narrow_paused=True));save(OUT/'status.json',dict(state=state,epoch=time.time(),physics=sum(counts),commands=commands,Actor_updates=0,optimizer_steps=0));E.N.manifest(OUT);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE';print('T424_FROZEN_NAVIGATION_COMPLETE',result['navigation_gate_pass'],flush=True)
if __name__=='__main__':main()
