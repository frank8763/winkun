"""Fresh full-Actor RL; acceptance on real native navigation, not action proxy."""
from pathlib import Path
import fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t525_student_state_recovery'));import t525_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t527_validated_recovery_learning'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
def event(which):return F.R.emit(which)

def evaluation(rt,p,model,reference=None):
    """Full arrays generated twice; compact evidence saved, all full hashes match.

    Explicit disk contract: evaluator full states are NOT saved/rebuilt offline.
    Native replay equality checks every full array/integration before discard.
    """
    rows=[];details=[];jobs=p['evaluation_jobs'];rt.role='EXACT_FULL32' if reference else 'AUTONOMOUS_FULL32'
    for off in range(0,len(jobs),2):
        batch=jobs[off:off+2];rt.guard()
        assert sum(rt.counts)+sum(j['case']['step_cap'] for j in batch)<=p['max_physics'] and rt.commands+len(batch)+2<=p['max_commands']
        rt.commands+=len(batch)
        requests=[dict(model=model['path'],job=j,seed=None,query=False,full_path=None) for j in batch]
        for i,((a,r),j,req) in enumerate(zip(rt.pool.request(requests),batch,requests)):
            assert r['policy_hash']==model['policy_hash'] and int(a['native_counts'].sum())==r['physics']
            r.update(job_id=j['id'],role=rt.role,seed=None,request=req,loaded_model_sha256=sha(model['path']),full=False,full_arrays_generated_but_not_saved=True)
            detail=F.detail(p,r,a) if j['id'].startswith('t502_') else None
            r['field_detail']=detail
            if reference:
                old=reference[off+i];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail']
                assert [r[k] for k in ['success','collision','timeout','remaining_m']]==[old[k] for k in ['success','collision','timeout','remaining_m']]
                r.update(exact=True,reference_path=old['path'],independent_new_sample=False);rt.replays.append(r);filename='replays.jsonl'
            else:
                path=Path(p['out'])/'episodes'/('eval_'+str(off+i)+'.npz');path.parent.mkdir(exist_ok=True)
                E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},p['compact_bytes'])
                r.update(path=str(path),sha256=sha(path));rt.rows.append(r);filename='cases.jsonl'
            with (Path(p['out'])/filename).open('a') as f:f.write(json.dumps(r)+'\n')
            rows.append(r);details.append(detail)
    return rows,details

def assess(p,rows):
    assert len(rows)==20;old=G.R.M.tally(rows[:8]);field=G.R.M.tally(rows[8:]);ids={r['job_id'] for r in rows if r['success'] and not r['collision'] and (r['field_detail'] is None or r['field_detail']['field_qualified_success'])};protected=set(p['initial_protected_ids'])
    passed=protected<=ids and p['probe_job']['id'] in ids and len(ids)==20
    return dict(old8=old,protected_field11=G.R.M.tally(rows[8:19]),target=G.R.M.tally(rows[19:]),panel20=G.R.M.tally(rows),panel_pass=passed,protected_success_ids=sorted(ids & protected),lost_ids=sorted(protected-ids),accepted=False,retained_research_candidate_for_future_full24=passed,field24=None,fullfield24_NOT_executed=True)

def main(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);P.frozen=L.A.frozen;E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        row=p['recovery_row'];assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        obj=L.make(p);shadow=L.make(p);before=P.state(*obj);assert digest(before)==p['training_initial_joint_hash'];refs=L.references(p)
        rt.role='VALIDATED_RECOVERY_IMITATION_WITH19_SUCCESS_REHEARSAL'
        aa=L.fit(obj,a,p,refs,lambda:rt.guard(True),F.R.emit('live'));bb=L.fit(shadow,a,p,refs,lambda:rt.guard(True),F.R.emit('shadow'));assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
        model=L.A.export(obj,folder/'proposal.pt.gz',p['model_bytes']);rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        rows,_=evaluation(rt,p,model);evaluation(rt,p,model,rows);result=assess(p,rows)
        result.update(model=model,update=aa,Actor_Adam=aa['Actor_Adam'],Value_Adam=0,new_policy_gradient_updates=0,new_training_navigation_samples=0,source_recovery_teacher_NOT_new_student_success=True,full_old275_six54_eight54_NOT_executed=True,fresh16=None,reload78=None,under8_gate=False)
        if not result['panel_pass']:
            for z in [obj,shadow]:P.load(z,before)
        result['final_joint_hash']=digest(P.state(*obj));state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:
            for z in [obj,shadow]:P.load(z,before)
        save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder);result=s['result'];rows=G.rows(folder);rep=G.rows(folder,'replays.jsonl');assert len(rows)==len(rep)==20
    a=E.N.S.arrays(p['recovery_row']['path']);obj=L.make(p);before=P.state(*obj);u=L.fit(obj,a,p,L.references(p),lambda:None,lambda *_:None);assert u==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
    by={r['path']:r for r in rows}
    for r in rows:
        assert not r['full'] and r['full_arrays_generated_but_not_saved'] and sha(r['path'])==r['sha256'];aa=E.N.S.arrays(r['path']);assert set(aa)=={'actions','native_counts','end_flags'} and int(aa['native_counts'].sum())==r['physics']
        for k,v in aa.items():assert digest(v)==r['array_hashes'][k]
        assert aa['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])] and not aa['end_flags'][:-1].any() and not r['administrative_censor']
    for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    for k,v in assess(p,rows).items():assert result[k]==v
    if not result['panel_pass']:P.load(obj,before)
    assert digest(P.state(*obj))==result['final_joint_hash'];steps=u['Actor_Adam'];assert G.rows(folder,'optimizer_calls.jsonl')==[[w,st,k] for w in ['live','shadow'] for k in range(steps) for st in ['BEFORE','AFTER']]
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==44
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,historical_discarded_Actor_Adam=steps,Value_Adam=0,validated_source_recovery_SHA_and_original_audit_locked=True,actual19_success_references_and_fullhistory_training_rebuilt=True,full_ActorAdam_rebuilt_and_rollback_exact=True,all20_native_fullarray_hashes_and_integrations_replay_exact=True,compact20_NOT_full120_reconstructed_offline=True,fullfield24_NOT_executed=True,physics=s['physics'],commands=s['commands']));print('T527_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
