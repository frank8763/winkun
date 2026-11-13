"""Frozen completion of 22 missing first-pass cases, no new policy updates."""
from pathlib import Path
import copy,fcntl,io,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t541_frozen_result_recovery';DOC=ROOT/'stage/docs/day_20261113'
sys.path.insert(0,str(ROOT/'stage/experiments/t539_terminal_success_pg'));import t539_runtime as Previous
sys.path.insert(0,str(ROOT/'stage/experiments/common'));from native_case_record import envelope
F,G,E,P,np,torch=Previous.F,Previous.G,Previous.E,Previous.P,Previous.np,Previous.torch
read,save,sha,digest=Previous.read,Previous.save,Previous.sha,Previous.digest
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
    for f,h in root['inputs'].items():assert sha(f)==h,f
def configuration(p):
    c=Previous.configuration(read(p['source_registration']));c.update(p)
    c['out']=str(OUT);c['initial']=p['candidate'];c['evaluation_jobs']=c['old8_jobs']+c['field_jobs'];c['remaining_jobs']=c['evaluation_jobs'][10:]
    assert c['out']==str(OUT) and c['initial']['sha256']==p['candidate']['sha256']
    return c
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
def pack(a):
    assert (a['native_counts'][:-1]==100).all() and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
    f=io.BytesIO();np.savez(f,actions=a['actions'].astype(np.float32),last_native_count=a['native_counts'][-1:].astype(np.uint8),terminal_flags=a['end_flags'][-1].astype(np.uint8));return f.getvalue()
def unpack(path):
    with np.load(path) as z:
        actions=z['actions'];n=len(actions);counts=np.full(n,100,np.uint8);counts[-1]=z['last_native_count'][0];flags=np.zeros((n,3),np.uint8);flags[-1]=z['terminal_flags']
    return dict(actions=actions,native_counts=counts,end_flags=flags)
def combined(p,rows):
    old=G.rows(Path(p['previous_cases']).parent);assert len(old)==10 and len(rows)==22
    whole=old+rows;ids={r['job_id'] for r in whole[8:] if r['field_detail']['field_qualified_success']};before=set(p['baseline_field_success_ids'])
    return dict(reason='FROZEN_ALL32_FIRSTPASS_COMPLETE_EXACT_PENDING',old8=G.R.M.tally(whole[:8]),field24=G.R.M.tally(whole[8:]),field_success_ids=sorted(ids),gained_ids=sorted(ids-before),lost_ids=sorted(before-ids),complete_original_start_first_pass=True,independent_native_exact32_NOT_executed=True,accepted=False,promoted=False,Actor_updates=0,full_old275_six54_eight54_fresh_reload_NOT_executed=True)
def main():
    root=read(OUT/'protocol.json');p=configuration(root);F.R.setup(OUT);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state='FAILED';error=None;rows=[];result=None
    try:
        rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for off in range(0,22,2):
            batch=p['remaining_jobs'][off:off+2];rt.role='FROZEN_REMAINING22_FIRST_PASS';rt.guard();assert sum(rt.counts)+sum(j['case']['step_cap'] for j in batch)<=p['max_physics'];rt.commands+=len(batch)
            requests=[dict(model=p['candidate']['path'],job=j,seed=None,query=False,full_path=None) for j in batch]
            for (a,r),job in zip(rt.pool.request(requests),batch):
                # Preserve worker outcome and actual compact trajectory BEFORE
                # caller-specific validation or optional derived diagnostics.
                index=10+len(rows)
                save(OUT/('raw_result_'+str(index)+'.json'),dict(request_job_id=job['id'],native_result=r))
                path=OUT/('case_'+str(index)+'.npz');blob=pack(a);assert len(blob)<=p['compact_bytes']
                with path.open('xb') as h:h.write(blob);h.flush();os.fsync(h.fileno())
                assert r['policy_hash']==p['candidate']['policy_hash'];r=envelope(job,r,a,p['field_limit_m']);detail=r['field_detail']
                detail['terminal_contacts_sha256']=digest(r['terminal_contacts'])
                row={k:r[k] for k in ['success','collision','timeout','administrative_censor','physics','remaining_m','array_hashes','integration_sha256','policy_hash']}
                row.update(complete=True,job_id=job['id'],path=str(path),sha256=sha(path),field_detail=detail,model_sha256=p['candidate']['sha256'],full120_generated_but_NOT_saved=True,compact_native_counts_reconstructible=True)
                with (OUT/'cases.jsonl').open('a') as h:h.write(json.dumps(row)+'\n')
                rows.append(row)
        assert len(rows)==22;result=combined(p,rows);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result,completed_new_cases=len(rows)));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error
def audit():
    p=configuration(read(OUT/'protocol.json'));F.R.setup(OUT);verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();G.stopped(OUT)
    rows=G.rows(OUT);assert len(rows)==22
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=unpack(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].sum())==r['physics'] and a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])] and not r['administrative_censor']
    assert s['result']==combined(p,rows) and s['physics']==sum(r['physics'] for r in rows)<=p['max_physics'] and s['commands']==26
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,source_candidate_unchanged=True,all32_firstpass_summary_verified=True,compact_flags_counts_actions_SHA_checked=True,unsaved_full120_NOT_rebuilt_offline=True,independent_native_exact32_NOT_executed=True,physics=s['physics'],commands=s['commands']));print('T541_FROZEN_FIRSTPASS_AUDIT_PASS_NOT_PROMOTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
