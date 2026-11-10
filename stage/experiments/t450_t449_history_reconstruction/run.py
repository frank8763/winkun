"""New bounded historical validation, not T449 continuation or new RL."""
from pathlib import Path
import copy,gzip,importlib.util,io,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t450_t449_history_reconstruction';DOC=ROOT/'stage/docs/review_20261110'
spec=importlib.util.spec_from_file_location('stage450_pinned449',ROOT/'stage/experiments/t449_offline_joint_recovery/run.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
P,torch,np=M.P,M.torch,M.np
read,save,sha,digest=P.read,P.save,P.sha,P.digest
def used(folder=None):return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert used(OUT)<=p['max_bytes'] and used()-used(OUT)+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t449_offline_joint_recovery';old=read(prior/'protocol.json');assert read(prior/'summary.json')['state']=='FAILED'
    assert read(ROOT/'stage/docs/overnight_20261109/POSTCHECK_20261110.json')['state']=='PASS_RAW_NAVIGATION_AND_FULL_HISTORY_SCOPE_ONLY'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T449.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T450_PLAN.md')}
    files|={str(prior/n) for n in ['protocol.json','summary.json','phases.json','tail_trace.json','recovery_trace.json','cases.jsonl','replays.jsonl','optimizer_calls.jsonl','corrections.json']}
    for e in read(prior/'phases.json'):files.add(e['candidate']['model']['path']);files.add(e['initial_model']['path'])
    for r in map(json.loads,(prior/'cases.jsonl').read_text().splitlines()):
        if r['full']:files.add(r['path'])
    p=dict(id='T450_T449_HISTORICAL_RECONSTRUCTION_ONLY',original_protocol=str(prior/'protocol.json'),out=str(OUT),max_bytes=3*2**20,model_bytes=2*2**20,
        main_seconds=1500,all_seconds=2400,max_native=0,new_navigation_updates=0,maximum_discarded_historical_BC_Adam=2524,maximum_discarded_historical_PPO_Actor_Adam=8,maximum_discarded_historical_PPO_V_Adam=8,
        original_failed_PPO_size=1587237,inputs={f:sha(f) for f in sorted(files)},user_authorization='Nov10 explicit start next step: complete missing historical validation before targeted328/329 repair; no old-run restart or new native.',
        scope='Exact original saved BC traces and Actor/V/matureAdam hashes for both completed phases; reconstruct unsaved PPO from source/data with independent clone, original unsaved hash cannot be verified. No policy promotion or native gate.')
    assert used()+p['max_bytes']<2**30;OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p);print('T450_REGISTERED_ZERO_NATIVE',flush=True)
def main():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'summary.json').exists();prior=Path(p['original_protocol']).parent;old=read(p['original_protocol']);start=time.time();calls=0;state='FAILED';error=None;completed=[]
    def guard():
        assert time.time()-start<p['main_seconds']-5 and used(OUT)<=p['max_bytes'] and used()-used(OUT)+p['max_bytes']<=2**30
    def progress(phase,k):save(OUT/'status.json',dict(state=phase,steps=k,historical_Adam=calls,new_native=0,epoch=time.time(),pid=os.getpid()))
    try:
        M.E.N.offline();data=M.dataset(old);groups=P.packs(data,'cuda');order=P.schedule(data,1500)
        for entry in read(prior/'phases.json'):
            obj=P.make(old);P.load(obj,P.checkpoint(entry['initial_model']['path'])['joint_state']);assert digest(P.state(*obj))==entry['initial_joint_hash'];expected=read(prior/f"{entry['phase']}_trace.json")
            corrections=P.packs([dict(id=r['job_id'],task=10,x=(a:=M.E.N.S.arrays(r['path']))['states'],y=a['teacher_actions']) for r in entry['correction_rows']],'cuda')
            for k,value in enumerate(expected):
                guard();actual=M.tail_step(obj,groups) if entry['phase']=='tail' else M.mix_step(obj,groups,order,corrections,k);calls+=1;assert actual==value,(entry['phase'],k)
                if k%32==0:progress('HISTORICAL_'+entry['phase'],k+1)
            assert digest(P.state(*obj))==entry['candidate']['model']['joint_hash'];verify(p);completed.append(dict(phase=entry['phase'],steps=len(expected),original_full_Actor_V_mature_Adam_hash_exact=True,model_sha256=sha(entry['candidate']['model']['path'])));save(OUT/'completed_phases.json',completed);del obj;torch.cuda.empty_cache()
        rows=[r for r in map(json.loads,(prior/'cases.jsonl').read_text().splitlines()) if r['role']=='PPO_TRAIN_0'];assert len(rows)==4;arrays=[M.E.N.S.arrays(r['path']) for r in rows]
        obj=M.initial(old)
        for g in obj[2].param_groups:g['lr']=old['actor_lr']
        shadow=P.make(old);P.load(shadow,P.state(*obj));a=M.anchored_ppo(obj,arrays,old,groups,order,0,guard,lambda *_:None);b=M.anchored_ppo(shadow,arrays,old,groups,order,0,guard,lambda *_:None)
        assert a==b and a['Actor_Adam']==4 and a['V_Adam']==4 and digest(P.state(*obj))==digest(P.state(*shadow));calls+=16
        record=P.export(obj,OUT/'historical_unsaved_PPO_reconstruction.pt.gz',p['model_bytes']);mem=io.BytesIO();torch.save(P.checkpoint(record['path']),mem);raw_bytes=len(mem.getvalue())
        assert raw_bytes+65536<p['model_bytes'];assert digest(P.checkpoint(record['path'])['joint_state'])==record['joint_hash']
        result=dict(state='PASS_COMPLETED_BC_HISTORY_AND_RECONSTRUCTED_UNSAVED_PPO',completed_phases=completed,discarded_historical_Adam=calls,new_native=0,new_navigation_updates=0,
            PPO_reconstruction=record,PPO_fit=a,original_unsaved_PPO_hash_NOT_available=True,original_failed_gzip_size=p['original_failed_PPO_size'],reconstructed_gzip_size_matches_original_error=record['bytes']==p['original_failed_PPO_size'],
            serialized_full_two_Adam_uncompressed_bytes=raw_bytes,raw_plus_64KiB_fits_future2MiB=True,reconstructed_policy_NOT_evaluated_or_promoted=True,old_T449_FAILED_NOT_overwritten=True,six_gate=False)
        save(OUT/'audit.json',result);state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,historical_Adam=calls,new_native=0))
    finally:
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,discarded_historical_Adam=calls,new_native=0,new_navigation_updates=0,error=error));save(OUT/'status.json',dict(state=state,epoch=time.time(),pid=os.getpid()))
    assert state=='COMPLETE',error;print('T450_HISTORICAL_VALIDATION_PASS',calls,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
