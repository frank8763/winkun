"""Independent finite log-tail re-evaluation, zero navigation and optimization."""
from pathlib import Path
import sys,os,json,time,io,signal,shutil,traceback
import stable_density as D
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t296_log_tail_support';DOC=ROOT/'stage/docs/distance_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t295_hindsight_support'))
from t295_contract import *

def save(f,v):Path(f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify(p):
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def evaluate(source,q,p):
    old=load(source['path']);z=load(source['witness_payload']);a={k.removeprefix('alternate_'):v for k,v in z.items() if k.startswith('alternate_')};n=len(a['states'])
    with torch.no_grad():
        oldmean=torch.cat([q.step(torch.tensor(x[None]))[1] for x in old['states']]);newmean=torch.cat([q.step(torch.tensor(x[None]))[1] for x in a['states']]);acts=torch.tensor(old['actions'])
        assert np.array_equal(oldmean.numpy(),old['means'])
        reset=torch.zeros(len(acts),dtype=torch.bool);reset[0]=True
        original=D.logp(oldmean,acts,torch.cat([oldmean[:1],oldmean[:-1]]),torch.cat([acts[:1],acts[:-1]]),reset).numpy()
        maximum=float(np.abs(original-old['logp']).max());assert maximum<=p['old_logp_abs_tolerance'],maximum
        ac=torch.tensor(a['actions']);rs=reset[:n]
        new=D.logp(newmean,ac,torch.cat([newmean[:1],newmean[:-1]]),torch.cat([ac[:1],ac[:-1]]),rs).numpy()
    delta=new-original[:n];prefix=delta.cumsum();assert np.isfinite(prefix).all()
    arrays=dict(original_stable_logp=original,newgoal_stable_logp=new,newgoal_means=newmean.numpy(),prefix_log_ratio=prefix)
    row=dict(id=source['id'],source_reset_id=source['source_reset_id'],newgoal_density_invertible=True,success=source['alternate_outcome']['success'],collision=source['alternate_outcome']['collision'],timeout=source['alternate_outcome']['timeout'],censored=source['alternate_outcome']['censored'],original_logp_record_max_difference=maximum,original_goal_identity_ratio_exactly_one=True,terminal_log_weight=float(prefix[-1]),prefix_log_min=float(prefix.min()),prefix_log_max=float(prefix.max()),max_abs_conditional_log_ratio=float(np.abs(delta).max()),auxiliary_environment_G0=float(S.monte_carlo(a['environment_rewards'],a['native_counts'])[0]),decisions=n)
    return arrays,row

def screen(p,rows):
    groups=[]
    for reset in p['reset_ids']:
        rr=[r for r in rows if r['source_reset_id']==reset];assert len(rr)==4;w=weights([r['terminal_log_weight'] for r in rr]);passed=all(not r['censored'] for r in rr) and w['ESS']>=p['ESS_min'] and w['max_weight']<=p['max_normalized_weight'] and max(abs(v) for v in w['log_weights'])<=p['max_abs_terminal_log_weight']
        groups.append(dict(reset_id=reset,passed=bool(passed),weights=w))
    return dict(passed=all(g['passed'] for g in groups),groups=groups)

def preflight():
    assert not (OUT/'preflight.json').exists() and not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());verify(p);torch.set_num_threads(1)
    result=D.reference_contract();assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes'];save(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,reference=result));print(json.dumps(result))

def main():
    assert not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());start=time.time();rows=[];state='COMPLETE'
    def alarm(signum,frame):raise TimeoutError('T296 wall budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    try:
        verify(p);torch.set_num_threads(1);q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash']
        for i,s in enumerate(p['sources']):
            save(OUT/'status.json',dict(state='RECOMPUTING_LOG_TAILS',pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),records=i,native_steps=0,optimizer_calls=0,Actor_updates=0))
            arrays,row=evaluate(s,q,p);buf=io.BytesIO();np.savez_compressed(buf,**arrays);b=buf.getvalue();assert len(b)<=p['episode_bytes'];f=OUT/(s['id']+'.npz');f.write_bytes(b);row.update(path=str(f),sha256=C.sha(f),array_hashes={k:S.digest(v) for k,v in arrays.items()});rows.append(row);save(OUT/'results.json',rows);verify(p)
        assert S.digest(q.state_dict())==p['policy_hash']
    except BaseException:
        state='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),records_completed=len(rows),native_steps=0,optimizer_calls=0,Actor_updates=0))
    finally:signal.alarm(0)
    result=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,records=len(rows),support_screen=screen(p,rows) if len(rows)==8 else None,native_steps=0,optimizer_calls=0,Actor_updates=0,policy_hash=p['policy_hash'],own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),six_metre_gate_evaluated=False,seven_or_eight_started=False,limitation=p['scientific_scope']);save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result))

def audit():
    dest=DOC/'T296_AUDIT.json';assert not dest.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE';verify(p);torch.set_num_threads(1)
    for pid in [s['pid'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused PID',pid))
    reference=D.reference_contract();q=policy(json.loads(Path(p['source_protocol']).read_text())).eval().requires_grad_(False);rows=json.loads((OUT/'results.json').read_text());again=[]
    for source,row in zip(p['sources'],rows):
        a,r=evaluate(source,q,p);assert C.sha(row['path'])==row['sha256'];saved=load(row['path']);assert all(np.array_equal(v,saved[k]) and S.digest(v)==row['array_hashes'][k] for k,v in a.items());assert r=={k:v for k,v in row.items() if k not in ['path','sha256','array_hashes']};again.append(r)
    assert screen(p,again)==s['support_screen'] and len(again)==8 and S.digest(q.state_dict())==p['policy_hash'];verify(p)
    save(dest,dict(state='PASS',native_added=0,optimizer_calls=0,Actor_updates=0,original_policy_and_all_input_SHA_verified=True,all8_original_and_newgoal_AR_probabilities_and_prefix_weights_recomputed=True,reference=reference,screen=s['support_screen'],all_original_processes_absent=True,summary_sha256=C.sha(OUT/'summary.json')));print('T296 independent audit PASS zero native/optimizer')

if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    elif sys.argv[1:]==['--audit']:audit()
    else:assert not sys.argv[1:];main()
