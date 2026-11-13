"""Native prefix -> persisted complete snapshot -> exact causal continuation."""
from pathlib import Path
import collections,contextlib,copy,fcntl,gzip,hashlib,json,os,pickle,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t534_native_state_restore';DOC=ROOT/'stage/docs/day_20261113'
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,A,E,P,np,torch=F.G,F.L,F.E,F.P,F.np,F.torch
read,save,sha,digest=G.read,G.save,G.sha,G.digest
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def verify(p,full=False):
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<p['absolute_all_stop'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15} and shutil.disk_usage(ROOT).free>=40*2**30
    assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']<=2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f

def append(name,row):
    with (OUT/name).open('a') as h:h.write(json.dumps(row)+'\n')

def continuation(s,first):
    # Preserve every generated array, including source 120 inputs, sensor history,
    # hidden and native-step witnesses, but only the five suffix decisions.
    a=E.payload(s);n=len(a['states']);out={}
    for k,v in a.items():
        if k in ['initial_native_integration','final_native_integration','last_hidden_after','task_goal_xy','scene_sha256']:out[k]=v.copy()
        elif k.startswith('history_'):out[k]=v[first:].copy()
        elif hasattr(v,'shape') and v.shape and v.shape[0]==n:out[k]=v[first:].copy()
        else:out[k]=v.copy()
    return out

def source_check(s,a,k):
    row=s.rows[k]
    for name,val in zip(['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards'],row):assert np.array_equal(np.asarray(val),a[name][k]),(name,k)
    assert np.array_equal(s.hb[k],a['hidden_before'][k])
    assert np.array_equal(s.proposal[k]['locations'],a['locations'][k])
    for val,key in zip(s.hist[k],['history_poses','history_goals','history_times']):assert np.array_equal(val,a[key][k]),(key,k)

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.N.offline();E.set_out(OUT);E.L.install_stop_handlers()
    F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
    config=read(p['actor_configuration']);counts=0;commands=0;start=time.time();deadline=min(start+p['main_seconds'],p['absolute_native_stop']);rows=[];state='FAILED';error=None;s=w=None
    def guard(physical=False):
        E.L.check_stop();assert time.time()<deadline-5 and counts<p['max_physics'];verify(p)
    def charge():
        nonlocal counts
        counts+=1;assert counts<=p['max_physics']
    # Full source hashing is done at main/audit boundaries; native guard checks
    # storage at most once/second so per-step verification cannot dominate physics.
    last=[0.]
    def fastguard(physical=False):
        E.L.check_stop();assert time.time()<deadline-5 and counts<p['max_physics']
        if time.time()-last[0]>1.:verify(p);last[0]=time.time()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        folder=OUT/'native';folder.mkdir()
        with (OUT/'native_console.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,reward=F.field_setup(folder,p);commands+=1;original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers()
            for wi,window in enumerate(p['windows']):
                verify(p,True);r=window['row'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());first=window['first'];stop=first+p['holds'];before=counts
                job=r['request']['job'];q=A.frozen(config,r['request']['model']);assert q.phash()==r['policy_hash'];q.sample=False;q.rng=None
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m'])
                s=F.FieldSession(job,native,create,w,fastguard,charge,q);commands+=1;E.L.install_stop_handlers();assert s.steps==0 and s.env.data.time==0.
                assert np.array_equal(s.initial,a['initial_native_integration'])
                for k in range(first):s.step();source_check(s,a,k)
                assert counts-before==first*100 and not s.done()
                snapshot=s.capture();assert np.array_equal(snapshot['packet'],a['states'][first])
                blob=gzip.compress(pickle.dumps(snapshot,protocol=5),mtime=0);assert len(blob)<=p['snapshot_bytes'];snapfile=OUT/('snapshot_'+str(wi)+'.pkl.gz');snapfile.write_bytes(blob)
                snapshot_digest=digest(dict(physics=snapshot['native']['physics'],packet=snapshot['packet'],hidden=snapshot['hidden'],meta=snapshot['meta'],steps=snapshot['steps'],previous=snapshot['previous'],pn=snapshot['pn']))
                for k in range(first,stop):s.step();source_check(s,a,k)
                continuous=continuation(s,first);assert len(continuous['states'])==p['holds']
                # Restore a deserialized disk artifact, never an in-memory alias.
                saved=pickle.loads(gzip.decompress(snapfile.read_bytes()));s.restore(saved);commands+=1
                assert np.array_equal(s.packet(),saved['packet']) and np.array_equal(s.hidden.numpy(),saved['hidden'].numpy())
                for k in range(first,stop):s.step();source_check(s,a,k)
                restored=continuation(s,first)
                exact=set(continuous)==set(restored) and all(np.array_equal(continuous[k],restored[k]) for k in continuous)
                payload={kind+'__'+k:v for kind,part in [('continuous',continuous),('restored',restored)] for k,v in part.items()}
                path=OUT/('suffix_'+str(wi)+'.npz');E.N.write_npz(path,payload,p['suffix_bytes'])
                row=dict(id=window['id'],first=first,source_path=r['path'],source_sha256=r['sha256'],policy_hash=q.phash(),physics=counts-before,prefix_physics=first*100,continuous_suffix_physics=p['holds']*100,restored_suffix_physics=p['holds']*100,snapshot_path=str(snapfile),snapshot_sha256=sha(snapfile),snapshot_digest=snapshot_digest,payload_path=str(path),payload_sha256=sha(path),array_hashes={k:digest(v) for k,v in payload.items()},exact=exact,source_prefix_and_suffix_exact=True,source_midstate_really_generated=True)
                rows.append(row);append('cases.jsonl',row);save(OUT/'status.json',dict(state='NATIVE_WINDOW_COMPLETE',window=wi,physics=counts,commands=commands,epoch=time.time(),pid=os.getpid()))
                assert exact,'Persisted state restoration failed';assert q.phash()==r['policy_hash'];s.close();s=None
        assert len(rows)==4 and counts==p['max_physics'] and all(r['exact'] for r in rows);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,physics=counts,commands=commands,raw_preserved=True,no_restart=True))
    finally:
        if s:s.close()
        if w:w.close()
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=counts,commands=commands,elapsed_s=time.time()-start,error=error,result=dict(reason='PERSISTED_NATIVE_RESTORE_ALL_FOUR_EXACT' if state=='COMPLETE' else 'STATE_RESTORE_CONTRACT_FAILED_STOP',restore_gate=state=='COMPLETE',windows=rows,Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,native_parameter_directions_NOT_executed=True,navigation_test=None)))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE',error
    print('T534_NATIVE_STATE_RESTORE_PASS',flush=True)

def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    assert s['physics']==p['max_physics']==sum(r['physics'] for r in s['result']['windows']) and s['commands']==9
    for r,window in zip(s['result']['windows'],p['windows']):
        assert sha(r['snapshot_path'])==r['snapshot_sha256'] and sha(r['payload_path'])==r['payload_sha256']
        snap=pickle.loads(gzip.decompress(Path(r['snapshot_path']).read_bytes()));source=E.N.S.arrays(window['row']['path']);first=window['first']
        assert snap['steps']==first*100 and snap['pn']==100 and np.array_equal(snap['packet'],source['states'][first])
        assert np.array_equal(snap['hidden'].numpy().reshape(64),source['hidden_before'][first].reshape(64))
        assert len(snap['rows'])==first and len(snap['qstate'][2])==first
        assert digest(dict(physics=snap['native']['physics'],packet=snap['packet'],hidden=snap['hidden'],meta=snap['meta'],steps=snap['steps'],previous=snap['previous'],pn=snap['pn']))==r['snapshot_digest']
        with np.load(r['payload_path']) as z:
            for k in z.files:
                assert digest(z[k])==r['array_hashes'][k]
                if k.startswith('continuous__'):assert np.array_equal(z[k],z[k.replace('continuous__','restored__')])
            for k in ['states','next_states','actions','native_counts','poses','next_poses','rewards','hidden_before','locations']:
                assert np.array_equal(z['continuous__'+k],source[k][first:first+p['holds']]),k
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,all_persisted_snapshots_and_two_full_suffix_arrays_verified=True,all_source_prefix_metadata_hidden_and_suffix_values_checked=True,restore_gate=True,native_direction_gate=None,navigation_test=None,physics=s['physics'],commands=s['commands']));print('T534_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
