"""Zero-native, zero-optimizer partial evidence audit of deadline-censored T338R."""
from pathlib import Path
import sys,json,time,os,signal,hashlib,base64
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t338_remaining_recovery'))
from t338r_core import *
HERE=Path(__file__).resolve().parent
PRODUCER=OUT;OUT=ROOT/'stage/runs/t338r_failure_review'

def prepare():
    assert not OUT.exists()
    files={str(f) for f in PRODUCER.rglob('*') if f.is_file()}
    original=read(PRODUCER/'protocol.json');files|=set(original['input_hashes'])
    files|={str(f) for f in HERE.glob('*.py')}
    files|={str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T338R.json')}
    p=dict(scope='T338R_ADMINISTRATIVE_DEADLINE_PARTIAL_EVIDENCE_NOT_COMPLETED_SCIENTIFIC_GATE',
        wall_seconds=180,final_bytes=128*1024,global_bytes=1<<30,min_free_bytes=40*1024**3,
        native_added=0,optimizer_calls=0,training_restart=False,input_hashes={f:K.C.sha(f) for f in sorted(files)})
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print('T338R partial review registered:180s/128KiB/0native0Adam')

def main():
    start=time.time();p=read(HERE/'protocol.json');assert not (OUT/'summary.json').exists()
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T338R review180s')));signal.alarm(175)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    s=read(PRODUCER/'summary.json');plan=read(PRODUCER/'protocol.json');launch=read(PRODUCER/'launch.json');workers=read(PRODUCER/'workers.json')
    assert s['state']=='FAILED' and s['all_workers_stopped']
    ids=[s['pid'],launch['pid'],launch['orchestrator_pid'],*workers['workers'],read(DOC/'T338R_WRAPPER.json')['pid']]
    assert all(not Path('/proc/'+str(pid)).exists() for pid in ids)
    assert s['updates']==dict(live_actor=0,shadow_actor=0) and not (PRODUCER/'optimizer_calls.jsonl').exists()
    assert read(PRODUCER/'preflight.json')['synthetic_discarded_Adam_calls']==4
    q=Actor(plan,'capg').requires_grad_(False);reference=arrays(plan['reference'])
    rows=[json.loads(v) for v in (PRODUCER/'cases.jsonl').read_text().splitlines()];assert len(rows)==4
    full=[];partial=[];total=0;drawcount=0
    def rewards_and_history(a,row):
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        steps=24000+np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.max(np.abs(a['states'][:,112]-(1-steps/40000)))<1e-7
        for k,x in enumerate(a['states']):
            assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x))<=2e-6
            d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            r=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0.)
            assert abs(r-a['rewards'][k])<1e-12 and r==a['environment_rewards'][k]
    for j,row in enumerate(rows):
        assert row['arm']=='capg' and row['role']=='training' and row['prefix']==240
        assert K.C.sha(row['path'])==row['sha256'];a=arrays(row['path'])
        assert {k:digest(v) for k,v in a.items()}==row['array_hashes']
        assert row['policy_hash']==digest(q.state_dict()) and np.array_equal(a['states'][0],reference['states'][240])
        assert np.array_equal(a['initial_native_integration'],arrays(plan['prefix_native_references']['240'])['initial_native_integration'])
        assert K.R.arrays_hash(a['final_native_integration'])==row['integration_sha256']
        assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
        assert list(map(bool,a['end_flags'][-1]))==[row['collision'],row['success'],row['timeout']]
        assert row['physics_steps']==24000+int(a['native_counts'].astype(np.int64).sum())
        z=np.random.default_rng(row['seed']).normal(size=(len(a['states']),2));assert np.array_equal(z,a['innovations'])
        with torch.no_grad():draw=q.draw(torch.tensor(a['states']),torch.tensor(z));loc,sd=q.stats(torch.tensor(a['states']))
        assert all(np.array_equal(v.numpy(),a[name]) for name,v in draw.items())
        with torch.no_grad():lp=q.logp(loc,sd,torch.tensor(a['raw_actions']),torch.tensor(a['atom']))
        assert np.array_equal(lp.numpy(),a['logp'])
        rewards_and_history(a,row);drawcount+=len(a['states'])
        full.append(dict(id=row['id'],success=row['success'],collision=row['collision'],timeout=row['timeout'],
            episode_elapsed_seconds=row['physics_steps']/500,suffix_elapsed_seconds=row['suffix_steps']/500,remaining_m=row['remaining_m']))
        folder=PRODUCER/'workers'/str(j);failure=read(folder/'failure.json');b=arrays(folder/'partial.npz')
        assert "deadline_epoch" in failure['error']
        assert not b['end_flags'].any(),'Partial replay must not be mislabeled terminal navigation failure'
        n=len(b['states']);assert 0<n<len(a['states'])
        charged=failure['actual_physics']-row['physics_steps'];stored=int(b['native_counts'].astype(np.int64).sum())
        tail=charged-24000-stored;assert 0<=tail<100
        for name,v in b.items():
            if name=='final_native_integration':continue
            limit=n+(1 if tail==0 else 0) if name.startswith('history_') else n
            if name.startswith('history_'):v=v[:limit]
            assert np.array_equal(v,a[name][:limit]),(j,name)
        rewards_and_history(b,row)
        charged=failure['actual_physics']-row['physics_steps'];stored=int(b['native_counts'].astype(np.int64).sum())
        tail=charged-24000-stored;assert 0<=tail<100
        # Tail steps were actually executed before the per-native deadline;
        # no absent decision/observation rows are invented for them.
        assert b['final_native_integration'].shape==(203,)
        partial.append(dict(worker=j,charged_physics=charged,exact_saved_suffix_decisions=n,exact_saved_suffix_physics=stored,
            unrecorded_native_tail_steps=tail,administratively_censored=True,full_replay_pass=False,final_history_frame_not_compared_if_partial_hold=tail>0,
            final_integration_sha256=digest(b['final_native_integration'])))
        total+=failure['actual_physics']
    assert total==s['physics_steps']==270334
    assert s['commands']==1+len(rows)+len(partial)==9 and s['combined_commands']==297
    assert s['combined_physics']==4449310==4178976+total
    ck=s['models']['capg'];assert K.C.sha(ck['path'])==ck['sha256']
    state=torch.load(io.BytesIO(gzip.decompress(Path(ck['path']).read_bytes())),weights_only=False,map_location='cpu')['state']
    initial={k:v for k,v in q.state_dict().items() if not k.startswith('fixed.')}
    assert digest(initial)==digest(state['actor']) and state['optimizer']['state']=={}
    safe=ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'
    assert K.C.sha(safe)=='63e63e2b444867b26622a2ab0f0c4227bc38d48a7cd2b5a3b378d9b0266006df'
    all_physics=sum(v['physics_steps'] for v in rows);new_suffix=sum(v['physics_steps']-24000 for v in rows)
    result=dict(state='COMPLETE',audit_state='PASS_PARTIAL_ONLY_PRODUCER_FAILED',pid=os.getpid(),all_workers_stopped=True,
        no_workers_spawned=True,all_producer_processes_absent=True,new_native=0,new_optimizer_calls=0,
        producer_commands=9,producer_physics=total,combined_commands=297,combined_physics=4449310,
        complete_first_executions=full,complete_success=1,complete_contact=0,complete_timeout=3,
        independently_completed_replays=0,censored_replays=partial,unrecorded_native_tail_steps=sum(v['unrecorded_native_tail_steps'] for v in partial),
        complete_actual_draws_exact=drawcount,complete_first_execution_physics=all_physics,new_policy_suffix_physics=new_suffix,
        live_Adam=0,shadow_Adam=0,discarded_preflight_Adam=4,combined_discarded_preflight_Adam=8,
        initial_policy_and_empty_Adam_exact=True,cached_tanh_data_available_but_not_optimized=True,
        original_six_gate_not_executed=True,stable_six=False,source_inputs_and_original_summary_unchanged=True,
        producer_summary_sha256=K.C.sha(PRODUCER/'summary.json'),elapsed_s=time.time()-start,
        limitations=['One sampled success is not learned policy gain or reliable advantage.','No full exact native replay completed; only recorded replay prefixes match.',
            'CAPG and tanh share four fixed noise seeds and frozen deterministic parent; distributions differ.','Training stopped at original cutoff; no next runtime restart.'])
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    save(OUT/'summary.json',result);save(DOC/'T338R_PARTIAL_AUDIT.json',result)
    assert K.S.used(OUT)<=p['final_bytes'] and K.S.used(ROOT/'stage/runs')<=p['global_bytes'];signal.alarm(0)
    print(json.dumps({k:v for k,v in result.items() if k not in ['complete_first_executions','censored_replays','limitations']}))

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/'failure.json',dict(error=traceback.format_exc(),new_native=0,new_optimizer=0))
        raise
