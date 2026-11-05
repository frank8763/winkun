from night_common import *
import ast,datetime,zoneinfo

def main():
    assert not OUT.exists();DOC.mkdir(exist_ok=True)
    assert read(DOC/'STORAGE_AUDIT.json')['state']=='PASS';assert read(ROOT/'stage/runs/t383_cold_night_reserve/summary.json')['state']=='COMPLETE'
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    for pid in [249810,249812,249813,249820,249821,260180,260181]:assert not Path('/proc',str(pid)).exists()
    source=ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json';base=read(source)
    inputs=[source,*HERE.glob('*.py'),ROOT/'stage/experiments/t382_receding_human_teacher/run.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py',
        OLD/'mixed.npz',OLD/'predictions.npz',OLD/'summary.json',OLD/'audit.json',DOC/'PLAN.md',DOC/'METHODS_AND_LIMITS.md',DOC/'STORAGE_AUDIT.json',
        ROOT/'stage/docs/long_distance_20261101/release_receipt_T382.json']
    ids=dict(learning=1793919600.,native=1793923200.,all=1793925000.)
    p=dict(id='T384_T386_OFFLINE_NIGHT_CHAIN',source_protocol=str(source),inputs={str(f):sha(f) for f in sorted(set(inputs))},
        learning_deadline=ids['learning'],native_deadline=ids['native'],all_deadline=ids['all'],
        limits=dict(physics=46000000,commands=9500,live_Adam=1000,shadow_Adam=1000,audit_discarded_Adam=1000,global_bytes=1024**3,min_free_bytes=40*1024**3),
        root_bytes=256*1024,teacher_specs=[dict(id='T384A',folder='teacher_A',mechanism='SHIFT_SELECTED'),dict(id='T385B',folder='teacher_B',mechanism='HUMAN_ANCHOR')],
        teacher=dict(source_protocol=str(source),seed=20261105382,noise_sigma=[.2,.4],max_replans=306,final_bytes=int(2.5*1024**2),
            main_seconds=10000,audit_seconds=300,max_physics=8000000,max_commands=4300,full_bytes=512*1024,prediction_bytes=1792*1024),
        student=dict(source_protocol=str(source),final_bytes=12*1024**2,main_seconds=5400,audit_seconds=900,max_physics=30000000,max_commands=900,
            live_steps=1000,shadow_steps=1000,audit_steps=1000,lr=.0001,fit='teacher.5+human361.25+human364.25+cyclicold211.1',trainable='encoder/core/head/adapter6864; frozenbase/futurehead'),
        offline_runtime=True,remote_upload='PENDING_UNTIL_RECONNECTED',seven_eight_implemented=False,
        rules='A complete nonGOAL permits B once; any numerical/source/evidence/restore failure stops entire chain; teacherGOAL+exactaudit+manifest before one student; noseedgrid or retries')
    used=K.S.used(ROOT/'stage/runs');reservation=p['root_bytes']+2*p['teacher']['final_bytes']+p['student']['final_bytes']
    assert used+reservation<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET_AUDIT.json',dict(state='FULL_WORST_CASE_RESERVATION_PASS',global_before=used,reservation_bytes=reservation,global_full_reserved=used+reservation,global_limit=1024**3,
        A_B_max_physics=16000000,student_max_physics=30000000,total_physics=46000000,A_B_max_commands=8600,student_max_commands=900,
        teacher_per_pass_physics=9400+306*12100,teacher_two_pass_commands=2*(1+306*7),
        full_worst_wall=2*(10000+300+120)+5400+900+120,absolute_stop_Beijing='2026-11-06 08:30',learning_stop='07:00',native_stop='08:00'))
    print('REGISTERED_OFFLINE_CHAIN')

def preflight():
    offline();p=read(HERE/'protocol.json');assert not (OUT/'preflight.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    base=science(p);configure(OUT);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    native,create,w,job=R.setup(base);s=None;calls=0
    try:
        s=R.Session(job,native,create,w,lambda **_:None,lambda:(_ for _ in ()).throw(AssertionError('zero native')),R.actor(base));snapshot=s.capture();s.restore(snapshot)
        oldp=load_npz(OLD/'predictions.npz');selected=choose(oldp['costs'][54],oldp['flags'][54],oldp['lengths'][54]);assert selected==0
        assert oldp['flags'][54,5,19,0] and not oldp['flags'][54,0].any()
        human=S.arrays(base['demonstration'])['actions'];assert equal(refseq(human,400)[-1],human[-1])
        proposal,_=C.proposal(oldp['controls'][54,0],54,p['teacher']);assert equal(proposal,oldp['controls'][54,:5])
        rng=np.random.default_rng(384);compact_=compact(p['teacher'])
        for a in compact_.values():
            if np.issubdtype(a.dtype,np.floating):a[:]=rng.normal(size=a.shape)
            else:a[:]=rng.integers(0,127,size=a.shape,dtype=np.uint8)
        predbytes=len(R.blob(compact_));assert predbytes<1792*1024
        q,opt=learner(base);q2,opt2=learner(base);assert len([x for x in q.parameters() if x.requires_grad])==12
        x=torch.tensor(S.arrays(base['demonstration'])['states'][:12]);b=q.base_actions(x);g=[dict(x=x,b=b,target=torch.tensor(human[:12],dtype=torch.float64),first=0,weight=1.)]
        before=learned_state(q,opt);t1=fit(q,opt,g,2,lambda:None,lambda *_:None);calls+=2;t2=fit(q2,opt2,g,2,lambda:None,lambda *_:None);calls+=2
        assert equal(t1,t2) and digest(learned_state(q,opt))==digest(learned_state(q2,opt2))
        q.load_learned(before['learned']);opt.load_state_dict(before['optimizer']);assert digest(learned_state(q,opt))==digest(before)
        with torch.no_grad():
            seq=q.sequence(x,b);h=None;loc=[]
            for row in x:_,v,h,_=q.step(row[None],h);loc.append(v[0])
            error=float((seq-torch.stack(loc)).abs().max());assert error<1e-10
        try:socket.create_connection(('example.com',443));raise AssertionError('external network was allowed')
        except AssertionError as e:assert 'external network disabled' in str(e)
        report=dict(state='PASS',native=0,geometry_commands=1,discarded_Adam=calls,snapshot_Witness_history_exact=True,
            locked_T382_first5_controls_and_selection0_exact=True,prediction_worst_incompressible_bytes=predbytes,student_parameters=6864,
            clone_Adam_live_shadow_and_rollback_exact=True,sequence_step_error=error,external_network_blocked=True,loopback_only_for_local_multiprocessing=True,
            actual_teacher_first5_is_historical_notnew=True)
        save(OUT/'preflight.json',report);save(DOC/'LOCAL_READINESS.json',report)
    except BaseException:
        save(OUT/'preflight_error.json',dict(error=traceback.format_exc(),native=0,discarded_Adam=calls));raise
    finally:
        if s:s.close()
        w.close()
    print('OFFLINE_PREFLIGHT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
