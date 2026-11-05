from night_common import *
from search import controls,candidates,objective,compact_search
import ast,multiprocessing as mp
def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    prior=ROOT/'stage/docs/overnight_20261105/FINAL_REPORT.json';old=read(prior)
    assert old['complete'] and old['student']=='NOT_RUN_TEACHER_GATE' and len(old['completed'])==2
    for pid in [read(ROOT/'stage/runs/t384_offline_elite_chain/launch.json')['pid'],old['pid']]:assert not Path('/proc',str(pid)).exists()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    source=ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json';base=read(source)
    inputs=[source,*HERE.glob('*.py'),ROOT/'stage/experiments/t381_native_mppi_teacher/run.py',ROOT/'stage/experiments/t382_receding_human_teacher/run.py',R.SOURCE/'centre.npz',R.SOURCE/'candidate.pt.gz',prior,
        DOC/'PLAN.md',DOC/'FULL_HORIZON_METHODS.md',ROOT/'stage/docs/long_distance_20261101/release_receipt_T384-NIGHT.json']
    p=dict(id='T387_FULL_TASK_CEM_OFFLINE_REVISION',source_protocol=str(source),inputs={str(f):sha(f) for f in sorted(set(inputs))},learning_deadline=1793919600.,native_deadline=1793923200.,all_deadline=1793925000.,
        limits=dict(physics=46000000,commands=9500),prior_actual=dict(physics=old['actual']['physics'],commands=old['actual']['commands']+1),root_bytes=256*1024,
        search_specs=[dict(id='T387C',folder='action_residual',representation='ACTION_RESIDUAL',initial_sd=[.12,.35],minimum_sd=[.03,.08]),
                      dict(id='T388D',folder='phase_warp',representation='PHASE_WARP',initial_sd=[.12,1.],minimum_sd=[.03,.2])],
        search=dict(source_protocol=str(source),seed=20261105387,iterations=6,candidates=12,elite=3,alpha=.5,knots=16,prefix_holds=20,remaining_holds=380,
            final_bytes=int(3.5*1024**2),main_seconds=9000,audit_seconds=300,max_physics=6000000,max_commands=160),
        student=dict(source_protocol=str(source),final_bytes=12*1024**2,main_seconds=5400,audit_seconds=900,max_physics=30000000,max_commands=900,live_steps=1000,shadow_steps=1000,audit_steps=1000,lr=.0001,
            source_rule='Successful native teacher→.5teacher+.25eachhuman+.1cyclicold; otherwise two verified human groups .5/.5 and NO old target, full currentbody6864, not failureteacherlabels'),
        offline=True,remote='PENDING_UNTIL_RECONNECTED',new_RL_updates=0,seven_eight_implemented=False)
    used=K.S.used(ROOT/'stage/runs');reserve=p['root_bytes']+2*p['search']['final_bytes']+p['student']['final_bytes'];assert used+reserve<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET_AUDIT.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reservation_bytes=reserve,total_reserved=used+reserve,limit=1024**3,
        search_each_per_pass_worst=2000+72*38000+38000,search_each_two_pass_worst=2*(2000+72*38000+38000),each_commands=2*(1+72+1),
        night_including_previous_actual_physics=old['actual']['physics']+2*6000000+30000000,night_command_max=old['actual']['commands']+1+320+900,
        full_worst_wall=25260,stop_Beijing='2026-11-06 08:30'))
    print('REGISTERED_FULL_TASK_OFFLINE_CHAIN')
def preflight():
    offline();p=read(HERE/'protocol.json');assert not (OUT/'preflight.json').exists();base=science(p);configure(OUT);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f,h in p['inputs'].items():assert sha(f)==h,f
    native,create,w,job=R.setup(base);s=None;calls=0
    try:
        s=R.Session(job,native,create,w,lambda **_:None,lambda:(_ for _ in ()).throw(AssertionError('zero physics')),R.actor(base));snapshot=s.capture();s.restore(snapshot)
        human=S.arrays(base['demonstration'])['actions'];zero=np.zeros((16,2));data_=compact_search();rng=np.random.default_rng(387)
        for a in data_.values():
            if np.issubdtype(a.dtype,np.floating):a[:]=rng.normal(size=a.shape)
            elif a.dtype==np.uint16:a[:]=rng.integers(0,380,size=a.shape,dtype=np.uint16)
            else:a[:]=rng.integers(0,256,size=a.shape,dtype=np.uint8)
        predbytes=len(R.blob(data_));assert predbytes<=2176*1024
        for spec_ in p['search_specs']:
            sp=dict(p['search'],**spec_);u=controls(zero,human,sp);assert u.shape==(380,2) and np.abs(u).max()<=1
            assert equal(u,human[np.minimum(20+np.arange(380),len(human)-1)])
            theta=candidates(zero,np.broadcast_to(sp['initial_sd'],(16,2)),None,0,sp);assert theta.shape==(12,16,2) and equal(theta,candidates(zero,np.broadcast_to(sp['initial_sd'],(16,2)),None,0,sp))
        q,opt=learner(base);q2,opt2=learner(base);x=torch.tensor(S.arrays(base['demonstration'])['states'][:12]);b=q.base_actions(x);g=[dict(x=x,b=b,target=torch.tensor(human[:12],dtype=torch.float64),first=0,weight=1.)]
        before=learned_state(q,opt);a=fit(q,opt,g,2,lambda:None,lambda *_:None);calls+=2;b_=fit(q2,opt2,g,2,lambda:None,lambda *_:None);calls+=2
        assert equal(a,b_) and digest(learned_state(q,opt))==digest(learned_state(q2,opt2));q.load_learned(before['learned']);opt.load_state_dict(before['optimizer']);assert digest(learned_state(q,opt))==digest(before)
        # Verify the human-only fallback cannot include a failed teacher source.
        demo=data(q,base,None,lambda:None);assert len(demo)==2 and [g['weight'] for g in demo]==[.5,.5]
        assert objective(np.zeros((1,2)),1,[0,1,0],.1)<objective(np.zeros((1,2)),1,[1,0,0],.1)
        report=dict(state='PASS',native=0,geometry=1,discarded_Adam=calls,snapshot_Witness_and_previous_exact=True,full_search_incompressible_bytes=predbytes,
            two_representations_zero_coeff_exact_human_commands=True,CEM_antithetic_reproducible=True,student6864_clone_Adam_and_rollback_exact=True,verified_human_only_fallback_sources_exact=True,
            runtime_network_disabled=True,previous_worker_startup_wrapper_reused_SHA_locked=True)
        save(OUT/'preflight.json',report);save(DOC/'LOCAL_READINESS.json',report)
    except BaseException:save(OUT/'preflight_error.json',dict(error=traceback.format_exc(),native=0,discarded_Adam=calls));raise
    finally:
        if s:s.close()
        w.close()
    print('FULL_TASK_OFFLINE_PREFLIGHT_PASS')
if __name__=='__main__':globals()[sys.argv[1]]()
