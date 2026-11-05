"""Actual-human prefix, early handoff, preserved native clock and history."""
from pathlib import Path
import sys,importlib.util,copy,json,time,os
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t363_early_learner_correction';DOC=ROOT/'stage/docs/narrow_20261105'
spec=importlib.util.spec_from_file_location('stage363_original_human_engine',ROOT/'stage/experiments/t360_narrow_human/recorder.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
K,R,W,M,np,torch=V.K,V.R,V.W,V.M,V.np,V.torch
read,save=V.read,V.save
V.HERE=M.HERE=W.HERE=R.HERE=HERE;V.OUT=M.OUT=W.OUT=R.OUT=OUT;V.DOC=M.DOC=DOC
Handler=V.Handler


class App(W.App):
    def begin(self):
        super().begin();self.state='HISTORICAL_HUMAN_PREFIX';self.write_status()
        a,initial,_=R.load_arrays(self.p['reference']);s=self.session;n=self.p['prefix_decisions']
        assert np.array_equal(s.initial,initial)
        for k in range(n):
            s.step(a['actions'][k],int(a['native_counts'][k]));assert not(s.term or s.trunc)
        actual=s.arrays()
        for name,v in actual.items():
            stop=n+1 if name.startswith('history_') else n
            assert np.array_equal(v,a[name][:stop]),name
        assert s.steps==2000 and len(s.rows)==20
        self.checkpoint();self.jpg=self.view.frame(s);self.state='HUMAN_RECORDING'
        blob=__import__('io').BytesIO();np.savez_compressed(blob,handoff_native_integration=K.R.integration(s.env),handoff_full120=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn))
        (OUT/'handoff.npz').write_bytes(blob.getvalue())
        save(OUT/'handoff.json',dict(steps=s.steps,sim_time_s=4.,remaining_sim_s=76.,
            actual_learner_prefix_reference=self.p['reference'],all12_history_prefix_exact=True,
            native_integration_sha256=K.R.arrays_hash(K.R.integration(s.env)),no_clock_goal_history_reset=True))
        self.write_status()

    def finish_attempt(self,reason):
        super().finish_attempt(reason)
        for row in self.attempts:row.update(source='ACTUAL_T362_LEARNER_PREFIX4_THEN_REAL_HUMAN',human_start_decision=20,
            policy_prefix_decisions=20,historical_human_prefix_decisions=0,not_autonomous_navigation=True)
        save(OUT/'attempts.json',self.attempts)


M.App=App


def prepare():
    assert not OUT.exists();assert read(ROOT/'stage/runs/t362_narrow_demonstration_learning/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T362.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/experiments/t360_narrow_human/protocol.json'))
    rows=read(ROOT/'stage/runs/t362_narrow_demonstration_learning/cases.json')
    r=next(v for v in rows if v['role']=='final' and v['job_id']=='narrow070_dev_0')
    assert r['full_arrays_saved'] and r['collision'] and not r['success']
    p.update(experiment='T363_EARLY_ACTUAL_LEARNER_STATE_HUMAN_CORRECTION',port=18776,prefix_decisions=20,prefix_seconds=4.,reference=r['path'],
        source_model=read(ROOT/'stage/runs/t362_narrow_demonstration_learning/summary.json')['model']['path'],
        course_id='LEFT6_NARROW070_80S_LEARNER_PREFIX4_REAL_CORRECTION',contract=(DOC/'T363_PLAN.md').read_text())
    files=set(p['input_hashes'])|{str(f) for f in HERE.iterdir() if f.suffix in ['.py','.html','.js']}
    files|={r['path'],str(DOC/'T363_PLAN.md'),str(DOC/'T362_FINDINGS.md'),str(ROOT/'stage/runs/t362_narrow_demonstration_learning/audit.json'),p['source_model']}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(OUT/'human_availability.json',dict(confirmed_by_user=True,source='CURRENT_CHAT_ONGOING_REAL_HUMAN_CORRECTION',confirmed_epoch=time.time()))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,prefix_sim_s=4.,human_sim_s=76.)))


def preflight():
    V.preflight();p=read(HERE/'protocol.json');s=read(OUT/'preflight.json')
    a,initial,_=R.load_arrays(p['reference']);assert K.C.sha(p['reference'])==p['input_hashes'][p['reference']]
    assert len(a['states'])==225 and np.sum(a['native_counts'][:20].astype(int))==2000
    assert not a['end_flags'][:20].any() and np.array_equal(a['states'][1:20],a['next_states'][:19])
    s.update(planned_actual_learner_prefix_decisions=20,planned_human_suffix_seconds=76.,source_prefix_array_SHA_verified=True,
        actual_prefix_native_replay_deferred_to_explicit_user_start=True,policy_prefix_decisions=20)
    save(OUT/'preflight.json',s)


def audit():
    p=read(HERE/'protocol.json');R.verify(p);s=read(OUT/'summary.json');dest=DOC/'T363_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc',str(pid)).exists()
    torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step;a0,initial0,_=R.load_arrays(p['reference']);physics=0
    ev=[json.loads(v) for v in (OUT/'request_events.jsonl').read_text().splitlines()] if (OUT/'request_events.jsonl').exists() else []
    inp=[json.loads(v) for v in (OUT/'human_inputs.jsonl').read_text().splitlines()] if (OUT/'human_inputs.jsonl').exists() else []
    assert len({e['request_id'] for e in ev})==len(ev)
    for row in s['attempts']:
        assert K.C.sha(row['path'])==row['sha256'];a,initial,final=R.load_arrays(row['path']);n=20
        assert {k:K.S.digest(v) for k,v in a.items()}==row['array_hashes'] and np.array_equal(initial,initial0)
        for name,v in a.items():assert np.array_equal(v[:n+1 if name.startswith('history_') else n],a0[name][:n+1 if name.startswith('history_') else n]),name
        assert np.array_equal(a['actions'][n:],np.asarray([e['action'] for e in ev],np.float32).reshape(-1,2))
        assert np.array_equal(a['actions'][n:],np.asarray([e['action'] for e in inp],np.float32).reshape(-1,2))
        assert [e['decision'] for e in ev]==list(range(n,len(a['actions']))) and [e['decision'] for e in inp]==list(range(n,len(a['actions'])))
        assert not a['end_flags'][:-1].any() and np.array_equal(a['states'][1:],a['next_states'][:-1])
        steps=np.r_[0,np.cumsum(a['native_counts'].astype(int))[:-1]];assert np.abs(a['states'][:,112]-(1-steps/40000)).max()<1e-7
        for k,x in enumerate(a['states']):
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            nn=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**nn;d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
            rw=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
            if c or g or t:rw+=K.R.GAMMA**(nn-1)*(float(g)-float(c)-.5*float(t))
            assert abs(rw-a['rewards'][k])<1e-12
        assert int(a['native_counts'].astype(int).sum())==row['physics_steps'] and K.R.arrays_hash(final)==row['integration_sha256']
        rep=next(r for r in s['replays'] if r['id']==row['id']);assert rep['all12_arrays_exact'] and rep['physics_steps']==row['physics_steps'];physics+=2*row['physics_steps']
    assert physics==s['physics_steps']<=80000 and s['commands']<=3
    save(dest,dict(state='PASS',new_native=0,new_optimizer=0,actual_learner_prefix_exact_not_human_labels=True,new_suffix_real_unique_requests_exact=True,
        full120_history_clock_reward_native_replay_exact=True,commands=s['commands'],physics_steps=physics,not_autonomous_or_designated_throat_pass=True))


if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'launch':V.launch,'main':M.main,'audit':audit}[sys.argv[1]]()
