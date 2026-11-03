"""Only the unused third T317 human attempt, original absolute deadline.

No extra native allocation, no autonomous driver, no learning. The preview
environment is the charged recording environment and is reused on human start.
"""
from pathlib import Path
import importlib.util,json,time,os,signal,traceback,contextlib,inspect
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('stage318_original_recorder',ROOT/'stage/experiments/t317_human_demonstration/recorder.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t318_remaining_human';DOC=ROOT/'stage/docs/distance_20261103'
R.HERE,R.OUT=HERE,OUT
K=R.K
read=R.read

class App(R.App):
    def __init__(self,p,meter,start,folder):
        # Charge initialization once, before any native environment is created.
        meter.begin(28000)
        super().__init__(p,meter,start,folder)
        self.geometry_commands=0
        self.deadline=p['absolute_deadline_epoch']
        self.write_status()
    def status(self):
        x=super().status();x.update(max_attempts=1,commands=self.meter.commands,
            previous_human_attempts=2,combined_commands=6+self.meter.commands,
            combined_physics_steps=112000+self.meter.physics,
            deadline_epoch=self.p['absolute_deadline_epoch'])
        return x
    def begin(self):
        self.guard();assert not self.active and self.attempt_number==0
        assert self.preview is not None and self.preview.steps==0
        self.session,self.preview=self.preview,None
        self.attempt_number=1;self.current_file=OUT/'attempt_01.npz'
        assert not self.current_file.exists()
        self.active=True;self.state='HUMAN_RECORDING'
        self.checkpoint();self.jpg=self.view.frame(self.session);self.write_status()
    def finish_attempt(self,reason):
        super().finish_attempt(reason);self.closed=True

def verify(p):
    R.verify(p)
    old=read(ROOT/'stage/runs/t317_human_demonstration/summary.json')
    assert old['state']=='COMPLETE' and old['commands']==6 and old['physics_steps']==112000 and len(old['attempts'])==2
    assert old['main_epoch']+1800==p['absolute_deadline_epoch']
    assert K.S.used(ROOT/'stage/runs/t317_human_demonstration')+p['final_bytes']<=3*1024**2
    assert read(DOC/'T317_AUDIT.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T317.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'

def preflight():
    p=read(HERE/'protocol.json');verify(p)
    assert not (OUT/'preflight.json').exists()
    # No new geometry. Exact original Session/View were physically initialized
    # and render-checked by T317. Test the changed begin path with inert fakes.
    class Fake:
        steps=0
    a=App.__new__(App);a.active=False;a.attempt_number=0;a.preview=Fake()
    a.guard=lambda:None;a.checkpoint=lambda:None;a.write_status=lambda:None
    class View:
        def frame(self,s):assert s is a.session;return b''
    a.view=View();preview=a.preview;a.begin()
    assert a.session is preview and a.preview is None and a.attempt_number==1 and a.active
    try:a.begin()
    except AssertionError:pass
    else:raise AssertionError('Second start accepted')
    assert time.time()<p['absolute_deadline_epoch']-300
    K.C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,geometry_commands=0,Adam_calls=0,gradient_calls=0,
        changed_begin_reuses_same_environment=True,second_start_rejected=True,
        original_Session_View_reused_byte_identically=True,original_geometry_render_preflight_sha256=K.C.sha(ROOT/'stage/runs/t317_human_demonstration/preflight.json'),
        worst_reserved_global_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'],
        protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print('T318 remaining-only engineering preflight PASS,0native0geometry0Adam')

def main():
    assert not (OUT/'status.json').exists()
    p=read(HERE/'protocol.json');start=time.time();meter=app=http=None;state='FAILED';reason='STARTING'
    def alarm(*_):raise TimeoutError('Original T317 absolute deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(p['absolute_deadline_epoch']-start)-5))
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS'
        assert start<p['absolute_deadline_epoch']-300
        R.torch.set_num_threads(1);R.torch.optim.Adam.step=R.U.forbid_step
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);meter.deadline=min(meter.deadline,p['absolute_deadline_epoch']);meter.charge_payload(p['payload_bytes'])
        with (OUT/'native_factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            app=App(p,meter,start,OUT/'runtime')
            http=R.HTTPServer(('127.0.0.1',p['port']),R.Handler);http.timeout=.5;http.app=app
            while not app.closed and time.time()<app.deadline-p['final_reserve_seconds']:
                http.handle_request();app.write_status()
            if app.failure:raise RuntimeError(app.failure)
            if app.active:app.finish_attempt('ADMINISTRATIVE_COLLECTION_DEADLINE')
        state='COMPLETE';reason='HUMAN_COLLECTION_ENDED' if app.attempts else 'NO_HUMAN_ATTEMPT_EXECUTED'
    except BaseException:
        K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0 if meter is None else meter.physics))
    finally:
        signal.alarm(0)
        if http is not None:http.server_close()
        if app is not None:app.close()
        if meter is not None:meter.close()
    s=dict(state=state,stop_reason=reason,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,
        deadline_epoch=p['absolute_deadline_epoch'],commands=0 if meter is None else meter.commands,
        physics_steps=0 if meter is None else meter.physics,attempts=[] if app is None else app.attempts,replays=[] if app is None else app.replays,
        all_workers_stopped=True,workers_spawned=0,all_renderers_closed=True,Actor_updates=0,Adam_calls=0,gradient_calls=0,
        autonomous_six_metre_pass=False,automatic_training=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    s.update(combined_commands=6+s['commands'],combined_physics_steps=112000+s['physics_steps'],combined_human_attempts=2+len(s['attempts']))
    assert s['combined_commands']<=8 and s['combined_physics_steps']<=168000 and s['combined_human_attempts']<=3
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T318_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Process still exists',pid))
    inputs=[json.loads(t) for t in (OUT/'human_inputs.jsonl').read_text().splitlines()] if (OUT/'human_inputs.jsonl').exists() else []
    physics=0;successful=[]
    for row in s['attempts']:
        assert K.C.sha(row['path'])==row['sha256'];a,initial,final=R.load_arrays(row['path'])
        assert {k:K.S.digest(v) for k,v in a.items()}==row['array_hashes']
        assert K.R.arrays_hash(initial)==p['initial_geometry']['integration_sha256'] and K.R.arrays_hash(final)==row['integration_sha256']
        assert R.np.array_equal(R.np.asarray([v['action'] for v in inputs],R.np.float32).reshape(-1,2),a['actions'])
        assert R.np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['end_flags'][:-1].any()
        for k,x in enumerate(a['states']):
            assert R.np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
            # Same verified reward helper as original independent T317 audit.
            spec=importlib.util.spec_from_file_location('stage318_reward_audit',ROOT/'stage/experiments/t295_hindsight_support/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            assert abs(m.reward(d0,d1,int(a['native_counts'][k]),*map(bool,a['end_flags'][k]))-a['rewards'][k])<1e-12
        physics+=int(a['native_counts'].astype('int64').sum())
        if len(a['actions']):
            rep=next(v for v in s['replays'] if v['id']==row['id']);assert rep['all12_arrays_exact'] and rep['integration_sha256']==row['integration_sha256'] and rep['physics_steps']==row['physics_steps'];physics+=rep['physics_steps']
        if row['success'] and not row['collision'] and row['complete']:successful.append(row['id'])
    assert physics==s['physics_steps']<=56000 and s['commands']<=2
    K.C.save(dest,dict(state='PASS',native_added=0,Adam_calls=0,all120_history_actions_reward_hashes_verified=True,
        native_replay_receipts_verified=True,successful_human_demonstration_ids=successful,not_autonomous_navigation_success=True,
        combined_commands=s['combined_commands'],combined_physics_steps=s['combined_physics_steps'],combined_attempts=s['combined_human_attempts'],
        original_absolute_deadline_unchanged=True,all_original_processes_absent=True,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T318 remaining-human audit PASS,0native0Adam')

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['preflight','main','audit']);globals()[a.parse_args().mode]()
