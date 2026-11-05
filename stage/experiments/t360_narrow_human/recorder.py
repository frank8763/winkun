"""A single original-start real-human demonstration in a proven local opening."""
from pathlib import Path
import sys,importlib.util,copy,json,os,time,socket,contextlib,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t360_narrow_human';DOC=ROOT/'stage/docs/narrow_20261105'
spec=importlib.util.spec_from_file_location('stage360_human_factory',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
K,R,W,np,torch=M.K,M.R,M.W,M.np,M.torch
read,save=M.read,K.C.save
M.HERE=W.HERE=R.HERE=HERE;M.OUT=W.OUT=R.OUT=OUT;M.DOC=DOC
Handler=M.Handler


class App(W.App):
    def finish_attempt(self,reason):
        super().finish_attempt(reason)
        for row in self.attempts:row.update(source='REAL_HUMAN_ORIGINAL_START_NARROW070',human_start_decision=0,policy_prefix_decisions=0,not_autonomous_navigation=True)
        save(OUT/'attempts.json',self.attempts)


M.App=App


def prepare():
    assert not OUT.exists();p=copy.deepcopy(read(ROOT/'stage/experiments/t347_original_start_human/protocol.json'))
    q=read(ROOT/'stage/runs/t357_narrow_origin_transfer/protocol.json');g=read(ROOT/'stage/runs/t357_narrow_origin_transfer/preflight.json')['geometries'][0]['geometry']
    assert read(ROOT/'stage/runs/t359_coherent_narrow/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T359.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p.update(experiment='T360_REAL_HUMAN_NARROW070_FROM_ORIGINAL_START',port=18774,job=copy.deepcopy(q['jobs'][0]),initial_geometry=g,
        circle_shift_x_m=q['new_scene_shift_x_m'],prefix_decisions=0,prefix_seconds=0.,course_id='LEFT6_NARROW070_80S_REAL_HUMAN_FROM_START',
        max_attempts=1,max_commands=3,max_main_commands=2,max_physics=80000,wall_seconds=1200,final_reserve_seconds=240,audit_seconds=300,all_work_seconds=2400,
        final_bytes=3*1024**2,payload_bytes=1536*1024,metadata_bytes=1536*1024,episode_bytes=768*1024,preflight_geometry_commands=1,
        contract=(DOC/'T360_PLAN.md').read_text())
    files=set(p['input_hashes'])|{str(f) for f in HERE.iterdir() if f.suffix in ['.py','.html','.js']}
    files|={str(DOC/'T360_PLAN.md'),str(DOC/'T359_FINDINGS.md'),str(ROOT/'stage/runs/t359_coherent_narrow/audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T359.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(OUT/'human_availability.json',dict(confirmed_by_user=True,source='CURRENT_CHAT_EXPLICIT_AVAILABILITY_REPLY',confirmed_epoch=time.time(),original_reply='现在方便操作'))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,global_full_reservation=used+p['final_bytes'])))


def preflight():
    p=read(HERE/'protocol.json');M.install(p);R.verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
    with socket.socket() as check:check.bind(('127.0.0.1',p['port']))
    folder=OUT/'preflight';folder.mkdir();s=view=None
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create,w,reward=M.setup(folder/'native')
        try:
            s=R.Session(p,native,create,w,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('No preflight native advance')))
            assert s.steps==0 and s.env.max_steps==40000 and s.env.max_wheel==5. and K.S.digest(s.initial120)==p['initial_geometry']['initial120_sha256']
            model=s.env.model;mj=R.mujoco;c=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'real_circle_obs_9');wall=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
            gap=float(model.geom_pos[c,0]-model.geom_size[c,0]-model.geom_pos[wall,0]-model.geom_size[wall,0]);assert abs(gap-.70)<1e-6
            assert np.array_equal(model.actuator_forcerange[[s.env.id_lr_wheel,s.env.id_rr_wheel]],np.array([[-120.,120.],[-120.,120.]]))
            active=[i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in s.env.robot_body_ids and model.geom_contype[i]]
            assert sum(int(model.geom_type[i])==int(mj.mjtGeom.mjGEOM_CYLINDER) for i in active)>=4
            view=R.View(s);assert len(view.frame(s))>1000 and s.steps==0
            boundary="  if(typeof document==='undefined')return;";previous=ROOT/'stage/experiments/t347_original_start_human/control.js'
            cls=(HERE/'control.js').read_text().split(boundary)[0];assert cls==previous.read_text().split(boundary)[0]
            save(OUT/'client_contract.json',dict(state='PASS_REUSED_IDENTICAL_CLIENT_CONTROL_CLASS',source_contract=str(ROOT/'stage/runs/t347_original_start_human/client_contract.json'),
                identical_control_class_sha256=hashlib.sha256(cls.encode()).hexdigest(),new_client_test_executed=False,native_commands=0,physics_steps=0,human_actions=0))
            save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,physics_steps=0,Adam_calls=0,initial120_scene_integration_exact=True,
                full_active_tyres=True,gap_m=gap,policy_prefix_decisions=0,copied_render_native_unchanged=True,reverse_available=False))
        finally:
            if view is not None:view.close()
            if s is not None:s.close()
            w.close()
    print(json.dumps(dict(state='PREFLIGHT_PASS',geometry=1,native=0,Adam=0)))


def launch():
    p=read(HERE/'protocol.json');R.verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and read(OUT/'http_contract.json')['state']=='PASS_DUMMY_HTTP_ONLY'
    assert read(OUT/'human_availability.json')['confirmed_by_user'];assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'run.log').open('xb') as log:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage real human narrow demonstration',
        'timeout','--signal=TERM','--kill-after=1s','1200s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'recorder.py'),'main'],cwd=ROOT,
        env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+1200,all_work_deadline_epoch=epoch+2400,source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree']))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid,url=f'http://127.0.0.1:{p["port"]}/')))


def audit():
    p=read(HERE/'protocol.json');R.verify(p);s=read(OUT/'summary.json');dest=DOC/'T360_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc',str(pid)).exists()
    torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step;physics=0
    events=[json.loads(v) for v in (OUT/'request_events.jsonl').read_text().splitlines()] if (OUT/'request_events.jsonl').exists() else []
    inputs=[json.loads(v) for v in (OUT/'human_inputs.jsonl').read_text().splitlines()] if (OUT/'human_inputs.jsonl').exists() else []
    assert len({e['request_id'] for e in events})==len(events)
    for row in s['attempts']:
        assert K.C.sha(row['path'])==row['sha256'];a,initial,final=R.load_arrays(row['path'])
        assert {k:K.S.digest(v) for k,v in a.items()}==row['array_hashes'] and K.R.arrays_hash(initial)==p['initial_geometry']['integration_sha256']
        assert row['scene_sha256']==p['initial_geometry']['scene_sha256'] and row['human_start_decision']==row['policy_prefix_decisions']==0
        assert np.array_equal(a['actions'],np.asarray([e['action'] for e in events],np.float32).reshape(-1,2))
        assert np.array_equal(a['actions'],np.asarray([e['action'] for e in inputs],np.float32).reshape(-1,2))
        assert [e['decision'] for e in events]==list(range(len(a['actions']))) and [e['decision'] for e in inputs]==list(range(len(a['actions'])))
        assert int(a['native_counts'].astype(np.int64).sum())==row['physics_steps'];physics+=row['physics_steps']
        assert K.R.arrays_hash(final)==row['integration_sha256']
        if len(a['states']):
            assert K.S.digest(a['states'][0])==p['initial_geometry']['initial120_sha256'] and np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['end_flags'][:-1].any()
            steps=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]];assert np.abs(a['states'][:,112]-(1-steps/40000)).max()<1e-7
            for k,x in enumerate(a['states']):
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n;d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
                rw=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
                if c or g or t:rw+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
                assert abs(rw-a['rewards'][k])<1e-12
            rep=next(v for v in s['replays'] if v['id']==row['id']);assert rep['all12_arrays_exact'] and rep['physics_steps']==row['physics_steps'];physics+=rep['physics_steps']
    assert physics==s['physics_steps']<=80000 and s['commands']<=3 and len(s['attempts'])<=1
    save(dest,dict(state='PASS',native_added=0,optimizer_added=0,all_actions_real_human_from_start=True,full120_request_history_reward_replay_exact=True,
        commands=s['commands'],physics_steps=physics,not_autonomous_six_acceptance=True,summary_sha256=K.C.sha(OUT/'summary.json')))


if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'launch':launch,'main':M.main,'audit':audit}[sys.argv[1]]()
