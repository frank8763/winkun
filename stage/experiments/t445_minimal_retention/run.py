"""Two representative frozen pairs after T444 I/O failure; shared original caps."""
from pathlib import Path
import copy,gzip,hashlib,importlib.util,json,os,subprocess,sys,time,traceback,contextlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage445_pinned_failed444',ROOT/'stage/experiments/t444_paired_retention/run.py')
K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
L,M,C,E,torch,np=K.L,K.M,K.C,K.E,K.torch,K.np
read,save,sha,digest=K.read,K.save,K.sha,K.digest
OUT=ROOT/'stage/runs/t445_minimal_retention';DOC=K.DOC;OLD=ROOT/'stage/runs/t444_paired_retention'
K.HERE=HERE;K.OUT=OUT

def verify(p,full=False):
    K.verify(p,full)
    assert K.used(OLD)+K.used(OUT)<=10*2**20
    assert read(OLD/'summary.json')['physics']+p['max_physics']<=800000
    assert read(OLD/'summary.json')['commands']+p['max_commands']<=80
    assert time.time()<p['inherited_all_stop']

def ram_model(p):
    import mujoco as mj
    raw=gzip.decompress(Path(p['geometry_original_gzip']).read_bytes());assert hashlib.sha256(raw).hexdigest()==p['geometry_original_sha256']
    return mj.MjModel.from_binary_path('stage444.mjb',assets={'stage444.mjb':raw})

def install_geometry(p):
    import mujoco as mj
    original=ram_model(p)
    def validate(model,filename,buffer):
        # Only geometry relevant to forward kinematics/distances is compared;
        # renderer textures need not be reserialized or written to disk.
        assert model.nq==original.nq and model.ngeom==original.ngeom
        for attr in ['geom_type','geom_bodyid','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity',
                     'body_parentid','body_pos','body_quat','jnt_type','jnt_bodyid','jnt_pos','jnt_axis','jnt_qposadr','qpos0']:
            assert np.array_equal(getattr(model,attr),getattr(original,attr)),attr
    mj.mj_saveModel=validate

class CheckpointPipe:
    """Preserve a completed full record before sending it through the pipe."""
    def __init__(self,pipe,p,index):self.pipe=pipe;self.p=p;self.index=index;self.request=None
    def recv(self):self.request=self.pipe.recv();return self.request
    def send(self,value):
        if isinstance(value,tuple):
            a,r=value;j=self.request['job'];kind=self.request['kind'];path=OUT/'episodes'/f'{kind}_{j["id"]}.npz'
            path.parent.mkdir(exist_ok=True);blob=E.N.R.blob(E.N.S.packed(a)) if hasattr(E.N.R,'blob') else None
            if blob is None:
                import io
                b=io.BytesIO();np.savez_compressed(b,**E.N.S.packed(a));blob=b.getvalue()
            assert len(blob)<=self.p['full_bytes']
            if path.exists():assert path.read_bytes()==blob
            else:path.write_bytes(blob)
            with (OUT/'workers'/str(self.index)/'completed.jsonl').open('a') as f:
                f.write(json.dumps(dict(job_id=j['id'],kind=kind,physics=r['physics'],sha256=sha(path),path=str(path),array_digest=r['array_digest']))+'\n')
        self.pipe.send(value)
    def close(self):self.pipe.close()

def worker(pipe,index,p,counts):
    install_geometry(p)
    K.worker(CheckpointPipe(pipe,p,index),index,p,counts)

class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();child=ctx.Process(target=worker,args=(b,i,p,counts));child.start();b.close();self.children.append(child);self.pipes.append(a)
            save(Path(p['out'])/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]))
            self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
        except BaseException:self.close();raise

def prepare():
    assert not OUT.exists();old=read(OLD/'summary.json');assert old['state']=='FAILED' and old['primary']==old['exact']==0
    ids=[old['pid']]+read(OLD/'workers.json')['workers']+[read(OLD/'driver_started.json')['pid'],read(OLD/'launch.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    p=copy.deepcopy(read(OLD/'protocol.json'));p['jobs']=[j for j in p['jobs'] if j['id'] in ['dev_task6_00','dev_task6_06']]
    receipt=read(OLD/'geometry_encoding_receipt.json');p.update(id='T445_MINIMAL_TWO_REPRESENTATIVE_FROZEN_PAIRS',out=str(OUT),
        max_bytes=4*2**20,max_physics=200000,max_commands=32,main_seconds=600,learning_seconds=0,audit_seconds=180,all_seconds=1500,
        inherited_all_stop=read(OLD/'launch.json')['all_stop'],geometry_original_gzip=str(OLD/'geometry.mjb.gz'),
        geometry_original_sha256=receipt['original_sha256'],preflight_commands=6,main_pipe_commands=12,
        maximum_primary_trajectories=4,maximum_exact_replays=4,maximum_new_closed_loop_pairs=2,
        scope='Engineering recovery narrowed to first representative of two previously known terminal-distance clusters; NOT all8 localization or a new algorithm. All T444 consumed resources remain charged.',
        original_missing_data='T444 155059 physical steps but0 retained completed trajectories; outcomes missing, not0% success.',
        user_authorization='Existing next-step diagnosis authorization; preserves original combined .8M/80cmd/10MiB and earlier absolute stop. No added failed budget or training.')
    p['worst_physics']=4*sum(j['case']['step_cap'] for j in p['jobs']);assert p['worst_physics']<=p['max_physics']
    files=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T445_PLAN.md'),str(OLD/'summary.json'),str(OLD/'failure.json'),str(OLD/'geometry_encoding_receipt.json'),p['geometry_original_gzip']}
    p['inputs']={f:sha(f) for f in sorted(files)}
    assert K.used(OLD)+p['max_bytes']<=10*2**20;assert K.used()+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p,True)
    save(DOC/'T445_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',original_failed_physics=old['physics'],
        new_max_physics=p['max_physics'],new_worst_physics=p['worst_physics'],combined_max_physics=old['physics']+p['max_physics'],
        original_commands=old['commands'],new_commands_max=32,new_actual_max_commands=18,combined_commands_max=80,
        original_own_bytes=K.used(OLD),new_own_bytes_max=p['max_bytes'],combined_own_bytes_max=K.used(OLD)+p['max_bytes'],
        inherited_all_stop=p['inherited_all_stop'],Actor_Adam=0,V_Adam=0))
    print('T445_REGISTERED_SHARED_ORIGINAL_LIMITS',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists();C.ban_learning()
    install_geometry(p);folder=OUT/'preflight_native';folder.mkdir();counts=M.mp.get_context('spawn').RawArray('q',9)
    pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+150,max_physics=1),counts,lambda:None)
    try:result=pool.request([dict(kind=kind,job=j,cancel_probe=True) for kind,j in zip(['safe_reset','student'],p['jobs'])]);assert all(x['state']=='CANCEL_PASS' for x in result) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Actor_Adam=0,V_Adam=0,commands=6,
        original_geometry_decoded_in_RAM_only=True,actual_constructed_native_geometry_matches=True,
        same_scientific_policy_contract_inherited_from_T444=True,two_workers_cancelled_and_stopped=True))
    print('T445_PREFLIGHT_PASS',flush=True)

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not (OUT/'main_started.json').exists();E.N.offline();C.ban_learning();E.L.install_stop_handlers();M.OUT=OUT;M.verify=verify
    rt=M.Runtime(dict(p,learning_seconds=p['main_seconds']));rt.commands=2;state='FAILED';error=None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=min(rt.deadline,p['inherited_all_stop'])),rt.counts,rt.guard)
        for kind in ['safe_reset','student']:
            model=p['safe_model'] if kind=='safe_reset' else p['student_model'];ph=p['safe_policy_hash'] if kind=='safe_reset' else model['policy_hash'];refs=[]
            for replay in (False,True):
                rt.role=kind+('_EXACT' if replay else '_PRIMARY');rt.guard();rt.commands+=2
                for i,((a,r),j) in enumerate(zip(rt.pool.request([dict(kind=kind,job=j) for j in p['jobs']]),p['jobs'])):
                    assert r['policy_hash']==ph and r['complete'];r.update(job_id=j['id'],role=rt.role,model_sha256=model['sha256'])
                    old=p['safe_historical_rows'][j['id']] if kind=='safe_reset' else p['student_historical_rows'][j['id']]
                    for key in ['actions','native_counts','poses','next_poses','end_flags','rewards']:assert digest(a[key])==old['array_hashes'][key],(kind,j['id'],key)
                    expected=E.N.K.R.arrays_hash(a['final_native_integration']) if kind=='safe_reset' else digest(a['final_native_integration']);assert expected==old['integration_sha256']
                    if replay:
                        assert r['array_digest']==refs[i]['array_digest'];r.update(reference_path=refs[i]['path'],exact=True,independent_new_sample=False);rt.replays.append(r);log='replays.jsonl'
                    else:
                        path=OUT/'episodes'/f'{kind}_{j["id"]}.npz';assert path.exists() and digest(E.N.S.arrays(path))==r['array_digest']
                        r.update(path=str(path),sha256=sha(path),full=True);refs.append(r);rt.rows.append(r);log='cases.jsonl'
                    with (OUT/log).open('a') as f:f.write(json.dumps(r)+'\n')
        results=K.analyze(p,rt.rows);save(OUT/'comparisons.json',results);state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,physics=sum(rt.counts),commands=rt.commands))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+6,
            main_commands=rt.commands,preflight_commands=6,primary=len(rt.rows),exact=len(rt.replays),
            safe_success=sum(r['success'] for r in rt.rows if r['kind']=='safe_reset'),student_collisions=sum(r['collision'] for r in rt.rows if r['kind']=='student'),
            error=error,all_native_workers_stopped=True,Actor_Adam=0,V_Adam=0,fresh16=None,reload78=None,six_gate=False,eight_started=False)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,epoch=time.time()));M.fcntl.flock(rt.lock,M.fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error
    print('T445_MAIN_COMPLETE',flush=True)

def audit():
    import mujoco as mj
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in [s['pid']]+read(OUT/'workers.json')['workers'])
    C.ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);model=ram_model(p);data=mj.MjData(model);pairs=read(OUT/'geometry.json')['pairs']
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()];replays=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()]
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert digest(a)==r['array_digest']
        command,loc=K.responses(p,a,r['kind']);assert np.array_equal(command,a['actions']) and np.array_equal(loc,a['locations'])
        assert int(a['native_counts'].sum())==r['physics'] and not a['ownership'].any() and not a['stochastic'].any()
        for i,x in enumerate(a['states']):
            assert np.abs(E.N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
            data.qpos[:]=a['qpos_before'][i];mj.mj_forward(model,data);distance,pair=K.clearance(model,data,pairs)
            assert abs(distance-a['body_clearance_capped_m'][i])<1e-10 and np.array_equal(pair,a['body_geom_pair'][i])
            n=int(a['native_counts'][i]);z=E.N.K.R.GAMMA**n;f=a['end_flags'][i];d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy'])
            rw=-.000002*(1-z)/(1-E.N.K.R.GAMMA)+.03*d0-z*.03*d1*(not f.any())
            if f.any():rw+=E.N.K.R.GAMMA**(n-1)*(float(f[1])-float(f[0])-.5*float(f[2]))
            assert abs(rw-a['rewards'][i])<1e-12
    assert len(rows)==len(replays)==4 and s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==18
    for r in replays:assert r['array_digest']==next(v for v in rows if v['path']==r['reference_path'])['array_digest']
    for row in read(OUT/'comparisons.json'):
        assert sha(row['comparison_path'])==row['comparison_sha256'];v=E.N.S.arrays(row['comparison_path'])
        for i,kind in enumerate(['safe_reset','student']):
            for j,actualkind in enumerate(['safe_reset','student']):
                raw=E.N.S.arrays(next(r for r in rows if r['job_id']==row['job_id'] and r['kind']==actualkind)['path'])
                command,loc=K.responses(p,raw,kind);assert np.array_equal(v[f'A{i}{j}'],command[:len(v['time_s'])].astype(np.float64))
        assert np.max(np.abs(v['total']-v['symmetric_policy']-v['symmetric_history']))<1e-12
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),physics=s['physics'],commands=s['commands'],new_native=0,new_Actor_Adam=0,new_V_Adam=0,
        primary=4,exact_replays=4,all_history_actions_rewards_geometry_and_SHA_verified=True,
        geometry_model_original_bytes_RAM_decoded=True,paired_algebra_NOT_causal=True,original_T444_missing_trajectories_NOT_recovered=True))
    print('T445_AUDIT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip();now=time.time();assert now+p['all_seconds']<=p['inherited_all_stop']
    with (OUT/'driver_console.log').open('xb') as log:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T445','--why=Minimal frozen failure localization',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=min(now+p['all_seconds'],p['inherited_all_stop']),source_remote=r['remote_commit'],root_tree=r['root_tree'],protocol_sha256=sha(OUT/'protocol.json')))
    print('T445_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
