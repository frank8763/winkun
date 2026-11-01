"""Zero-optimizer source-semantics and recorded future-endpoint contract."""
from future_data import *
import ast,collections,io,os,shutil,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t275_future_target_contract'


def source_semantics(plan):
    tree=ast.parse(Path(plan['environment_source']).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Phase10SceneE2ELidarEnv')
    names=['_goal_state','_history_frame','_observation','_clearance']
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(methods)==4
    # Execute only four observation-construction methods on a synthetic dummy;
    # never import/construct a MuJoCo environment or call native physics.
    module=ast.Module(body=[ast.ClassDef(name='ObservationDummy',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ns=dict(np=np,math=math);exec(compile(ast.fix_missing_locations(module),'<locked-observation-methods>','exec'),ns)
    Dummy=ns['ObservationDummy'];rng=np.random.default_rng(2026111501)
    def make(goal_offset):
        d=Dummy();d.prev_action=np.array([.25,-.15],dtype=np.float32);d.history=collections.deque(maxlen=4);d.HISTORY_FRAMES=4
        d._lidar48=lambda:np.linspace(.05,.95,48,dtype=np.float32)
        for k in range(4):
            p=(.1*k,-.2*k,.2*k,.12,-.03,.08);d._pose=lambda p=p:p
            d.goal_xy=np.array([2.+k*.1,-3.])+goal_offset
            d._observation(update_history=True)
        d._pose=lambda:(.45,-.7,.67,.13,-.02,.07)
        d.goal_xy=np.array([3.5,-4.])+goal_offset
        return np.r_[d._observation(update_history=False),np.array([.4,.1,-.1,.2,-.2,.3,-.4,1.],dtype=np.float32)]
    a=make(np.zeros(2));b=make(np.array([1.25,-.8]))
    assert a.shape==(120,) and np.array_equal(encode(a),encode(b))
    assert set(np.flatnonzero(a!=b)).issubset(set(GOAL_COLUMNS)) and all(a[k]!=b[k] for k in GOAL_COLUMNS)
    changed=a.copy();changed[GOAL_COLUMNS]=rng.normal(size=13)*100
    assert np.array_equal(encode(a),encode(changed))
    assert encode(a).shape==(107,)
    c=a.copy();c[53]+=.1;assert not np.array_equal(encode(a),encode(c))
    # Coordinate-frame invariance: same native relative query under global SE2.
    pose=np.array([.4,-.5,.6]);point=np.array([1.8,2.]);g=relative(pose,point)
    theta=.93;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    trans=np.array([10.,-6.]);pp=np.r_[rot@pose[:2]+trans,pose[2]+theta]
    assert np.allclose(g,relative(pp,rot@point+trans),atol=2e-15,rtol=0)
    return dict(four_locked_source_methods_executed_without_native_environment=True,
        direct_goal_fields_exactly13_and_107_retained=True,history_and_current_goal_change_invariance=True,
        non_goal_motion_field_retained=True,global_SE2_relative_query_invariance=True,
        no_claim_statistical_goal_independence_or_reconstructed_Actor120=True)


def main():
    assert not (OUT/'status.json').exists(),'No restarted contract or expanded data'
    p=json.loads((HERE/'protocol.json').read_text());start=time.time();reason='COMPLETE';details={}
    try:
        for path,h in p['input_hashes'].items():assert sha(path)==h,path
        assert used(ROOT/'stage/runs')-used(OUT)+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_envs_created=0,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        save(OUT/'status.json',dict(state='CHECKING_SEMANTICS',pid=os.getpid(),physics_steps=0,optimizer_calls=0))
        details['source_contracts']=source_semantics(p);arrays,stats=build(p);details.update(stats)
        x,y,e=features(p,arrays);assert len(x)==len(y)==stats['pairs'] and x.shape[1]==111
        # Each target is a realized endpoint after the anchor. No prior-target
        # query point, contact endpoint, unknown suffix, or original success is invented.
        buf=io.BytesIO();np.savez_compressed(buf,**arrays);blob=buf.getvalue();assert len(blob)<=p['pairs_bytes']
        (OUT/'future_pairs.npz').write_bytes(blob)
        with np.load(OUT/'future_pairs.npz') as z:assert all(np.array_equal(v,z[k]) for k,v in arrays.items())
        details.update(future_pairs_sha256=sha(OUT/'future_pairs.npz'),future_pairs_bytes=len(blob),feature_width=111,
            source_state_width=120,retained_state_width=107,removed_goal_columns=GOAL_COLUMNS.tolist(),
            labels='log1p recorded physical time to a noncontact decision endpoint; NOT shortest/first-hit/reachability probability',
            feature_arrays_retained_in_RAM_only=True)
        assert time.time()-start<p['wall_seconds']
    except BaseException:
        reason='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0,optimizer_calls=0))
    result=dict(state=reason,physics_steps=0,optimizer_calls=0,Actor_updates=0,native_environments_created=0,
        all_workers_stopped=True,elapsed_s=time.time()-start,own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),
        protocol_sha256=sha(HERE/'protocol.json'),report=details,ready_for_bounded_future_time_pilot=reason=='COMPLETE',
        original_six_m_success_added=False,first_hit_or_safe_goal_probability_claim=False)
    save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
