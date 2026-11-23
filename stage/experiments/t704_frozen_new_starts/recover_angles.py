"""Only four zero-step angle-representation checks; no policy/physics repeat."""
from pathlib import Path
import copy,gzip,json,math,sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare as P
K,A,R,D=P.K,P.A,P.R,P.D
read,save=K.read,K.save

def main():
    start=time.time();K.old_stopped();K.unit_stopped('stage-t704-preflight.service')
    p=read(R/'prepared_protocol.json');initials=read(D/'INITIALS.json');old=read(D/'angle_preflight_original/INITIALS.json')
    assert len(old['invalid'])==4 and old['valid']==220 and not old['duplicates']
    proposed=read(D/'PROPOSED_INITIALS.json');by={j['id']:j for j in proposed['jobs']}
    expected={'T704_t502_field_sector1_offset2_'+ring+'_'+str(angle) for ring in ['near','outer'] for angle in [45,225]}
    assert {r['id'] for r in old['invalid']}==expected
    import mujoco,t698_native as N
    def forbidden(*args,**kw):raise AssertionError('Angle check has no physical integration or model action')
    mujoco.mj_step=forbidden
    def guard():assert time.time()<start+110;K.C.resources()
    C,F=N.C,N.F;C.ACTIVE=p;folder=R/'angle_preflight_recovery';folder.mkdir()
    module,native,create,w,_=C.C.consistent_reset.setup(F,folder,p,C.C.CANONICAL_SETUP);original=copy.deepcopy(module.ACTIVE)
    recovered=[]
    try:
        for jid in sorted(expected):
            guard();job=by[jid];angle=job['fullfield_start_pose'][2];assert angle>math.pi
            wrapped=math.atan2(math.sin(angle),math.cos(angle))
            assert abs(math.atan2(math.sin(angle-wrapped),math.cos(angle-wrapped)))<1e-14
            job['requested_unwrapped_yaw']=angle;job['fullfield_start_pose'][2]=wrapped
            module.ACTIVE=copy.deepcopy(original)
            if 'circle_shift_x_m' in job:module.ACTIVE['circle_shift_x_m']=job['circle_shift_x_m']
            current=None
            try:
                current=C.Session(job,native,create,w,guard,forbidden,None)
                pose=list(current.env._pose()[:3]);packet=current.packet()
                assert not current.env._collision() and not current.done()
                assert A.np.linalg.norm(current.goal-A.np.asarray(pose[:2]))>.2
                assert not current.hidden.any() and current.steps==0 and current.env.data.time==0
                assert max(abs(v) for v in pose[:2])<=7
                assert A.np.max(A.np.abs(A.np.asarray(pose)-job['fullfield_start_pose']))<1e-10
                nativehash=A.digest(current.initial);scenehash=A.digest(A.np.frombuffer(bytes.fromhex(current.scene),dtype=A.np.uint8));goalhash=A.digest(current.goal)
                combined=P.key(nativehash,scenehash,goalhash)
                assert combined not in {r['combined_initial_key'] for r in initials['contracts']}
                # Match original history dedup scope; no trajectories are generated.
                for item in old['history_metadata']:
                    for line in gzip.decompress(Path(item['path']).read_bytes()).decode().splitlines():
                        h=json.loads(line).get('array_hashes',{})
                        if all(k in h for k in ['initial_native_integration','scene_sha256','task_goal_xy']):
                            assert combined!=P.key(h['initial_native_integration'],h['scene_sha256'],h['task_goal_xy'])
                assert nativehash not in {r['native_hash'] for r in read(K.ROOT/'stage/runs/t703_projected_local/protocol.json')['new_initial_contracts']}
                initials['contracts'].append(dict(id=jid,pose=pose,packet_hash=A.digest(packet),native_hash=nativehash,scene=current.scene,goal_hash=goalhash,combined_initial_key=combined))
                recovered.append(dict(id=jid,requested_unwrapped_yaw=angle,canonical_yaw=wrapped,periodic_angle_error=abs(math.atan2(math.sin(angle-wrapped),math.cos(angle-wrapped))),
                    zero_step_contact_free=True,packet_hash=A.digest(packet),native_hash=nativehash))
            finally:
                if current is not None:current.close()
    finally:w.close()
    order=[j['id'] for j in proposed['jobs']];bycontracts={r['id']:r for r in initials['contracts']}
    initials.update(valid=224,invalid=[],exact_count=28,contracts=[bycontracts[k] for k in order],
        engineering_correction='Four original failed raw-yaw comparisons were periodic-angle representation errors, same physical requested poses canonicalized; no replacement',
        original_preflight_preserved=str(D/'angle_preflight_original'))
    save(D/'INITIALS.json',initials);save(D/'PROPOSED_INITIALS.json',proposed)
    p.update(new_jobs=proposed['jobs'],new_exact_jobs=[j for j in proposed['jobs'] if j['prospective_ring']=='near' and j['prospective_direction_deg']==45],
        new_invalid_initials=[],new_initial_contracts=initials['contracts'],prospective_all224=proposed['jobs'])
    p.pop('inputs',None);p.pop('input_stats',None)
    (R/'prepared_protocol.json').write_text(json.dumps(p,separators=(',',':'))+'\n')
    pre=read(D/'PREFLIGHT.json');pre.update(valid=224,invalid=0,duplicates=0,
        periodic_angle_recovery='Only four original zero-step failed representations rechecked, zero repeats of220 valid checks or model forwards',
        storage_worst_bytes=4*(224*p['fresh_bytes']+28*p['compact_bytes']+32*2**20)+32*2**20)
    save(D/'PREFLIGHT.json',pre)
    save(D/'ANGLE_RECOVERY.json',dict(state='PASS',checks=recovered,elapsed_seconds=time.time()-start,new_native=0,new_Actor=0,new_model_forward=0,
        no_source_or_point_selection=True,original_preflight_preserved=True))
    print('T704_FOUR_PERIODIC_ANGLES_PASS_ALL224_VALID',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:save(D/'ANGLE_RECOVERY_REJECTION.json',dict(error=traceback.format_exc(),new_native=0));raise
