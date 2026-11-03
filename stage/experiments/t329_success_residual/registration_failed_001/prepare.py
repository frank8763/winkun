from learning import *
def main():
    assert not OUT.exists();base=read(ROOT/'stage/experiments/t322_full_network_steering/protocol.json');course=read(ROOT/'stage/experiments/t328_wide_gap_80s/protocol.json')
    assert read(DOC/'T328_AUDIT.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T328.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    demo=ROOT/'stage/runs/t328_wide_gap_80s/attempt_01.npz';a=arrays(demo);assert a['end_flags'][-1].tolist()==[0,1,0]
    old,_=H.anchors(base);allx=np.concatenate([a['states'],old]).astype(np.float64)
    wide=dict(course['job'],id='wide_left6_80s',source_reset_id='wide_left6_80s',wide_course=True,
        expected_initial120_sha256=course['initial_geometry']['initial120_sha256'])
    originals=[j for j in base['development_jobs'] if j['id'] in ['dev6_LEFT_13','dev6_RIGHT_13']];assert len(originals)==2
    training=[dict(wide,id='fresh_'+str(k),source_reset_id='wide_left6_80s',seed=202611033290+k,family='wide_left6',side='LEFT',split='train',arm='center',first_steer_innovation_offset=0.) for k in range(8)]
    # Five fixed translations, all development. No fresh/unseen claim.
    course_dev=[dict(wide,id='wide_dev_'+str(k),source_reset_id='wide_dev_'+str(k),delta_x_m=dx,delta_y_m=dy,
        expected_initial120_sha256=None if dx or dy else wide['expected_initial120_sha256']) for k,(dx,dy) in enumerate([(0,0),(-.03,0),(.03,0),(0,-.03),(0,.03)])]
    p=dict(experiment='T329_SUCCESS_DEMO_FROZEN_BASE_RESIDUAL_PILOT',base_protocol=str(ROOT/'stage/experiments/t322_full_network_steering/protocol.json'),
        course_protocol=str(ROOT/'stage/experiments/t328_wide_gap_80s/protocol.json'),demo=str(demo),frozen_policy_hash=base['policy_hash'],
        seed=20261103329,input_mean=allx.mean(0).astype(np.float32).tolist(),input_scale=np.maximum(allx.std(0),1e-3).astype(np.float32).tolist(),
        anchor_rows_each=32,bc_steps=100,bc_lr=.001,pg_lr=.0001,retention_weight=100.,
        wide_job=wide,training_jobs=training,course_dev_jobs=course_dev,original_jobs=originals,old_jobs=base['acceptance_guard_jobs'],
        wall_seconds=2700,training_seconds=2100,final_reserve_seconds=600,audit_seconds=1200,
        max_commands=160,max_physics=6400000,final_bytes=16*1024**2,payload_bytes=13*1024**2,metadata_bytes=2*1024**2,
        training_episode_bytes=768*1024,compact_bytes=12*1024,patch_bytes=512*1024,global_bytes=1024**3,min_free_bytes=40*1024**3)
    files={f for f in base['input_hashes'] if Path(f).suffix=='.py'}|set(course['input_hashes'])
    files|={str(f) for f in HERE.glob('*.py')}
    files|={str(demo),base['checkpoint'],base['initial_base'],base['retention_states'],p['base_protocol'],p['course_protocol'],str(ROOT/'stage/experiments/frozen_base_residual/actor.py'),str(ROOT/'stage/experiments/t308_iid_native_goal_value/t308_core.py'),str(DOC/'T329_PLAN.md'),str(DOC/'T328_AUDIT.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T328.json')}
    for j in [*p['old_jobs'],*originals,wide]:files.add(j['exact_snapshot'])
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',full_reserved_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))
if __name__=='__main__':main()
