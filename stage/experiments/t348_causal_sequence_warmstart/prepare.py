from t348_core import *


def main():
    assert not OUT.exists()
    assert read(DOC / 'T347_AUDIT.json')['state'] == 'PASS'
    receipt = ROOT / 'stage/docs/long_distance_20261101/release_receipt_T347.json'
    assert read(receipt)['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old = read(ROOT / 'stage/experiments/t342_balanced_consolidation/protocol.json')
    p = {k: old[k] for k in ['base_protocol','safe_policy_hash','old_states','input_mean','input_scale','course_protocol','old_jobs','course_dev_jobs','original_jobs']}
    p.update(experiment='T348_MATCHED_CAUSAL_SEQUENCE_RESET_DEMONSTRATION_WARMSTART',
             arms=['reset','memory'], sequence_seed=20261104348, sequence_lr=.001, fit_steps=600,
             embedding=32, hidden=16, dtype='float64_trainable_head_float32_frozen_base',
             sources_old_episodes=211, old_total_weight=.5, new_total_weight=.5,
             sequence_sources=[
                 dict(name='human_T347_matched_start',path=str(ROOT/'stage/runs/t347_original_start_human/attempt_01.npz'),label_start=0),
                 dict(name='human_T328_center',path=str(ROOT/'stage/runs/t328_wide_gap_80s/attempt_01.npz'),label_start=0),
                 dict(name='human_T331_correction_only',path=str(ROOT/'stage/runs/t331_policy_prefix_retry/attempt_01.npz'),label_start=150),
                 dict(name='self_T340_center',path=str(ROOT/'stage/runs/t341_wheel_speed_transfer/episodes/navigation_x1_primary.npz'),label_start=0),
                 dict(name='self_T342_matched_start',path=str(ROOT/'stage/runs/t344_neighbour_recovery/episodes/support_regeneration_known_T342_wide_dev1.npz'),label_start=0)],
             wall_seconds=3600, final_reserve_seconds=1200, audit_seconds=1200, all_work_seconds=7200,
             per_batch_admission_seconds=90, geometry_commands=7, max_commands=120, max_physics=4520000,
             final_bytes=5*1024**2, full_bytes=384*1024, max_full_cases=4,
             compact_bytes=4096, max_compact_cases=55, partial_bytes=384*1024, model_bytes=256*1024,
             metadata_bytes=1024*1024, logs_bytes=256*1024, global_bytes=1024**3, min_free_bytes=40*1024**3,
             live_supervised_Adam=1200, shadow_supervised_Adam=1200, preflight_discarded_Adam=8,
             audit_historical_discarded_Adam=1200, new_RL_updates=0, new_training_native_collection=0,
             contract='One bounded new mechanism test: trainable causal GRU16 residual on frozen safeT255,matched reset vs memory. Same parameters/init/data schedule/LR600Adam each,full actual episodes from time0,no stored stale hidden. Old211 exact episode boundaries; balancedold/new source groups,5real successful sources inclnewT347,onlyhuman suffix T331>=150 labelled. Base/no observation/reward/plant change:120inputs/wheel5force120/wide1.17m80s,narrow56s. Memory carries every real0.2s decision,reset zeros eachdecision; trueBPTT causal episodes,float64head for reliable sequence/step path,outputfloat32. This is supervisedwarmstart0newRL/not NiSAC or R2D2 reproduction/no novelty. Shared frozenbase initialwide5; eacharm finalold20/wide5/originalnarrow2 plus27independentnative modelreadbackexact. 5+2*(27+27)+7geometry=120commands/4.52Mallphysics,allrepeats counted.4fullcases center+matchedwide perarm include120/history/hidden;remaining55primarycompact,54readbackallarrays exact inRAM/hashrecord. WholeActorAdam deepcopied beforeblock,inputhashbeforeupdates,shadowexact; interruption rollback. Main3600 last1200evaluation/audit1200/all7200/own5MiB with4partialslots/global1GiB/free40GiB. No old275/six54/fresh16/reload78 inpilot,null,no stableoriginal6/7/8. No planner/map/waypoint/runtime teacher/router/Phase,no extra seeds/epochs/heads/noise grid or restart.')
    assert p['max_commands']==7+5+2*(27+27)
    assert 4*p['full_bytes']+55*p['compact_bytes']+4*p['partial_bytes']+2*p['model_bytes']+p['metadata_bytes']+p['logs_bytes'] <= p['final_bytes']
    files=set(old['input_hashes']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T348_PLAN.md'),str(DOC/'T347_AUDIT.json'),str(receipt),str(ROOT/'stage/experiments/t344_neighbour_recovery/t344_core.py')}
    files |= {r['path'] for r in p['sequence_sources']}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,reserved_global=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()
