from t342_core import *

def main():
    assert not OUT.exists();assert read(DOC/'T341_AUDIT.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T341.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json');source=read(ROOT/'stage/runs/t341_wheel_speed_transfer/summary.json')
    success=next(r for r in source['cases'] if r['id']=='navigation_x1_primary');assert success['success'] and not success['collision']
    keep=['learning_protocol','initial_patch','initial_policy_hash','course_protocol','reference','input_mean','input_scale','base_protocol',
        'old_jobs','course_dev_jobs','original_jobs','final_old_jobs','executed_std','tanh_sigma','seed','six_jobs','fresh_jobs','reload_jobs']
    p={k:old[k] for k in keep};base=read(p['base_protocol']);lp=read(p['learning_protocol'])
    p.update(experiment='T342_BALANCED_SINGLE_ACTOR_CONSOLIDATION',arms=['capg'],student_initial_model=str(ROOT/'stage/runs/t340_demo_supported_recovery/final.pt.gz'),
        student_initial_policy_hash=success['policy_hash'],old_states=base['retention_states'],safe_policy_hash=lp['frozen_policy_hash'],
        new_success_states=success['path'],new_success_source=success,consolidation_steps=3000,consolidation_lr=.001,consolidation_seed=20261104342,
        old_rows_each=16,new_rows=128,old_total_weight=.5,new_total_weight=.5,hash_interval=100,
        optimizer_kind='fresh Adam; correction MLP only; fixed logscale and T333 base',max_live_supervised_Adam=3000,max_shadow_supervised_Adam=3000,
        preflight_discarded_Adam=4,RL_updates=0,training_native=0,wall_seconds=5400,final_reserve_seconds=2000,audit_seconds=1800,all_work_seconds=8400,
        per_batch_admission_seconds=120,optimizer_admission_seconds=120,max_commands=500,max_physics=18000000,preflight_geometry_commands=2,
        final_bytes=8*1024**2,payload_bytes=int(4.5*1024**2),metadata_bytes=int(3.5*1024**2),
        compact_bytes=4096,max_compact_cases=498,partial_bytes=384*1024,model_bytes=512*1024,
        global_bytes=1024**3,min_free_bytes=40*1024**3,no_scene_or_speed_change=True,
        numerical_supervised_batch_vs_deployment_tolerance=3e-6,
        contract='Fixed-data balanced consolidation,0newRL/onlinecollection. InitialT340 known wide6 retained as teacher states; safeT255 old28629 states/8families totalweight.5, new complete deterministic wide6 trajectory teacher locations weight.5. Same canonical CAPG Actor/frozenT333/base/capacity/logscale, only correction MLP supervised Adam changes; no probability or BC-onpolicy claim. 3000live+3000exactshadow,0newQ/V, no grids. Initial old20+widecenter measured, final old20/wide5/original2 then27exactmodel native readbacks. Same candidate allold20/wide5/narrow2 pass before fullold275+six54,allbefore16fresh then78independent model native reload. Missing remainsnull. No7/8 before full strict gate/audit/backup. Deployment1Actor120hiddenreset/no map/planner/waypoint/runtime teacher/routing. Phaseisolated. No speed2x in this consolidation; T341 speed transfer is separate and preserved.')
    p['max_registered_commands']=2+21+27+27+275+54+16+78
    p['worst_registered_physics']=21*40000+54*40000+(275+54+16+78)*28000
    assert p['max_registered_commands']==p['max_commands'] and p['worst_registered_physics']<=p['max_physics']
    assert 498*p['compact_bytes']+4*p['partial_bytes']+p['model_bytes']+p['metadata_bytes']<p['final_bytes']
    files=set(old['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={p['old_states'],p['student_initial_model'],p['new_success_states'],str(ROOT/'stage/runs/t341_wheel_speed_transfer/summary.json'),str(ROOT/'stage/experiments/t341_wheel_speed_transfer/run.py'),str(ROOT/'stage/experiments/t341_wheel_speed_transfer/protocol.json'),str(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json'),str(DOC/'T342_PLAN.md'),str(DOC/'T341_AUDIT.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T341.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',RL=0,max_supervised_live=3000,max_native_commands=500,reserved_global=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()
