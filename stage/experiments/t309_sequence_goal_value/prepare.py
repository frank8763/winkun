from sequence_value import *
import ast

def main():
    assert not OUT.exists();base=read(ROOT/'stage/runs/t308_iid_native_goal_value/protocol.json');jobs=[];goals=[];inputs=dict(base['input_hashes'])
    dev=read(ROOT/'stage/runs/t299_conditional_goal_likelihood/protocol.json')['development_jobs']
    for side,task in [('LEFT',6),('RIGHT',7)]:
        old=ROOT/f'stage/runs/t255_overnight/episodes/final_s2_reset_dev_task{task}_13.npz';actual=H.resolve(old);inputs[str(actual)]=C.sha(actual)
        with np.load(actual) as z:poses=z['next_poses'].copy();origin=z['poses'][0,:2].copy();flags=z['end_flags'].copy()
        assert not flags[:-1].any() and np.array_equal(flags[-1],[0,1,0]);distance=np.linalg.norm(poses[:,:2]-origin,axis=1)
        source=next(j for j in dev if j['id']==f'dev6_{side}_13');inputs[source['exact_snapshot']]=C.sha(source['exact_snapshot'])
        raw=next(r for r in read(ROOT/'stage/runs/t308_iid_native_goal_value/cases.json') if r['side']==side and r['split']=='train' and r['arm']=='center')
        for band in [.75,1.5,3.5,6.]:
            index=int(np.argmin(np.abs(distance-band))) if band!=6 else None
            for split in ['train','test']:
                k=None if index is None else index+(3 if split=='test' else 0)
                assert k is None or k<len(poses)-1
                goal=raw['goal_xy'] if k is None else poses[k,:2].tolist()
                family=f'{side}_d{band:g}_{split}'
                g=dict(id=family,side=side,split=split,distance_band=band,active_goal_xy=goal,coordinate_source='original6mgoal' if k is None else str(actual),coordinate_index=k,
                    nominal_distance_is_family_label=True,delta_x_m=.015 if split=='test' else 0.,delta_y_m=-.015 if split=='test' else 0.)
                goals.append(g)
                for noise in range(4):
                    seed=2026110310+(0 if side=='LEFT' else 1000)+int(band*100)+noise
                    fam=family+f'_noise{noise}';arms=[('center',0.)] if split=='train' else [('center',0.),('minus',-1.),('plus',1.)]
                    for arm,offset in arms:
                        j=dict(source,**{k:g[k] for k in ['side','split','distance_band','active_goal_xy','delta_x_m','delta_y_m']},id=fam+'_'+arm,source_reset_id=family,family=fam,seed=seed,arm=arm,mode='iid',first_steer_innovation_offset=offset,aux_eta=0.,policy_hash=base['policy_hash'])
                        j.pop('expected_initial120_sha256',None);jobs.append(j)
    for f in [ROOT/'stage/runs/t308_iid_native_goal_value/protocol.json',ROOT/'stage/runs/t308_iid_native_goal_value/cases.json',
        ROOT/'stage/docs/overnight_20261102/T308_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T308.json',
        DOC/'LONG_DISTANCE_METHOD_PROPOSAL.md',DOC/'T309_PLAN.md',DOC/'T309_LITERATURE.md',
        ROOT/'stage/experiments/t308_iid_native_goal_value/t308_core.py',ROOT/'stage/experiments/t290_exact_duplicate_store/store.py',*HERE.glob('*.py')]:inputs[str(f)]=C.sha(f)
    for f,v in read(ROOT/'stage/docs/overnight_20261102/T308_POSTHOC_NUMERICS.json')['protected_files'].items():
        assert C.sha(f)==v['expected'];inputs[f]=v['expected']
    p=dict(experiment='T309',policy_protocol=base['policy_protocol'],policy_hash=base['policy_hash'],input_hashes=inputs,jobs=jobs,task_goals=goals,
        seed=2026110313,distillation_seed=2026110314,evaluation_seed=2026110315,head_steps=1500,lr=.001,batch_size=64,expectile=.95,H=3,
        validation_batches=16,distance_bands=[.75,1.5,3.5,6.],kernel_width_m=.2,value_margin=1e-8,
        max_commands=272,main_max_commands=256,max_physics=7168000,wall_seconds=7200,final_reserve_seconds=1200,audit_seconds=1800,
        final_bytes=22*1024**2,payload_bytes=int(19.75*1024**2),metadata_bytes=2*1024**2,episode_bytes=256*1024,compact_bytes=24*1024,
        model_bytes=256*1024,evidence_bytes=256*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,
        Actor_updates=0,automatic_next_training=False,
        contract='NEW multi-real-goal fixedH3CRL+DQC feasibility pilot from safeT255. Not restart/extension ofT308. Original120hiddenresetActor fixed, rho0iid sigma(.035,.03) matchesT308, noActoroptimizer. 16fixed actualtaskfamilies LEFT/RIGHT x nominaldistance .75/1.5/3.5/6 x train/test; near/mid targets fixed beforecollection from priorownSAFE5m recorded endpoints nearest nominal radialdistance, testindex+3 and initialoffset(+.015,-.015), SIXuses original6mgoal andnewtestoffset; alltaskcap28000not historical5m cap. Goals are separatefinaltrainingtasks, neverruntimewaypoints/teacheractions. Goal13historyfields rebuilt withrealnativeposes beforeepisode, nativeintegrationunchanged, allgoalsconstant throughoutepisode. 4fixednoise streams/family:32train+32testcenters, 64testfirststeerinnovation minus/plus1sigma branches, then sameiidpolicy/commonfutureinnovations. 128unique+128independentexactreplays allcharged; 16zero-stepfactorygeometry commands inpreflight alsocharged=272total. Center64savefull12arrays+history3+innovations<=256KiB, branch64compact<=24KiB plusfullarrayhashes+integration, replaycompletearraysnotduplicated on disk,4partialfull256KiB reserved. Single/sequence/blind heads128-64-32 query6-32-32 sameinit/samebatch1500Adam each; raw120+zero latent2+padded6actions, singleonlyfirst2,blindzeros6. Allanchorsrequire3consecutiveactions beforetruetermination/no cross episode; futurelaw identicalacrossheads fromanchor, contactpriority/goalsabsorbed/truetimeout/censorunknown; nofirsthitQclaim. Factualgoalcontext retained; newquerystrategy transfer andobserved futureaction conditioning notopenloopcausalQ are limitations. DQCstudent stateactionencoder128-64-32 initializedsameprototype, copiedteacherquery frozen, teacherfullfrozen,1500Adam onexpectile.95 target stopgradient;notarithmeticaverage/unbiasedorcalibratedprobability. 6000realheadAdam+6000independentdisposableaudit, noselectcheckpoint/grid. Representationgate ALLsequenceNCE<=.95blind AND<=.98single, eachside<=1.02blind. Canonicalbatch1control uses fixedactualtaskgoals chosen BEFORE rollouts, kernel.2mfiniteoccupancy; pergoalband/side report all informativepairs/noisegroups, ALLdistilled rank>=.90 with>=6pairs eachside andtopselectedrealvalue>tieuniform. Correlatedpairs notindependentCI, bandlackofinformation cannotcount asvalidatedsix. No Actor allowed evenifcombinedgate passes; nextActor protocol mayuseonlyexplicitvalidatedgoalranges andneeds actualsixnavgate andremainingcapacity/source registration. Old275/orig6m54/fresh16/reload78 NOT measured here; no7/8/map/planner/runtime teacher/router/Phase. 22MiB own;64full16MiB+4partial1MiB+64compact1.5MiB+4heads1MiB+evidence.25MiB=19.75payload, metadata2MiB+buffer.25MiB. 256nativecommands7.168Mphysics+16zero-stepgeometries,7200wall last1200final/audit1800,global1GiB/free40GiB. First actual source cleancommit/fullrootremote+preflight required beforeuniquelaunch; afterallstopped independentarrays/goalhistory/density/6000headAdam/models/controlmetrics exactaudit andalloriginalRAMremote everymemberSHA; currentmodels/videos/failures untouched, no coldencodingdeletion.')
    assert len(jobs)==128 and len(goals)==16
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    for f,h in inputs.items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:C.save(f,p)
    print('T309 registered H3 representation and single-action distillation; zero native/Adam')

if __name__=='__main__':main()
