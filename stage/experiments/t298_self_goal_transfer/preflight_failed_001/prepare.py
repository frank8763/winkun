"""Register a finite supervised/RL composite pilot, no source-goal IS shortcut."""
from t298_learning import *
import ast,gzip,json,shutil
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t298_self_goal_transfer'

def main():
    assert not OUT.exists();p=json.loads((ROOT/'stage/runs/t291_hard_guard_ablation/protocol.json').read_text());inputs={};enc=[]
    for f,h in p['input_hashes'].items():
        f=Path(f)
        if not f.exists():
            zipped=Path(str(f)+'.gz');assert zipped.exists() and __import__('hashlib').sha256(gzip.decompress(zipped.read_bytes())).hexdigest()==h;enc.append(dict(original=str(f),original_sha256=h,encoded=str(zipped),encoded_sha256=C.sha(zipped)));f=zipped
        inputs[str(f)]=C.sha(f)
    a=json.loads((ROOT/'stage/runs/t295_hindsight_support/protocol.json').read_text());r=json.loads((ROOT/'stage/runs/t295_hindsight_support/cases.json').read_text());aux=[];auxjobs=[]
    for source,row in zip(a['sources'],r):
        assert row['alternate']['success'] and not row['alternate']['collision'];aux.append(dict(id=source['id'],path=row['path'],sha256=row['sha256']))
        if source['id'].endswith('_s0'):
            with np.load(row['path']) as z:initial=z['alternate_states'][0]
            auxjobs.append(dict(source['job'],id='aux_'+source['source_reset_id'],source_reset_id='aux_'+source['source_reset_id'],auxiliary_goal_xy=source['auxiliary_goal_xy'],expected_initial120_sha256=S.digest(initial)))
        inputs[row['path']]=row['sha256']
    for f in [ROOT/'stage/docs/long_distance_20261101/T297_LOG_ENCODING_RECEIPT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T297.json',ROOT/'stage/docs/distance_20261102/T297_AUDIT.json',ROOT/'stage/docs/distance_20261102/T295_AUDIT.json',ROOT/'stage/docs/distance_20261102/T296_AUDIT.json',ROOT/'stage/experiments/t295_hindsight_support/run.py',ROOT/'stage/experiments/t295_hindsight_support/t295_contract.py',ROOT/'stage/experiments/t280_local_cost/t280_worker.py',ROOT/'stage/experiments/t282_capped_balance/balanced_learning.py',ROOT/'stage/experiments/t291_hard_guard_ablation/t291_codec.py',*HERE.glob('*.py')]:inputs[str(f)]=C.sha(f)
    p.update(experiment='T298',arms={'self_goal_transfer':.1},rounds=1,seed=2026111801,actor_attempts_per_round=2,
        supervised_steps=500,supervised_seed=2026110208,auxiliary_sources=aux,auxiliary_centers=auxjobs,
        hard_old20_acceptance_required=True,preflight_clone_optimizer_max_calls=8,postrun_clone_optimizer_max_calls=1004,
        max_commands=600,max_physics=16800000,max_training_payloads=16,max_compact_payloads=520,
        training_episode_bytes=192*1024,compact_episode_bytes=10*1024,model_bytes=int(7.25*1024**2),metadata_bytes=int(1.5*1024**2),
        payload_bytes=16*192*1024+520*10*1024+int(7.25*1024**2),final_bytes=17*1024**2,
        wall_seconds=3600,final_reserve_seconds=1400,external_group_watchdog_seconds=3600,overnight_stop_epoch=10**12,overnight_learning_stop_epoch=10**12,
        historical_control=dict(experiment='T287',scope='Frozen safeT255 and consumed original6m data only; not concurrent randomized GCSL/RL ablation'),
        input_hashes=inputs,encoded_original_inputs=enc,
        contract='T298 independently bounded composite engineering pilot, not novel GCSL/HPG or pureRL. FromsafeT255 fullActor120hiddenreset; fixed8 T295 ownbehavior successful prefixes under2fixedauxiliarygoals, complete13goalhistory/107nongoal/firsthit true data verified.500supervised live+500matchedshadow Adam, equal8wholetrajectory sampling8rows each plus old8tasks4anchors each; loss auxiliary modeactionMSE+100oldmodeMSE, original baseLR3e-6/residual1e-4/gradclip1. Supervised Adam discarded, freshRLAdam; no importanceweight truncation/offpolicy PPO. Explicit supervisor fitMSE<=half baseline AND true old20 allsafe AND twoauxiliarycenter autonomoussafeGOAL before any newRL; failure stopsacquisition, still exportcandidate/fullold275six54. OneRLround atmost16new onpolicy conditionalAR episodes+16twelvearraynativeexactreplays,2live+2shadowRLAdam proposals/actualold20hardguard/numerictrust/PCGrad32/soft100/timealignedRLOO/originaleta.1 rewards/LR/factors unchanged. Single policy deploy no auxgoalrouting/waypoints/maps/planner/teacher/Phase; auxiliarygoals are separate training and diagnostic tasks, never runtime finalgoal switching. First16 aftersupervision newpolicydata, historical same seed doesnot makeoldpolicy behavior onpolicy; logp/mu recomputed batch1 and acceptedsourceSHA. Original nearestcandidatecontrol T287consumed historical notmatched concurrentcausal proof; supervised+RLeffects confoundedpilot. Finalold275 and6m54 regardlesspretrain gate unlessphysical/adminfailure, missing/censored/invalidseparate; allsafe then16fresh78independentnative reload, no7/8trainingorpromotioninthispilot. Fixed500supervised/1RLround/2proposals/no seed/noise/lossweight/model/round grids. 600allcommands16.8Mallphysical3600s learning2200s/final1400,own17MiB/global1GiB/free40GiB;16fullstatechain192KiB+520compact10KiB(fivearrays,rewards exactlyderivedenv-eta*aux)+7.25MiBActorAdam+1.5MiBmetadata. StoreonlyfinalActorAdamwithsupervisedhashhistory, duplicate supervised model fullyreconstructedpostrun, up to1004disposableauditcalls. Zero-nativepreflight8disposableoptimizermax/shadow/rollback/codec/globalnames/resources; registersourcefullrootremote beforeuniquelaunch, nativeexclusive noexternalrunswrites/sourcesync. Endallprocessesabsent auditallraw/partial/counts/source/ActorAdam and exact500supervisedpair+2RLpair reconstruction, fullRAMbinaryremotememberSHA, sourcesync. SafeT255/HDvideos/originalfailureskept; all supervisedupdates reportedseparatelyfromRL. No safetycertificate/originalalgorithmclaim or auxiliaries/centers replacing6m stability.')
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes'];assert len(aux)==8 and len(auxjobs)==2
    for f,h in inputs.items():assert C.sha(f)==h,f
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir()
    for dest in [HERE/'protocol.json',OUT/'protocol.json']:dest.write_text(json.dumps(p,separators=(',',':'))+'\n')
    print(json.dumps(dict(state='REGISTERED_ZERO_NATIVE',global_if_full=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'],supervised_live=500,supervised_shadow=500,conditional_RL_live_max=2,conditional_RL_shadow_max=2)))

if __name__=='__main__':main()
