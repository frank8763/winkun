from recover import *
import ast
def main():
    assert not OUT.exists();p=read(ORIGINAL/'protocol.json');s=read(ORIGINAL/'summary.json')
    assert s['state']=='FAILED' and s['live_Actor_calls']==s['shadow_Actor_calls']==0
    assert read(DOC/'T312_FAILED_ENTRY_AUDIT.json')['state']=='PASS_COMPLETED_RAW_AND_V_RECONSTRUCTION_ORIGINAL_PRODUCER_FAILED'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T312.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    inputs=dict(p['input_hashes'])
    for f in [*HERE.glob('*.py'),ORIGINAL/'protocol.json',ORIGINAL/'summary.json',ORIGINAL/'failure.json',ORIGINAL/'cases.jsonl',ORIGINAL/'screens.json',ORIGINAL/'learning/mc_0.npz',ORIGINAL/'learning/gae_0.npz',
        DOC/'T312_FAILED_ENTRY_AUDIT.json',receipt,DOC/'T312_REMAINING_PLAN.md',ROOT/'stage/experiments/t312_event_contract/event_contract.py']:
        inputs[str(f)]=C.sha(f)
    for m in s['models'].values():inputs[m['path']]=m['sha256']
    for r in map(json.loads,(ORIGINAL/'cases.jsonl').read_text().splitlines()):inputs[r['path']]=r['sha256']
    p.update(experiment='T312R_REMAINING_ENGINEERING_RECOVERY',input_hashes=inputs,original_main_epoch=s['main_epoch'],absolute_deadline_epoch=s['absolute_deadline_epoch'],training_deadline_epoch=s['training_deadline_epoch'],original_empty_Actor_Adam_hash=s['models']['mc']['optimizer_hash'],
        max_commands=1172-s['commands'],max_physics=32816000-s['physics_steps'],final_bytes=34*1024**2,payload_bytes=int(29.5*1024**2),metadata_bytes=4*1024**2,
        recovery_contract='Explicit user continue afterreview. New independent recovery output; originalFAILED source/protocol/error/models unchanged. No originaldirectoryrestart. Only real eventadapter fixed via common validated append_event. Reuse32 originaltraining trajectories/32exactreplays/78geometry/bothpassedV heads/4000Vcalls, not newphysical samples or optimizerupdates. Round0 enters same originallyplanned Actor steps fromsafeActor/freshAdam, cachedV/advantages exactSHAs; round1data/Vupdates onlyas originallyregistered, no extra seeds/rounds/LRs/thresholds/noise. Original combined1172commands/32.816Mphysics/43MiB remain; old142commands913812physics/8.13MiB charged, new1030commands31.902188Mphysics/34MiB atmost. Original absolutelearning12:25:57.691733 andtotal12:55:57.691733 Beijing untouched; no launchunless>=300s learningtime remains, externaltimeoutearlierfromlaunchstrict. PreflightrealfullActorfunction+sameactualeventdiskwriter withexplicitNoStep object,0native/optimizer; not anotherAdamcloneornewlearning. SourceinputSHA/cleancommit/fullremote beforeuniquelaunch. Existing2arms2roundstotal remains,firstbatchVnotrefit; allactualnewActorAdam+secondroundVlive/shadow hashes/rollback audited. Finalfull275old/54six/fresh16/78rules unchanged, no7/8training. Originalfailedpipeline notretroactivelyCOMPLETE; aggregateledgerseparatescached/new/replayed andproducerstates. Original78geometryzero repeated. New29.5MiBpayload (32newfull7.25+4partials1+2models15+.25valueevidence+6compact),4metadata+.5buffer=34MiB. Alloriginalfailures/models/safe5m/video preserved,Phasefullyisolated. Endallprocessesstoppedthenindependentnewoptimizer/rawaudit/fullremoteeverymemberSHAbackup/source; no automaticrestart/extension.')
    assert p['max_commands']==1030 and p['max_physics']==31902188
    assert time.time()<p['training_deadline_epoch']-600,'Do not rush beyond original learning deadline'
    assert S.used(ORIGINAL)+p['final_bytes']<=43*1024**2 and S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    for f,h in inputs.items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'learning').mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:C.save(f,p)
    print('T312R remaining engineering recovery registered,0native0optimizer; original deadlines fixed')
if __name__=='__main__':main()
