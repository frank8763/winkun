from recorder import *
import ast
def main():
    assert not OUT.exists();old=ROOT/'stage/runs/t316_frozen_native_validation';p=read(old/'protocol.json');assert read(DOC/'T316_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T316.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    job=next(j for j in p['development_jobs'] if j['id']=='dev6_LEFT_13');geometry=next(j for j in read(old/'geometry.json') if j['id']=='dev6_LEFT_13');inputs=dict(p['input_hashes'])
    for f in [*HERE.glob('*.py'),HERE/'index.html',DOC/'T317_LITERATURE_AND_METHOD.md',receipt,old/'geometry.json',DOC/'T316_AUDIT.json',ROOT/'stage/experiments/t295_hindsight_support/run.py',ROOT/'stage/experiments/t288_five_metre_presentation/run.py']:
        inputs[str(f)]=K.C.sha(f)
    p.update(experiment='T317_HUMAN_LEFT6_DEMONSTRATION',input_hashes=inputs,job=job,initial_geometry=geometry,port=18763,max_attempts=3,max_commands=8,max_physics=168000,wall_seconds=1800,final_reserve_seconds=180,audit_seconds=300,
        episode_bytes=512*1024,preview_bytes=128*1024,final_bytes=3*1024**2,payload_bytes=2*1024**2,metadata_bytes=896*1024,
        contract='Userexplicitlyacceptedfewhuman demonstrationsintraining, deploymentstill1Actor/no map/planner/waypoints/runtime teacher/router/Phase. Thiscollector0Actor/Adam/gradient/training. OriginalLEFT6center13/nativeinitial120+integration+scene+goal exactT316,56simseconds/28kphysical/.20m/collisionpriority/hold100native unchanged. Humanbrowserexplicitopeninterval[-.999,.999] drive/steer only, no autoActordriver/no AI drivingorplanner. Start/advance buttons only; beforeadvance0physics. Overheadrenderer privilegedhumantrainingview disclosed, copiedmodeldata unchangednativeintegration; no visual/mapinputtoActor. Up to3humanattempts each<=28k and1exact nativeactionreplay each,168kallphysical; preflight1geometry+mainpreview1geometry+3record+3replay=8allcommands. NativehumanGOAL notautonomoussuccess; collision/timeout/admincensor/missing separate; no expertlogp/Actoronpolicyfiction. Full120statesnextstates/real3history/actions/counts/poses/nativeenvironment reward/initialfinalintegration losslesslysaved,rawhumaninputpersisted beforestep; perdecision atomiccheckpoint<=512KiB,3files1.5MiB+max.5MiBpartial=2MiBpayload,metadata.875MiB(protocol<=.5+preview.125+inputlog.125+logs.03125+summary.09375),buffer.125=3MiB,globalallruns1GiB/free40GiB neverexpanded. No diskvideo or newmodelcopy. Main1800last180reserve,externalearlierstrict; no attemptunless2capsremainingphysical/commands; no silentseed/attempt/noiseextension. Usermaypausebutwallstillcounts, ends3attempts/userfinish/absolute window. Sameactions replay full12arrays/terminalintegrationexact chargedphysical,notindependentnavsample. Zero-step preflight copiedactualrenderer/nativehash/openactions/noAdam. Exactsource/inputSHA/cleancommit/fullremoterootbeforeunique serverlaunch. Activeexclusivewriter nooutside runswrites/sourcesync; no archivewhilewriting. Endallprocesses/renderer/serverstop thenindependent0physicsraw/history/reward/humaninput/hashaudit andfullrawRAMremoteeverymemberSHA/source. Models/T299/HD/allfailuresevidencepreserved. No auto BC/RL/old275/six54/fresh16/reload78 or7/8; obtaininggooddemonstrationisnextdata gate,not6mstableororiginalinnovation.')
    assert 3*p['episode_bytes']+p['episode_bytes']==p['payload_bytes']
    assert 512*1024+128*1024+128*1024+32*1024+96*1024<=p['metadata_bytes']
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes'];used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=p['global_bytes']
    for f,h in inputs.items():assert K.C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:K.C.save(f,p)
    assert (OUT/'protocol.json').stat().st_size<512*1024
    print('T317 humancollector registered,0physics0optimizer,full worst reserve',used+p['final_bytes'])
if __name__=='__main__':main()
