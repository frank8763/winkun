from run import *
import ast
def main():
    assert not OUT.exists();p=read(ROOT/'stage/runs/t313_fisher_feasibility/protocol.json');assert read(DOC/'T313_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T313.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    inputs=dict(p['input_hashes'])
    for f in [*HERE.glob('*.py'),DOC/'T314_PLAN.md',DOC/'T313_AUDIT.json',receipt,*[ROOT/'stage/runs/t313_fisher_feasibility'/n for n in ['protocol.json','summary.json','metrics.json']]]:inputs[str(f)]=U.K.C.sha(f)
    for v in read(ROOT/'stage/runs/t313_fisher_feasibility/summary.json')['models'].values():inputs[v['path']]=v['sha256']
    p.update(experiment='T314_READ_ONLY_DIRECTIONAL_AUDIT',input_hashes=inputs,max_autograd_calls=73,streamed_gradient_norm_relative_tolerance=1e-5,wall_seconds=900,audit_seconds=900,final_bytes=1024**2,payload_bytes=0,metadata_bytes=768*1024,max_commands=0,max_physics=0,
        contract='Independent read-only causeaudit afterT313 jointgatefailed; no optimizer/parameters/proposals/native and no bypass ofgate. Samefixedfirst16/MCV/GAE/oldanchors. Two full and minibatch gradientvectors at originalsafeActor via exactratio1 firstorder identity, streaming16trajectoryalgebra doubleaccumulation; 64trajectoryautograd+8oldfull+1oldmini=73 main,73audit. Fixedrelative1e-5 numerical normcheck vs original batched gradient, not looser density. Four existing offline lastcandidateweights read asvectors only, NEVERload candidate intoActor or produceanotherproposal. Dot full/minigrad with actualfloat32 candidatedelta; per6goal directionalderivatives, retentionderivative andactualfinite-stepcost separated; finitechange-minuslinearterm notHessianproof. ExplicitAdam.step forbidden. Zero new directions/CG/backtracking/trials/grid or training evenif favorable. preflight1scoregrad0nativeAdamproposal. Main900/audit900/own1MiB/global1GiB/free40GiB; no deletion or compression ofcurrentevidence. SourceSHA/cleancommit/fullremoterootbeforeunique launch; active noother runswrite/sourcesync. Endallprocessesabsent independentvectors/derivatives/hash audit/rawfullremoteeverymemberSHA. Stageonly no Phase/map/planner/waypoints/runtimecriticteacherrouter; deploymentunchanged. Old275/six54/fresh16/reload78 not executed; no successrate/new6mvideo/7or8.')
    assert U.K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    for f,h in inputs.items():assert U.K.C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:U.K.C.save(f,p)
    print('T314 read-only gradient audit registered,0native0Adam0proposal')
if __name__=='__main__':main()
