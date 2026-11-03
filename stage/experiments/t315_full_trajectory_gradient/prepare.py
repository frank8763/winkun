from t315_core import *
import ast

def main():
    assert not OUT.exists();p=read(ROOT/'stage/runs/t313_fisher_feasibility/protocol.json')
    for name in ['T313','T314']:
        assert read(DOC/(name+'_AUDIT.json'))['state']=='PASS'
        assert read(ROOT/('stage/docs/long_distance_20261101/release_receipt_'+name+'.json'))['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    source=ROOT/'stage/experiments/t313_fisher_feasibility/t313_core.py'
    old={n.name:ast.get_source_segment(source.read_text(),n) for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)}
    current=HERE/'t315_core.py';new={n.name:ast.get_source_segment(current.read_text(),n) for n in ast.parse(current.read_text()).body if isinstance(n,ast.FunctionDef)}
    for n in ['flat','put','cg','baseline_batch','fisher_rows','measure']:assert old[n]==new[n],n
    inputs=dict(p['input_hashes'])
    for f in [*HERE.glob('*.py'),source,DOC/'T315_PLAN.md',DOC/'T314_AUDIT.json',DOC/'T314_FINDINGS_GRADIENT_REFERENCE.json',ROOT/'stage/runs/t313_fisher_feasibility/metrics.json',ROOT/'stage/runs/t314_directional_audit/metrics.json',
              *[ROOT/('stage/docs/long_distance_20261101/release_receipt_'+n+'.json') for n in ['T313','T314']]]:inputs[str(f)]=K.C.sha(f)
    p.update(experiment='T315_FULL_TRAJECTORY_NEW_TASK_GRADIENT',input_hashes=inputs,wall_seconds=1800,audit_seconds=1800,
        max_full_gradient_Adam_calls=2,max_backtracking_parameter_proposals=32,max_total_candidate_parameter_updates=34,
        final_bytes=14*1024**2,payload_bytes=12*1024**2,metadata_bytes=1024**2,model_bytes=3*1024**2,max_commands=0,max_physics=0,
        contract='Explicit user start. Independent bounded zero-native fulltrajectorynewtaskgradient factor diagnostic afterT313/T314 stopped/audited/fullbackup. Samefirst16 actualgoal/onpolicy histories/MCV/GAE/cachedV/normalizer/sigma/rho0/reward/hiddenreset120/safeActor/freshAdam. ONLY newtaskgradient sampling changes from16rows pertrajectory to complete16episodes equaltrajectory/goal; sameoriginal128oldretention sampledrows in direction, no simultaneousoldcoverage/softweight change. Ratio1 logprob firstorder identity with unchanged .2clip objective; streamdoublepertrajectory gradients then castfloat32/clipnorm1, numerical accumulation path disclosed. Fulloldanchor objective gate unchanged. T313same128scorematrix andselection SHA required, same.1damping/CG10/tol1e-12/.02quadraticinit/bothdirections8x.5backtracks/k3.02/maxabs1/fullobjective>1e-10/eachsixgoalnondegrade1e-10. No lambda/LR/noise/threshold/seed/CGgrid. 2NEWofflinefullgradientAdam rawcandidates (not historicalreconstructions) plus<=32normalizedparameterproposals,<=34totalcandidateupdates, deployedsafeunchanged. RawAdamstate/exactgradient/policySHA preservedbydeterministicrecipe+inputSHA andindependentaudit; 4lastpolicies savedincludingfailure, no resumableAdamstate falselyassigned to normalizedpolicies. No Vrefit/native/old275/six54/fresh16/reload78/7or8 andno automaticnative budget evenifjointproxygatespass. EmpiricalFisher feasiblebotharms andAdam feasiblebotharms separate finiteproxy screens, notnavigationornewalgorithmadvantage. Main1800/audit1800/own14MiB/global1GiB/free40GiB actualworstreserve mandatory,4policy3MiBeach+metadata1+buffer1,no giantmatrixfile. Preflight19autograd0native0Adam0proposal coversrealallrowdensity/fullgradreference/clipfirstorderidentity/SPDsolver/Actorunchanged. Main sourceSHA/cleancommit/fullremoterootbeforeuniquelaunch. Active singlewriter nooutside runswrites/sourcesync/protocolchanges. Endallprocessesabsent independenteveryactual2newAdam/scorematrix/fullgrad/CG/proposals/models exactreconstruction+allrawRAMremoteeachmemberSHA/source. Preserveoriginalsafe/T299/failures/HDvideo,Phasefullyisolated,no maps/planner/waypoints/runtimeCriticteacherrouter.')
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=p['global_bytes'],(used,p['final_bytes'])
    for f,h in inputs.items():assert K.C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:K.C.save(f,p)
    print('T315 registered, matched helpers exact,0native0Adam0proposal; worst reserved',used+p['final_bytes'])
if __name__=='__main__':main()
