from t304_core import *
import ast


def main():
    assert not OUT.exists()
    source_protocol = ROOT/'stage/runs/t299_conditional_goal_likelihood/protocol.json'
    base = read(source_protocol)
    checkpoint = ROOT/'stage/runs/t299_conditional_goal_likelihood/conditional_goal_likelihood_final_checkpoint.pt.gz'
    source_cases = source_protocol.parent/'cases.jsonl.gz'
    rows = list(map(json.loads,gzip.decompress(source_cases.read_bytes()).decode().splitlines()))
    ph = 'd104ea9dc4aaeef6fbdc65947392ce05f3334101c6393e18194ad9eb03ffebb0'
    inputs = dict(base['input_hashes']); sources = []
    for side in ['LEFT','RIGHT']:
        for number in [3,9,18,24]:
            name = f'dev6_{side}_{number:02d}'
            r = next(r for r in rows if r['role']=='development' and r['source_reset_id']==name)
            assert r['complete'] and r['policy_hash']==ph and C.sha(r['path'])==r['sha256']
            with np.load(r['path']) as z:
                actions = z['actions'].copy()
            decision = (len(actions)-3)//2
            original = actions[decision:decision+3].copy()
            chunks = {}
            for arm,sign in [('minus',-1),('center',0),('plus',1)]:
                a = original.copy(); a[:,1] = np.clip(a[:,1]+sign*.03,-.999999,.999999)
                chunks[arm] = a.tolist()
            assert len({str(v) for v in chunks.values()})==3
            job = next(j for j in base['development_jobs'] if j['id']==name)
            sources.append(dict(id=name, side=side, split='test' if number==24 else 'train',
                reference=r, job=job, branch_decision=decision, chunks=chunks))
            inputs[r['path']] = r['sha256']
    for f in [source_protocol,checkpoint,source_cases,
              ROOT/'stage/docs/distance_20261102/T299_AUDIT.json',
              ROOT/'stage/docs/long_distance_20261101/release_receipt_T299.json',
              ROOT/'stage/docs/distance_20261102/T303_AUDIT.json',
              ROOT/'stage/docs/distance_20261102/T304_PLAN.md',
              ROOT/'stage/docs/distance_20261102/T304_LITERATURE.md',*HERE.glob('*.py')]:
        inputs[str(f)] = C.sha(f)
    p = dict(experiment='T304', source_protocol=str(source_protocol),source_checkpoint=str(checkpoint),
        policy_hash=ph,sources=sources,input_hashes=inputs,arms=['minus','center','plus'],
        chunk_length=3,steer_offset=.03,workers=4,seed=2026110209,
        head_steps=500,head_learning_rate=.001,value_margin=1e-8,
        max_commands=48,max_physics=1344000,wall_seconds=2400,final_reserve_seconds=300,
        audit_seconds=600,final_bytes=int(7.25*1024**2),payload_bytes=6016*1024,metadata_bytes=1024**2,
        episode_bytes=192*1024,model_bytes=256*1024,evidence_bytes=128*1024,
        global_bytes=1024**3,min_free_bytes=40*1024**3,
        contract='Native fixedT299 researchActor original6m interventional value pilot, not new navigation learning/SGCRL or action-chunked CRL reproduction, not grid/continuationT294. Fixedsame LEFT/RIGHT consumeddev ids03/09/18/24, one frozenpolicy/goal120/time/native termination throughout; parentcontrols exact originalT299 full12arrays+integration. Atdecisionfloor((parentdecisioncount-3)/2), prescribethreeopenloop atomic actions copiedfromparent trajectory, samedrive andsteer offsets minus.03/0/plus.03 clippednativebounds, then sameT299 deterministicfeedback atactualbranch states. Currentgoalneverchanges, no state teleport, waypoint/map/planner/runtime teacher/router orPhase. 24uniquephysicaltrajectories+24independentexactreplays allcharged, not48 independent seeds. Beforebranch full120/prefix mustmatchsamecontrol; firstchunk precommitted andnot feedback-conditionedfutureactionlabels; postchunk everyaction sameActor. Deterministiclogp0 placeholders neverARonpolicy. DefineQ3 of thisprescribedchunk followedbyfixedActor, actualeta.1MCfrombranchstart underoriginalreward/cost, targetadvantage Gbranch-Gcenter, notmixedbehavior firsthitprobability/density orDPGgradient. 6train resetfamilies18rows vs2consumedtestfamilies6rows(id24eachside), repetitions nottraining rows. Trainmatched4609param MLP126-32-16-1 action vsblindzero6 sameinit/fullbatch500Adam each/clip5/lr.001 seedfixed; trainingonlynormalize/targetRMSfloor1e-6. Strict heldoutrelativeMSE/rank/value margin fixed, regardlesspass noActorupdate/6mpromotion7or8. Threeprimitiveholds.6s motivatedby2026chunkwork, notassertactuatorinertia provedrootcause orclaimednewalgorithm. Independent1000disposableheadAdam audit exactmodeloptimizer/loss/predictions, all24full12012arrays/replays/sourceSHA/actionsemantics/counts. 48commands1.344Mphysics2400s last300forfit/final/own7.25MiB/global1GiB/free40GiB; fourpartialreplays192KiBeach reserved, failuremodels kept no restart. 24x192KiBfull +2x256KiBmodels +128KiBevidence+4x192KiBpartial =6016KiB; metadata1MiB, buffer384KiB. Fourworkers countsrealphysicalinclinflight, source sync onlyafter allstopped/audit/fullRAMremotemembers. AllsafeT255/T299/HDvideos unchanged.')
    assert len(sources)==8 and p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    for f,h in inputs.items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:C.save(f,p)
    print('T304 registered: fixed native3-action Q/pairedhead pilot, zero native/optimizer so far')


if __name__=='__main__':main()
