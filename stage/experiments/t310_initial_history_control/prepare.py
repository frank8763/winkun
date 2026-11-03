from run import *
import ast

def main():
    assert not OUT.exists();base=read(ROOT/'stage/runs/t309_sequence_goal_value/protocol.json')
    old=read(ROOT/'stage/runs/t287_time_aligned_credit/protocol.json')
    raw=list(map(json.loads,gzip.decompress((ROOT/'stage/runs/t287_time_aligned_credit/cases.jsonl.gz').read_bytes()).decode().splitlines()))
    sources=old['old_centers']+[next(j for j in old['development_jobs'] if j['id']==f'dev6_{s}_13') for s in ['LEFT','RIGHT']]
    refs={};jobs=[];inputs=dict(base['input_hashes'])
    for j in sources:
        r=next(r for r in raw if r['source_reset_id']==j['id'] and r['role'] in ['old_regression','development'])
        assert r['policy_hash']==base['policy_hash'];refs[j['id']]=dict(array_hashes=r['array_hashes'],integration_sha256=r['integration_sha256'])
        inputs[j['exact_snapshot']]=C.sha(j['exact_snapshot'])
        for context in ['native','aligned']:
            jobs.append(dict(j,id=j['id']+'_'+context,case_id=j['id'],source_reset_id=j['id'],context=context,mode='deterministic',seed=0,aux_eta=0.,policy_hash=base['policy_hash']))
    for f in [*HERE.glob('*.py'),ROOT/'stage/experiments/t280_local_cost/t280_worker.py',
        ROOT/'stage/experiments/t309_sequence_goal_value/sequence_value.py',ROOT/'stage/runs/t309_sequence_goal_value/cases.json',
        ROOT/'stage/runs/t309_sequence_goal_value/protocol.json',ROOT/'stage/runs/t287_time_aligned_credit/cases.jsonl.gz',
        DOC/'T309_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T309.json',DOC/'T310_PLAN.md']:
        inputs[str(f)]=C.sha(f)
    p=dict(experiment='T310',policy_protocol=base['policy_protocol'],policy_hash=base['policy_hash'],input_hashes=inputs,
        jobs=jobs,historical_references=refs,max_commands=40,max_physics=sum(j['case']['step_cap'] for j in jobs)*2,
        wall_seconds=900,final_reserve_seconds=120,audit_seconds=300,final_bytes=7*1024**2,
        payload_bytes=int(5.5*1024**2),metadata_bytes=1024**2,episode_bytes=256*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,
        optimizer_calls=0,Actor_updates=0,automatic_next_training=False,
        contract='New frozen initial goal-history control, NOT extension of failedT309 value heads. SafeT255 unchanged; original8centers+original6m2centers, nativefactoryprehistory vs initial4frames aligned toactualfinalgoal using witnessed realphysicalposes/timestamps. Currentgoal/completeintegration/remainingtime/nonhistory112inputbits identical atreset; no runtimegoalchange/waypoint/map/planner/teacher/router/Phase. Fullgoalhistory then evolves natively. 20deterministicunique+20independentexactreplays=40commands; historicalnative10all12arrays+terminalintegration mustmatchT287safe. Replays/historicalcontrol notnewindependentseeds. NoActor/critic/valuehead/optimizer, no RLdata fromdeterministic0means/logp placeholders. Everyunique full120 native12arrays+3history arrays losslesschain <=256KiB,20full5MiB+2partial.5MiB,metadata1MiB+buffer.5MiB=7MiB. Finite900s last120reserve/audit300s; maxphysical fromallcaps,global1GiB/free40GiB. Source/inputSHA/preflight0native0optimizer/cleancommit/exactremote beforeunique launch. Actualbaselinefails orpartialstops, preserveallraw; no grids/restarts/promotion/7or8. Afterparenttimeoutinhibitorabsent, independent0native0optimizer audit allarrays/history/paircontrols/rewards/deterministicmode/historicalhashes; fullRAMremote everymemberSHA then sourcesync. Centers are diagnostic not old275/6m54/fresh16/78 gate; different initialobservationcontext notchangedphysics. Do not call context sensitivity training leakage or universalrootcause without further evidence.')
    assert p['max_physics']==784008 and len(jobs)==20
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    for f,h in inputs.items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:C.save(f,p)
    print('T310 paired history control registered;0native0optimizer, not navigation training')
if __name__=='__main__':main()
