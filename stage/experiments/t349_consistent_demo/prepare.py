from t349_core import *


def main():
    assert not OUT.exists() and read(DOC/'T348_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T348.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/experiments/t348_causal_sequence_warmstart/protocol.json'))
    p.update(experiment='T349_SINGLE_CONSISTENT_HUMAN_STYLE_SEQUENCE_RESET',
        sequence_sources=[dict(name='human_T347_matched_start',path=str(ROOT/'stage/runs/t347_original_start_human/attempt_01.npz'),label_start=0)],
        wall_seconds=1800,final_reserve_seconds=600,audit_seconds=600,all_work_seconds=3600,
        workers=2,final_bytes=int(3.25*1024**2),full_bytes=384*1024,max_full_cases=2,
        compact_bytes=4096,max_compact_cases=57,partial_bytes=384*1024,model_bytes=256*1024,
        metadata_bytes=768*1024,logs_bytes=128*1024,
        contract='One new source-consistency hypothesis after T348,not extra oldtraining budget. Same safeT255/new6306parameters/float64GRU16head/twoarmsreset-memory/initseed20261104348/LR.001/600Adam each/old211 episode schedule and totalold-new.5:.5. Newsource onlyT347real full314success,removes all4other sourcegroups. T347 exposures120->600 changes withsourcecount;not pure proof conflictinglabels caused failure. Wholecausal episode currentparameter BPTT,realhiddencarry vsreset, no new observations/Q/RL. Same wheel5force120/wide1.17m80s/originalnarrow56s. Two nativeworkers, same120commands/4.52Mphysical allincluding7geometry/5initial/2*(27+27);onlymatchedwide fullsavedperarm,rest57compact/54readbackfullarrayhiddenhistoryexact inRAMhash. 1200live supervised+1200shadow/preflight8discarded/audit1200historical,0newRL or physicaltraincollection. Quota outsidefullreserve+flocksinglewriter,exactown checks>=.5s/fulloutside manifest<=10s and finalexact,changedoutsidefails;syntheticoverflow/mutation preflight. Main1800last600evaluation/audit600/all3600/own3.25MiB including2partial/global1GiB/free40GiB,no quota expansion/deletion. Full275/54/fresh16/reload78 unexecutednull,no stableoriginal6/no7or8. No planner/map/waypoint/runtime teacher/router/Phase. No seed/epoch/weight/modelgrid,restart or automatic follow-onif fail.')
    assert 2*p['full_bytes']+57*p['compact_bytes']+2*p['partial_bytes']+2*p['model_bytes']+p['metadata_bytes']+p['logs_bytes']<=p['final_bytes']
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T349_PLAN.md'),str(DOC/'T348_AUDIT.json'),str(receipt),str(ROOT/'stage/experiments/t348_causal_sequence_warmstart/protocol.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,only_new_source='T347',reserved_global=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()
