from run import *
import gzip

def main():
    assert not OUT.exists();source=ROOT/'stage/runs/t299_conditional_goal_likelihood';plan=json.loads((source/'protocol.json').read_text());summary=json.loads((source/'summary.json').read_text());exp=summary['arms']['conditional_goal_likelihood']['export'];rr=list(map(json.loads,gzip.decompress((source/'cases.jsonl.gz').read_bytes()).decode().splitlines()));rows=[r for r in rr if r['role']=='development' and r['success'] and not r['collision']];assert len(rows)==10
    inputs=dict(plan['input_hashes']);sources=[]
    for r in rows:
        job=next(j for j in plan['development_jobs'] if j['id']==r['source_reset_id']);sources.append(dict(id=r['source_reset_id'],job=job,reference=r))
    for f in [source/'protocol.json',source/'summary.json',source/'cases.jsonl.gz',Path(exp['path']),ROOT/'stage/docs/distance_20261102/T300_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/T300_LOG_ENCODING_RECEIPT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T300.json',*HERE.glob('*.py')]:inputs[str(f)]=C.sha(f)
    p=dict(experiment='T301',source_protocol=str(source/'protocol.json'),source_checkpoint=exp['path'],policy_hash=exp['policy_hash'],sources=sources,input_hashes=inputs,max_commands=10,max_physics=280000,wall_seconds=900,episode_bytes=192*1024,metadata_bytes=512*1024,payload_bytes=10*192*1024,final_bytes=int(2.5*1024**2),global_bytes=1024**3,min_free_bytes=40*1024**3,contract='Exact recovery of existing10T299 RIGHT6m successful deterministic development trajectories, not new independent data/model improvement/training/promotion. Source frozenT299 checkpoint/hash, same initial snapshot/goal/native protocol; full12arrays including120states/nextstates andintegration must exactmatchoriginal registeredSHA.10commands280kphysics900s/own2.5MiB/global1GiB/free40GiB,0optimizer/Actorupdates. Fullstatechain192KiB each, partial/rawfailures preserved no restart. Nativeexclusive nooutside runswrites/sourcesync; source exactremote beforeuniquelaunch, endedallprocessesabsent zeronativeaudit fullSHA/counts andfullRAMremotememberreadback. SafeT255/HDvideos/alloriginals retained, Phaseisolated/no map/planner/waypoint/teacher/router/7or8. Output data for separatelyregistered bounded next learning, old10/54 not stability.')
    for f,h in inputs.items():assert C.sha(f)==h,f
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'];OUT.mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:save(f,p)
    print('T301 registered 0native0optimizer')

if __name__=='__main__':main()
