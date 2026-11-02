"""Read-only coverage inventory for the next LEFT/right mechanism decision."""
from compare import ROOT,C,S,np,Path,json,gzip
import re,hashlib

def main():
    dest=ROOT/'stage/docs/distance_20261102/T299_COVERAGE_INVENTORY.json';assert not dest.exists();left=[];seen=set()
    for folder in ['t263_adaptive_goal_course','t267_causal_credit_course']:
        f=ROOT/'stage/runs'/folder/'training.jsonl.gz';records=list(map(json.loads,gzip.decompress(f.read_bytes()).decode().splitlines()))
        for r in records:
            if not(r.get('complete') and r.get('success') and not r.get('collision') and 'LEFT' in r['id']):continue
            distance=re.search(r'course([0-9.]+)_',r['id']);assert distance
            if float(distance[1])<5.5:continue
            path=Path(r['path']);assert C.sha(path)==r['sha256']
            with np.load(path) as z:a={k:z[k].copy() for k in z.files}
            assert len(a['states'])==len(a['actions']) and a['states'].shape[1]==a['next_states'].shape[1]==120
            assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and np.array_equal(a['end_flags'][-1],[0,1,0]) and not a['end_flags'][:-1].any()
            assert all(S.digest(v)==r['array_hashes'][k] for k,v in a.items())
            key=(r['array_hashes']['states'],r['array_hashes']['actions'],r['integration_sha256']);duplicate=key in seen;seen.add(key)
            left.append(dict(experiment=folder,id=r['id'],source_reset_id=r.get('source_reset_id'),nominal_goal_distance_m=float(distance[1]),actual_initial_goal_distance_m=float(a['states'][0,48]*10),source_path=str(path),source_sha256=r['sha256'],complete_full120_and_endflags_SHA_verified=True,source_policy_hash=r.get('policy_hash'),physics_steps=r['physics_steps'],complete=True,safe=True,exact_historical_duplicate=duplicate,key=list(key)))
    current=ROOT/'stage/runs/t299_conditional_goal_likelihood';rows=list(map(json.loads,gzip.decompress((current/'cases.jsonl.gz').read_bytes()).decode().splitlines()));success=[r for r in rows if r['role']=='development' and r['success'] and not r['collision']];ids={r['source_reset_id'] for r in success};prior=[]
    for directory in (ROOT/'stage/runs').iterdir():
        m=re.match(r't(\d+)',directory.name)
        if not directory.is_dir() or not m or not 256<=int(m[1])<=298:continue
        for f in directory.rglob('*'):
            if not f.is_file() or not(f.name.endswith('.jsonl') or f.name.endswith('.jsonl.gz')):continue
            try:lines=(gzip.decompress(f.read_bytes()) if f.suffix=='.gz' else f.read_bytes()).decode().splitlines()
            except (OSError,UnicodeError):continue
            for line in lines:
                if 'dev6_RIGHT_' not in line:continue
                try:r=json.loads(line)
                except ValueError:continue
                if not isinstance(r,dict) or not(r.get('complete') and r.get('success') and not r.get('collision')):continue
                if any(v in (r.get('id','')+' '+str(r.get('source_reset_id',''))).lower() for v in ['course','auxiliary','local1m','remaining']):continue
                sid=r.get('source_reset_id')
                if sid not in ids:
                    sid=next((v for v in ids if r.get('id','').endswith('_'+v)),None)
                if sid in ids:prior.append(dict(id=sid,path=str(f),record_id=r['id'],role=r.get('role')))
    prior_ids=sorted({r['id'] for r in prior});new_ids=sorted(ids-set(prior_ids));timeouts=[r for r in rows if r['role']=='development' and r['group']=='6m_RIGHT' and r['timeout']]
    dest.write_text(json.dumps(dict(scope='Posthoc finite read-only evidence inventory, not new training/gate or new physical verification. LEFT only T263/T267 explicit completed safe nominal>=5.5m full120SHA; original data remain read-only. RIGHT prior search StageT256-T298 native JSONL with exact source_reset_id or id suffix, not all chats/unnamed geometries. New successes are consumed development, not independent unseen test.',native_steps=0,optimizer_calls=0,Actor_updates=0,verified_LEFT_bridge_records=left,unique_LEFT_bridge_trajectory_keys=len(seen),prior_current_success_ids=prior_ids,new_current_success_ids_in_scan=new_ids,prior_success_records=prior,initial_inventory_correction='Initial posthoc id-suffix selector also included course5.25/5.5 records endingdev6RIGHT13; initialJSON/source preserved. This version excludes course/auxiliary/local1m/remaining before exactid match. Current 6m results unchanged; historical stochastic successes remain distinct from deterministic development.',current_RIGHT_safe_ids=sorted(ids),current_RIGHT_timeout_remaining_m=[dict(id=r['source_reset_id'],remaining_m=r['remaining_m']) for r in timeouts],current_LEFT_original6m_safe=0,source_sha256=C.sha(Path(__file__))),indent=2)+'\n');print('Coverage inventory:',len(left),'LEFTbridge records/',len(seen),'unique;',len(new_ids),'newRIGHT6m devpoints relative explicitscan')

if __name__=='__main__':main()
