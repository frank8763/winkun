from pathlib import Path
import json,gzip,hashlib,time,numpy as np
R=Path('/home/linghu/Documents/Codex/2026-10-30/new-chat/winkun');D=R/'stage/docs/day_20261124/t716';start=time.time();end=start+180
save=lambda f,v:f.write_text(json.dumps(v,indent=2)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(rel):
 p=R/rel;assert json.loads((p/'audit.json').read_text())['state']=='PASS';return {r['job_id']:r for r in map(json.loads,gzip.decompress((p/'cases.jsonl.gz').read_bytes()).decode().splitlines()) if not r.get('exact')}
safe=lambda r:r['success'] and not r['collision'] and not r.get('field_boundary')
selection=json.loads((R/'stage/docs/day_20261124/t714/SOURCE_SELECTION.json').read_text())['rows'];compact={**rows('stage/runs/t703_projected_local/new_proj8'),**rows('stage/runs/t711_fresh_starts/new_compact')};replay=rows('stage/runs/t712_retention_tradeoff/new_replay50');fixed=[];missing=[];maxphysics=0
for entry in selection:
 assert time.time()<end;jid=entry['job_id'];b=entry['baseline'];c=compact[jid];teacher=c if safe(c) else replay[jid];assert safe(teacher)
 for k in ['initial_native_integration','scene_sha256','task_goal_xy']:assert b['array_hashes'][k]==teacher['array_hashes'][k]
 for r in [b,teacher]:assert sha(r['path'])==r['sha256'] and sha(r['request']['model'])==r['loaded_model_sha256']
 with np.load(b['path'],allow_pickle=False) as f:bp=f['poses'].copy();bc=f['native_counts'].copy();be=f['end_flags'].copy()
 with np.load(teacher['path'],allow_pickle=False) as f:tp=f['poses'].copy();tc=f['native_counts'].copy();te=f['end_flags'].copy()
 bt=np.r_[0,np.cumsum(bc[:-1])];tt=np.r_[0,np.cumsum(tc[:-1])];bytime={int(t):i for i,t in enumerate(tt)};first=None;ti=None
 for i,t in enumerate(bt):
  if int(t) not in bytime:continue
  j=bytime[int(t)]
  if i>0 and j>0 and not be[:i].any() and not te[:j].any() and np.linalg.norm(bp[i,:2]-tp[j,:2])>=.01:first=i;ti=j;break
 record=dict(id=jid,assigned_teacher='compact' if teacher is c else 'replay50',baseline=b,teacher=teacher,recovery_first=first,teacher_reference_decision=ti,equal_native_prefix_steps=int(bt[first]) if first is not None else None,initial_scene_native_goal_exact=True)
 if first is None:missing.append(jid)
 fixed.append(record);maxphysics+=2*int(b['request']['job']['case']['step_cap'])
assert len(fixed)==18 and sum(r['assigned_teacher']=='compact' for r in fixed)==14
save(D/'FIXED_ASSIGNMENT.json',dict(state='QUALIFIED_EXISTING_ORIGINAL_START_SUCCESSES_ONLY',rows=fixed,ids=18,compact_ids=14,replay50_ids=4,unavailable_recovery_points=missing,maximum_recovery_and_exact_physics=maxphysics,no_recovery_result=True,new_native=0,new_optimizer=0,new_model_forward=0,elapsed_seconds=time.time()-start))
print(json.dumps(dict(ids=18,compact=14,replay50=4,available_recovery_points=18-len(missing),unavailable=missing,maxphysics=maxphysics,new_native=0,new_learning=0,elapsed_seconds=time.time()-start)))
