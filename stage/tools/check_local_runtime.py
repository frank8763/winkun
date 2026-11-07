"""Read-only Stage repository readiness; no simulation or model unpickling."""
from pathlib import Path
import hashlib,json,sys
import numpy,torch,mujoco
from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a_dense_env import Phase10Stage1ADenseEnv
root=Path(__file__).resolve().parents[2]
m=json.loads((root/'stage/manifests/archive_manifest.json').read_text());bad=[];cold_count=0
cold_file=root/'stage/manifests/remote_only_20261107.json';cold={}
if cold_file.is_file():
 for record in json.loads(cold_file.read_text())['records']:
  for original in record['rows']:
   cold[original['archive_path']]=record
for row in m['files']:
 path=root/row['archive_path']
 if not path.is_file():
  record=cold.get(row['archive_path'])
  if record and record['state']=='REMOTE_ONLY' and record['bytes']==row['bytes'] and record['sha256']==row['sha256'] and record['remote_member_sha256']==row['sha256'] and record['asset']['state']=='FULL_REMOTE_ARCHIVE_AND_MEMBER_SHA_VERIFIED':cold_count+=1;continue
  bad.append([row['archive_path'],'MISSING']);continue
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if path.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:bad.append([row['archive_path'],'HASH_MISMATCH'])
print(json.dumps({'state':('READY_WITH_REMOTE_HISTORY' if cold_count else 'READY_PAUSED') if not bad else 'FILES_NEED_ATTENTION','repository':str(root),'python':sys.executable,'torch':torch.__version__,'numpy':numpy.__version__,'mujoco':mujoco.__version__,'manifest_files_checked':len(m['files']),'verified_remote_only_history_files':cold_count,'historical_restore_command':'python stage/tools/restore_cold.py <repo-relative-path>','failures':bad,'simulation_steps':0,'training_updates':0,'automatic_resume':False},indent=2))
sys.exit(bool(bad))
