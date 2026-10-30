"""Read-only Stage repository readiness; no simulation or model unpickling."""
from pathlib import Path
import hashlib,json,sys
import numpy,torch,mujoco
from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a_dense_env import Phase10Stage1ADenseEnv
root=Path(__file__).resolve().parents[2]
m=json.loads((root/'stage/manifests/archive_manifest.json').read_text());bad=[]
for row in m['files']:
 path=root/row['archive_path']
 if not path.is_file():bad.append([row['archive_path'],'MISSING']);continue
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if path.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:bad.append([row['archive_path'],'HASH_MISMATCH'])
print(json.dumps({'state':'READY_PAUSED' if not bad else 'FILES_NEED_ATTENTION','repository':str(root),'python':sys.executable,'torch':torch.__version__,'numpy':numpy.__version__,'mujoco':mujoco.__version__,'manifest_files_checked':len(m['files']),'failures':bad,'simulation_steps':0,'training_updates':0,'automatic_resume':False},indent=2))
sys.exit(bool(bad))
