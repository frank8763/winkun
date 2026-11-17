from pathlib import Path
import json,subprocess,os,io,time
import t627_learning as M
import t627_stream as S
def download(p,receipt):
 c=subprocess.run([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t627_stream.py'),'download'],cwd=M.ROOT,input=json.dumps(receipt).encode(),capture_output=True,timeout=p['stream_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));assert c.returncode==0,c.stderr.decode(errors='replace')[-1000:];return S.unpack(c.stdout)
def prior(p,verify=True):
 groups=[]
 for i in range(8):
  receipt=M.read(Path(p['reuse_parent_root'])/f'batch_{i:02d}_receipt.json');files,manifest=download(p,receipt);group=[]
  for r in manifest['rows']:
   b=files[Path(r['path']).name];assert S.digest(b)==r['sha256'];a=M.P.Base.G.E.N.S.arrays(io.BytesIO(b));assert all(M.digest(v)==r['array_hashes'][k] for k,v in a.items());assert r['policy_hash']==p['student_initial']['policy_hash'];group.append(a)
  assert len(group)==4;groups.append(group)
 return groups
