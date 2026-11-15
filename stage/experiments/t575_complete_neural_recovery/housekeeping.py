"""Readiness archival and reversible remote-only eviction of fixed ended caches."""
from pathlib import Path
import importlib.util,json,sys,time,os,io,tarfile,hashlib
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261115/t575';LEDGER=DOC/'COLD_CACHE.json'
def read(f):return json.loads(Path(f).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def readiness_backup():
 a=loadmod('_t575_readiness_transport',ROOT/'stage/experiments/t493_wide8_night/transport.py');a.OLD.install();T=a.OLD.T
 def stopped():
  s=read(ROOT/'stage/runs/t570r_multicore_launch_fix/summary.json');w=read(ROOT/'stage/runs/t570r_multicore_launch_fix/workers.json');assert s['all_workers_stopped'];assert all(not Path('/proc',str(x['pid'])).exists() for x in w)
 a.stopped=stopped;stopped();release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
 for name in ['T570','T570R']:
  folder=ROOT/'stage/runs'/('t570_multicore_readiness' if name=='T570' else 't570r_multicore_launch_fix');files=sorted(f for f in folder.rglob('*') if f.is_file());r=a.archive(name,files,release,time.time()+300,T);print(name,r['state'])
def evict():
 assert not LEDGER.exists();B=loadmod('_t575_old_cold',ROOT/'stage/experiments/t552_joint_robust_night/cold_cache.py');T=B.backend();records=[];archives={};groups=['T551','T552'];protected=set()
 for group in groups:
  files=read(ROOT/f'stage/docs/long_distance_20261101/source_manifest_{group}.json')['files'];eligible=[x for x in files if '/episodes/full_' in x['path'] and x['path'].endswith('.npz') and (ROOT/x['path']).exists()];assert eligible
  blob,archive=B.fetch(group,T);archives[group]=archive
  with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
   for x in eligible:
    f=ROOT/x['path'];assert str(f) not in protected;data=f.read_bytes();assert len(data)==x['bytes'] and sha(data)==x['sha256'] and tar.extractfile(x['path']).read()==data;records.append(dict(x,group=group))
 total=sum(x['bytes'] for x in records);assert total==10388806,(total,len(records));ledger=dict(state='JOURNALED_BEFORE_EVICTION',archives=archives,records=records,completed=[],released_bytes=0,all_original_bytes_fresh_remote_verified=True,restore_script='stage/experiments/t575_complete_neural_recovery/housekeeping.py restore PATH',current_models_and_inputs_untouched=True,epoch=time.time());save(LEDGER,ledger)
 for x in records:
  f=ROOT/x['path'];assert sha(f.read_bytes())==x['sha256'];f.unlink();ledger['completed'].append(x['path']);save(LEDGER,ledger)
 ledger.update(state='COMPLETE_REMOTE_RESTORABLE_COLD_CACHE',released_bytes=total);save(LEDGER,ledger);print('T575_COLD_RELEASE',total)
def restore(path):
 B=loadmod('_t575_restore_old',ROOT/'stage/experiments/t552_joint_robust_night/cold_cache.py');B.LEDGER=LEDGER;B.restore(path)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
