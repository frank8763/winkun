"""Bounded byte-exact reserve, no model tensors or optimizer values altered."""
from pathlib import Path
import importlib.util,json,os,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('_t529_original_codec',ROOT/'stage/experiments/t321_cold_checkpoint_reserve/codec.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
LEDGER=ROOT/'stage/docs/day_20261113/T529_STORAGE_LEDGER.json';REF=ROOT/'stage/runs/t507_joint_success_transfer/pilot/block0_proposal.pt.gz'
FOLDERS=['t511_frozen_parent_residual','t515_decoder_recovery_transfer','t516_nonlinear_compatibility','t517_hard_output_constraint','t518_active_tangent_recovery','t519_fresh_full_actor_rl','t520_old_center_repair','t522_lunch_joint_relay','t523_joint_failure_signal','t524_task_gradient_balance']
def compact():
 assert not LEDGER.exists();start=time.time();ref=REF.read_bytes();assert C.sha(ref)=='ba3d7a21467b16686bd2e583f5e12a75e07e15c80c6ff0fb889ced69bf2dbe18'
 # Known latest Stage pipeline must be completely stopped, no Phase inspection.
 last=ROOT/'stage/runs/t528_module_transfer_probe';s=C.read(last/'SEALED.json');assert all(not Path('/proc',str(i)).exists() for i in s['scientific_pids']+[C.read(last/'launch.json')['pid']])
 index={}
 for m in (ROOT/'stage/docs/long_distance_20261101').glob('source_manifest_T*.json'):
  rec=m.with_name(m.name.replace('source_manifest_','release_receipt_'))
  if not rec.exists():continue
  rr=C.read(rec)
  if rr.get('state')!='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' or rr.get('manifest_sha256')!=C.sha(m.read_bytes()):continue
  for x in C.read(m).get('files',[]):index[x['path']]=(x,m,rec)
 records=[];cache={}
 for name in FOLDERS:
  for f in sorted((ROOT/'stage/runs'/name).rglob('*.pt.gz')):
   assert time.time()-start<90
   if f.name.startswith(('preflight','zero_')):continue
   relative=str(f.relative_to(ROOT));x,m,rec=index[relative];blob=f.read_bytes();assert C.sha(blob)==x['sha256'] and len(blob)==x['bytes'];encoded=C.encode(blob,ref);assert C.decode(encoded,ref)==blob
   if len(encoded)>=len(blob):continue
   dest=f.with_name(f.name+'.t529dxz');assert not dest.exists();record=dict(original_path=relative,original_sha256=x['sha256'],original_bytes=len(blob),encoded_path=str(dest.relative_to(ROOT)),encoded_bytes=len(encoded),encoded_sha256=C.sha(encoded),prior_manifest=str(m.relative_to(ROOT)),prior_manifest_sha256=C.sha(m.read_bytes()),prior_receipt=str(rec.relative_to(ROOT)))
   records.append(record);cache[relative]=(blob,encoded)
 assert sum(x['original_bytes']-x['encoded_bytes'] for x in records)>7*10**6
 protected={str(f.relative_to(ROOT)):C.sha(f.read_bytes()) for f in [REF,ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz',ROOT/'stage/runs/t527_validated_recovery_learning/pilot/proposal.pt.gz']}
 ledger=dict(state='JOURNALED_BEFORE_BYTE_EXACT_ENCODING',records=records,reference=str(REF.relative_to(ROOT)),reference_sha256=C.sha(ref),source_codec=str(Path(C.__file__).relative_to(ROOT)),source_codec_sha256=C.sha(Path(C.__file__).read_bytes()),protected_hashes=protected,completed=[],native=0,Adam=0)
 C.write(LEDGER,ledger)
 for x in records:
  assert time.time()-start<120;f=ROOT/x['original_path'];dest=ROOT/x['encoded_path'];blob,encoded=cache[x['original_path']];assert C.sha(f.read_bytes())==x['original_sha256'];ledger['current']=x['original_path'];C.write(LEDGER,ledger);f.rename(dest)
  with dest.open('r+b') as h:h.write(encoded);h.truncate();h.flush();os.fsync(h.fileno())
  assert C.decode(dest.read_bytes(),ref)==blob;ledger['completed'].append(x['original_path']);ledger['current']=None;C.write(LEDGER,ledger)
 # Independent second pass through persisted encodings, not cached blob objects.
 for x in records:
  encoded=(ROOT/x['encoded_path']).read_bytes();assert C.sha(encoded)==x['encoded_sha256'];original=C.decode(encoded,REF.read_bytes());assert C.sha(original)==x['original_sha256'] and len(original)==x['original_bytes']
 assert all(C.sha((ROOT/f).read_bytes())==v for f,v in protected.items())
 ledger.update(state='COMPLETE_ALL_ORIGINAL_GZIP_RAWZIP_AND_ADAM_BYTES_EXACT',released_payload_bytes=sum(x['original_bytes']-x['encoded_bytes'] for x in records),independent_persisted_decode_pass=True,new_encoded_upload_not_required_original_remote_full_SHA_verified=True,elapsed_s=time.time()-start,global_bytes=C.used());C.write(LEDGER,ledger);assert C.used()<2**30
 print('T529_STORAGE_EXACT_PASS',len(records),ledger['released_payload_bytes'],C.used(),flush=True)
def load_original(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 if p.exists():return p.read_bytes()
 ledger=C.read(LEDGER);x=next(x for x in ledger['records'] if x['original_path']==str(p.relative_to(ROOT)));b=C.decode((ROOT/x['encoded_path']).read_bytes(),REF.read_bytes());assert C.sha(b)==x['original_sha256'];return b
if __name__=='__main__':compact()
