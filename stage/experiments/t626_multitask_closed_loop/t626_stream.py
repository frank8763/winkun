"""RAM-only GitHub release members: full SHA readback before eviction."""
from pathlib import Path
import io,sys,importlib.util,json,gzip,tarfile,time,urllib.parse,hashlib
ROOT=Path(__file__).resolve().parents[3]
def digest(b):return hashlib.sha256(b).hexdigest()
def unpack(data):
 files={}
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for m in tar.getmembers():assert m.isfile() and '..' not in Path(m.name).parts;files[m.name]=tar.extractfile(m).read()
 manifest=json.loads(files.pop('manifest.json'));assert set(files)=={z['path'] for z in manifest['files']}
 for z in manifest['files']:assert len(files[z['path']])==z['bytes'] and digest(files[z['path']])==z['sha256']
 return files,manifest
def archive(members,metadata):
 manifest=dict(metadata,files=[dict(path=name,bytes=len(b),sha256=digest(b)) for name,b in members.items()]);buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w') as tar:
  for name,b in dict(members,**{'manifest.json':json.dumps(manifest,separators=(',',':')).encode()}).items():
   assert '..' not in Path(name).parts;i=tarfile.TarInfo(name);i.size=len(b);i.mtime=0;tar.addfile(i,io.BytesIO(b))
 data=gzip.compress(buf.getvalue(),compresslevel=1,mtime=0);unpack(data);return data
def client():
 spec=importlib.util.spec_from_file_location('t626_network_transport_unique',ROOT/'stage/experiments/t493_wide8_night/transport.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.OLD.install();T=a.OLD.T
 import urllib.request
 T.B.OPENER=urllib.request.build_opener();return T
def main():
 action=sys.argv[1];data=sys.stdin.buffer.read();T=client()
 if action=='download':
  receipt=json.loads(data);pieces=[]
  for part in receipt['parts']:
   a=T.B.api('GET',f"releases/assets/{part['asset_id']}");b=T.readback(a,part['bytes']);assert digest(b)==part['sha256'];pieces.append(b)
  blob=b''.join(pieces);assert digest(blob)==receipt['sha256'];unpack(blob);sys.stdout.buffer.write(blob);return
 assert action=='upload';assert len(data)<=8*2**20;files,manifest=unpack(data);tag=sys.argv[2];assert tag.startswith('T626')
 release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');parts=[];returned=[]
 for k,off in enumerate(range(0,len(data),4*2**20)):
  b=data[off:off+4*2**20];label=f"stage-{tag.lower()}-{digest(data)[:12]}-{k:03d}.bin"
  a=json.loads(T.req('POST',f"https://uploads.github.com/repos/{T.A.SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(label),b,{'Content-Type':'application/octet-stream'}));a=T.B.api('GET',f"releases/assets/{a['id']}");assert a['size']==len(b) and a['digest']=='sha256:'+digest(b);z=T.readback(a,len(b));assert z==b;returned.append(z);parts.append(dict(asset_id=a['id'],asset_url=a['browser_download_url'],bytes=len(b),sha256=digest(b)))
 blob=b''.join(returned);assert blob==data;unpack(blob);print(json.dumps(dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',sha256=digest(data),bytes=len(data),parts=parts,manifest=manifest,temporary_disk_archive_bytes=0,epoch=time.time())))
if __name__=='__main__':main()
