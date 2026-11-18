"""Raw immutable member bytes in a bounded RAM broker, read twice independently."""
from pathlib import Path
import hashlib,io,json,os,signal,socket,struct,subprocess,tarfile,threading,time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def digest(b):return hashlib.sha256(b).hexdigest()
def exact(sock,n):
    result=bytearray()
    while len(result)<n:
        part=sock.recv(n-len(result));assert part,'Incomplete RAM transfer';result.extend(part)
    return bytes(result)
def send(sock,data):sock.sendall(struct.pack('!Q',len(data))+data)
def receive(sock,cap):
    n=struct.unpack('!Q',exact(sock,8))[0];assert n<=cap
    return exact(sock,n)


def member(blob,row):
    assert len(blob)==row['bytes'] and digest(blob)==row['sha256'],row['id']
    return blob


def fetch(receipt,seconds):
    child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t645r_fetch.py'),receipt],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    try:
        data,error=child.communicate(timeout=seconds)
        assert child.returncode==0,error.decode(errors='replace')[-1000:]
        assert digest(data)==read(receipt)['sha256']
        return data
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise


def preload(p,out,guard):
    started=time.time();raw={};groups={};network=[]
    for row in p['memory_sources']:
        groups.setdefault(row.get('receipt','LOCAL'),[]).append(row)
    for key,rows in groups.items():
        guard();assert time.time()-started<p['preload_seconds']
        if key=='LOCAL':
            for row in rows:raw[row['path']]=member((ROOT/row['path']).read_bytes(),row)
        else:
            data=None
            for attempt in range(4):
                guard();left=p['preload_seconds']-(time.time()-started)
                assert left>15
                try:data=fetch(key,min(180,left-5));break
                except Exception:
                    if attempt==3:raise
                    time.sleep(min(2.,.5*(attempt+1)))
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
                for row in rows:raw[row['path']]=member(tar.extractfile(row['path']).read(),row)
            network.append(dict(receipt=key,sha256=digest(data),members_selected=len(rows)))
            del data
        total=sum(map(len,raw.values()));assert total<=p['raw_RAM_cap_bytes']
        save(out/'DATA_PREPARATION.json',dict(state='PREPARING_DATA_NOT_SCIENCE',verified_members=len(raw),raw_RAM_bytes=total,
             network_groups_done=len(network),epoch=time.time(),native=0,optimizer=0))
    assert len(raw)==384 and set(raw)=={r['path'] for r in p['memory_sources']}
    for row in p['memory_sources']:member(raw[row['path']],row)
    save(out/'DATA_READY.json',dict(state='ALL384_RAW_BYTES_READY_AND_SHA_VERIFIED',members=384,raw_RAM_bytes=sum(map(len,raw.values())),
         network_groups=network,elapsed_s=time.time()-started,temporary_disk_cache_bytes=0,native=0,optimizer=0))
    return raw


class Server:
    def __init__(self,path,raw,rows,deadline):
        self.path=Path(path);assert not self.path.exists()
        self.raw=raw;self.rows={r['path']:r for r in rows};self.deadline=deadline
        self.calls=[];self.error=None;self.stopping=threading.Event()
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.bind(str(path));os.chmod(path,0o600)
        self.sock.listen(4);self.sock.settimeout(.2)
        self.thread=threading.Thread(target=self.serve,daemon=True);self.thread.start()

    def serve(self):
        try:
            while not self.stopping.is_set() and time.time()<self.deadline:
                try:connection,_=self.sock.accept()
                except socket.timeout:continue
                with connection:
                    connection.settimeout(15)
                    req=json.loads(receive(connection,8192));phase=req['phase'];path=req['path']
                    valid=phase in ['main','audit'] and path in self.raw and (phase,path) not in self.calls
                    if not valid:send(connection,b'REJECT');continue
                    blob=member(self.raw[path],self.rows[path]);send(connection,blob);self.calls.append((phase,path))
        except BaseException as e:self.error=type(e).__name__+': '+str(e)[:200]

    def close(self):
        self.stopping.set();self.thread.join(timeout=2);self.sock.close()
        if self.path.exists():self.path.unlink()
        assert not self.thread.is_alive()


def get(path,row,phase):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(15);sock.connect(str(path));send(sock,json.dumps(dict(phase=phase,path=row['path'])).encode())
        data=receive(sock,2**21)
    return member(data,row)
