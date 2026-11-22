"""Seven CPU workers; external physical exactness across source architecture."""
from pathlib import Path
import gzip,json,os
import t698_actor as A
import t696_native as Parent
C,G,F,P,np=Parent.C,Parent.G,Parent.F,Parent.P,Parent.np
ORIGINAL_INSPECT=Parent.ORIGINAL_INSPECT
EXTERNAL_PHYSICAL_KEYS=[k for k in Parent.PHYSICAL_KEYS if k not in ['hidden_before','last_hidden_after']]

def worker(pipe,index,p,counts):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    assert not A.torch.cuda.is_initialized()
    C.ACTIVE=p;real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=C.Session
    G.B.native_setup=lambda folder,pc:C.C.consistent_reset.setup(F,folder,pc,C.C.CANONICAL_SETUP)
    P.frozen=A.frozen
    G.T.worker(pipe,index,p,counts)

class Pool(A.L.T.R.N.Pool):
    def __init__(self,p,counts,guard):
        self.p,self.guard=p,guard;self.children=[];self.pipes=[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for index in range(7):
                a,b=ctx.Pipe();child=ctx.Process(target=worker,args=(b,index,p,counts));child.start();b.close();self.children.append(child);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(7)]
            actual=[sorted(os.sched_getaffinity(c.pid)) for c in self.children]
            assert actual==[[c] for c in p['stage_cpus'][:7]] and all(v['state']=='READY' for v in self.ready)
            (Path(p['out'])/'worker_generations.json').write_text(json.dumps([dict(parent=os.getpid(),workers=[c.pid for c in self.children],actual_affinity=actual)])+'\n')
        except BaseException:self.close();raise
    def request(self,items):
        assert 1<=len(items)<=7
        for index,request in enumerate(items):self.pipes[index].send(request)
        result=[]
        for index,request in enumerate(items):
            arrays,row=self.receive(index)
            with gzip.open(Path(self.p['out'])/('native_'+request['record_key']+'.json.gz'),'xb') as f:f.write(json.dumps(dict(request=request,native_result=row)).encode())
            result.append((arrays,row))
        return result

def reference(p,key):
    found=[r for r in p['protected410_rows']+p['novel_rows'] if r['id']==key];assert len(found)<=1
    return found[0] if found else None

def keys(p,key):
    # Same-policy retained histories also keep their own64 hidden bytewise;
    # only the two foreign-policy memories are excluded cross-source.
    return EXTERNAL_PHYSICAL_KEYS if key in p['source_ids'] else Parent.PHYSICAL_KEYS

def inspect(p,a,row):
    ORIGINAL_INSPECT(p,a,row)
    ref=reference(p,row['job_id'])
    if ref:
        assert A.sha(ref['path'])==ref['sha256'];source=A.L.T.R.E.N.S.arrays(ref['path'])
        assert all(np.array_equal(a[k],source[k]) for k in keys(p,row['job_id'])),row['job_id']
        assert A.same_float32(a['actions'],source['actions'])

def audit_source_hashes(p,row):
    ref=reference(p,row['job_id'])
    if ref:
        assert A.sha(ref['path'])==ref['sha256'];source=A.L.T.R.E.N.S.arrays(ref['path'])
        assert all(row['array_hashes'][k]==A.digest(source[k]) for k in keys(p,row['job_id'])),row['job_id']
