"""Stage-only new bounded actual-recovery support mechanism; inert imports."""
from pathlib import Path
import sys,os,json,time,copy,shutil,traceback,contextlib,fcntl,multiprocessing as mp,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t432_offline_night';DOC=ROOT/'stage/docs/overnight_20261108'
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E
sys.path.insert(0,str(ROOT/'stage/experiments/t422_feedback_recovery_night'))
import reference as R
import t432_learning as LEARNING
np,torch=E.np,E.torch
read,save,sha,digest=E.read,E.save,E.sha,E.digest
ALIASES={k:k for k in ['data','b','c','gate','eight']}
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert E.N.K.S.used(OUT)<=p['root_limits']['max_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert time.time()<p['deadline_all']
def frozen(p,model=None):
    q=E.safe_initial(E.N.science(p),p).eval()
    if model:q.load_learned(E.N.S.read_model(model))
    q.sample=False;q.reset_noise();LEARNING.params(q);return q
def teacher_actor(path,p):return frozen(p,path).requires_grad_(False)
def model_decode(path):return LEARNING.load_complete_state(path)
def ban_learning():
    def no(*a,**k):raise AssertionError('Native-only process cannot optimize')
    torch.optim.Adam.step=no;torch.optim.AdamW.step=no;torch.optim.SGD.step=no;torch.Tensor.backward=no
def gradient_sha(q):return digest({n:None if v.grad is None else v.grad.detach().clone() for n,v in q.named_parameters() if v.requires_grad})
def initial_record(p):
    value=model_decode(p['student_start_model'])
    return dict(path=p['student_start_model'],sha256=p['student_start_sha256'],hashes=value['hashes'],policy_hash=value['policy_hash'])
def selected_arm(p):
    choices=[]
    for order,name in enumerate(['b','c']):
        f=OUT/name/'summary.json'
        if not f.exists():continue
        summary=read(f)
        if summary['state']!='COMPLETE':continue
        result=summary['result']
        if result['accepted']:
            choices.append((len(result['selected_success_ids']),-order,name,result))
    if not choices:return None
    return max(choices,key=lambda x:x[:2])[2:]
def selected_model(p):
    chosen=selected_arm(p)
    return chosen[1]['model'] if chosen else initial_record(p)
def worker(pipe,index,p,counts):
    E.N.offline();ban_learning();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);E.set_out(Path(p['out']))
    log=(folder/'console.log').open('w');stdout,stderr=sys.stdout,sys.stderr;s=w=None
    def guard(physical=False):E.L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,_=E.N.S.P.C.B.setup(folder/'native',E.N.science(p));original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers();pipe.send(dict(state='READY',pid=os.getpid(),cpu=index))
            while True:
                req=pipe.recv()
                if req is None:break
                assert req.get('mode','student')=='student'
                q=frozen(p,req['model']).requires_grad_(False);q.rng=None if req.get('seed') is None else np.random.default_rng(np.random.SeedSequence(req['seed']))
                job=req['job'];mode=req.get('mode','student');first=req.get('first',0);module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m']);before=counts[index+1]
                s=E.L.Session(job,native,create,w,guard,charge,q);E.L.install_stop_handlers()
                if req.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise E.L.StageStop('T432 zero native cancellation')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Missing cancellation')
                    except E.L.StageStop:pass
                    assert s.steps==0 and not s.rows;s.close();s=None;save(folder/'cancel_pass.json',dict(state='CANCEL_PASS',native=0,pid=os.getpid()));pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                while not s.done():
                    q.sample=req.get('seed') is not None;s.step(owner=0)
                a=E.payload(s)
                r=s.result();r.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=digest(a['final_native_integration']),pid=os.getpid(),cpu=index,teacher=False,mode=mode,first=first)
                s.close();s=None;pipe.send((a,r))
    except BaseException:
        save(folder/'failure.json',dict(error=traceback.format_exc(),physics=counts[index+1]))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(E.payload(s)),p['full_bytes'])
            except BaseException:save(folder/'partial_save_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if s:s.close()
        if w:w.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()
class Pool(E.NATIVE_AUDIT.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=mp.get_context('spawn')
        try:
            for i in range(4):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            save(Path(p['out'])/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]));self.ready=[self.receive(i) for i in range(4)];assert all(x['state']=='READY' for x in self.ready)
        except BaseException:self.close();raise
