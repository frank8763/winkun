"""Stage-only bounded offline recovery/retention pipeline dependencies."""
from pathlib import Path
import sys,os,json,time,copy,shutil,traceback,contextlib,fcntl,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t425_offline_day_chain';DOC=ROOT/'stage/docs/day_20261108'
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E
sys.path.insert(0,str(ROOT/'stage/experiments/t422_feedback_recovery_night'))
import reference as R
import t425_learning as LEARNING
np,torch=E.np,E.torch
read,save,sha,digest=E.read,E.save,E.sha,E.digest
ALIASES=dict(data='data',arm_a='a',arm_b='b',gate='gate',eight='eight')
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert E.N.K.S.used(OUT)<=p['root_limits']['max_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
def frozen(p,model=None):
    q=E.safe_initial(E.N.science(p),p).eval()
    if model:q.load_learned(E.N.S.read_model(model))
    q.sample=False;q.reset_noise();return q
def model_decode(path):return LEARNING.load_complete_state(path)
def ban_learning():
    def no(*a,**k):raise AssertionError('Native process cannot optimize')
    torch.optim.Adam.step=no;torch.optim.AdamW.step=no;torch.optim.SGD.step=no;torch.Tensor.backward=no
def gradient_sha(q):return digest({n:None if v.grad is None else v.grad.detach().clone() for n,v in q.named_parameters() if v.requires_grad})
def anchors(p):
    return [(E.N.S.arrays(x['path']),x['row']) for x in p['anchors']]
def selected_model(p):
    for name in ['b','a']:
        f=OUT/name/'summary.json'
        if f.exists() and read(f)['state']=='COMPLETE':return read(f)['result']['model']
    stored=model_decode(p['student_start_model']);return dict(path=p['student_start_model'],sha256=p['student_start_sha256'],hashes=stored['hashes'],policy_hash=stored['policy_hash'])
def worker(pipe,index,p,counts):
    E.N.offline();ban_learning();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True);folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);E.set_out(Path(p['out']));log=(folder/'console.log').open('w');stdout,stderr=sys.stdout,sys.stderr;s=w=None
    def guard(physical=False):E.L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,_=E.N.S.P.C.B.setup(folder/'native',E.N.science(p));original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers();pipe.send(dict(state='READY',pid=os.getpid(),cpu=index))
            while True:
                req=pipe.recv()
                if req is None:break
                q=frozen(p,req['model']).requires_grad_(False);q.rng=None if req.get('seed') is None else np.random.default_rng(np.random.SeedSequence(req['seed']));job=req['job'];mode=req.get('mode','student');first=req.get('first',0);module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m']);before=counts[index+1]
                s=E.L.Session(job,native,create,w,guard,charge,q);E.L.install_stop_handlers();progress=first;ds=[];ref=None
                if req.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise E.L.StageStop('T425 zero native cancel')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Missing cancellation')
                    except E.L.StageStop:pass
                    assert s.steps==0 and not s.rows;s.close();s=None;pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                if mode=='feedback':
                    ref=R.load_reference(req['reference_path'],req['reference_sha256']);assert E.equal(ref['goal'],s.goal)
                    # Neighbour reference has a declared different start. Never
                    # transplant its native state, history or recurrent hidden.
                while not s.done():
                    i=len(s.rows);q.sample=req.get('seed') is not None
                    if mode=='student' or i<first:s.step(owner=0);ds.append((-1,0.));continue
                    x=s.packet()
                    if mode=='safe_reset':
                        with torch.no_grad():action=q.base_actions(torch.tensor(x[None]))[0].clamp(-1,1).float().numpy()
                        ds.append((-1,0.))
                    else:
                        # Current planar speed (m/s), matching reference x56.
                        # x113 is the LF steering encoder, never vehicle speed.
                        assert mode=='feedback';speed=float(x[56]);action,progress,_=R.feedback(ref,np.asarray(s.env._pose()[:3]),speed,s.steps*.002,progress,req['gains'],'progress');ds.append((progress,speed))
                    s.step(action,owner=1,actor_active=True)
                a=E.payload(s)
                if mode!='student':a.update(feedback_progress=np.asarray([d[0] for d in ds],np.int16),feedback_speed=np.asarray([d[1] for d in ds],np.float32))
                r=s.result();r.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=digest(a['final_native_integration']),pid=os.getpid(),cpu=index,teacher=mode!='student',mode=mode,first=first)
                s.close();s=None;pipe.send((a,r))
    except BaseException:
        save(folder/'failure.json',dict(error=traceback.format_exc(),physics=counts[index+1]))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(E.payload(s)),p['full_bytes'])
            except BaseException:pass
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
