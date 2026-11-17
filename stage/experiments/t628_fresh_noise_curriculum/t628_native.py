from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_native as N
import t628_learning as M
def install():
 N.L.frozen=M.frozen;N.N.L.frozen=M.frozen;N.P.frozen=M.frozen
def worker(pipe,index,p,counts):
 os.environ['CUDA_VISIBLE_DEVICES']='';assert not N.torch.cuda.is_initialized();install();real=N.G._real_affinity;cpus=p['stage_cpus']
 def pin(pid,mask):return real(pid,{cpus[index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
 os.sched_setaffinity=pin;N.G.T.Session=N.Session;N.G.B.native_setup=N.F.field_setup;N.E.payload=N.N.payload;N.G.T.worker(pipe,index,p,counts)
