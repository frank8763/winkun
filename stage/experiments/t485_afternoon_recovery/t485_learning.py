"""Interface-only adapter; exact T481 learning functions and frozen Vtrace."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t481_fresh16_rehearsal'))
import t481_learning as F
B,P,E,M,R,torch,np=F.B,F.P,F.E,F.M,F.R,F.torch,F.np
read,save,sha,digest=F.read,F.save,F.sha,F.digest
make,references,memory,keep_loss,replay_losses=F.make,F.references,F.memory,F.keep_loss,F.replay_losses
batch,epoch,fit=F.batch,F.epoch,F.fit
vtrace=F.B.vtrace
