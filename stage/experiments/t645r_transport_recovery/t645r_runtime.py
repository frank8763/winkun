"""Byte-identical T645 scientific functions, new output root only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t645r_transport_recovery';DOC=ROOT/'stage/docs/day_20261118/t645r'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import t645_runtime as Original
Original.HERE=HERE;Original.OUT=OUT;Original.DOC=DOC
Original.R.HERE=HERE;Original.R.OUT=OUT;Original.R.DOC=DOC
Original.Mem.HERE=HERE;Original.Mem.OUT=OUT;Original.Mem.DOC=DOC


if __name__=='__main__':
    def no_network(*args,**kwargs):raise RuntimeError('Verified RAM only during science')
    Original.R.D.remote_bytes=no_network;Original.R.E.N.offline()
    getattr(Original,sys.argv[1])()
