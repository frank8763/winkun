"""Zero-optimizer actual early-departure recovery diagnostic."""
from pathlib import Path
import sys,os,json,time,copy,shutil,traceback,contextlib,fcntl,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t429_feedback_recovery';DOC=ROOT/'stage/docs/feedback_20261108'
sys.path.insert(0,str(ROOT/'stage/experiments/t427_fixed_representation'))
import t427_core as PREVIOUS
E,R,LEARNING=PREVIOUS.E,PREVIOUS.R,PREVIOUS.LEARNING
np,torch=E.np,E.torch
read,save,sha,digest=E.read,E.save,E.sha,E.digest
Pool,frozen,teacher_actor,ban_learning=PREVIOUS.Pool,PREVIOUS.frozen,PREVIOUS.teacher_actor,PREVIOUS.ban_learning
initial_record=PREVIOUS.initial_record;model_decode=PREVIOUS.model_decode
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for path,value in p['inputs'].items():assert sha(path)==value,path
    assert E.N.K.S.used(OUT)<=p['root_limits']['max_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
