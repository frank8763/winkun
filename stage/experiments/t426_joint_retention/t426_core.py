"""Bounded Stage joint old-task recovery and six-metre retention dependencies."""
from pathlib import Path
import sys,os,json,time,copy,shutil,traceback,contextlib,fcntl,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t426_joint_retention';DOC=ROOT/'stage/docs/evening_20261108'
sys.path.insert(0,str(ROOT/'stage/experiments/t425_offline_day_chain'))
import t425_core as PREVIOUS
import t426_learning as LEARNING
E,R=PREVIOUS.E,PREVIOUS.R
np,torch=E.np,E.torch
read,save,sha,digest=E.read,E.save,E.sha,E.digest
Pool=PREVIOUS.Pool;frozen=PREVIOUS.frozen;ban_learning=PREVIOUS.ban_learning;gradient_sha=PREVIOUS.gradient_sha
ALIASES={k:k for k in ['data','learn','gate','eight']}
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for path,value in p['inputs'].items():assert sha(path)==value,path
    assert E.N.K.S.used(OUT)<=p['root_limits']['max_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30

def model_decode(path):return LEARNING.load_complete_state(path)
def initial_record(p):
    value=model_decode(p['student_start_model'])
    return dict(path=p['student_start_model'],sha256=p['student_start_sha256'],hashes=value['hashes'],policy_hash=value['policy_hash'])
def selected_model(p):
    path=OUT/'learn/summary.json'
    return read(path)['result']['model'] if path.exists() and read(path)['state']=='COMPLETE' else initial_record(p)
