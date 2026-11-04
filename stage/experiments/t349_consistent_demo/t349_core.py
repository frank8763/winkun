"""One consistent demonstrated style; matched sequence/reset source ablation."""
from pathlib import Path
import sys, time, shutil, tempfile, fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t348_causal_sequence_warmstart'))
import t348_core as C
from t348_core import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t349_consistent_demo';DOC=ROOT/'stage/docs/review_20261104'
C.OUT=OUT


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


class ReservedBudget:
    """Full outside reservation plus exact own size; changed outside fails closed."""
    def __init__(self,root,own,own_limit,global_limit,min_free,interval=10.):
        self.root,self.own=Path(root),Path(own);self.own_limit,self.global_limit,self.min_free=own_limit,global_limit,min_free
        self.interval=interval;self.last_full=self.last_own=0.;self.full_checks=self.own_checks=0
        self.lock=(self.root/'.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.outside=self.index();self.other_bytes=sum(v[0] for v in self.outside.values())
        assert self.other_bytes+own_limit<=global_limit
        self.check(force=True)

    def index(self):
        result={}
        for p in self.root.rglob('*'):
            if not p.is_file() or self.own==p or self.own in p.parents:continue
            s=p.stat();result[str(p.relative_to(self.root))]=(s.st_size,s.st_mtime_ns)
        return result

    def check(self,force=False):
        now=time.monotonic()
        if force or now-self.last_full>=self.interval:
            current=self.index();self.full_checks+=1;self.last_full=now
            assert current==self.outside,'Outside Stage run changed during single-writer reservation'
        if force or now-self.last_own>=.5:
            own=K.S.used(self.own);self.own_checks+=1;self.last_own=now
            assert own<=self.own_limit and self.other_bytes+own<=self.global_limit
            assert shutil.disk_usage(self.root).free>=self.min_free

    def stats(self):return dict(outside_files=len(self.outside),reserved_other_bytes=self.other_bytes,
        own_limit=self.own_limit,global_limit=self.global_limit,full_outside_checks=self.full_checks,exact_own_checks=self.own_checks)

    def close(self):
        if self.lock is not None:fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();self.lock=None


def quota_contract():
    results={}
    # Only generated disposable bytes under the new run; original evidence untouched.
    with tempfile.TemporaryDirectory(prefix='quota_contract_',dir=OUT) as name:
        root=Path(name);own=root/'own';own.mkdir();peer=root/'frozen';peer.mkdir();f=peer/'source';f.write_bytes(b'abc')
        m=ReservedBudget(root,own,10,16,0)
        (own/'generated').write_bytes(b'1234567890');m.check(force=True);results['full_own_reservation_allowed']=True
        try:
            (own/'generated').write_bytes(b'12345678901');m.check(force=True);raise RuntimeError('quota rejection missing')
        except AssertionError:results['own_overlimit_rejected']=True
        (own/'generated').write_bytes(b'12');f.write_bytes(b'abcd')
        try:m.check(force=True);raise RuntimeError('outside mutation rejection missing')
        except AssertionError:results['outside_mutation_rejected']=True
        m.close()
        try:ReservedBudget(root,own,20,16,0);raise RuntimeError('worst reservation rejection missing')
        except AssertionError:results['worst_case_global_overreservation_rejected']=True
    return dict(state='PASS_DISPOSABLE_QUOTA_CONTRACT',native=0,optimizer=0,results=results)
