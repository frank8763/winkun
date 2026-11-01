"""Reserved single-writer output accounting; expensive scans only at boundaries."""
from pathlib import Path
import fcntl,json,shutil,time


def logical(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def room(outside,own_limit,global_limit,payload_after,payload_limit,metadata_limit):
    return outside+own_limit<=global_limit and payload_after<=payload_limit and payload_after+metadata_limit<=own_limit


class Budget:
    def __init__(self,root,out,plan,start_epoch=None):
        self.root,self.out,self.p=Path(root),Path(out),plan
        self.lock=(self.root/'stage/runs/.stage_native_exclusive.lock').open('a+b')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.deadline=(time.time() if start_epoch is None else start_epoch)+plan['wall_seconds'];self.physics=0;self.commands=0;self.full_scans=0
        self.outside=logical(self.root/'stage/runs')-logical(self.out)
        self.payload_bytes=0
        assert room(self.outside,plan['final_bytes'],plan['global_bytes'],0,plan['payload_bytes'],plan['metadata_bytes'])
        self.checkpoint()

    def checkpoint(self):
        total=logical(self.root/'stage/runs');own=logical(self.out);self.full_scans+=1
        assert total-own==self.outside,'Noncooperating Stage write detected outside the exclusive output'
        assert total<=self.p['global_bytes'] and own<=self.p['final_bytes']
        assert shutil.disk_usage(self.root).free>=self.p['min_free_bytes']
        assert time.time()<self.deadline

    def begin(self,cap):
        self.checkpoint();assert self.commands+1<=self.p['max_commands']
        assert self.physics+cap<=self.p['max_physics'];self.commands+=1

    def running(self):return time.time()<self.deadline and self.physics<self.p['max_physics']

    def hold(self):
        # O(1), no recursive rglob/stat. Known writes remain within reserved
        # payload+metadata space under this explicit Stage-exclusive-writer scope.
        assert shutil.disk_usage(self.root).free>=self.p['min_free_bytes']

    def charge_payload(self,n):
        assert room(self.outside,self.p['final_bytes'],self.p['global_bytes'],self.payload_bytes+n,self.p['payload_bytes'],self.p['metadata_bytes'])
        self.payload_bytes+=n

    def close(self):
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
