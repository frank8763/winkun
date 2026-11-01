"""Unique launch after source registration; no resumed/duplicate directories."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t271_frozen_feature_exploration'


def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert receipt['root_tree']==tree and receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    log=(OUT/'run.log').open('xb')
    cmd=['systemd-inhibit','--what=sleep','--mode=block','--why=Stage T271 bounded frozen diagnostic',
         str(ROOT/'run_stage.sh'),str(HERE/'run.py')]
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    stamp=time.time()
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=proc.pid,launch_epoch=stamp,
        source_commit=source,remote_source_commit=receipt['remote_commit'],single_launch=True,
        optimizer_budget=0,wall_seconds=1500,outer_latest_deadline_epoch=stamp+1500,
        protocol_sha256=json.loads((OUT/'preflight.json').read_text())['protocol_sha256']),f,indent=2)
    log.close();print(json.dumps(dict(pid=proc.pid,launch_epoch=stamp)),flush=True)


if __name__=='__main__':main()
