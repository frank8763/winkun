from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t296_log_tail_support'

def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json']);p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();r=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text());assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as f:proc=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=1s',str(p['wall_seconds'])+'s',str(ROOT/'run_stage.sh'),str(HERE/'run.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=proc.pid,launch_epoch=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=r['remote_commit'],wall_seconds=p['wall_seconds'],single_launch=True),f,indent=2)
    print(json.dumps(dict(pid=proc.pid)))

if __name__=='__main__':main()
