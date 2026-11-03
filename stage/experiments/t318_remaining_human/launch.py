from pathlib import Path
import json,os,subprocess,time,math
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t318_remaining_human'
def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    remaining=math.floor(p['absolute_deadline_epoch']-time.time());assert remaining>=300
    with (OUT/'run.log').open('xb') as f:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage remaining third human demonstration',
            '/usr/bin/timeout','--signal=TERM','--kill-after=1s',str(remaining)+'s','/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'recorder.py'),'main'],
            cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'launch.json').write_text(json.dumps(dict(pid=proc.pid,launch_epoch=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        remote_commit=r['remote_commit'],single_launch=True,remaining_attempt_only=True,original_absolute_deadline=p['absolute_deadline_epoch']),indent=2)+'\n')
    print(json.dumps(dict(pid=proc.pid,absolute_deadline=p['absolute_deadline_epoch'])))
if __name__=='__main__':main()
