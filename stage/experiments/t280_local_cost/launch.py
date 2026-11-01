from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t280_local_cost'


def main():
    assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    log=(OUT/'run.log').open('xb');env=os.environ.copy();env['STAGE_PROCESS_CPU']='0'
    process=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage T280 paired local acquisition',str(ROOT/'run_stage.sh'),str(HERE/'run.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=process.pid,launch_epoch=time.time(),single_launch=True,wall_seconds=4800,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=receipt['remote_commit'],
        max_commands=1600,max_physics=45000000,own_bytes=64*1024**2),f,indent=2)
    log.close();print(json.dumps(dict(pid=process.pid)),flush=True)


if __name__=='__main__':main()
