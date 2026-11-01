from pathlib import Path
import ast,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t283_capped_task_balance'


def main():
    assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json'])
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    doc=ROOT/'stage/docs/long_distance_20261101'
    backup=json.loads((doc/'release_receipt_T280.json').read_text())
    assert backup['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    manifest=doc/'source_manifest_T280.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==backup['manifest_sha256']
    for record in json.loads(manifest.read_text())['files']:
        data=(ROOT/record['path']).read_bytes()
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    log=(OUT/'run.log').open('xb');env=os.environ.copy();env['STAGE_PROCESS_CPU']='0'
    process=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage T283 capped task balance',str(ROOT/'run_stage.sh'),str(HERE/'run.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    with (OUT/'launch.json').open('x') as f:json.dump(dict(pid=process.pid,launch_epoch=time.time(),single_launch=True,wall_seconds=4800,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=receipt['remote_commit'],
        max_commands=800,max_physics=22400000,own_bytes=35*1024**2),f,indent=2)
    log.close();print(json.dumps(dict(pid=process.pid)),flush=True)


if __name__=='__main__':
    if len(__import__('sys').argv)>1:
        assert __import__('sys').argv[1:] == ['--validate-guard-only']
        ast.parse(Path(__file__).read_text())
        assert not (OUT/'launch.json').exists() and not (OUT/'status.json').exists()
        assert not (ROOT/'stage/docs/long_distance_20261101/release_receipt_T280.json').exists()
        print(json.dumps(dict(state='EXPECTED_LAUNCH_BLOCKED_MISSING_T280_REMOTE_BACKUP',physics_steps=0,optimizer_calls=0)))
    else:main()
