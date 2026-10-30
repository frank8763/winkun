"""Verify bounded Stage T189/T190 raw evidence and export compact Git summaries."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import ast,hashlib,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'stage/docs/results_20261030_head_curriculum'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def write(path,v):path.write_text(json.dumps(v,indent=2)+'\n')

def main():
    torch.set_num_threads(1)
    metrics={'state':'PASS','protocols':{},'experiments':{},'automatic_promotion':False,'incumbent':'T159'}
    for name in ['t189_head_training','t190_adaptive_start_curriculum']:
        src=ROOT/'stage/experiments'/name;out=ROOT/'stage/runs'/name;plan=read(src/'protocol.json');summary=read(out/'summary.json')
        assert summary['state']==read(out/'status.json')['state']=='COMPLETE'
        for p,h in plan['input_hashes'].items():assert sha(p)==h,p
        assert read(out/'launch.json')['protocol_sha256']==sha(src/'protocol.json')
        assert sha(src/'protocol.json')==sha(out/'protocol.json')
        for p in src.glob('*.py'):ast.parse(p.read_text())
        process=read(out/'workers.json')
        for pid in [process['parent']]+process['pids']:assert not Path('/proc',str(pid)).exists(),pid
        training=rows(out/'episodes.jsonl');evaluation=rows(out/'evaluation.jsonl')
        assert len(training)%8==0
        for i in range(0,len(training),8):
            for j in range(4):
                a,b=training[i+j],training[i+j+4]
                assert a['source']==b['source'] and a['initial_reset_signature']==b['initial_reset_signature']
        physics=sum(r['steps'] for r in training);decisions=sum(r['decisions'] for r in training)
        batch_files=list(out.glob('*/batch_*.npz')) if name.startswith('t189') else list(out.glob('round*_batch*.npz'))
        batch_decisions=batch_physics=0
        for p in batch_files:
            with np.load(p,allow_pickle=False) as z:
                assert all(np.isfinite(z[k]).all() for k in z.files)
                batch_decisions+=len(z['states']);batch_physics+=int(z['native_counts'].sum())
        assert batch_decisions==decisions and batch_physics==physics
        if name.startswith('t189'):
            assert len(training)==128 and len(evaluation)==258
            for arm,spec in summary['training'].items():
                assert sha(spec['checkpoint'])==spec['sha256']
                ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
                assert ck['protocol_sha256']==sha(src/'protocol.json')
                if arm=='frozen_head':assert torch.count_nonzero(ck['policy']['direct_head.weight'])==0
                else:assert torch.count_nonzero(ck['policy']['direct_head.weight'])>0
            updates=sum(v['actor_updates'] for v in summary['training'].values())
            assert physics==sum(v['physics_steps'] for v in summary['training'].values())
            diagnostics=summary['frozen_evaluation_physics']+summary['preflight_physics']
            write(DOC/'t189_verification.json',read(out/'verification.json'))
        else:
            assert len(training)<=256 and physics==summary['training_physics']<=plan['max_training_physics']
            assert decisions==summary['training_decisions']
            assert sha(out/'checkpoint_latest.pt')==summary['checkpoint_sha256']
            final=[r for r in evaluation if r['evaluation_tag']=='final'];assert len(final)==86
            assert sum(r['steps'] for r in final)==summary['final_evaluation_physics']
            assert sum(r['steps'] for r in evaluation if r['evaluation_tag']!='final')==summary['probe_physics']
            updates=summary['optimizer_updates'];diagnostics=summary['probe_physics']+summary['final_evaluation_physics']
        assert not list(out.rglob('checkpoint_previous.pt'))
        metrics['protocols'][name]={'sha256':sha(src/'protocol.json'),'inputs_verified':len(plan['input_hashes'])}
        metrics['experiments'][name]={'training_physics':physics,'training_decisions':decisions,'optimizer_updates':updates,'training_episodes':len(training),'diagnostic_evaluation_physics':diagnostics,'scores':summary['scores']}
        write(DOC/(name+'_summary.json'),summary)
    # Previous archived results must remain byte-identical.
    previous=read(ROOT/'stage/manifests/control_robustness_20261030_files.json')
    for row in previous['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
    metrics['previous_archived_files_unchanged']=len(previous['files'])
    metrics['all_new_runs_bytes']=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert metrics['all_new_runs_bytes']<1024**3
    metrics['total_new_training_physics']=sum(x['training_physics'] for x in metrics['experiments'].values())
    metrics['total_new_training_decisions']=sum(x['training_decisions'] for x in metrics['experiments'].values())
    metrics['total_optimizer_updates']=sum(x['optimizer_updates'] for x in metrics['experiments'].values())
    write(DOC/'verification_and_metrics.json',metrics);print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
