"""Summarize completed T205/T206 evidence without simulator steps."""
from pathlib import Path
import json,collections,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[3];B5=ROOT/'stage/runs/t205_temporal_intervention';B6=ROOT/'stage/runs/t206_feature_direction';DOCS=ROOT/'stage/docs/results_20261031_frontier'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def safe(r):return r['complete'] and r['success'] and not r['collision']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a,b=read(B5/'summary.json'),read(B6/'summary.json');assert a['state']==b['state']=='COMPLETE' and a['all_workers_stopped'] and b['all_workers_stopped']
 for base in [B5,B6]:
  for path,h in read(base/'protocol.json')['input_hashes'].items():assert sha(path)==h,path
 r5=lines(B5/'cases.jsonl');r6=lines(B6/'rollouts.jsonl');source=read(B5/'source.json');replays=read(B6/'replay_checks.json') if (B6/'replay_checks.json').exists() else []
 assert source['physics_steps']+sum(r['physics_steps'] for r in r5)==a['physics_steps']
 assert sum(r['physics_steps'] for r in r6)+sum(r['physics_steps'] for r in replays)==b['physics_steps']
 for r in r5+r6:
  if r.get('path'):assert sha(r['path'])==r.get('sha256',r.get('full_sha256'))
 candidates=lines(B6/'candidates.jsonl');proposals=lines(B6/'proposals.jsonl');pmap={r['id']:r for r in proposals}
 metrics={'T205':{'physics_steps':a['physics_steps'],'cases':len(r5),'interventions':len([r for r in r5 if r['kind']=='intervention']),'groups':a['groups'],'collision_objects':dict(collections.Counter(c['other_geom'] for r in r5 for c in r['contacts'])),'source_exact':read(B5/'preflight.json')['original_trace_exact']},'T206':{k:v for k,v in b.items() if k!='scope'},'candidate_outcomes':[{'id':r['id'],'steer':r['steer'],'throttle':r['throttle'],'offline_old7_RMSE':r['offline_old7_RMSE'],'safe_GOAL':safe(r['LEFT']),'collision':r['LEFT']['collision'],'timeout':r['LEFT']['timeout'],'remaining_m':r['LEFT']['remaining_m'],'nearest_m':r['LEFT']['nearest_m'],'eligible':r['eligible']} for r in candidates],'optimizer_updates':0,'new_training_physics_steps':0}
 folder=B6/'analysis';folder.mkdir(exist_ok=True);(folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');shutil.copyfile(folder/'metrics.json',DOCS/'metrics.json');shutil.copyfile(B6/'summary.json',DOCS/'t206_summary.json');shutil.copyfile(B6/'offline_direction.json',DOCS/'t206_offline_direction.json')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 with np.load(ROOT/'stage/runs/t196_left_exploration/left_baseline_trace.npz') as z:parent=z['poses']
 for col,t in enumerate([0.,-.4]):
  ax=axes[col];ax.plot(parent[:,0],parent[:,1],'k--',linewidth=1.5,label='T194 parent')
  for row in candidates:
   if row['throttle']!=t:continue
   path=row['LEFT'].get('path')
   if path:
    with np.load(path) as z:xy=np.vstack([z['poses'][:,:2],z['next_poses'][-1:,:2]])
    ax.plot(xy[:,0],xy[:,1],linewidth=1,label=f"steer {row['steer']:+.1f}")
  goal=read(B5/'source.json')['snapshots'][0]
  with np.load(B5/'traces/source.npz') as z:g=z['goal_xy']
  ax.scatter(g[0],g[1],marker='*',s=80,color='green',label='fixed final goal');ax.set(xlabel='x (m)',ylabel='y (m)',title=f'T206 fixed neural head: throttle {t:+.1f}')
  ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=7)
 fig.savefig(folder/'policy_paths.png',dpi=160);plt.close(fig)
 summary='''\n## T206实际结果\n\n'''+f"T206已完成，停止原因{b['stop_reason']}，{b['episodes']}个冻结闭环回合，{b['physics_steps']}诊断物理步，零训练/优化器。12组候选中原始LEFT安全GOAL {b['LEFT_safe_GOAL']}，同时保持old7者{b['old7_preserving_LEFT_GOAL']}。保存重载old8完成{b['saved_old8_completed']}、安全GOAL {b['saved_old8_safe_GOAL']}；54扰动完成{b['robustness_completed']}、安全GOAL {b['robustness_safe_GOAL']}；稳定5m门槛={b['stable_5m_gate_passed']}。未执行项不是实测失败。\n\n"
 summary+='| 候选 | 转向系数 | 油门系数 | old7离线动作RMSE | LEFT终止 | 最近目标距离(m) |\n|---|---:|---:|---:|---|---:|\n'
 for r in metrics['candidate_outcomes']:summary+=f"| {r['id']} | {r['steer']:+.1f} | {r['throttle']:+.1f} | {r['offline_old7_RMSE']:.5f} | {'GOAL' if r['safe_GOAL'] else 'CONTACT' if r['collision'] else 'TIMEOUT'} | {r['nearest_m']:.4f} |\n"
 summary+='\n所有输入和原始轨迹哈希通过；模型、失败及未通过结果保留，不晋升T159/T194。当前只检验一个固定方向，不能将失败推广为整个神经参数空间不可学；离线可分性和小锚点误差也不能直接保证绕障成功。结果用于选择下一训练问题。该12候选网格不重复、不自动追加。\n'
 with (DOCS/'findings.md').open('a') as f:f.write(summary)
 assert sum(p.stat().st_size for p in B6.rglob('*') if p.is_file())<read(B6/'protocol.json')['final_bytes']
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
