"""Reconcile completed paired parameter-PG evidence, no new interactions."""
from pathlib import Path
import json,collections,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t207_parameter_pg';DOCS=ROOT/'stage/docs/results_20261031_parameter_pg'
def lines(name):
 p=OUT/name;return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def safe(r):return r['complete'] and r['success'] and not r['collision']
def main():
 summary=json.loads((OUT/'summary.json').read_text());plan=json.loads((OUT/'protocol.json').read_text());assert summary['state']=='COMPLETE' and summary['all_workers_stopped']
 for path,h in plan['input_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 rows=lines('rollouts.jsonl');replay=json.loads((OUT/'replays.json').read_text()) if (OUT/'replays.json').exists() else [];gens=lines('generations.jsonl')
 assert sum(r['physics_steps'] for r in rows)+sum(r['physics_steps'] for r in replay)==summary['physics_steps']
 train=[r for r in rows if r['role']=='parameter_training'];assert sum(r['physics_steps'] for r in train)==summary['training_physics_steps']
 assert sum(r['complete'] for r in train)==summary['training_episodes'];assert len(rows)==summary['episodes']
 arms={}
 for arm in plan['arms']:
  sampled=[r for r in rows if r['arm']==arm and r['role'] in ['parameter_training','frozen_exploration_control']];final=[r for r in summary['final_old8'] if r['arm']==arm]
  arms[arm]={'sampled_episodes':len(sampled),'sampled_safe_GOAL':sum(safe(r) for r in sampled),'sampled_CONTACT':sum(r['collision'] for r in sampled),'sampled_TIMEOUT':sum(r['timeout'] for r in sampled),'sampled_partial':sum(not r['complete'] for r in sampled),'final_old8_completed':len(final),'final_old8_safe_GOAL':sum(safe(r) for r in final) if final else None,'final_old8':[{'case':r['case']['id'],'safe_GOAL':safe(r),'collision':r['collision'],'timeout':r['timeout'],'remaining_m':r['remaining_m']} for r in final],'final_coefficients':summary['mean_coefficients'][arm]}
 result={'state':'COMPLETE','stop_reason':summary['stop_reason'],'parameter_update_attempts':summary['parameter_update_attempts'],'completed_generation_records':len(gens),'physics_steps':summary['physics_steps'],'training_physics_steps':summary['training_physics_steps'],'training_episodes':summary['training_episodes'],'decision_count':sum(r['decisions'] for r in rows),'arms':arms,'witness':summary['witness'],'selection':summary['selection'],'stable_5m_gate_passed':summary['stable_5m_gate_passed'],'saved_reload_completed':len(summary['saved_reload']),'robustness_completed':summary['robustness_completed'],'robustness_safe_GOAL':summary['robustness_safe_GOAL'],'own_bytes':summary['own_bytes'],'elapsed_s':summary['elapsed_s'],'scope':'Paired one-seed low-dimensional PGPE-style fine-tuning of a pre-trained deep neural policy; frozen feature backbone, six parameter-distribution means. Sampled success and learned central success are different evidence. Not novel algorithm/from-scratch DRL or independent generalization.'}
 (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copyfile(OUT/'metrics.json',DOCS/'metrics.json');shutil.copyfile(OUT/'launch.json',DOCS/'launch.json')
 report=['# T207结果（2026-10-31）','',f"停止原因{result['stop_reason']}；参数分布更新尝试{result['parameter_update_attempts']}次，{result['episodes'] if 'episodes' in result else summary['episodes']}全部回合/{result['physics_steps']}全部物理步。learned组新鲜训练交互{result['training_episodes']}回合/{result['training_physics_steps']}物理步。全部模拟与工作进程退出，原始数据计数对账通过。",'', '| 组 | 参数采样安全GOAL | 最终中心old8 |','|---|---:|---|']
 for arm,d in arms.items():
  score=f"{d['final_old8_safe_GOAL']}/{d['final_old8_completed']}" if d['final_old8_completed'] else '未执行'
  report.append(f"| {arm} | {d['sampled_safe_GOAL']}/{d['sampled_episodes']} | {score} |")
 report += ['',f"原始LEFT成功见证={summary['witness'] is not None}；稳定5m门槛={summary['stable_5m_gate_passed']}；重载old8完成{len(summary['saved_reload'])}，扰动完成{summary['robustness_completed']}/54，安全GOAL {summary['robustness_safe_GOAL']}。未执行项不得计作实测失败。",'', '随机样本成功和中心策略成功分开报告；若frozen组成功，不能当作梯度更新贡献。仅一配对种子、同图开发任务和既有冻结特征的低维微调，不支持算法创新/从零DRL/独立泛化结论。T159/T194保留，无自动模型晋升、续训或长距离。','', '方法、协议与预算见protocol_notes.md；完整逐回合/每代梯度、噪声、行为参数、模型和失败以Release附件备份，保留哈希与临时包清理收据。']
 (DOCS/'findings.md').write_text('\n'.join(report)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['witness','arms']},indent=2))
if __name__=='__main__':main()
