"""Read stored Stage evidence using stdlib only; never import a simulator/learner.

--check prints a deterministic verdict. --write adds only small review docs.
No protocol registration, native step, Actor forward, optimizer, launch, model
export, compression or removal exists in this module.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,collections,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/review_20261109'

def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def successful(rows):return {r['job_id'] for r in rows if r['success'] and not r['collision']}
def tally(rows):
    return dict(valid=sum(not r['administrative_censor'] for r in rows),success=sum(r['success'] and not r['collision'] for r in rows),contact=sum(r['collision'] for r in rows),timeout=sum(r['timeout'] for r in rows),censored=sum(r['administrative_censor'] for r in rows))
def beijing(epoch):return datetime.fromtimestamp(epoch,ZoneInfo('Asia/Shanghai')).isoformat()

def review():
    base=ROOT/'stage/runs/t432_offline_night';p=read(base/'protocol.json');initial=lines(p['current_cases_path']);parents=successful(initial)
    assert len(initial)==329 and len(parents)==322 and parents==set(p['initial_success_ids'])
    assert {r['policy_hash'] for r in initial}=={p['expected_policy_hash']}
    assert {r['loaded_model_sha256'] for r in initial}=={p['student_start_sha256']}
    assert sha(p['student_start_model'])==p['student_start_sha256']
    byid={r['job_id']:r for r in initial};dataset=read(base/'data/summary.json')['result'];arms={};inputs={}
    root_summary=read(ROOT/'stage/docs/overnight_20261108/FINAL_REPORT.json');launch=read(base/'launch.json');actual=dict(physics=0,commands=p['preflight_zero_physics_commands'],main_wall_s=0.)
    for name in ['data','b','c']:
        folder=base/name;s=read(folder/'summary.json');audit=read(folder/'audit.json');command=lines(folder/'commands.jsonl')
        assert s['state']=='COMPLETE' and audit['state']=='PASS' and s['all_workers_stopped'] and s['all_sessions_closed']
        assert sum(r['physics'] for r in command)==s['physics'] and len(command)==s['commands']
        for k,v in [('physics',s['physics']),('commands',s['commands']),('main_wall_s',s['elapsed_s'])]:actual[k]+=v
        for f in [folder/'summary.json',folder/'audit.json',folder/'commands.jsonl']:inputs[str(f.relative_to(ROOT))]=sha(f)
        if name=='data':continue
        result=s['result'];rows=lines(folder/'cases.jsonl');dev=read(folder/'development.json');ids=successful(rows);assert rows==dev['rows'] and len(rows)==329
        assert dev['wide54']==dict(tally(rows[:54]),invalid=0) and dev['old275']==dict(tally(rows[54:]),invalid=0)
        assert result['accepted']==bool(parents<=ids and ids-parents)
        assert result['accepted'] is False and result['development_all_safe'] is False
        for model in ['initial_model','proposed_model','model']:
            m=result[model];assert sha(m['path'])==m['sha256']
        assert result['model']['hashes']==result['initial_model']['hashes'] and result['model']['policy_hash']==p['expected_policy_hash']
        anchors={e['row']['job_id'] for e in dataset['anchors_'+name]};lost=parents-ids;gained=ids-parents;trace=result['fit']['trace'];events=lines(folder/'optimizer_calls.jsonl');after=collections.Counter(e['which'] for e in events if e['where']=='AFTER')
        assert dict(after)==dict(live=120,shadow=120) and result['fit']['new_RL']==0
        arms[name]=dict(six=dev['wide54'],old=dev['old275'],left=tally([r for r in rows[:54] if '_LEFT_' in r['job_id']]),right=tally([r for r in rows[:54] if '_RIGHT_' in r['job_id']]),accepted=False,lost_ids=sorted(lost),gained_ids=sorted(gained),lost_ids_present_in_preservation=sorted(lost&anchors),lost_ids_outside_preservation=sorted(lost-anchors),remaining_original_failures=sorted(set(byid)-parents-ids),initial_corrective_loss=trace[0][3],final_corrective_loss=trace[-1][3],corrective_loss_relative_reduction=1-trace[-1][3]/trace[0][3],preservation_losses_first=trace[0][:3],preservation_losses_last=trace[-1][:3],live_Adam=after['live'],shadow_Adam=after['shadow'],retained_updates=0,new_RL=0,training_array_evidence='Full source anchors/recoveries saved; full new autonomous329 state arrays absent.',failures=[dict(id=r['job_id'],collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],duration_s=r['physics']*.002,in_anchor=r['job_id'] in anchors,already_failed_at_initializer=r['job_id'] not in parents) for r in rows if r['job_id'] not in ids])
    assert actual==root_summary['actual']
    receipt_path=ROOT/'stage/docs/long_distance_20261101/release_receipt_T432.json';receipt=read(receipt_path)
    assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert sha(ROOT/'stage/docs/long_distance_20261101/source_manifest_T432.json')==receipt['manifest_sha256']
    for alias in ['gate','eight']:assert not (base/alias/'main_started.json').exists()
    draft=ROOT/'stage/runs/t434_night_relay';draft_states={f:(draft/f).exists() for f in ['protocol.json','preflight.json','launch.json','driver_started.json']};assert not any(draft_states.values())
    assert not (ROOT/'stage/runs/t435_relay_reserve').exists()
    # Small future diagnostic, not registered, launched or authorized by review.
    chosen=['wide117_80_dev6_LEFT_00','wide117_80_dev6_LEFT_07','dev_task6_09','dev_task6_24']
    full={r['job_id']:r for r in initial if r['full']};full.update({e['row']['job_id']:e['row'] for e in dataset['anchors_c']})
    needed=[dict(job_id=i,policy='C_proposed_rejected_diagnostic_only',step_cap=byid[i]['request']['job']['case']['step_cap']) for i in chosen]
    reused=[]
    for i in chosen:
        if i in full:
            row=full[i];assert sha(row['path'])==row['sha256'];reused.append(dict(job_id=i,path=row['path'],sha256=row['sha256']))
        else:needed.append(dict(job_id=i,policy='T427_source',step_cap=byid[i]['request']['job']['case']['step_cap']))
    proposal=dict(state='DRAFT_NOT_REGISTERED_NOT_STARTED',frozen_task_ids=chosen,source_full_existing_references=reused,new_native_rollouts_needed=needed,exact_replays=len(needed),all_native_command_ceiling=4+2*len(needed),all_physical_step_ceiling=2*sum(j['step_cap'] for j in needed),optimizer_updates=0,full_files=len(needed),full_file_byte_ceiling=384*1024,output_ceiling_bytes=3*2**20,main_seconds_ceiling=600,audit_seconds_ceiling=300,scope='Locate first actual history/action/vehicle-clearance drift; do not relax goal/collision/time or train a rejected model.',requires_new_authorized_window=True)
    assert proposal['all_physical_step_ceiling']==300004 and len(needed)==5
    # A two-dimensional arithmetic counterexample: aggregate gradient sign can
    # change under an anisotropic positive preconditioner. Not an actual Adam
    # reconstruction or measured causal attribution for this experiment.
    reference=(1.,1.);gradient=(2.,-1.);preconditioned=(.2,-1.)
    toy=dict(reference=reference,raw_gradient=gradient,raw_dot=sum(a*b for a,b in zip(reference,gradient)),preconditioned_gradient=preconditioned,preconditioned_dot=sum(a*b for a,b in zip(reference,preconditioned)),synthetic_only=True,not_actual_Adam_evidence=True)
    return dict(state='READ_ONLY_REVIEW_COMPLETE_NO_NEW_EXPERIMENT',new_native_physics=0,new_optimizer_updates=0,new_Actor_forwards=0,initial_six=tally(initial[:54]),initial_old=tally(initial[54:]),source_success_ids=322,initial_failed_ids=sorted(set(byid)-parents),arms=arms,night_actual=actual,approved_physics=40000000,approved_commands=2000,approved_main_wall_s=18000,physics_budget_fraction_used=actual['physics']/40000000,main_budget_fraction_used=actual['main_wall_s']/18000,launched_beijing=beijing(launch['epoch']),finished_beijing=beijing(root_summary['finished_epoch']),deadline_beijing=beijing(p['deadline_all']),main_minutes=actual['main_wall_s']/60,elapsed_minutes=root_summary['elapsed_s']/60,unused_wall_after_finite_chain_s=p['deadline_all']-root_summary['finished_epoch'],relay_files_present=draft_states,relay_executed=False,relay_missing_required_docs=[str((ROOT/'stage/docs/overnight_20261108_relay'/f).relative_to(ROOT)) for f in ['PLAN.md','METHOD_SOURCES.md'] if not (ROOT/'stage/docs/overnight_20261108_relay'/f).exists()],encoding_executed=False,remote_receipt=dict(path=str(receipt_path.relative_to(ROOT)),state=receipt['state'],archive_sha256=receipt['sha256'],manifest_sha256=receipt['manifest_sha256'],members=receipt['members']),aggregate_gradient_counterexample=toy,proposed_next_diagnostic=proposal,inputs_sha256=inputs,safety='No existing evidence/models/protocols modified. Only read-only analysis docs may be written.')

def markdown(d):
    b,c=d['arms']['b'],d['arms']['c']
    return f'''# Stage 11月9日只读复盘：六米提案退化与接力未执行

本次遵照用户“先复盘，不启动”：只读取既有JSON/源码与文件哈希；新物理步、Actor前向、优化器更新均为0。未创建新run、未注册/启动T434、未执行T435冷编码。机器时间/状态复核属于事实检查，不是新科学实验。

## 已执行结果

| 同一329项开发任务 | 宽六米54 | 左27 | 右27 | 旧275 | 接受 |
|---|---:|---:|---:|---:|---|
| 原研究初始化T427 proposed | 51/54 | 24/27 | 27/27 | 271/275 | 历史未晋升 |
| T432 B 均值保持 | 50/54 | 23/27 | 27/27 | 269/275 | 拒绝并整Actor+Adam回退 |
| T432 C 增3旧轨迹＋最坏轨迹保持 | 52/54 | 25/27 | 27/27 | 269/275 | 拒绝并整Actor+Adam回退 |

C六米96.30%、旧97.82%仅是这一固定开发集的提案测量；完整联合验收失败，不能称六米稳定或真机成功。原安全T255不被替换。fresh16、reload78、八米均未执行，无成功率可填；七米按用户原要求跳过。

B修复{len(b['gained_ids'])}项、丢失{len(b['lost_ids'])}项；C修复{len(c['gained_ids'])}项、丢失{len(c['lost_ids'])}项。C丢失5项中2项就在保持集合：dev_task6_09、wide117_80_dev6_LEFT_00。C六米两失败为LEFT00（本轮新增退化）与LEFT07（原有未解决）；旧6失败中2项原有未解决、4项是本轮新增退化。因此不能把全部旧失败算成本轮遗忘，也不能只凭52/54认为只剩两条任务。

每臂120 live＋120 shadow Adam，最终保留0更新；独立审计各重构240历史丢弃Adam。源码fit明确new_RL=0，B/C都属于恢复监督＋功能保持提案，不是新增PPO/SAC在线训练。不能将所有Adam计数统称“强化学习训练量”。

## 哪些原因已有证据，哪些仍是假设

1. **新旧行为干扰已被逐任务集合证实。** 修好原失败同时丢掉原成功，且已保持的轨迹也会退化。“保持集合覆盖不足”是可能因素，但不足以单独解释所有退化。
2. **平均监督目标与闭环成功存在缺口。** C纠正损失从{c['initial_corrective_loss']:.9f}降至{c['final_corrective_loss']:.9f}（下降{100*c['corrective_loss_relative_reduction']:.2f}%），仍未整体通过。C的max选择的是每条完整轨迹平均KL中的最大值；每条轨迹内部依然跨时间和动作维取均值，并非逐时刻控制边界。少数关键动作/反馈状态漂移是否主导当前失败，尚需完整状态轨迹，不能靠平均损失断言。
3. **当前保持梯度规则不提供逐任务保证。** 代码先把old/six/broad合成一个reference，再对纠正梯度做单半空间投影，之后执行Adam。初始保持损失约1e-31、近零；聚合保护和Adam后的实际位移均不能当安全证书。REVIEW.json给出一个2维正对角预条件反例，只说明数学上无普遍保证，不是对本次真实Adam的因果证明，也不证明更强约束必能学会。
4. **场景不可通过没有得到支持。** 当前各已知失败任务已有安全教师/恢复执行证据；教师使用额外训练信息不代表单Actor一定可学会。冻结encoder/GRU是否使任务难以区分、是否需要新观测，都还未单独检验。当前应避免将结果笼统归因“场景太难”“没记忆”或单纯课程灾难性遗忘。

B/C同时改变保持覆盖和mean/max目标，不能将C比B多两个六米成功归因单独某个模块；更不能据这次开发集比较宣称新算法创新。

## 昨晚执行复盘

T432在{d['launched_beijing']}实际启动，{d['finished_beijing']}结束（含审计/备份约{d['elapsed_minutes']:.2f}分钟）；仿真/训练主管线约{d['main_minutes']:.2f}分钟。全部物理{d['night_actual']['physics']:,}、命令{d['night_actual']['commands']}，分别没有用满40M/2000/18000秒批准上限。剩余预算不等于存在已实现的接力。

T434只有源代码草案，无protocol、preflight、launch或driver_started；PLAN.md/METHOD_SOURCES.md也未落地。准备回合结束前未完成登记、独立数值合同、源码锁定和实际launch，所以它无法离线自行接棒。这是接力没有落实，不是T434训练失败或MuJoCo算力不足。T432完成后至08:30约{d['unused_wall_after_finite_chain_s']/3600:.2f}小时没有T434新训练。不能把这个空档解释成算法已经进行了整夜失败尝试。

后续无人值守必须在用户离开前交付完整有限分支＋共享预算＋所有前置合同，并以真实driver/worker回执与心跳确认启动；不能以“代码正在准备”或口头安排替代运行。断网只允许执行预编程分支，不依赖临时LLM继续编程。用户本次选择只复盘，今天没有自动启动授权。

## 下一步最小方案，仅建议未注册

先冻结T427原研究模型和C已拒绝提案作对照诊断，选4项：LEFT00（已保持仍新碰撞）、LEFT07（原未学会）、dev_task6_09（补进保持仍碰撞）、dev_task6_24（新超时，剩0.23869m）。只诊断、不继承或晋升失败提案。

已有三条T427完整原轨迹可复用哈希与旧审计；另补T427 dev_task6_24和4条C完整轨迹，再各一次精确原生重生。最多5条生成＋5条重生、14条含setup命令、300,004物理步、3MiB、600秒主流程＋300秒零新物理审计、0优化器。该预算是草案，尚未注册，也不表示今天会执行。

定位顺序：同一初态/场景/目标/时钟 → 动作首次变化 → 车体/前轮净空与航向变化 → 新状态上的动作误差 → 终止。原C仅compact动作/计数/终止和全数组哈希，不能用这些哈希重建缺失状态，必须明确证据边界。

若确认少数关键窗口变化导致碰撞，下一学习方案优先针对实际Adam提案后关键窗口函数变化与自主接管成功做受控比较；若已监督历史能拟合而自主新状态仍失控，则验证真实新学生状态的数据聚合＋分离的on-policy RL；若完整历史特征可分辨性不足，才检验表示适配。每次只改变已定位机制，固定验收，不一次叠加三机制后无法解释结果。论文创新应围绕可验证的问题和消融证据提出，标准PPO/GRU/DAgger或保护梯度本身不能作为原创主张。

## 证据与限制

原始B/C cases、development、optimizer_calls、summary、audit与完整Release逐成员收据相互核对。原始输入SHA、修复/丢失ID、丢失项保持覆盖、损失和预算见[REVIEW.json](REVIEW.json)。T432原始737成员完整远端读回收据已存在，归档SHA为`{d['remote_receipt']['archive_sha256']}`。这次没有重新运行原审计/优化器，也没有覆盖任何历史summary。T422历史训练复现缺口仍保留，不能用T432本轮审计通过宣称整个历史训练链已完全复现。
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');parser.add_argument('--check',action='store_true');a=parser.parse_args();d=review()
    if a.write:
        DOC.mkdir(exist_ok=True);(DOC/'REVIEW.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');(DOC/'REVIEW.md').write_text(markdown(d))
    print(json.dumps(dict(state=d['state'],scientific_verdict='FULL_GATE_FAIL_CONFIRMED',relay_verdict='NOT_LAUNCHED',C_fixed=len(d['arms']['c']['gained_ids']),C_lost=len(d['arms']['c']['lost_ids']),C_lost_inside_anchor=len(d['arms']['c']['lost_ids_present_in_preservation']),new_native=0,new_optimizer=0),ensure_ascii=False))
if __name__=='__main__':main()
