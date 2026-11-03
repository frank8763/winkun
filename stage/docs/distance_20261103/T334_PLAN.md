# T334：已学习候选在真实纠正交接状态的固定复核

2026-11-03。T332/T333配对学习结束、独立全部1200历史Adam审计PASS。control旧17/20，recovery旧20/20，但两臂宽5均0/5接触，原窄中心左接触右超时。原始失败与模型保留，不追加该机制学习预算。下一要区分：前段自主运行把策略带离人控纠正覆盖，还是即使在真实纠正状态也未学会后段。

固定T331真正成功混合轨迹的前150决策/30秒，实际native执行同原actions/counts，每条比较全部12数组/history至交接逐位一致，交接native积分必须等于原handoff.json。随后control或recovery各自单Actor canonical batch1 mode独立执行剩余原80秒时限，直到原GOAL/接触/超时；不reset/改goal/clock/history、无runtime人工纠正。每条另一次全12数组/终态积分精确重生。**这是受控前缀后缀诊断，不是原起点全程自主六米或部署教师/路由**。T323曾测试旧父接手40秒状态失败，本轮测试的是经过新真实恢复学习的两候选及其实际30秒学生状态，不能拿旧父结果代替。

文献与反证复核沿用本轮刚读[DAgger正式页面](https://proceedings.mlr.press/v15/ross11a.html)状态分布与[HG-DAgger v2 §I](https://arxiv.org/pdf/1810.02890)人控反馈和共享控制问题。真实恢复数据仍未使原起点通过，不能直接推断灾难性遗忘/场景无解；只有固定真实交接对照才能更具体描述覆盖缺口。本项没有新算法代码来源或作者方法复现，不称原创/安全保证。

模型取T332 control_candidate和T333 recovery_candidate原字节，基础safeT255/normalization不动；0梯度/优化器/Actor学习。1零物理真实native几何与两训练模型载入/首动作预检+2派生×2精确=5命令，总160k所有物理（每次都计前缀），600秒末120收尾/own3MiB/global1GiB/free40GiB。2完整NPZ各≤768KiB+至多1partial同界共2.25MiB，metadata.5MiB+buffer.25MiB=3MiB；无本地临时tar。单进程CPU0，绝对截止硬停止，不加交接时刻/模型/种子网格，失败部分如实保留。

两候选、T331真实轨迹/原协议/handoff与历史源、T333独立audit/远端完整备份、代码和本计划输入SHA全部锁定；干净源码完整远端精确后唯一launch。结束所有父/timeout/inhibitor退后独立零物理完整数组/history/clock/reward/所有后缀Actor动作/原生重生与命令核验，再全部RAM远端逐成员SHA/source收尾。末段通过也不晋升、不执行完整275/54/fresh16/78或进入七八米。Phase完全隔离。
