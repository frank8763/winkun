# Stage 当前问题证据分层（2026-11-13约13:00）

本轮按用户要求只检查与检索，不启动新修复/训练。T524原16live/shadow已完成，原32exact/audit仍允许按原13:17硬截止收尾；本报告T524数字为完整首次32验收，最终状态以原summary/audit为准。

## 实际成绩与任务范围

历史T501同一模型原old271/275、宽six54/54、宽eight54/54，不能把该已消费开发集成功等同全场地、泛化、全383通过（实际379/383、另fresh16/reload78未执行）。当前接受fullfield模型T507经T508冻结补验old8/8、field11/24；这不是同模型old275/six/eight重验全过。T519研究12/24旧7/8不晋升；T520修旧后field11/24旧8/8；T522 BC field9/24旧8/8；T523 field10/24旧8/8；T524首次field10/24旧8/8，与T52332项终止类别逐ID全同。不得把任何研究候选的最好指标拼接成一个部署模型成绩。

## 已观测问题

1. 学习后能力回退：修旧/新有取舍，符合遗忘现象。尚未分离覆盖旧知识、任务间干扰、有限优化/样本或闭环敏感性因果。不是证明单网络不能兼容。
2. 离线动作拟合与闭环结果脱节：T522固定两成功history BC损失从.00185269降至.0000658293（约下降96.45%），field却9/24。历史输入上的输出误差不是学生新访问状态上的专家纠正。soft19参考实际刷新为T519父模型在历史输入的当前函数，该父模型旧7/8；因此参考不是全部旧任务的已验证恢复专家或成功证书。
3. 近目标捕获失败：T524 sector0_offset2最近决策端点.347152m（78.8秒），仍在.2m成功区域外，后远离49.424秒，最终2.83456m超时。不能说已经到达/改判成功，也不能只说找不到目标。是否转弯半径、速度/转向时序或状态估计问题尚未因果确认。
4. 早期绕障失败：sector2_offset2在51.122秒余5.264785m与real_circle_obs_0接触；不是多等一会就能修复的超时。全field其他失败还包含不同障碍/更长路径，不能以此单例归纳全部碰撞原因。
5. 覆盖与评价差距：T522–T524配对分析实际共享原8条轨迹，只两个起终点，重复优化/精确重生不增加独立训练分布。训练探索成功不是无噪确定性模式成功，4/4不等于场地稳定。大量实验编号包括零Actor诊断/预检/被guard拒绝，并非大量新的长周期广覆盖RL。
6. 单因素解释尚不充分：T523第一任务梯度cos.021235（首步未反向）、范数old约2.486倍new。T524按固定初始范数平衡，首次验收仍同10/24，只说明此固定平衡对本批没有提高通过率，不否定全部梯度方法/遗忘。

## 相关一手研究与匹配边界

- CLEAR NeurIPS2019 §4.1区分forgetting/interference，并用同时/顺序/分任务训练对照；研究DMLab/Atari，不是本Ackermann直接复现：https://arxiv.org/html/1811.11682v2 。
- DAgger AISTATS2011强调策略改变后状态访问分布改变，需要学生访问状态上的专家标签；低专家轨迹误差不能直接当自主成功：https://proceedings.mlr.press/v15/ross11a.html 。
- DART CoRL2017直接讨论偏离示范后误差复合/恢复数据，实际MuJoCo运动+HSR抓取而非本导航：https://proceedings.mlr.press/v78/laskey17a.html 。
- TaSIL NeurIPS2022指出小动作误差可能放大为闭环失败，理论依赖专家增量稳定性及所需阶数匹配，尚未在本仿真验证这些假设：https://proceedings.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf 。
- PCGrad NeurIPS2020分析负余弦、幅值差和曲率组合；不支持只凭幅值差判定已查明冲突，更不证明固定归一化必然解决：https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf 。
- Learning Navigation Behaviors End-to-End with AutoRL RA-L2019：作者页面明确讨论DRL导航catastrophic forgetfulness，LiDAR到线/角速度及仿真/真机；点到点与路径跟随分列，自动奖励/网络大规模搜索，不等于我们的单Actor完整全场地图已被证明可复现：https://research.google/pubs/learning-navigation-behaviors-end-to-end-with-autorl/ 。大范围PRM-RL另含规划，不能移称无规划纯端到端。
- Deep RL That Matters AAAI2018分析continuous control评估/种子/实现方差，支持谨慎解释少数样本，不给本场景所需样本数定理：https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553 。
- Nonholonomic Narrow Dead-End Escape with DRL, arXiv2025，Ackermann曲率/不可原地转向和前后退序列是相近物理问题。仅核摘要及公开项目声明，不宣称本项目已验证它或本局部状态不可达：https://arxiv.org/abs/2511.22338 。

目前最有依据的总体表述是：单策略在扩大任务分布时，旧能力保持、自己访问状态上的恢复控制、与真实闭环验收之间尚未打通。遗忘是症状之一，数据覆盖和闭环鲁棒性是合理高优先级假说；尚未证明是SAC/PPO某个名字、GRU容量、相机缺失、仿真引擎或场景本身唯一导致。相关文献证明这类问题被系统研究，不能替本项目给成功保证。本次不新增修复或预算。
