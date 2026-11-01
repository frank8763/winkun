# 长距离扩展的机制、相关工作与论文证据边界

检索日期：2026-11-01。本轮阅读一手论文页面、可访问正文与作者仓库 README；未声称穷尽全网、复现实验或审计全部外部代码。不下载外部模型或安装新依赖。

## 目前真正支持的诊断

1. 五米成功只覆盖当前地图八个任务族和局部扰动，冻结 T255 为211有效开发+64留出全过；不是任意位置五米或全地图。冻结父模型未学习六米前已经失败，所以灾难性遗忘不是充分解释。
2. T278 固定六米终点的1米新起态也超时、3米起态接触：末段状态控制有缺口。该重置历史不是原轨迹完整后缀，不能夸大为全局不可达或所有短距均失败。
3. T280 有界端点代价已打破同 cap 超时回报等价，但两臂仍只1/54且成功点已有历史记录；T281 显示左右/末段梯度尺度严重不均。T283 将任务尺度平衡后0保留更新，4次真实 old20 为19/20且同 dev_task6_24 接触。说明新学习信号、旧行为保持和闭环状态偏离均需区分。
4. 不存在“场景不可解”证据；也没有充分证据把问题归结为单个算法名字、单纯时长或单纯遗忘。T216 去掉保持也未学会旧困难任务，不能推断放开保护就能成功。

## 一手近邻及适配

|来源|可借鉴机制|Stage 的边界|
|---|---|---|
|[CAPS, ICRA 2021](https://ai.bu.edu/caps/)，[论文](https://arxiv.org/abs/2012.06644)|动作时间/空间平滑正则|平滑、降低抖动不等于旧任务保持或长距离成功，不能包装为原创|
|[SR²L, ICML 2020](https://proceedings.mlr.press/v119/shen20b/shen20b.pdf)|局部扰动下策略平滑/鲁棒性|观测各维有物理相关性，不宜给120列独立任意噪声冒充真实可达状态|
|[CLEAR, NeurIPS 2019](https://proceedings.neurips.cc/paper/8327-experience-replay-for-continual-learning.pdf)|新 on-policy 学习和旧 replay/行为保持|已有 soft 保持，增加 replay/自蒸馏本身非创新；旧轨迹不能无校正当当前 on-policy|
|[Fine-tuning RL…Forgetting, ICML 2024](https://arxiv.org/abs/2402.02868)，[作者仓库](https://github.com/BartekCupial/finetuning-RL-as-CL)|微调初期状态覆盖缺口与旧能力保持|读了论文摘要/部分正文及 README；未复现 NetHack，也不能据此断言 Stage 的初始六米失败是遗忘|
|[OGD, AISTATS 2020](https://proceedings.mlr.press/v108/farajtabar20a.html)|用旧输出 Jacobian 约束参数更新|投影已是现有方法；一阶零变化不保证真实闭环保持，Stage T211–T215 已有相关失败|
|[稳定性约束模仿学习, L4DC 2022](https://proceedings.mlr.press/v168/tu22a.html)|闭环增量稳定性与时间跨度/样本复杂度关系|Stage 未验证理论所需增量稳定性假设，不能据此颁发安全保证|
|[TaSIL, NeurIPS 2022](https://arxiv.org/html/2205.14812)，[作者代码](https://github.com/unstable-zeros/TaSIL)|专家策略导数匹配、处理策略诱导的分布偏移|正文方法/相关工作和仓库 README 已查；导数匹配不新，Stage 需要真实 RL 收益且无外部教师部署|
|[Distributionally Robust and Safe IL, 2026预印本](https://arxiv.org/abs/2607.13436)|TaSIL 联合分布鲁棒自适应控制|本轮仅核对摘要，未审计实现；说明“导数保持+鲁棒安全”已有更近组合，且其自适应控制结构不直接符合当前单 Actor 部署|
|[Horizon Generalization, ICLR 2025](https://horizon-generalization.github.io/static/pdf/horizon_generalization.pdf)|长距离推广需要相关短程状态区域能力与条件|固定八个起点五米全过不满足这些覆盖条件，不保证十米；不能引入外部规划器来混淆端到端结果|

前一轮 PCRC/TMD/TOPPO/CLI-PPO 和作者 README 的检索见 EVENING_REVIEW_AND_DECISION.md；新课程、时间距离头、记忆或梯度平衡都已有近邻，且当前部分代理头已有负结果，不盲目拼接。

## 可研究的候选贡献，而非已成立的创新

研究问题：怎样用真实闭环反例分配旧行为保持预算，让同一个无地图 Actor 学习更远目标，同时保持已有短距能力？

候选方法应把真实轨迹的反馈偏离证据变成**学习更新约束**，而不是只报告动作 MSE。可考察关键状态的输出/方向导数保持，约束实际 Adam 参数位移而不是只投影原始梯度（预条件会改变方向），随后非线性读取与原生接受检查。这仍接近 OGD、TaSIL、信赖域、已有 Stage 投影，尚不能声称原创。需证明额外机制、计算代价与效果；若没有优于匹配对照，应放弃该创新叙事。

T284 只恢复一个已拒绝候选并精确分解两类差异，先检验这个方向是否有事实基础。不能因一个比值较大就跳到算法/因果收益。

必须同时解决新目标末段的学习信号与覆盖。闭环保持只防退化，不能凭空创造六米后段能力；现有1米原地重置和端点代价均未成功，要先区分候选是否真的改善新目标，不能只优化通过旧20。

## 支撑高水平论文还缺什么

- 主结果：同一 Actor 真实完成6→7→8→10米；每级冻结验收，未见起态/目标/障碍组合，失败类型和物理预算完整。现阶段六米仍未过，谈不上论文完成。
- 因果对照：匹配数据/物理预算/初始化，原 RL、普通 replay/保持、标准平滑/投影、所提闭环机制；等数均匀状态 vs 闭环关键状态，保持机制与新目标学习分开消融。
- 泛化与可信度：后续另注册多个训练种子、未见地图与动态扰动及置信区间；不能复用已消费留出充当新测试。当前单地图结果不能外推全地图。
- 部署：保持120因果输入单 Actor，无地图/规划器/路点/运行时教师/路由；历史兼容组件与真实RL/辅助监督明确标注。若最终主要靠模仿修补，不能写成纯新端到端DRL算法。
- 理论/解释：限定假设的误差传播或更新保持分析，直接检验其适用条件；不把缓存、逐位重建、有限病例零碰撞写成安全证书。
- 验收后才生成新的 MuJoCo 展示；不拿旧五米视频、探索偶然成功冒充长距离稳定。上述清单不是自动启动多seed/多地图的授权协议。

## 补充检索：实际 Adam 更新投影也不是新概念

[Adam-NSCL, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Training_Networks_in_Null_Space_of_Feature_Covariance_for_Continual_CVPR_2021_paper.html)明确将 Adam 生成的参数更新投影到历史层输入特征协方差的近似零空间；[作者仓库](https://github.com/ShipengWang/Adam-NSCL)提供实现。本轮核对论文摘要/方法首页及作者 README，未执行外部代码。其视觉分类结果不直接证明 Stage 闭环效果，但足以排除“Adam 后投影本身原创”的叙事。

[Fisher-Orthogonal Projected Natural Gradient, 2026预印本](https://arxiv.org/abs/2601.12816)也提出 Fisher 几何下的保持约束；本轮仅核对摘要，不能声称实现/理论已验证。新的候选必须围绕闭环数据如何选择约束、适应能力与保持之间的实测权衡建立独立证据，不能只是换一个投影度量。

## T285负结果与时间信用候选

T285两个18名义行约束（有效秩18/17）将选中状态误差降至5.2e−7/6.8e−7，但关键旧五米仍接触；0新Actor/Adam训练。不得把闭环证据选择已经有效作为论文主张，也不加锚点/秩网格。下一T287独立检验时间对齐reward-to-go的同行LOO基线，保持现有真实RL架构/接受门槛，四轮64上限与历史T283两轮32不同，只有前两轮可作阶段描述，不能夸大为完整匹配单因素试验。

[Back to Basics: Revisiting REINFORCE Style Optimization, 2024](https://arxiv.org/abs/2402.14740)为RLOO已有近邻，本文任务是LLM反馈优化，不能直接当作导航有效性证据；[Munos: Policy Gradient in Continuous Time, JMLR 2006](https://www.jmlr.org/papers/v7/munos06b.html)表明物理时间梯度研究已有长期基础。本轮仅核对两者论文页/摘要，不声称复现。物理时间对齐、reward-to-go或独立baseline本身均不能包装原创。T287完整算法还含批次依赖RMS、PPO剪裁、PCGrad和接受门槛，不以未归一化基线的零期望score恒等式宣称整体无偏或普遍降方差。
