# 五米成功、六米失败：遗忘与未学会的区别

2026-11-01检索一手正式论文/作者页面；本轮核对以下摘要，不声称完整复现论文或穷尽全网，不引入作者代码。已有Stage原始证据及T278方案是项目推断，不是论文对本项目的结论。

- [Horizon Generalization in Reinforcement Learning，ICLR2025](https://arxiv.org/abs/2501.02709)，作者Myers/Ji/Eysenbach。正式摘要研究近目标训练到远目标的泛化，理论需假设和planning invariance。支持将距离泛化当作独立问题；不意味着本Actor满足假设，也不授权运行时规划器/路点。已有Stage输入与动作对目标距离敏感，而非同动作自然延长。
- [Reverse Curriculum Generation for Reinforcement Learning，CoRL2017](https://proceedings.mlr.press/v78/florensa17a.html)，正式摘要从近目标起态逐步扩展。采用其固定最终目标、研究起态能力的思路，当前仅四个冻结probe；不实现其完整自动课程，不称原创。历史Stage T184/T190已有相关失败；完整动态状态、起态几何及能力连通性必须重新核验，近起态成功不等于原六米通过。
- [A Study of Plasticity Loss in On-Policy Deep RL，NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce7984e36d58659211a8dc7d5457cd6f-Abstract-Conference.html)，Juliani/Ash，正式摘要区分在线网络拟合新任务能力下降，且若干外部补救在on-policy中失败/更差。说明“新任务学不进”不能直接等同“旧能力被忘掉”，也不应未经匹配检验直接重置神经元、加正则或堆模块。本项目未证明可塑性损失，无自动再生机制。

证据链：T256零新优化器六米失败→先存在新距离能力缺口；T259旧一项退化→确有局部遗忘/负迁移风险；T267/T268旧275全保留但六米0/54→保持旧技能并不足以学到新技能。当前最强结论是**遗忘不是六米失败的充分解释**，不是“遗忘绝不存在”或“唯一根因已找到”。下一优先取得真实六米终点后段的冻结能力/有效几何证据，不继续用预测分数改善替代导航成功。

该诊断不能支撑顶刊原创性。若未来主张固定终点课程、失败感知目标表示或保持与适应的新机制，必须与这些近邻区分，给出真实端到端Actor学习、相同物理预算对照、消融及独立种子/场景证据；此处只是证据与试验设计，不自动启动大规模网格。

补充作者资料核对：已查看[Florensa作者仓库README](https://github.com/florensacc/rllab-curriculum)，其逆课程示例入口为arm3d_key_brownian.py，框架为rllab，机器人/任务/状态和Stage小车并不相同；未下载、移植或声称复现作者算法。[Horizon Generalization作者页面](https://horizon-generalization.github.io/)说明随机策略也可形式上planning invariant却不具备远距泛化，且实测方法都未完美泛化。这是重要限制：不能靠“局部相似动作”或一组短任务通过就推定长任务全过。页面明确的确定性MDP/特定目标奖励/全体近邻最优等条件不等于本项目已有保证。
