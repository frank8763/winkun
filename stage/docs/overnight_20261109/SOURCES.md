# 一手近邻与本晚证据边界

- [DAgger, AISTATS2011](https://proceedings.mlr.press/v15/ross11a.html)：学习策略自身诱导状态上的专家查询/数据聚合，直接针对序列误差累积。这里先学生1秒再教师完成并验证GOAL，是有界恢复混合数据，不声称完整DAgger复现或onpolicy RL。
- [DART, CoRL2017](https://proceedings.mlr.press/v78/laskey17a.html)：展示分布偏移与恢复覆盖的重要性。没有复现其优化噪声算法，不能把普通iid探索称DART。
- [TaSIL, NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f10c3d66c3b7863a9cda255dcac5bb7-Abstract-Conference.html)：动作拟合与闭环稳定性间需要额外条件。这里不实现Taylor导数匹配，不借其理论声称安全。
- [Group DRO, ICLR2020](https://openreview.net/pdf?id=ryxGuJrFvS)：平均与困难组风险、正则化不能混同。本轮是固定回合内部upper20%经验尾部损失与均值各半，不称标准groupDRO复现/原创，也没有独立等计算对照。
- [DQfD, AAAI2018](https://ojs.aaai.org/index.php/AAAI/article/view/11757)：示范与真实强化学习组合的已知先例。本晚连续控制PPO+BC不是离散Q-learning DQfD，不借论文名称或结论代替本项目验收。

可证伪目标：尾部拟合后完整闭环是否保住原成功集合并新增成功；实际学生早期偏离后T255能否恢复；独立纯学生探索产生的PPO更新是否能保持功能成功集合。loss下降、教师成功、噪声探索成功、几个中心成功均不能代替最终完整门槛。所有组件已有近邻，本晚结果不足支撑顶刊新颖性或sim2real有效性。
