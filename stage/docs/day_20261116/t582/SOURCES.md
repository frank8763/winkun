# T582一手来源，2026-11-16重新核读

[Rolnick等CLEAR，NeurIPS2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)§1指出可同时生成各任务数据时可直接同时学习；§3结合fresh RL、off-policy replay和行为/价值克隆。本文采用“旧任务也有当前新鲜RL数据”的基础对照，不复现CLEAR的IMPALA/V-trace/价值蒸馏，不把只保持动作损失的T581叫完整CLEAR。代码为本库已审计CAPG/PPO实现，无新增外部代码。旧任务新鲜采样是可访问仿真条件下的选择，非未知真实终身学习保证。

[Schulman等PPO原论文](https://arxiv.org/abs/1707.06347)交互采样/裁剪目标；新旧样本都由此次更新前同一行为策略采集，组内四同行时间baseline是本项目既有方法，不称整体无偏、普遍降方差或安全证书。两组相同新任务系数，防止把新任务权重缩小当保旧机制；增加旧梯度仍可能冲突，需自主逐ID检验。

项目反证：T581完全冻结T550 body/末新任务KL仅4.87e-7，仍丢5场地；27静态保持初始梯度0，不能证明当前旧任务执行不退。四左六米探索无成功，超时在失败同行中获正相对优势；不能承诺再加旧RL就学会新任务。T580 MER无新增、匹配步长ER虽恢复八米右却丢场地，故本轮不追加元更新。

[DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)与[DART](https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf)已在T581复核，需要有能力的纠正监督者。T550当前未具备左六米成功恢复资格，本轮不用失败教师标签。部署始终同一Actor，无外部规划器/路点/地图/运行教师/路由。

已核读CLEAR方法正文，PPO本次重新核对原论文条目/方法原则，具体密度/代码来源T581/T519既有逐位审计。未声称穷尽近期研究、完整论文复现或新颖性；原公式与标准训练采样本身不能支撑顶刊创新。
