# T458：冻结warmup的缺失对照

2026-11-10，Stage限定，只读文献复核；本说明未运行原生、模型前向或优化器。

原T449研究父候选328/329。T455做128BC后仅测11中心，10/11；其PPO0经T456完整测为305/329，PPO1为245/329。T457从同一warmup做均衡22条新交互后为297/329。因此目前不能把这些完整失败全部记为PPO新增退化。项目依据：[T455记录](T455_FINDINGS.md)、[T456记录](T456_FINDINGS.md)、[T457末报告](T457_FINAL_REPORT.json)。

先对**已保存同一warmup**冻结全测原329；“零训练”指本次不追加BC/PPO或教师查询，已有128BC并未抹去。锁定模型/源SHA、任务签名、初态、时限、动作与终止契约，并复现原11中心。每个候选逐ID与warmup配对，分别列出两者皆成功、warmup已失败且候选仍失败、warmup成功而候选失败、warmup失败而候选修复；另按原328成功集合与目标00分层。总分可能掩盖相互抵消的修复和损伤，已失败ID也可能从超时变碰撞。

这能定位失败在预热后是否已存在，并描述warmup至各候选的净变化；父候选至warmup的差异属于预热阶段。它**不能独立分解**PPO裁剪、Value/GAE、辅助监督、Adam、覆盖、归一化或多epoch的效应，也不能证明128BC是唯一根因。T457同时改变多项训练设计；单个训练历史不能估计训练随机性，更不能由失败直接推出网络容量不足。以下文献只提供解释框架。

- Schulman等，*Proximal Policy Optimization Algorithms*，[作者v2全文](https://arxiv.org/pdf/1707.06347v2)，实际读§2–3、§5及Algorithm 1：先由固定旧策略交互，再以有限批次优势和旧密度比优化多epoch。裁剪下界针对代理目标；这些段落不提供原328成功ID逐项保持保证，平均KL也不能替代闭环验收。
- Ross、Gordon、Bagnell，*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*，[AISTATS正式全文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)，实际读§1、§2正文、§2.1、§3含Algorithm 3.1与定理条件：动作影响未来输入，专家分布监督误差与学生诱导分布表现不同。故冻结encoder/GRU仍不能阻止头部改变轨迹和hidden；这是本地推论，不把离散损失界直接套给连续控制，也不把本次冻结评估称为DAgger复现。
- Cobbe等，*Quantifying Generalization in Reinforcement Learning*，[ICML正式全文](https://proceedings.mlr.press/v97/cobbe19a/cobbe19a.pdf)，实际读§1、§3.1–3.2：用未见测试关卡、测试时不微调来区别泛化。Stage原329已反复参与开发选择，同地图未直接训练的扰动也不自动成为独立留出；零训练评估不会恢复其独立性。CoinRun结论不直接证明本车跨地图能力。

329若仅保留compact终止、动作/步数摘要与SHA，独立审计只能按该证据核对逐ID结果和原面板复现，不能宣称329完整120维history/hidden独立原生重建，更不能抵作未执行的16fresh/78重载。此对照补齐阶段归因缺口；无论结果如何，均不证明新算法、全局安全或充分解决导航，不自动晋升、扩距或追加训练。
