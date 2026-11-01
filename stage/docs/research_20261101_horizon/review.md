# Stage距离外推、安全保持与论文研究更新（2026-11-01）

本轮按长时域GCRL、失败/删失、安全保持三条关键词检索一手来源。没有穷尽全网，14项来源的阅读深度逐项列出；摘要证据不写成完整算法审计。除P04机构技术报告和明确标注的预印本外，会议归属按正式页面/作者稿说明。未来上线算法须另查完整方法、作者代码、许可和适配契约。

| 来源 | 本轮阅读范围 | 与Stage的关系 |
|---|---|---|
| [P01 Automatic Goal Generation for Reinforcement Learning Agents](https://proceedings.mlr.press/v80/florensa18a.html) · ICML 2018 | 摘要、PDF问题/假设/目标难度章节 | 按能力选择目标，适合更细课程；不声称复现GAN或独创课程 |
| [P02 Horizon Generalization in Reinforcement Learning](https://horizon-generalization.github.io/) · ICLR 2025 / 作者项目页 | 项目页定义/定理概述 | 近距离能力不自动推出远距离能力；理论有假设，不引入路点规划 |
| [P03 Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://papers.neurips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract-Conference.html) · NeurIPS 2022 | 正式摘要 | 表征内积可表达目标条件价值；需要合适轨迹/目标标签与独立Critic验证 |
| [P04 Distributional Distance Classifiers for Goal-Conditioned Reinforcement Learning](https://publications.ri.cmu.edu/distributional-distance-classifiers-for-goal-conditioned-reinforcement-learning) · ICML 2023 workshop / CMU硕士报告 | 机构摘要、workshop PDF可读检索片段 | 到达时间与到达概率必须区分；不能把成功样本时长当所有策略的可达距离 |
| [P05 Scaling Goal-conditioned Reinforcement Learning with Multistep Quasimetric Distances](https://proceedings.iclr.cc/paper_files/paper/2026/hash/eddc0fab5a42f8d8de6eb5566cd9f1d3-Abstract-Conference.html) · ICLR 2026 | 正式摘要 | 多步MC拟合准度量；暂不复制价值梯度，更不引入图搜索 |
| [P06 Dual Goal Representations](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f8cd7eb180deb5c6ec876de2b442fa72-Abstract-Conference.html) · ICLR 2026 | 正式摘要 | 时间关系表示属于已有研究；新增goal encoder不是原创性证明 |
| [P07 SVL: Goal-Conditioned Reinforcement Learning as Survival Learning](https://arxiv.org/html/2604.17551v2) · ICML 2026；arXiv v2 | 摘要、方法/删失/层次actor目录与开篇 | hazard/到达时间/删失已存在；借鉴标签契约，原层次Actor不照搬 |
| [P08 Survival Reinforcement Learning: Toward Scalable Self-Supervised RL](https://arxiv.org/abs/2605.31273) · arXiv 2026 v2，未据此声称正式录用 | 原稿摘要 | 目标驻留与一次到达不同；本项目不静默改变terminate-on-GOAL目标 |
| [P09 Arrive and Survive: Scaling Safe Goal-Conditioned Policy Learning from One-Bit Failure Signals](https://arxiv.org/html/2608.26571v1) · arXiv 2026 v1，未据此声称正式录用 | 摘要、方法/实现/失败vs删失技术条件目录与开篇 | 失败质量校正是强近邻；碰撞不能当普通时间截断，失败预测头本身非新 |
| [P10 SafeAdapt: Provably Safe Policy Updates in Deep Reinforcement Learning](https://arxiv.org/html/2604.09452v1) · arXiv 2026 v1 | 摘要、相关工作、方法/假设/安全规格节目录与开篇 | 已研究安全保持和参数投影；有限/离散及unsafe标签等假设不能搬成连续车保证 |
| [P11 Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](https://arxiv.org/abs/2605.11975) · ICML 2026（作者稿标注） | 作者原稿摘要 | 概率reach-avoid与成本共同优化已有理论；当前原生回归不是该证书 |
| [P12 Safe Multi-Agent Navigation guided by Goal-Conditioned Safe Reinforcement Learning](https://arxiv.org/abs/2502.17813) · ICRA 2025 | 作者摘要、作者仓库README | 联合距离/安全价值有先例；其图和路点执行与Stage契约冲突，不采用 |
| [P13 Offline Goal-Conditioned Reinforcement Learning for Safety-Critical Tasks with Recovery Policy](https://arxiv.org/abs/2403.01734) · ICRA 2024 | 作者原稿摘要 | 安全GCRL已有研究；恢复策略方案不作为单Actor运行时控制器 |
| [P14 Success-Conditioning as Policy Improvement: The Optimization Problem Solved by Imitating Success](https://proceedings.mlr.press/v306/russo26a.html) · ICML 2026 | 正式摘要 | 精确success-conditioning的理论不等于有限神经BC；T231/T243等失败不能被理论抹去 |

## 对当前六米问题的判断

现有证据支持三个相互关联但未完成因果分离的问题：左六米缺少新鲜安全到达覆盖；失败终止的完整回报不按最终距离排序；旧动作函数误差很小仍可能造成旧左五米接触。T261排除了已观测数据中的奖励符号/折扣实现错误，不代表目标信息充分。T259只改变旧能力接受条件，没有改表示/课程/奖励；先让其按锁定预算完成，完整275+54门槛比少数中心重要。

直接把SAC改名为PPO、加入GRU/PCGrad、课程、survival头、准度量或成功BC，都不能单独当原创性。SVL/Safe-CRL/SafeAdapt是必须正面对照的近邻；安全GCRL和不依赖运行时shield的安全保持也不能说首次。

短期仍先六米：若完整旧任务保持且左侧到达覆盖很少，下一优先候选是预声明的更细独立目标课程，保持同父模型/同训练回合上限/同奖励与优化器以考查覆盖变化。若已有充足成功却确定性未学，改查梯度/随机行为与确定性部署差异。T197距离项已试而失败，不开相同罚项网格。

中期算法候选是以校准过的“到达先于碰撞且在时限内”的事件价值指导扩距更新，并在训练阶段约束旧任务的行为退化；它目前只是可证伪研究假设，未证明优于已有方法或具备新颖性。原生闭环接受作为有限证据兜底，不是全局理论证书。部署保持单Actor，无地图/外部规划器/路点/运行时教师或路由。

T262只完成事件标签、物理时间和概率质量的零步数据合同；没有训练价值模型/Actor。六米数据缺完整120维状态，需要另行有界精确恢复和目标/历史重标记验证，不能根据标签预检就宣布新的DRL算法已经跑起。详见candidate_event_method.md、novelty_and_evidence.md和t262_contract.md。
