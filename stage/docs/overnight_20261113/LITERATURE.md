# 2026-11-13：单 Actor 持续导航的一手文献与有界分支

本文件是研究建议，不是实验注册、运行授权或成功报告。只读 Stage 上下文、RESEARCH_STATUS 前100行及 T539/T540 最新报告；未读取或使用另一研究线。未新增 runs、模型、原生交互或优化器步骤。文献检索日期 2026-11-13；只以论文、作者页面及原作者机构代码作为技术依据。

## 当前证据先于算法选择

[T539_FINAL_REPORT](../day_20261113/T539_FINAL_REPORT.json) 证明一次真实全 Actor SGD、一次 shadow 和一次历史重构，复用四条 T519 完整轨迹，旧8初验8/8；field 只有两例1GOAL/1TIMEOUT，完整24例和32次 exact 都未完成。[T540_FINAL_REPORT](../day_20261113/T540_FINAL_REPORT.json) 是结果适配器工程失败，63,476物理步的新增结局未知，不能当两项导航失败；候选仍未晋升。修复见 [FINDINGS_AND_FIX](../day_20261113/T540_FINDINGS_AND_FIX.md)。因此先把现存候选的完整成绩查明，比复制训练更能消除当前不确定性。

## 一手研究实际支持什么

| 来源 | 可直接采用的机制 | 本项目的适用边界 |
|---|---|---|
| [Rennie 等，SCST，2017，§3–4，式4–9](https://arxiv.org/html/1612.00563) | 整回合回报乘完整序列 log-probability 的 REINFORCE；以同一个当前模型的测试时确定性结果作 action-independent baseline，无需训练 Value。 | 原实验是图像描述，不是连续导航或保持旧技能的证明。迁移的是估计器。二值安全GOAL可以直接作为任务指标。 |
| [Schulman 等，PPO，2017，§2–3](https://arxiv.org/pdf/1707.06347) | 旧/新策略概率比和 clipped surrogate，支持同批数据多轮有限优化；论文明确指出反复优化原始 PG loss 缺乏依据且可能造成过大更新。 | PPO clipping 不是旧任务闭环安全保证，也不创造未采到的成功。无需把“PPO”与某个失败的 Value 实现绑定。 |
| [OpenAI Baselines 的 PPO2 原始机构代码](https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py) | 明确保存 OLDNEGLOGPAC，计算 exp(oldneglogp−newneglogp)，取 clipped/unclipped loss 的保守项，并记录 approximate KL/clipfrac。 | 该实现另有 Value、优势标准化、Adam；只借 clipped policy loss 时须明示 actor-only adaptation，不能声称完整复现。 |
| [Rolnick 等，CLEAR，NeurIPS 2019，§3](https://papers.nips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf) | 新经验学习与历史 replay 混合；历史策略分布到当前策略的 forward KL 和 Value cloning 保护输出；无需输入任务编号。 | 完整 CLEAR 是 actor-critic + V-trace + policy/value cloning。只做“fresh PG + 历史策略保持”是简化改编，不叫完整 CLEAR。论文区分 interference 与 forgetting，不能仅由一次旧任务退化断言唯一原因。 |
| [Espeholt 等，IMPALA，ICML 2018，§4](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) | V-trace 显式处理行为策略与学习策略的差异，截断重要性权重控制 off-policy 更新。 | 历史成功轨迹不能不加说明反复当 current on-policy RL；引入完整 V-trace/Value 是另一机制，今晚不宜顺带重做。 |
| [Kapturowski 等，R2D2，作者页面](https://willdabney.com/publication/r2d2/)，及 [DeepMind Acme R2D2 learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/learning.py) | 作者研究 recurrent state staleness/representational drift；机构代码以当前和 target 参数分别 unroll burn-in，然后才处理训练序列。 | 原算法是 recurrent DQN，不能直接声称适配连续 Actor。可借鉴的是历史依赖合同；stored hidden 不因数组可读取就仍与更新后参数相容。 |
| [Chiang 等，AutoRL，RA-L 2019，Google Research 原文入口](https://research.google/pubs/learning-navigation-behaviors-end-to-end-with-autorl/) | noisy lidar→机器人速度的端到端 point-to-point 策略已有先例，作者在仿真和机器人测试。 | 它依赖大规模 reward/architecture 搜索；不是现有固定 GRU 今晚少量更新必然成功的依据。此处不引入其 path-following 或外部路径。 |

## 实施前必须排除的三个假比较

1. **一次初始梯度不是 PPO clipping 对照。** 在同一参数 θ=θold、同优势、相同样本权重和优化器下，ratio=1，clipping 未激活，PPO 与 REINFORCE 的首步梯度相同。若结果不同，先检查 loss reduction、优势标准化、optimizer、hidden 或密度，而非称 clipping 有效。这是对 PPO 公式的直接推论。
2. **初始 forward KL 的首步梯度为零。** 若历史标签恰是初始化父策略在同历史上的分布，KL(parent||current) 在初始化处最小。把它加到唯一的一次 SGD 中不能改变这一步；动作均值 MSE 也有同一问题。要研究保持作用，至少需第二个固定优化步骤，或明示先 PG 再一次 proximal 保持修正；后者的原始PG和修正成本分开记账。这是损失函数的数学推论，不是 CLEAR 的导航保证。
3. **二值 baseline 的信号不同。** 令 Z=1 当且仅当原生安全GOAL，g=(1/B)Σ_i(Z_i−b_i)Σ_t∇logπ(a_it|history_it)。b=0 时所有失败梯度严格为零；b=1 时成功梯度为零、失败受抑制。真正 SCST 的 b 必须来自同目标、同起态、当前确定性策略；旧任务历史成功不自动意味着更新后 b=1。baseline stop-gradient，不按 sampled action 选择。整回合求和再按回合均值，改为每回合除以长度会改变任务权重。依据上列 SCST 式4–9及代数。

## 建议的有限优先顺序（推论，尚未注册）

“离线过夜”应明确为**本机不依赖网络/LLM**，还是**完全不新增采样**。后者可以做已知行为策略的一次配对更新/有限 surrogate 改编，不能在参数变化后无限重复称作 on-policy 学习。

| 优先级 | 固定分支与硬上限建议 | 要证伪的假设 / 失败时动作 |
|---|---|---|
| 0 | 冻结 T539 原候选，仅补有证据缺失的 original-start 验收；0新训练。先通过真实 worker schema、落盘后派生汇总的合同。 | 一次 terminal success PG 是否已保持旧8和原field11并新增任务。完整结果不增或旧成功丢失即拒绝；工程缺测保持 null，不重训掩盖。 |
| 1 | 一个预锁定父模型；最多2轮，每轮8条完整 fresh stochastic 轨迹（旧/新任务各4），一次全 Actor terminal-SCST SGD；最多16新训练回合/2 live 更新，shadow/audit 另计，不扫LR/seed。每轮 baseline 由当前确定性策略按相同task/reset取得，或已有同参数同起态完整记录精确复用。 | 对“真实整回合成功指标能否兼顾新旧”的最小测试。没有非零优势则明文 NO_SIGNAL 停，不加样本填钟；旧保护失败或两轮无新增即停。 |
| 2，仅预算独立容纳时 | 与分支1首批相同的父/完整轨迹，最多2个 full-batch PPO-surrogate 步，固定优势与旧 logp；再做一对“无replay保持 / 加父策略forward-KL保持”候选，各同2步、无新Value；只选一个预注册KL权重。 | 第二步开始 clipping/保持能否改变保留—获得权衡。若首步不一致先判工程/定义差异；若二步后全任务无改善，结束该机制，不追加 epoch/权重网格。此分支是配对局部机制试验，不是独立重复或完整 CLEAR。 |

上述是算法数量上限，**不能替代**主协议的绝对截止、每项总物理步/命令、own/global容量、CPU/GPU、审计与远端备份预算。每个完整分支须在开始前容纳最坏采样+old8+field24+exact+审计+备份，做不到则 NOT_ADMITTED；不得让未完成的训练结果消耗验收保留量。与今晚任何已有注册预算重叠时从严，不自动追加。

## 120输入 GRU 的共用合同与判定

- 不变更部署输入/执行器映射/控制时长/原生终止；单 Actor 输出 drive/steer，无任务路由、地图、规划器、路点或运行时教师。训练 baseline 的独立 rollout 不是部署控制器。
- 用完整因果120维历史，按本 Actor 原训练的 hidden reset/carry 语义重算；若有截断BPTT，明记截断长度和 detach，不能称完整 recurrent policy gradient。replay 旧观测不能混入新策略反事实动作历史；但可在固定历史上计算当前条件分布作 off-policy/保持目标。AR探索须用实际条件密度（含前序均值依赖和合法变换），零logp占位与干预轨迹不作PG数据。
- 采样成功、监督标签拟合、旧历史KL很小、确定性原点整回合成功分别报告；保持按旧8及原field11的逐ID集合，新获得按先前失败ID的新增成功。通过研究面板也不代表 old275/six54/eight54/fresh16/78 已执行或未知场景泛化。
- 本次检索未找到能保证这个小样本固定 GRU 同时严格保留全部旧闭环并获得新导航的一手定理。现成组件已有名称；新颖性尚不成立。最需要的证据是受控更新后**同一Actor完整闭环的旧成功集合保留＋新成功集合扩大**。

访问限制：Williams 作者文献目录返回403，R2D2 OpenReview正文遇验证页；未声称读完这两份正文。REINFORCE 推导实际依据 SCST 全文，R2D2 仅依据作者摘要和已查看的 DeepMind learner burn-in 代码；未把第三方复现当作者实现。
