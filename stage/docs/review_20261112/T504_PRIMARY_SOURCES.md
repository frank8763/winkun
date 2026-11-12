# T504 一手文献：探索成功如何转为确定性闭环到达

日期：2026-11-12。范围：Stage；本研究记录工作零模拟器步进、零优化器更新、未启动任何实验。以下建议不是已获效果验证的 T504 结果；末节另记录主代理随后选定的有界设计，最终注册/启动以主协议与凭证为准。

## 读取范围与本地事实

读取工作区及仓库规则、迁移/上下文记录、RESEARCH_STATUS 最新段落、T503_PLAN 全文，以及 T503_FINAL_REPORT 的 training、update、field24、old8、成功 ID、模型和计账字段。对八条 samples 仅读取标量终止/任务/来源摘要，未展开巨大历史数组，未加载模型或 NPZ；没有重新审计 T503 科学结果。未读取 Phase 内容，未读取全部原始 Stage 聊天。

T503 的本地报告显示：8 条自身探索为 6 GOAL、0 接触、2 超时；sector5_offset1 为 2/4 GOAL、2 超时，sector2_offset1 为 4/4 GOAL。16 次 Actor Adam、0 Value Adam 后，确定性 field24 从 T502 的 5 GOAL/11 接触/8 超时变为 6/10/8，新增 sector2_offset1，原五成功 ID 未丢，old8 为 8/8。sector5_offset1 不在最终成功 ID 中，因此探索中的两次 GOAL 尚未变成该任务的确定性 GOAL。完整 old275/six54/eight54 与 fresh16 未执行。来源：[T503_PLAN](T503_PLAN.md)、[T503_FINAL_REPORT](T503_FINAL_REPORT.json)、[RESEARCH_STATUS](../RESEARCH_STATUS.md)。

不能从这些摘要认定“只有均值抵消这一根因”。对 sector5 的两条成功路线是否同模态、何时分离、成功动作与确定性动作的真实差异，此次没有读取完整轨迹证明。

## 原始论文实际支持什么

### SIL：超过价值估计才模仿，不是“成功必学”

Oh、Guo、Singh、Lee，ICML 2018，[Self-Imitation Learning 原文](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf)。精读范围：引言的反例、§3/Eq.1–3、Algorithm 1 与 replay 说明；非全篇逐页审读。

其 Actor 项为 `−log πθ(a|s) · max(R−Vθ(s),0)`，并有正部价值损失；存储自己的既往回合，配合在线 Actor-Critic 与 replay。好坏由 return 相对 V 决定，不由原生 GOAL 标志直接决定；成功也可能零权重，失败的有利片段也可能正权重。论文确实报告 PPO+SIL 的 MuJoCo 改善，但也明确提示局部诱惑奖励可能让反复利用有害。它不证明一次监督更新能把随机成功变为确定性到达。作者的 [TensorFlow 仓库 README](https://github.com/junhyukoh/self-imitation-learning) 核实了 A2C+SIL/PPO+SIL 入口；本次未审计实现内部、未运行作者代码。

本项目推论：Value 冻结且未经此任务验证时，不宜为了使用 SIL 名称引入它的 `(R−V)+` 筛选。直接按自身原生成功标志作正权重似然更新，应准确称“成功条件自身经验似然/监督整合”，或 SIL-inspired；不是原论文完整复现。

### AWR：正权重回归与 signed PPO 的关键差异

Peng、Kumar、Zhang、Levine，2019，[Advantage-Weighted Regression 原文](https://xbpeng.github.io/projects/AWR/AWR_2019.pdf)，[作者项目页](https://xbpeng.github.io/projects/AWR/index.html)。精读范围：§3/Algorithm 1、§3.1 约束近似、实践权重裁剪段、§5.2 baseline/replay 反证与附录 C；非全文审读。

AWR 先回归 replay 的 return 得到状态价值，再最大化 `E[log πθ(a|s) · exp((R−V(s))/β)]`。即使 advantage 为负，权重仍为正；它是降低模仿权重，不是 signed PG 的反向推离。实践中裁剪过大权重。其理论动机使用行为状态分布近似，依赖策略接近；消融报告移除 baseline 或只用较小在线数据会退化，不能据此预言六条成功的小批次必有效。作者 [仓库 README](https://github.com/xbpeng/awr) 指向 `learning/awr_agent.py::_update()`，本次只核对 README、未审计该函数。

本项目推论：T503 的“同 task、同物理时间的其他三个回合”不是当前完整 history 的 `V(h)`；同时间的车位、航向、速度、GRU 状态可不同。把同一个 peer advantage 放入指数只是 AWR-inspired，不应称原 AWR。尤其 sector2 四条全成功，彼此 return 差距小；相对优势可弱于直接“成功动作要可复现”的信号，但 T503 已让该任务确定性成功，不能把相对基线概括成完全无用。

### 多模态成功标签与闭环分布转移是两个独立风险

Lynch 等，[Learning Latent Plans from Play](https://proceedings.mlr.press/v100/lynch20a/lynch20a.pdf)，CoRL 2019/PMLR 2020。精读 §3.1–3.2 及引言覆盖范围：同一当前状态/目标可有多种有效高层行为，产生互相冲突的动作轨迹标签；作者为此引入 latent plan。它是任务不同的原始实证/方法动机，不是 Stage 存在多模态的证据。本任务没有因此自动采用 latent plan、外部规划器或运行时路由。

Ross、Gordon、Bagnell，[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)，AISTATS 2011。读取摘要与引言的诱导分布/误差累积论证，未审阅全文证明：动作影响未来观测，所以在示范分布上预测准，不等于在学生自己访问的状态分布上闭环成功。本文的 DAgger 需要相应专家查询/假设；本项目的自身成功回放不满足这些条件，不能继承其保证。

本项目推论：固定方差单峰 Gaussian 的加权最大似然，在未裁剪的普通位置参数情形等价于加权平方误差，最优位置是标签条件均值；两条绕障成功模式的中间动作未必成功。这是数学例子，不是当前动作分布合同的直接等价证明：Stage 有相关噪声、边界 atom、真实 carry，不能把实际 loss 偷换成普通 MSE。即使两个标签属于同一模式，微小早期误差也可把学生带离成功状态分布。二者都解释了为何“似然提高”不足以验收。

## 对下一有界试验的具体建议

**优先检验成功条件似然整合，保留 T503 signed peer-PPO 为明确对照；暂不增加未经验证的 Value 或多模态架构。** 这是对当前已有成功覆盖的低变量检验，不是算法创新或效果保证。

1. **先锁同一个起点和数据。** 最清楚的对照是两候选均从 T503 的同一 `rl_initial` Actor/成熟 Adam 出发，复用完全相同八条已核验自身轨迹：原 T503 signed peer-PPO 为对照，另一候选仅变 Actor 数据项。保留 T503 已接受模型及 T501 最佳原件。若改从 T503 最终模型开始，应明确这是另一项研究，旧数据是旧行为数据，不能把它称为新 on-policy 采集或直接借用 T503 原对照。
2. **推荐的候选数据项：** 对六条原生安全 GOAL 完整回合最大化原动作条件 log-likelihood；task 等权、task 内成功回合等权、回合内按实际原生持续时间归一，预先固定。示意 `Lsucc = −mean_task mean_success_episode Σt Δt log pθ(a_t,atom_t | causal_history_t) / Σt Δt`。超时不作成功标签、不删除原始失败。该式是工程成功条件 BC，既不是原 SIL，也不是原 AWR；它故意检验正向成功信号，不能声称无偏原奖励策略梯度。只有两个任务/六条成功，不能声称覆盖全场地。
3. **必须沿用真实密度与循环状态合同。** 用当前参数重算因果完整 history 和真实 GRU carry，保持原 AR 噪声、raw/action 变换、边界 atom 与 logp 语义；不跨轨迹拼接 hidden，不把未来状态输入 Actor，不将含噪动作改成手工平滑动作。明确训练的随机条件似然与部署的零噪声 location 是不同对象。若实际选择普通 raw-action MSE，应单独说明目标改变，而不是冒称等价。
4. **保持其他条件一致并限制一次候选。** 固定原成功保持组、权重、11906 可训参数、Value 冻结、成熟 Adam、最大更新数和 KL 止学规则；不因末段失败追加 seed/温度/步长/轮数搜索。对照和候选计入相同总预算；独立 shadow/reconstruction 与完整失败保留沿用仓库协议。此建议不授予运行预算，主协议必须另锁物理/墙钟/容量/唯一 writer 边界。
5. **先做零交互可解释诊断，再看闭环。** 分任务/回合报告实际成功权重、有效样本量、裁剪 atom 占比、每条轨迹 NLL 的变化和成功保持误差；不要只给全局平均 loss。对 sector5 两条成功查完整 history 上 drive/steer 标签是否相互冲突，及参数变化在确定性原轨迹对应历史上的方向；相似时刻不等于相同状态，不从图形相关直接宣布因果。日志应分开记录“在成功 history 上拟合”和“在学生 history 上闭环”。
6. **验收与止步条件：** 一次最终冻结 field24，明确报告 sector5 是否确定性原生安全 GOAL、原六成功 ID 是否保留、接触数与 old8；与候选相配 exact replay 仍不算新独立成功样本。候选不改善或丢失保护项则整 Actor/Adam 回退，保留失败，不续同机制预算。未做完整 old275/six54/eight54 就保持 null；新增未见起终点只能另立预先锁定的泛化验收。

如果未来确实测试 AWR，则先另行验证 history-conditioned V 的校准/覆盖并预定 β、权重上限及 ESS，再对同数据检验；不要在本次同时改 Value、成功筛选、奖励和策略结构。若改用 `exp(peer_adv/β)`，应标注 peer-weighted likelihood，并单列失败也有正权重的事实。

**为何仍不保证：** 成功筛选带来选择偏差；有限自身轨迹可能不包含可稳定部署的单一反馈模式；随机成功可依赖一整段一致的噪声，而去噪 location 不会自动复现；循环策略重算后的 hidden 与轨迹原 hidden 会变化；软保持/KL 约束只覆盖采样 history；真正学生访问分布、原生接触与 0.2m GOAL 边界仍须确定性闭环检验。以上是本项目待检验推论，论文没有为当前 Stage 合同提供成功保证。

## 主代理选定设计的文献边界（输入锁定前补记）

主代理随后告知选择：从 T503 已接受最终模型初始化，限定尚未确定性成功的 sector5_offset1 两条既有自身 GOAL；不新采训练回合、不新增策略梯度。对原 censored-action likelihood 使用一半全序列均值、一半真实距离≤1m 的末段均值，64 次监督 Adam；显式 LR=1e-4，保留成熟 optimizer 计数 912/272 与 moments。固定 backbone，使用真实因果 history features。软保持参考当前 T503，在旧379条历史成功和当前6条场地成功的385个 history 上约束；旧379不冒充由当前 T503 全部重验。一次 full24+24exact 与 old8；须保原6成功、总GOAL>6、contact≤10，否则整 Actor/Adam 回退。此段是收到的设计说明，本研究子任务没有审计这些输入或计数。

该设计可准确称**有界、末段加权的自身成功轨迹监督整合**：它是旧行为数据对新初始化的 off-policy likelihood 学习，既无 SIL 的 `(R−V)+`/价值更新，也无 AWR 的拟合 V 与指数优势权重；不是原 SIL/AWR 复现、不是新增 on-policy PPO、不是原创性证据。与 T503 的目标、初始化和更新次数也不同，因此只能作后续工程 pilot，不能把结果差异归因于“成功似然优于 peer-PPO”的受控因果结论。末段权重是预先声明的工程偏好；distance 仅作训练样本权重，不能增加未来信息到 Actor 输入。

两条成功仍可能含冲突动作或一致噪声依赖；应逐轨迹报告全程/末段 NLL、实际末段样本数量/持续时间、边界 atom、保持误差，并确认两条均有非空≤1m片段。若≤1m集合为空须停止并保留错误，不能临时改变距离阈值。完整序列项有助避免只学末段而遗漏进场行为，但不提供闭环保证；旧历史上小保持误差也不证明旧379闭环全部保留。最终科学结论只由实际注册验收决定。
