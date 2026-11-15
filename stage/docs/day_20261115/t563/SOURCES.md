# T563 一手来源与适用边界：保存历史的尾部拟合及经验接受检查

检索日期：2026-11-15。Stage only；本任务只阅读文档和公开一手来源、写本文件，未执行训练、仿真、优化器、进程检查或模型修改。已读仓库 `AGENTS.md`、实验文献工作流及 [T562 结果](../t562/FINDINGS_AND_NEXT.md)。下述协议来自本次任务输入，不能据此称 T563 已注册、运行或通过。

## 项目问题与拟采用机制

T562 的因果 encoder＋GRU＋SIREN 在 27 条既有教师成功历史、6779 个保存决策上，27/27 满足命令 RMS≤.02，24/27 同时满足 max≤.1；剩下三条最大误差约 .109/.183/.163，共 20 个决策至少一维超 .1。它支持“平均拟合已改善、少数尾部误差仍阻止动作门”这一有限判断，不能证明碰撞原因或未见任务能力。

拟议试验沿用 T562 权重与同一 93186 个可训变量，fresh Adam，27 条完整历史 BPTT；目标为一半 mean location MSE、一半全局 top32 平方执行命令误差。固定最多 240 个提案，每提案固定回溯因子 1/.25/.0625；已达标的 24 条保持原 RMS/max 门，全部 27 条 RMS≤.02，且全局最大执行命令误差相对当前已接受模型至少下降 1e-6，才接受。否则 Actor 与 Adam 整体回退；连续 8 个拒绝或全部动作门通过即停。具体常数属于本项目有界工程选择，以下论文不提供这些常数的最优性或成功保证。

## 1. 尾部/难例拟合有明确已有近邻

- **Fan、Lyu、Ying、Hu，NeurIPS 2017，Learning with Average Top-k Loss。** 阅读正式论文定义、§2 与 Lemma 1。ATk 是最大的 k 个样本损失的平均，包含均值与最大值两个端点；其聚合形式支持在少数高误差上增加监督压力。凸性是对损失向量的性质；经非线性循环网络参数化后，不能直接宣称全局凸、收敛或所有动作门必达。采用“均值＋top32”是已知聚合思路的项目改写。[正式论文](https://proceedings.neurips.cc/paper_files/paper/2017/file/6c524f9d5d7027454a783c841250ba71-Paper.pdf)
- **Rockafellar、Uryasev，Optimization of Conditional Value-at-Risk，作者公开 1999-09-05 稿。** 阅读 CVaR 定义、辅助阈值形式与 Theorem 1–2。它提供尾部平均风险的经典优化表述。对均匀的有限损失集合，`top-k mean = min_tau[tau + sum(max(loss_i-tau,0))/k]`；这里的有限样本恒等式也直接见上篇 Lemma 1。因此可称“经验尾部损失/CVaR 形式的近邻”，不能称碰撞风险 CVaR、真实部署风险或分布外保证。原稿的连续分布论述不应不加说明套到有限离散历史。[作者稿](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)
- **Shrivastava、Gupta、Girshick，CVPR 2016，Online Hard Example Mining。** 阅读论文动机与方法、作者实现 `get_ohem_minibatch` 和 `select_hard_examples`：源码按当前 loss 降序，取规定数量难例，并可用 NMS 去重。它支持“重新识别当前难例再分配梯度”的已有机制；对象检测 RoI、SGD、空间去重与本项目完整序列 Adam/BPTT 不同，不能照搬其检测收益或称复现 OHEM。未复制或运行作者代码；读取的是未锁提交的 master 页面，只作概念核对。[论文](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)、[作者实现](https://github.com/abhi2610/ohem/blob/master/lib/roi_data_layer/minibatch.py#L132-L170)、[作者仓库说明](https://github.com/abhi2610/ohem)

**本项目推论和反例：** 降低 top32 平均值不等于降低最大值。例如两个尾部平方误差从 (.04,.04) 变为 (.05,.02)，平均下降而最大上升；更大的 k 同样可构造。因此全局 max 下降检查有独立作用。top-k 成员会随参数变化，排序交点可不光滑；需要固定并记录平局处理，不能把选中的旧 20 个决策永久当作全部难例。

**必须在协议/实现中明确的单位：** 若 top32 对两个命令维度展平，则 N=13558、经验尾部质量为 32/13558；若先按决策聚合则 N=6779，目标不同。20 个超标决策并不等于 20 个超标命令坐标。mean location MSE 的历史/位置权重也必须沿用明确定义。这些是可复现性条件，不是建议扫描两种目标。

## 2. 旧历史守门有遗忘研究近邻，但保护范围有限

**Lopez-Paz、Ranzato，NeurIPS 2017，Gradient Episodic Memory。** 阅读 §3 的过去任务记忆损失约束、其局部线性/记忆代表性假设；核对作者 `gem.py` 中旧梯度存储和 QP 投影函数。GEM 为减少遗忘而约束记忆样本上的损失变化。T563 不是 GEM：它没有复现其投影，而是在有限候选更新后，重新前向计算全部既有历史的实际非线性指标再接受/拒绝。该区别允许经验检查不依赖梯度内积近似，但不能推出未保存状态上的保持。[论文](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)、[作者实现](https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py)

**本项目边界：** 24 条保护的是原有 RMS≤.02/max≤.1 的合格状态，不代表每条 RMS/max 数值都不退化；全部 27 条 RMS 门也不限制每一坐标变化。重复用于 loss、接受和停止的 27 条都是训练/开发数据，不是独立测试。完整历史前向/BPTT仍使用保存的观测与既定重置；学生自主动作改变后续观测的闭环结果必须另外验收。

这一分布偏移限制有直接一手依据：**Ross、Gordon、Bagnell，AISTATS 2011，DAgger 论文**指出，序列预测的动作影响后续输入，普通独立同分布监督假设会失效。这里只采用该反证边界，不采用其在线教师查询或运行时教师混合机制，也不称 T563 为 DAgger。[正式论文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)

## 3. 接受/回溯及 Adam 回退属于受限经验优化

**Achiam 等，ICML 2017，Constrained Policy Optimization。** 阅读 CMDP/策略散度约束与理论保证的范围。CPO 的保证针对其规定的期望回报/成本及散度界；T563 是固定历史的监督损失、RMS/max 数值接受检查，未建立 CMDP 成本、占用分布界或其理论前提。因此既不能称 CPO 复现，也不能借用“安全策略优化”“全局安全证书”或导航安全保证。[正式论文](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)

**Kingma、Ba，Adam，ICLR 2015 / arXiv v9（2017）。** 阅读 Algorithm 1：更新依赖一阶矩、二阶矩与步数及偏差修正。由此推得，若拒绝候选只恢复 Actor 权重而保留已前进的 Adam 状态，下一提案不是从同一训练状态重试。完整 Actor＋Adam 回退是本协议的事务语义要求；论文没有提出本项目这套接受策略，也不保证有限三档回溯总能找到可行更新。[论文及 Algorithm 1](https://arxiv.org/pdf/1412.6980)

**本项目推论：** fresh Adam 表示从 T562 权重启动一个新优化阶段，不是恢复 T562 的完整 Adam 续训。每个回溯因子须相对同一已接受状态解释，不能无意累计三次 optimizer step；拒绝、异常和中断的恢复范围需一致。通过守门最多保证已接受检查点在这组保存历史、固定计算约定下满足阈值及 max 的声明下降；8 连拒或 240 提案到限只说明有界搜索停止，不是不可拟合证明。全动作门通过后的自主评估仍是下一道必要证据。

## 4. 执行裁剪与饱和导数需分别说明

核对 **PyTorch v2.5.1 固定标签源码 `clamp_backward`**：对输入在裁剪区间外的部分回传零梯度，在边界选定子梯度。这是公开实现语义参照，不是已检查本机训练依赖版本。[固定版本源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/autograd/FunctionsManual.cpp#L1100-L1143)

**由该语义的数学推论：** 若预测执行命令为 `a=clip(z,l,u)`，损失 `(a-y)^2` 对 z 的直接导数在严格区间外为零；即使执行命令有误差，单纯增大 top-k 权重也未必能从该分支把 z 拉回。恰在边界的导数不可由数学唯一确定，必须记录实际实现。若只裁剪固定教师标签 y、预测 z 没经过 clip，则标签裁剪本身不使预测导数为零；不能把两者混为一谈。饱和后相同执行命令也不代表 preclip 输出相同。应报告真实执行域误差，并明确优化导数实际经过什么裁剪路径；本说明不授权更换导数、直通估计或扫描新目标。

## 可支持的结论

该方向可被准确命名为“同容量模型上，以全历史监督尾部损失提出候选、用保存历史经验门接受并回退 Actor/Adam 的有界拟合试验”。已有 top-k、难例与遗忘约束近邻，不能据组合称原创。通过只能先报告 27 条保存历史的动作门，不能报告新自主 GOAL、恢复能力、六/八米通过、泛化或安全证书；失败也不能推为网络理论不可学或场景无解。

本轮查看上述论文的列明部分和两个作者实现片段，未穷尽全网、未声称读完所有引用或原始聊天。正式协议、输入/源码 SHA、预算与独立审计仍须由主任务锁定。
