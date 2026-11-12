# 2026-11-12 夜间：动作模块消融与保持优先梯度的一手依据

2026-11-12，Stage 专用。采用 research 技能；仅写本文件，没有读取 Phase、启动仿真、执行优化器或修改实验源码/原始证据。协调者确认当前联网可用，已在注册前选择性核对以下一手论文及作者实现；夜间执行必须使用预先锁定的本地代码和资料，不依赖网络，不临场改算法。本文不是启动凭证、预算扩展或新颖性声明。

## 本地事实与问题

依据 [当前研究状态](../RESEARCH_STATUS.md)、[T509 最终报告](../review_20261112/T509_FINAL_REPORT.json)及 [T509 方法合同](../research_20261112_t509/PRIMARY_SOURCES.md)：接受父模型为 old8=8/8、field24=11 GOAL/9 CONTACT/4 TIMEOUT；T509 两固定任务各四次 sigma=.2 新探索共 8/8 安全 GOAL，八次 exact 不增加独立样本。第一块 16 次联合 BC 后，21 项 guard 为 17 GOAL/2 CONTACT/2 TIMEOUT，丢失原保护 ID `dev_task6_13`、`t502_field_sector0_offset1`、`t502_field_sector2_offset2`，已完整回退 Actor/Adam。新任务 sector1/offset1 成功，sector0/offset2 未成功；没有全 field24 最终结果。

这说明已有成功数据尚未转化为兼顾旧能力的确定性策略；它不证明数据不足、梯度冲突是唯一原因、两个动作模块中任何一个必然有错，或任务不可达。保持损失起点约 1e-31，随后增大，也直接提示初始保持梯度几乎为零，不能指望第一步投影提供实质保护。

## 一手来源与已核对范围

| 来源 | 本次实际查看与支持内容 | 适配边界/反证 |
|---|---|---|
| Yu 等，**Gradient Surgery for Multi-Task Learning**，2020，[论文 §2.3–2.4、Algorithm 1](https://arxiv.org/html/2001.06782v2)；[作者 TensorFlow 源码](https://raw.githubusercontent.com/tianheyu927/PCGrad/master/PCGrad_tf.py) | 原 PCGrad 对冲突的逐任务原始梯度作投影，再交给 SGD/Adam；作者源码的 `compute_gradients` 投影后由被包装优化器执行。查看两任务凸情形的 Theorem 1 及单步比较的条件。 | 原法对各任务作对称处理；本轮只修改 repair、保持 keep 的优先版本是适配。原定理不提供成熟 Adam、GRU、非凸 BC、逐 ID 导航或安全保证。 |
| Liu 等，**Conflict-Averse Gradient Descent**，NeurIPS 2021，[正式全文 §2–3、式(2)–(3)、Theorem 3.2](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf)；[作者仓库](https://github.com/Cranial-XIX/CAGrad) | 在平均梯度附近优化最差局部改善；式(2)明确使用小步 Taylor 近似。定理需 Lipschitz 梯度、固定合适步长等条件，非凸结论是平均损失的驻点性质。仓库仅查看首页/README，未移植或审计代码。 | 平均损失下降不等于逐任务保持；不能将摘要中的 minimum 解读为任意神经网络的全局最优或闭环无碰撞。本夜不再加入第四个 CAGrad 臂。 |
| Kingma、Ba，**Adam**，[原文 Algorithm 1、§2](https://arxiv.org/pdf/1412.6980) | 实际位移由累积一阶矩、二阶矩、步数和当前梯度共同决定，坐标缩放不同。 | 原始梯度去冲突后，成熟 Adam 的实际位移仍可能增大 keep 损失；重置 Adam 会改变本次比较的问题。 |
| Rolnick 等，**CLEAR**，NeurIPS 2019，[正式全文 §3](https://papers.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf) | 以旧经验回放、历史策略 KL/Value 函数保持减轻遗忘，并用 V-trace 处理回放的分布差异。 | 支持保留真实历史与函数目标；本轮没有其 Value/V-trace/在线学习合同，不能称完整 CLEAR 或继承实验性能。有限历史保持不是全部旧能力证书。 |
| Ross、Gordon、Bagnell，**DAgger**，AISTATS 2011，[原文引言、§3、Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)；Pfrommer 等，**TaSIL**，NeurIPS 2022，[原文 §2–3](https://papers.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf) | 动作会改变后续状态分布；TaSIL 的正面闭环结论依赖专家增量稳定性及相应邻域/导数误差条件。 | 均方误差很小、冻结骨干、固定旧历史输出近似不变，都不能直接推出新闭环轨迹或碰撞结果相同。本轮无专家查询或这些稳定性证明。 |
| Oh 等，**Self-Imitation Learning**，ICML 2018，[原文 §3、式(1)–(3)、Algorithm 1](https://proceedings.mlr.press/v80/oh18b/oh18b.pdf) | 原 SIL 用 `(R−V)+` 加权历史动作对数概率并更新 Value。 | 八条成功回合提供可用行为标签；普通成功 BC 没有逐动作回报优势，不能把整回合成功当成每个噪声动作的正因果贡献，或称原 SIL/新策略梯度。 |

没有声称阅读全文、穷尽最新文献或完成新颖性检索。以上各来源的实验数字不直接迁移至 Stage。

## 三个固定臂的机制含义

三个臂分别从同一接受父模型与完整成熟 Adam（drive step1025 / steer step385，lr=1e-4）的独立副本开始；父模型和父优化器不可变。数据完全相同：T509 已验证八条自身成功与原 old8+field11 的 19 条自身保持历史，不增加探索、seed、噪声幅度、参数或教师。最终夜间设计为每臂仅一个提案、固定 32 次监督更新，无第二块或重试；原骨干冻结、已有 drive/steer 两残差总计 11,906 参数的结构不变。

1. **只学习 drive，冻结 steer。** 检验仅改变速度动作是否足以把已有噪声成功整合为新的确定性成功。冻结模块的参数、梯度通路与 Adam 状态均须真正不更新；仅把梯度置零仍可能因旧动量发生位移。同一旧历史上的 steer 函数不变，不等于新闭环中的 steer 动作序列不变。
2. **只学习 steer，冻结 drive。** 反向检验转向适配是否足够；同样保留完整成熟 Adam，冻结模块不得执行动量更新。两臂可定位“哪种受限适配在本数据/预算有效”，不能单独证明失败的唯一物理原因；有效参数子空间与优化难度不同。
3. **联合学习，keep 优先的单侧 PCGrad 适配。** 保持 T509 的数据、分组权重、损失与超参；仅改变梯度合成。令 `gR`、`gK` 已包含注册的 repair/keep 权重，若 `gR·gK<0` 且 `||gK||²` 超过预注册数值阈值，则 `gR*=gR−(gR·gK/||gK||²)gK`，否则 `gR*=gR`；交给原成熟 Adam 的梯度为 `gK+gR*`。keep 梯度不投影，不随机排列，不在夜间换成对称 PCGrad/CAGrad。零/近零保持梯度直接跳过投影，不归一化放大噪声。

对第三臂，原始梯度代数满足 `gK·(gK+gR*)≥0`；这是欧氏负梯度小步的局部代理性质。必须记录实际 `Δθ=θ_after−θ_before` 的 `gK·Δθ`、逐组 keep 损失前后值、梯度范数/夹角、投影是否发生与冻结模块参数/Adam 哈希。`gK·Δθ>0` 说明连一阶 keep 代理方向也未被实际 Adam 保住；`≤0` 仍不能保证有限步损失、每个历史/任务或闭环安全。该记录是诊断，不是中途追加回退系数/换优化器的开关。

## 接受合同与解释边界

每臂唯一提案完成后执行锁定的 21 项原生闭环 guard +21 exact。必须逐 ID 保留最初的 19 个成功；总 GOAL 数相同不能抵消成功集合替换。丢保护 ID 即按主协议回退完整 Actor/Adam 并结束该臂，不因平均监督误差下降追加更新。保护通过且至少增加一个确定性成功，才检查余下 11 个 field 任务与全 field24 的 24 exact；最终仍要求原 field11 全保、old8 全安全、field GOAL>11、CONTACT≤9。新增成功与最终完整集合须记录；下一臂仍从共同父模型开始。完整 old275/six54/eight54、fresh/reload 未安排就保持未测。

成功噪声动作的回归可能形成未实际执行过的折中控制；这是本地待检验机制，不能由 T509 失守直接认定。执行动作标签与裁剪边界语义必须保持原注册约定。三个臂不更改奖励、折扣、终止或 GOAL 半径；成功 BC 也没有识别哪一段动作导致固定奖励的 credit assignment。因此结果只能支持/否定本次有限子空间与梯度处理方案，不能称已解决奖励归因、泛化、全场地稳定或算法原创。

所有臂的次序、停止/选择规则、输入与代码 SHA、墙钟/物理/存储上限由主协议预先锁定；代码在无人值守前全部实现并预检。三臂顺序执行，CPU14/15 合计一核、GPU 禁用；root52MiB、child28MiB 的实际完整最坏容量须逐臂准入，空间不足允许跳过并记未执行。离线队列至 2026-11-13 09:00 强制停止；失败、拒绝和未执行证据保留，不加填充任务、加机制、加轮数、重复提案或续旧 T509。
