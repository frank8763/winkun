# T393：保持五米基线函数的循环残差初始化与学习预算

2026-11-06，一手资料定向核查；只写本文，0 仿真、0 优化器、未写 runs。用户已选 **safe T255 五米基线初始化，先学习新 0.70 m 缺口／80 s 六米任务，完整 old275 放在学习后**。这是新协议的依据，不是实验启动／通过收据。不重做 [T392_PRIMARY_METHODS.md](T392_PRIMARY_METHODS.md) 的 CAPG、PPO、逆课程、DAPG 广泛检索。

## 1. 采用零输出残差，但“安全”只指已保留基线及初始化关系

Silver 等 *Residual Policy Learning* §IV-A 明确把残差最后一层置零，使初始策略等于原策略；§IV-B 同时指出，后续不良 Critic 更新仍可能破坏原策略。§IV-C 实际采用当前／上一状态的短历史近似，并非完整 GRU/BPTT 验证。可借用零残差思想，不能声称本文方法复现其 DDPG+HER，或获得了训练不退化定理。[论文 §IV–V-C](https://arxiv.org/pdf/1812.06298)

作者仓库 `ab330ee425463459f1dd7daa4b571104ae449118` 的 `nn_last_zero` 仅最后层 kernel 为零，前层 Xavier；`actor_critic_last_zeros.py` 用该函数生成残差。本文仅阅读，不复制／执行代码；这里不是所有层全零初始化。[utils.py L25–43](https://github.com/k-r-allen/residual-policy-learning/blob/ab330ee425463459f1dd7daa4b571104ae449118/tensorflow/experiment/models/utils.py#L25-L43)、[Actor L31–35](https://github.com/k-r-allen/residual-policy-learning/blob/ab330ee425463459f1dd7daa4b571104ae449118/tensorflow/experiment/models/actor_critic_last_zeros.py#L31-L35)

**项目代数推导，非论文对本项目的保证：** 保留 T255 原推理路径为冻结 `a0(Ht)`，新增因果 `ht=GRU_phi(xt,h[t-1])`、`r=W ht+b`。若最终命令为 `float32(clip(a0+r,-1,1))`，初始 `W=b=0` 且 hidden 有限，则确定性命令应等于 T255。新分支可携带历史；底座仍保持 T255 的原 hidden-reset、输入归一化、传感／previous-action、数值路径和 clip/cast 顺序。若直接让 T255 原 GRU 携带 hidden，即便权重相同也不属于此函数保持关系。两分支是同一 Actor 内部计算，不引入教师选择、规划器或路点。

具体采用建议：

- 只零新输出 `W,b`，保留非零循环特征；底座参数排除在 optimizer 外并核对 SHA。`r=W h+b` 在 `W=0` 时，第一步梯度可到输出头，但进入 GRU/body 的任务梯度为零；输出头改变后再检查循环梯度。不能要求所有循环张量在第一个 backward 就非零，也不能把这个暂时的零梯度误判为永久断路。
- Gaussian location 域须单独锁定。用 T255 **实际 clipped 命令**作 `mu0`，可保持零残差的确定性命令，但不保持旧 preclip 函数；用旧 preclip 作 `mu0` 也可能确定性相同，却在其远超边界时继续让随机动作饱和。RPL 不替本项目解决该 CAPG 参数化选择。按 [T392 §1](T392_PRIMARY_METHODS.md) 报真实执行分位数、边界质量和反向命令频率。
- 零残差只保持无噪声初始读出；`sigma>0` 后的采集不是安全基线原行为。更新后亦无 old275 保证。零物理的完整历史逐步／序列、真实 dtype 和序列化读回等价证据可支持减少重复初始化评估，不能代替最终原生验收，也不能把有限历史检查称全状态形式证明。

## 2. 示范辅助用于支撑真实 on-policy 学习

DAPG §IV-C 将新策略轨迹的 PG 与示范 score 辅助结合，并明确示范优势需要额外 rollout／假设，论文用衰减启发式权重。作者 `mjrl` 的 `dapg.py` L49–107 则使用标准化 on-policy 优势、常数衰减 `demo_adv` 和 NPG。它支持“示范继续参与 RL”，不保证某个固定 MSE 权重有效。[论文 §IV-C，PDF 第6页](https://arxiv.org/pdf/1709.10087)、[作者实现，commit 3871d93763d3b49c4741e6daeaebbc605fe140dc](https://github.com/aravindr93/mjrl/blob/3871d93763d3b49c4741e6daeaebbc605fe140dc/mjrl/algos/dapg.py#L49-L107)

采用 T392 已核查的 mask／完整 history 语义：固定成功前缀是训练起态课程，前缀真物理全部计费且保留原 clock／执行器／历史；仅学生随机后缀进入 CAPG/PPO 比例，当前参数从 episode zero 重建 hidden，后缀梯度可经前缀观测 BPTT。示范 MSE 是独立辅助，不伪造示范 on-policy 优势；截断分布的边界 NLL 不等于 MSE。固定 `.1*MSE`、LOO-MC、PPO 与循环残差的组合只能称工程 pilot。

负面边界：T391 已显示监督损失下降而闭环接触；示范和前缀缩短 credit horizon 不保证原起点能力。50 s 前缀在正式 80 s 任务仅余 30 s，近终点成功单独列为条件起态成绩；只有 0 s 自主全程能报告主任务到达。[T391_FINDINGS.md](T391_FINDINGS.md)、[T392_PRIMARY_METHODS.md §2–4](T392_PRIMARY_METHODS.md)

## 3. 先学习、后完整回归是预算分配选择

作者 `mjrl/utils/train_agent.py` 先以 `N=num_traj/num_samples` 调 `train_step`，再可选调用 `evaluation_rollouts` 的 `eval_mode=True` 采集，两者明确分开；这不是“所有真实交互都免费”或“无需最终评估”。Henderson 等强调策略优化视角须用同一个冻结目标策略跑完整任务轨迹，不能用跨更新拼接的局部片段代替。[作者 train_agent L95–108](https://github.com/aravindr93/mjrl/blob/3871d93763d3b49c4741e6daeaebbc605fe140dc/mjrl/utils/train_agent.py#L95-L108)、[论文 Reporting Evaluation Metrics；附录 Experimental Setup](https://arxiv.org/pdf/1709.06560)

**项目采用推断：** 学习配额、最终完整验收配额和总硬限预先分开；初始化只做函数／状态合同与预注册的少量原生抽检，已有 T255 历史结果注明复用，不能冒称本轮重测 275/275。保留独立的最终 old275、主任务开发／留出／模型原生读回时间；开始一个新学习 batch 前须同时满足其最坏开销和最终验收 reserve，额度不足就停止，不把行政删失赋成假任务 timeout。全部真实前缀、探索、重生、回归均计物理和墙钟，另报**真正学生新随机后缀**，不能以总物理步充学习量。论文没有推出固定 old275-before-RL 要求，也没有确定本项目该分配多少秒；数值由正式 T393 预算锁定。

## 4. 必须保留的历史反证与差异

| 历史 | 已有证据及 T393 的可检验差异 |
|---|---|
| T336 | 最终六米仅历史 RIGHT18 的 1/54；reverse 首4成功在该臂 RL 前已有。饱和前沿采样与 mode 差中位约 `1.19e-7`。T393 不能只更换 residual 名称，须验证实际动作域探索；本轮也不是该旧预算续跑。[原 posthoc](../overnight_20261103/T336_POSTHOC_ACTION_SUPPORT.json)、[研究状态 T336 条目](../RESEARCH_STATUS.md) |
| T351 | T349 循环学生初始化已 old275=240/275、old20=9/20；在 wide_dev2/4 **原起点**各4条探索，不是已成功末段前缀课程。8 live 更新后最终 old20=9/20、wide5=2/5、原窄0/2。[计划](../review_20261104/T351_PLAN.md)、[结果](../review_20261104/T351_FINDINGS.md) |
| T353 | 同 T349 初始化比较 iid／AR，并共同修改成功保持 target；各臂最终 wide5=5/5、原窄0/2，完整 old275／six54 未执行。该结果不证明成功前缀课程无效，也不支持 AR 必优。T393 同时更换安全初始化、任务版本／课程等，不能归因单个变量。[执行协议](../overnight_20261104/LOCAL_EXECUTION_SPEC.md)、[原结果 T353段](../overnight_20261104/FINAL_REPORT.md) |
| T392 | 使用 T391 学生初始化；用户在 `initial_old275` 阶段取消，取消快照 rounds=0、live=0、shadow=0。它是**零实际 RL**，不能算 CAPG/PPO 或示范辅助学习的失败。快照物理／命令不是取消后最终总量；不在此猜测。[取消收据](T392_CANCEL_REQUEST.json)、[原计划](T392_PLAN.md) |

safe T255 自身只在已有五米局部任务域得到验证；新任务障碍／时限改变不算算法收益。学习后旧回归缺口需如实报告且不替换 safe T255；标准组件已有先例，本笔记不作原创、普遍安全或六米已学会主张。

## 5. 实际阅读范围与限制

- 本地：仓库 AGENTS、迁移 context／progress／cleanup／MIGRATION、实验流程；当前研究状态的新近相关条目；上述 T392 计划／方法／取消、T351 计划／结果、T353 协议及结果段、T336 posthoc／状态、T391 结果。未逐项重新审计这些实验原始 runs，未读取 Phase 或全部 ChatGPT 历史。
- 新外部核查：RPL PDF §IV-A/B/C、§V-C 及作者三个网络源文件（另定向查看 residual/base optimizer 相关片段）；DAPG PDF §IV-C、完整 `dapg.py`、完整 `train_agent.py`；Henderson PDF 的 Reporting Evaluation Metrics 和附录 Experimental Setup。链接可访问，未执行作者工程。RPL 与 mjrl 源 commit 已锁定如上；不把仓库文件名里的 freeze 当作底座确实冻结的证据，项目以 optimizer 参数集合和前后 hash 为准。
- 另外只看了 Net2Net 官方摘要和 Johannink 等 residual RL 搜索摘要，未作为本结论依据；没有声称读完这些全文、穷尽当前文献，或完成新颖性调查。正式 T393 的实现、真实预检、预算和最终结果由独立协议／收据决定。
