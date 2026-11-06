# 多车并行采样：先保留 Stage 物理，再测吞吐

2026-11-06。Stage only；只读文献与现有协议形成的方案，不是已启动实验。T399 已注册、零原生/零优化器预检 PASS，尚无 launch；本说明不修改或并发执行 T399。内部依据：[研究状态](../RESEARCH_STATUS.md)、[T399 协议](T399_PLAN.md)、[GPU 与算法谱系核验](ALGORITHM_LINEAGE_GPU_AND_NEXT.md)。

## 结论与术语

可以让许多“克隆车”同时积累经验，训练同一个策略。每个车实例有独立的世界状态、任务、随机流、历史与 recurrent hidden；采样后汇成一个训练批次。这是**多环境并行采样**，不是多机器人协同，也不是把很多车塞进一个相互碰撞的世界。复制运行状态不需要复制仓库、模型文件或每车完整视频。

Gymnasium 是环境接口/向量化工具，MuJoCo 是物理引擎，两者能组合使用；换成 Gymnasium 并不自动替换或加速物理。官方 `AsyncVectorEnv` 用多个 CPU 进程和进程通信并行执行环境，可共享观测内存；它不自动把原生 MuJoCo 变成 GPU 仿真。[Gymnasium 官方 API](https://gymnasium.farama.org/api/vector/async_vector_env/)

MuJoCo 将模型结构 `mjModel` 与可变运行数据 `mjData` 分开，官方支持共享只读模型、每线程独立数据的并行计算。Stage 若先用进程隔离，可各自初始化同一模型，优先避免现有全局变量、碰撞记录与渲染上下文串扰；这不要求重造世界或磁盘复制。[MuJoCo 官方仿真与多线程说明](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#multi-threading)

## 这台机器适合先做什么

主任务本轮只读核验的硬件/软件是 Ryzen 9 8945HX、32 逻辑 CPU、32 GB RAM（约 23 GB 可用），RTX 5060 Laptop 8151 MiB；Torch 2.11/cu128 实际 CUDA 运算 PASS，MuJoCo 3.9、Gymnasium 已安装。JAX、MJX、Warp、Isaac Lab、Isaac Sim 未安装。这说明具备 CPU 并行和后续 GPU 试验条件，不能推出支持多少个环境或已有 GPU 仿真加速。

| 路线 | 能解决的问题 | 对当前 Stage 的判断 |
| --- | --- | --- |
| 现有原生 MuJoCo，1/2/4 个独立 CPU 环境 | 同时采集多个轨迹，保留当前物理与任务合同 | **第一选择**：安装和迁移成本最低，先测瓶颈与等价性 |
| CPU 环境＋GPU 批量学习/推理 | 神经网络计算确实占时才有收益 | 第二步：先测传输、批处理与数值/梯度合同；当前小网络逐步调用未必获益 |
| MJX-Warp / MuJoCo Warp | NVIDIA GPU 上批量物理和后续批量像素观测 | 后续独立兼容性 pilot；不把新后端当原历史轨迹精确续跑 |
| Isaac Lab | 另一套完整机器人学习框架、GPU 后端和工具 | 目前迁移工作超出先测并行吞吐的需要；暂不安装 |

MJX 当前官方文档区分 MJX-JAX 和面向 NVIDIA GPU 的 MJX-Warp，并列出接触、传感器、ray 等功能差异；MJX-Warp 有批量 RGB/depth 渲染，但不提供物理自动微分。Stage 无须可微物理来做普通策略梯度。现有 MJCF 可以成为迁移起点，却仍须核对执行器、碰撞/接触、ray 观测、积分器、求解器和 reset，不能把另一后端默认当逐位一致。[MuJoCo MJX 官方文档：实现、批量渲染与功能表](https://mujoco.readthedocs.io/en/stable/mjx.html)

NVIDIA 已将旧 Isaac Gym 标记为 deprecated、停止支持，不建议为了本任务新建旧版依赖。[NVIDIA 官方状态](https://developer.nvidia.com/isaac-gym) 但也不能据此把今天的 Isaac Lab 一概说成“必须装 Omniverse/Isaac Sim、至少 16 GB 显存”：**当前 `main` 安装文档默认 Newton/Warp，可不装 Isaac Sim**；其内存取决于任务、并行数与相机，16 GB 显存的工作站规划针对可选 Isaac Sim。该文档会变，未来安装应锁定具体版本和驱动合同，而非直接覆盖现有 Torch 环境。[Isaac Lab 当前官方安装要求](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html#system-requirements)

## 第一阶段：另行注册的短 CPU pilot

建议在 T399 收尾、单 writer 退出后另立协议，**总墙钟不超过 10 分钟**，固定 1→2→4 环境三档，不扫描更多并发数。先冻结同一策略，不做优化器更新；沿用相同任务清单、每轨迹种子、动作/物理时钟、120 维因果观测和当前单轨迹推理路径，关闭显示与录屏。每档同样的总轨迹/总物理预算，初始化开销单列；重放和 setup 也计费。额度不足时不启动，不能边跑边增加预算。

记录全部物理步/秒、策略前向/环境步/通信/保存各耗时、总墙钟与初始化、峰值 RSS、CPU 使用、进程退出及输出字节。相同任务轨迹逐项核对状态、动作、reward、终止、历史、hidden、native 积分与 SHA；差异即保留失败并停止，不用平均成功率遮盖语义变化。只有 2/4 环境在相同总工作量下节省墙钟且保持证据完整，才选最快的**已测**档位。32 逻辑线程不等于应该直接开 32 个仿真。

初始每 worker 固定单计算线程并明确进程启动方式，不继承活动 native/GPU/渲染上下文；避免每个 worker 又开满 Torch/BLAS 线程。只保留一份策略权重版本与必要检查点，每车保存必要完整证据，不默认每车视频。若日后任务输入需要真实相机，必须渲染该传感器；仅关掉展示窗口不等于可以删掉策略输入。

## 并行后仍须遵守的学习合同

1. 一批采样固定一个 policy SHA/version，主 learner 收齐后再更新；下一批广播新版本。异步收集若引入旧版本轨迹，须另定算法合同，不能混作当前 on-policy 数据。多个探索流不是多个独立训练 seed，精确重生也不是新增独立样本。
2. 每环境保存并传递其实际 `(history, hidden, previous-action, executor, clock)`。一个环境 reset 只处理它自身；同回合不能每步清 hidden，也不能把别车 hidden 或旧参数生成的 hidden 当当前值。模型更新后若需从 episode 头重建，必须用该环境自己的完整因果序列和当前参数，记录来源。
3. 明确选择和测试 vector autoreset 行为，保存 reset 前最后观测与终止标记。`terminated` 与 `truncated` 分开；价值 bootstrap 依任务是否真正终止决定，不能把超时和碰撞一律合成 `done`。当前 Stage 原始 reward/时限/成功/碰撞语义原样保留。[Gymnasium autoreset API](https://gymnasium.farama.org/api/vector/async_vector_env/)、[官方时间限制教程](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
4. 吞吐提高不等于样本效率或导航效果提高。比较学习时固定总真实物理预算、任务分布、更新次数及 batch 合同；旧能力、窄通道和原起点导航各自门槛保留。完整 old275/主54/fresh16/reload78 未执行仍为 null，局部接管 GOAL 不算全程自主，未过门不进 7/8 米。
5. global new-runs **≤1 GiB、磁盘 free≥40 GiB** 与本轮完整最坏容量 admission 均先通过；多环境共用总预算，不能每 worker 各给一份。原模型、失败、全部必要原始证据、独立审计与远端逐成员 SHA 核验照常保留，禁止为并行吞吐删证据或绕配额。

先完成上述 CPU 证据，再决定是否值得做 GPU 批量 learner。若物理仍主导而 CPU 扩展已饱和，下一步才给 MJX-Warp 单独的小型兼容性/内存/吞吐协议；本说明没有安装、运行原生模拟器、训练或修改优化器。
