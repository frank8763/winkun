# T550：训练期无地图 LiDAR 反应式避障教师与有限保护迁移

日期：2026-11-14。本文使用 research 技能，是主任务明确分派的背景研究。只写本文件；0 原生物理、0 仿真命令、0 优化器、0 拟合，不启动、重启或扩展任何实验。未读取 Phase 研究、模型或结果，未宣称读全 Stage 聊天。本文不是新授权、预算或已执行协议；8 小时窗口及唯一 launch 由主任务另行锁定。

结论：可以研究一个**仅用于训练标签的固定 LiDAR 反应式教师**，但不能先假定它能解开现有障碍接触。APF 的吸引/排斥相加有明确反例；最近点 safety bubble 加最大角隙也不构成车辆可通过证明。必须先在保持真实因果前缀、完整执行器/hidden/history、原目标和原时限的原生接入中取得安全 GOAL，再考虑监督迁移。部署仍为统一的单 Actor，教师、地图、路径、路点和任务 ID 路由均不进入推理。

## 1. 实际核对的主来源与阅读深度

| 来源 | 本次实际查看 | 可支持与边界 |
|---|---|---|
| Koren & Borenstein, ICRA 1991, DOI 10.1109/ROBOT.1991.131810：[作者站论文链接](https://ykoren.engin.umich.edu/wp-content/uploads/sites/122/2014/05/4-Potential-Field-Methods-and-Their-Inherent-Limitations-for-Mobile-Robot-Navigation.19853.pdf)，[CMU 托管原论文全文](https://www.cs.cmu.edu/~./motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf) | 原文 §3–5，尤其 PDF 第7–9页的四类问题、实验条件；作者站直开403，实际全文来自 CMU 镜像。 | APF/VFF 有局部极小、窄门拒入、障碍旁振荡、窄通道振荡。其 CARMEL 是 synchro-drive；实验为可重复而手工置入网格障碍，不能当作 Stage Ackermann/LiDAR 传感条件的直接定量结论。 |
| Sezer & Gokasan, RAS 2012, DOI 10.1016/j.robot.2012.05.021：[出版商预览](https://www.sciencedirect.com/science/article/abs/pii/S0921889012000838)，[作者所属 ITU 的论文记录](https://research.itu.edu.tr/en/publications/a-novel-obstacle-avoidance-algorithm-follow-the-gap-method/) | 出版商 Abstract、Introduction 和各节 snippets；ITU 摘要/书目信息。**未取得全文，未核对所有公式或证明。** | 原 FGM 同时考虑最大间隙、目标、视场与非完整约束，使用外包圆建模，并在 Ackermann 车上实验；原文摘要声称没有局部极小问题。不能把该声称泛化为任意离散 scan bubble 实现的全局完备性。 |
| [F1TENTH 官方 Lab 4 模板 README](https://github.com/f1tenth/f1tenth_lab4_template)，[官方课程 Lab 4](https://f1tenth-coursekit.readthedocs.io/en/stable/assignments/labs/lab4.html) | README §III–IV；课程主页。2026-11-14实际查看，网页可变。 | 官方步骤为预处理 scan、最近点 bubble、连续非零最大 gap、选 gap 内方向、发布 Ackermann 命令；README 明示最近点 bubble 不保证车能通过 gap，并要求双向测试死端口袋。它是教学反应式算法，不是安全定理。 |
| [CTU-IIG/f1t-ftg 作者仓库](https://github.com/CTU-IIG/f1t-ftg)，[作者源码 follow_the_gap.cpp](https://raw.githubusercontent.com/CTU-IIG/f1t-ftg/master/src/follow_the_gap.cpp) | README；`FollowTheGapMethod`、`Callback`、`CalculateGapCenterAngle`，本次网页行34–86、227–282、552–590。 | 该实现由 Anders Solberg Pedersen/Jaroslav Klapálek 编写，**不是 Sezer/Gokasan 的原作者代码**。它明确修改原法以处理实赛问题，包含 no-gap/NaN 拒绝、额外同侧 gap 几何、重试及 corner fallback；可作为实现反证，不能称原算法精确复现或直接移植方案。 |
| Ross, Gordon & Bagnell, AISTATS 2011：[PMLR 原论文 PDF](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | §1–3、Algorithm 3.1、Theorems 2.1–2.2/3.1–3.4 及 §4 开头；不是只读摘要。 | 支持 learner 诱导状态上的专家标注/聚合和最终确定性策略；理论有损失、no-regret、采样与专家恢复条件，不能给本项目少量连续动作标签直接颁发同样保证。 |
| Pfrommer et al., NeurIPS 2022, TaSIL：[会议原论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf) | Introduction、§2–3 的问题设定/稳定性动机，未声称独立复核全部附录证明。 | 其传递闭环相近性依赖专家增量输入到状态稳定性及局部高阶匹配；有限点动作相等不是该条件。Stage 未建立这些条件，也未实现 TaSIL。 |
| Broomhead & Lowe, Complex Systems 1988：[出版方原文 PDF](https://content.wolfram.com/sites/13/2018/02/02-3-5.pdf) | 原文摘要及 §2–3 的函数插值、径向基与线性权重构造；未审全部数值实验。 | RBF 固定基加线性读出是既有函数逼近方法；该谱系不支持把本轮轨迹插值称为新 RL 算法。 |

## 2. APF 不应成为无条件“已合格教师”

原论文 §5 的具体反例是：U 形死端中吸引与排斥导致陷阱；两障碍合成排斥可能把车推离本来可通行的门；车体动态、采样延迟及速度会使障碍旁/窄走廊发生振荡。原论文调用全局规划器解陷阱的做法**不属于 Stage 可接入方案**。见 [Koren/Borenstein §5](https://www.cs.cmu.edu/~./motionplanning/papers/sbp_papers/integrated1/borenstein_potential_field_limitations.pdf#page=7)。

对 Stage 的推论：单纯把目标吸引项与 LiDAR 排斥项相加，或在失败后临时增大斥力，并不能推出会绕过当前障碍。增大斥力还可能拒入窄隙。固定 APF 若发生零合力、持续零进展、左右换向振荡或接触，应保留为本机制失败；不以新增路点、地图或事后参数搜索“修好”后仍称原协议通过。论文中的四类问题是警示，不是已经证明 Stage 当前失败属于哪一类。

## 3. Follow-the-gap / bubble 能借什么，仍缺什么

可借的是**从当下 scan 产生避障方向**的反应式结构。官方教学流程先屏蔽最近点 bubble 再取最大角隙，但官方 README 同时提醒车体是否能通过要另查；最大连续角度不是净通行宽度。参见 [F1TENTH README §III–IV](https://github.com/f1tenth/f1tenth_lab4_template)。原 FGM 考虑了非完整约束，不应把它简化成“任一最远射线直接当转角”；其全文仍有访问缺口，见 [出版商预览](https://www.sciencedirect.com/science/article/abs/pii/S0921889012000838)。

以下是针对 Stage 的工程反例/推论，**不是声称原 FGM 已被本文推翻**：

- 只膨胀最近一个点，旁边未被膨胀的近障碍仍可碰到车角；全方向 scan 有效距离大也不说明旋转中的矩形车身安全。
- 同样角宽的 gap，近处实际间距更小。远射线可能穿过车宽不足的狭缝；单个漏检或离散射线间障碍也可能产生假 gap。
- 最大 gap 可通向死端或偏离最终目标；目标方向和 gap 方向加权后的中间方向也可能落回被障碍占据的方向，必须重新验证，而不能仅验证两个端方向。
- 相邻 gap 大小随 scan 抖动交替，输出可以来回跳变；固定平滑/滞回若采用，必须事先锁定并计作教师内部状态，不能在失败后追加。
- no-gap、无效/非有限 scan、全屏蔽 scan 需要明确拒绝结果。把无效测量填成无限远，会把未知空间误作已知安全；它不是原生可达性证据。

CTU 作者实现也显式处理 no-gap、NaN 及原公式未覆盖的同侧边界，并使用多级 fallback；这说明“几十行 gap 逻辑即覆盖所有几何”没有实现依据。Stage 不应照搬其重试网格或 corner fallback 后仍称一个固定教师。见 [作者 `Callback` 与 `CalculateGapCenterAngle`](https://raw.githubusercontent.com/CTU-IIG/f1t-ftg/master/src/follow_the_gap.cpp)。

## 4. Ackermann 曲率、轮廓和动作语义必须同时成立

对 Stage 所需的是车辆能执行的两维原生动作，而非二维平面上的期望移动向量。名义无侧滑单轨几何为 `κ=tan(δ)/L`、`Rmin=L/tan(δmax)`；LiDAR 选出的车身相对方位角不是前轮转角。具体 Stage 轴距/限角和 T548 饱和反例来自 [T549_CURVATURE_AND_SUPPORT_SOURCES.md](T549_CURVATURE_AND_SUPPORT_SOURCES.md)，本文未新增动力学标定。

原 FGM 的外包圆建模是轮廓处理思路，见 [原文 Problem definition 预览](https://www.sciencedirect.com/science/article/abs/pii/S0921889012000838)。以下要求是本项目推论：

1. 必须区分 LiDAR 原点、车身中心、后轴参考点，并固定 scan 的角度方向、尺度与车体坐标变换。车体外包圆须包含全部真实碰撞轮廓；只按半车宽膨胀可能漏掉车头/车尾。
2. 可以预声明固定的 scan 角度屏蔽/车体膨胀规则，但离散测量覆盖、遮挡、量程和 LiDAR 安装偏置仍是缺口；这种屏蔽不是地图，也不是连续空间安全证明。
3. `|δ|≤δmax` 仅是命令可表达；转向速率、滞后、实际速度、侧滑及转弯内外侧扫掠仍可能使车角接触。减速可减小动态误差，但不消除几何最小半径，也不能把前向车变成可原地旋转的圆点。
4. 原生动作映射、下限、限幅和执行保持必须原样使用。不能假定 raw 最小驱动就是真停车或倒车，不能把教师输出的物理速度直接混作网络归一化标签。

允许的教师输入应仅为当下有效 LiDAR、相对**最终**目标、已有因果车体/执行器状态及预先声明的教师内部记忆；不读取全场障碍真值、地图、未来碰撞、任务 ID、历史成功路线或预制路点。训练恢复必须从唯一预定学生前缀真实接入，沿用原 clock、goal 与成功判据。原生安全 GOAL 和完整精确重放是当前工程资格，不能被一张可视化 gap 图或未饱和命令代替。

## 5. DAgger 启发的训练标签，不需要运行时教师

[DAgger Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf#page=4)在当前 learner 访问状态上取得专家动作，并聚合训练；最终返回学习策略，专家混入是可选的训练采集机制。原文 §1–2 解释动作错误会改变后续输入分布，有限 sample 定理另有条件。**因此可以把教师完全留在训练端；不能由此声称一次成功后缀插值就是完整 DAgger 或自动获得其界。**

对本项目的推论：首批标签可来自预定学生失败前缀的真实恢复，但后续学生自己产生的新状态若不再等同教师轨迹，已有标签并不覆盖它们。仅在教师自己行走的状态上学习属于恢复 BC；只有实际收集 learner 诱导状态并在那里标注，才具备 DAgger 的关键数据机制。教师在某个学生状态无法安全恢复时，也没有合格的“专家”标签可硬填。

训练时必须分别保存教师实际输入、hidden/history 演化、动作语义、执行命令及真实终止，不能把 teacher forcing 下的小误差当学生闭环成功；恢复后缀与原前缀不可把时钟/动作历史拼成一个从未实际执行的状态序列。候选冻结后须由**同一个 Actor 从原起点全程自主**执行，教师代码不可被推理调用，不能按任务/时间/失败检测进行教师切换。

## 6. 保护性紧支撑 RBF 是窄记忆，不是鲁棒 RL

固定基函数加线性读出已有 [Broomhead/Lowe §2–3](https://content.wolfram.com/sites/13/2018/02/02-3-5.pdf)的明确谱系。Stage T549 的实际工程证据为：168新中心、一次闭式监督解、0新Adam/RL/Value；old8=8/8、field24=13/24（10接触、1超时）、保留原成功ID、只新增sector0_offset2，严格研究门未过、accepted=false。见 [T549_FINAL_REPORT.json](T549_FINAL_REPORT.json)。该结果不能表述为已掌握场地避障。

以下是支持几何的直接推论，不是文献赋予的系统保证。若新残差在全部旧保护特征点恰为0，固定环境、精度、hidden递推、动作映射与运算顺序都一致时，可以通过逐步归纳解释旧确定性轨迹为何保持。这个保证只针对精确访问点；新策略出现微小状态偏差后，归纳前提便不再自动成立。

若半径按最近保护点/其他中心距离的0.49倍隔离，中心矩阵可以是单位阵，但支持越窄，越容易在 scan/hidden 微扰后退出支持而回到失败父动作；中心间还有未覆盖空隙。基函数变化率按系数/半径缩放，条件数1不限制闭环灵敏度。相同因果特征若要求不同动作，确定性 Actor 也无法靠更多中心消除混叠。见既有直接证明及反例 [T549 来源笔记 §6](T549_CURVATURE_AND_SUPPORT_SOURCES.md)。

[TaSIL 原文 Introduction/§2–3](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f10c3d66c3b7863a9cda255dcac5bb7-Paper-Conference.pdf)强调专家稳定性和局部高阶匹配，而非只在采样点复制动作；这些假设在 Stage 尚未验证。RBF 教师迁移最多是“新增有限已知闭环成功、保护有限旧成功”的工程证据。若后续研究新 RL，须另列真实交互、策略梯度与保留更新数及匹配对照；不能把闭式监督解、教师控制或精确物理重放计作 RL 更新，不能以开发集通过冒充噪声鲁棒性、独立泛化、全场安全或算法原创。

## 7. 给主协议的可证伪决策

先锁定一个教师、一个接入规则、全部常数/无效输入处理、完整因果输入和原动作语义；在预定预算内只检验其真实原生恢复资格。没有安全 GOAL 就停止这个固定教师，不追加APF权重、bubble半径、gap选择或接入时间网格。资格通过后再做一次有界保护迁移；必须分别报告教师资格、离散点函数合同、候选原起点自主测试、精确重放与任何未执行验收。当前来源支持研究这条机制，不保证它会解决剩余10次接触，也不提供新的时间/物理/存储授权。
