# 复杂场景中的经典雷达端到端导航：一手论文核验

检索日期：2026-11-06。问题限定为雷达/相对目标/历史观测到连续控制的移动机器人导航；区分学习策略、定位来源、运行时混合控制和全局规划。本次只增研究笔记，未运行仿真、训练或修改历史证据。不是穷尽检索，也未核验所有论文附件及视频。

**最直接的先例是 Long 等 ICRA 2018：学习策略处理有静态障碍的窄走廊双向交会。** Fan 等 IJRR 2020 提供更丰富的复杂场景和真机材料，但其完整部署系统是 Hybrid-RL。下述机器人主要为差速底盘，不能把它们的通过率移植到 Stage 的油门/舵角动力学车。

## 1. Long 等，ICRA 2018：优先阅读的近邻

**Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning**，Pinxin Long、Tingxiang Fan 等。会议及页码为 ICRA 2018，6252–6259，DOI 10.1109/ICRA.2018.8461113；[作者所属机构记录](https://scholars.cityu.edu.hk/en/publications/towards-optimally-decentralized-multi-robot-collision-avoidance-v/)。技术内容查阅 [arXiv v3，2018-05-20](https://arxiv.org/abs/1709.10082v3) 的 [9页 PDF](https://arxiv.org/pdf/1709.10082v3)。

- 输入为最近三帧、每帧512束、180°/4m雷达，以及目标相对极坐标和当前线/角速度；输出连续线速度 `v∈(0,1)m/s`、角速度 `ω∈(-1,1)rad/s`，禁止倒车。训练机器人为同半径圆盘、非完整约束差速模型；**该版本正文只给半径符号 R，未核实数值半径、执行器动态或走廊精确净宽**。见 PDF §III、§IV-A。
- PPO/GAE共享策略：先20机器人在无静态障碍随机场景学习互避，再58机器人跨多场景继续训练；不是用SL基线的示范数据预训练该RL策略。§IV-C、§V-A。
- §V-B4/Fig.7 明确：两群机器人经带两个障碍的窄走廊互换，仅第二阶段策略完成；§V-C另展示100机器人及异形/不合作机器人。统计指标为限时无碰撞到达比例、额外时间、额外距离、平均速度，常规案例50次重复；**不能把50次重复自动解释为窄走廊独立成功率或100机器人统计保证**。§V-B/C。
- 原ICRA正文的实验是仿真。相对目标直接提供引导，未见该策略使用全局子目标/规划器；它也未证明任意复杂拓扑中的全局可达性。以上均据[论文方法及实验](https://arxiv.org/pdf/1709.10082v3)，不是 Stage 同场景复现。

[作者项目页 DRLMACA](https://sites.google.com/view/drlmaca) 提供论文和演示入口，并把真机材料另链至后续工作；本次未观看嵌入视频。常见 [Acmece/rl-collision-avoidance](https://github.com/Acmece/rl-collision-avoidance) **不是作者代码**：README 的 Notice 明确自述不属于作者团队；它是 PyTorch 复现，而正文说明原实现为 TensorFlow。未将其参数反填成论文原始配置。

## 2. Fan 等，IJRR 2020：复杂场景强证据，须拆开 RL 与 Hybrid-RL

正式题名 **Distributed multi-robot collision avoidance via deep reinforcement learning for navigation in complex scenarios**，Tingxiang Fan、Pinxin Long、Wenxi Liu、Jia Pan，*The International Journal of Robotics Research* 39(7):856–892，2020，DOI [10.1177/0278364920916531](https://doi.org/10.1177/0278364920916531)。年份/卷页由[HKU机构库](https://hub.hku.hk/handle/10722/285104)及[作者 Hybrid-MRCA 页](https://sites.google.com/view/hybridmrca)交叉核验。

**访问限制：**出版商正文未成功取得；方法细节来自 [arXiv:1808.03841v1，2018-08-11，30页](https://arxiv.org/pdf/1808.03841v1)，其题名以 *Fully Distributed…Safe and Efficient…* 开头。未逐页比对2020最终37页版本，不能把预印本全部数值无条件标为最终刊本结果。

- 预印本 §4.4/Algorithms 2–3：运行时按场景切换 PID、保守安全策略、RL。因此 Hybrid-RL 真机表现不等于单个Actor独立完成。§5.3.5 同时报告 learned-only 的 RL Stage-2 和 Hybrid-RL 能过窄走廊，并明确最终目标向量在复杂建筑拓扑中可能失效。
- §5.1、§5.4.3：训练圆盘半径0.12m，仿真步长0.1s、控制10Hz；这是扩展预印本配置，不能据此补写 Long 2018 未报告的数字。
- §6.1/6.2：46cm边长方形真机、雷达+UWB定位，测试对象明确为 Hybrid-RL；含静/动态障碍和仓储任务，§6.3/6.4含密集人群。§5.4还检查尺寸、密度、速度、频率变化；这些是经验鲁棒性测试，非安全/完备性证明。[技术来源：预印本](https://arxiv.org/pdf/1808.03841v1)。

作者项目页提供视频入口；本次未验证作者发布的完整训练代码，也未执行第三方实现。

## 3. Tai 等，IROS 2017：接口最接近的奠基工作

**Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation**，Lei Tai、Giuseppe Paolo、Ming Liu，IROS 2017。查阅[最终提交版 arXiv v4，2017-07-21](https://arxiv.org/abs/1703.00420v4)及[6页 PDF](https://arxiv.org/pdf/1703.00420v4)。

- 10维稀疏雷达+2维相对目标+上一步2维动作，共14维输入，直接输出差速底盘线/角速度；从零 ADDPG，无先验示范。V-REP 的两个10×10m环境训练，约0.8M训练步/20h；没有核实车体精确半径/执行器参数。§III、§IV-A。
- 测试包括未知仿真布局及狭窄办公室真机，Env-2策略完成预设目标序列；报告控制查询频率、行驶时间和距离，并描述窄道中旋转恢复。§IV-B。
- **“无地图策略”不等于整套实验无地图依赖：真机用 AMCL 提供位姿计算相对目标，任务按预设0→9目标逐段执行。**该论文不是任意终点的长程拓扑导航证明；讨论明确承认无观察记忆/长期预测、轨迹曲折及大规模复杂环境的限制。§IV-B、§V。[技术来源：论文](https://arxiv.org/pdf/1703.00420v4)。

本次未验证可归属作者的完整训练仓库；不将搜索到的复现库称为官方代码。

## 4. Fan 等，ICRA 2020：窄道与传感器故障下的学习控制

**Learning Resilient Behaviors for Navigation Under Uncertainty**，Tingxiang Fan 等。[arXiv v3，2020-06-03，7页](https://arxiv.org/abs/1910.09998v3)注明 accepted to ICRA 2020；方法核验据[该版 PDF](https://arxiv.org/pdf/1910.09998v3)。

- 雷达/速度历史经两层GRU预测运动与不确定性，再连同相对目标进入连续线/角速度策略；用 SAC 和温度衰减训练。预测器另以旧导航策略收集约25,000轨迹监督训练，故不是所有模块从零仅靠RL、也不是单个前馈Actor。§III、§IV、§V-A。
- §V-B包含狭窄静态走廊和不合作动态障碍：过度保守策略会卡在入口，平衡策略可进入。量化实验为12×12平面20机器人、雷达180°/1°，测试有效FOV降至150/120/90/60°，每项50次，分报成功/卡住/碰撞；60°时仍有卡住。§V-C。
- 本版未核实精确车体尺寸/窄道净宽；实验材料为仿真，没有据此宣称真机验证或复杂迷宫完备导航，也未核实作者代码。相对目标输入中未见全局路点，但部署有学习预测器。[技术来源：论文](https://arxiv.org/pdf/1910.09998v3)。

## 对 Stage 的判断（本笔记的推论）

这些论文足以回答“有没有相似论文”：**有，而且包含窄走廊、静动障碍混合和拥挤交互；Long 2018 尤其接近无规划器的学习控制合同。**“复杂”仍须逐维对照：同一策略能否通过、车宽与通道净宽、差速或舵轮车、是否可原地转向、观测时间跨度、目标是否分段提供、训练布局覆盖、测试独立性，不能仅看场景照片或机器人数量。

对当前“新0.70m/80s教师曾成功、学生仍0/5”的事实，这些文献不能证明 Stage 已可由现有学生稳定解决，也不能据此断言 RL 无效。可借鉴 Long 的跨场景课程，以及不确定性导航论文的到达/碰撞/卡住分别统计；现阶段的教师到学生闭环差异仍须由本项目证据解释。引用先例不自动构成新实验授权。

## 本次实际查看范围与未完成项

以下为网页工具提取的 PDF 页码（从1计）及相关段落范围，**不是“全文所有细节均已读完”**：

| 来源/版本 | 本次核验范围 |
|---|---|
| Tai 1703.00420v4 | 摘要/版本记录；PDF p2–5 的 §III、§IV-A/B、§V 方法/实验/限制段；工具提取行158–323、343–521中的相关窗口 |
| Long 1709.10082v3 | 摘要/版本记录；PDF p3–8 的 §III、§IV-A/B/C、§V-A/B/C相关窗口；行218–279、396–704、704–849、850–914、997–1061 |
| Fan 1808.03841v1 | 摘要/版本记录；PDF p9–12 §4.4/5.1、p15–18 §5.3.5/5.4.3、p20–26 §6相关窗口；另核作者项目页与HKU期刊元数据 |
| Fan 1910.09998v3 | 摘要/版本记录；PDF p2–6 §III–V及结论相关窗口，提取行164–372、478–687 |
| 代码身份 | Acmece仓库README与Notice；没有克隆、安装或运行代码 |

未做：最终刊本与预印本逐项差异审计、窄道图像比例测量、视频逐帧核验、作者代码复现、Stage性能比较或新训练。未核实的参数已留空，未用常见第三方配置填补。
