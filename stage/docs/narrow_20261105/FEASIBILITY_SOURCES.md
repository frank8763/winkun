# 原窄六米：整车可通过性与最小原生诊断依据

2026-11-05，Stage only。这里只读原证据、换算已存JSON坐标、查询包版本并检索一手资料；**没有创建原生环境、步进、优化、启动作业或修改runs**。不引入地图、规划器、路点或运行时教师。

**建议先验证整车局部通行，再谈从原起点56秒到达。** T326的0.57米缺口与0.54米车身宽度不构成整车通行证明：已存轮胎碰撞体中立外宽约0.61米。反过来，圆柱边界与有限墙段也使一个横向宽度不足以证明整个原场景不可达。

## 本项目证据与必须补齐的轮廓

- [T323](../distance_20261103/T323_FINDINGS.md)的两条40秒人工前缀后由safeT255接手均碰撞；[T324](../distance_20261103/T324_CAUSE_FINDINGS.md)没有证明左六米不可达，并明确原目标控制与新状态覆盖缺口。它们不是几何无解证明。
- [T325](../distance_20261103/T325_FINDINGS.md)人工原窄尝试没有成功；[T326](../distance_20261103/T326_FINDINGS.md)只测得circle9左缘与rect1右缘净x间隔0.569999978米。后续[T327](../distance_20261103/T327_FINDINGS.md)、[T328](../distance_20261103/T328_FINDINGS.md)分别是宽1.17米的56秒超时与80秒人工成功，不能替代原窄56秒验收。
- [最新晨检](../overnight_20261104/MORNING_VERIFIED_20261105.md)：选中T353 iid循环单Actor在固定宽课程5/5；原窄LEFT于48.034秒接触、余2.394243米，RIGHT于56秒超时、余0.201518414米；旧20仅10/20。完整old275/six54/fresh16/reload78未执行，不能填失败率或宣称原窄稳定。

直接读取[T326原geometry.json](../../runs/t326_gap_geometry/geometry.json)，以chassis_collision旋转矩阵逆变换原有world_pos，得到下表；这是已存坐标的代数换算，非新仿真。MuJoCo的box size为半尺寸，cylinder size为半径与半轴长。[官方MJCF形状定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-type)

| 原geom | 形状/几何 | contype / conaffinity |
|---|---|---|
| chassis_collision，id23 | box半尺寸(.43,.27,.10)，即长.86、宽.54米 | 1 / 1 |
| 四轮碰撞体，ids25/27/29/31 | cylinder半径.12、半轴宽.035米；中心在车体坐标(±.30,±.27,−.12)米 | 各1 / 1 |
| chassis_visual及轮胎可视mesh | 仅可视模型不能代替上述碰撞体 | 0 / 0 |
| rect1 id2 / circle9 id20 | 原墙box / 原圆柱；circle9半径.25米 | 各1 / 3 |

由四轮中心与轴宽得中立轮胎横向外包约`2*(.27+.035)=.610m`，大于车身宽度；转向后轮胎投影还会改变。原文件还有lidar/机械臂等机器人碰撞geom，应按原robot_body_ids与三维高度检查，不能只挑车身或把所有geom投影无限拉高。上述掩码满足`(contype1 & conaffinity2) || (contype2 & conaffinity1)`，因此轮胎不能因“车身才算碰撞”而省略；仍须核对原显式pair/exclude及实际终止调用链。[MuJoCo碰撞选择与过滤](https://mujoco.readthedocs.io/en/stable/computation/index.html#selection)

## 为什么宽度、静态姿态和可执行通行不同

LaValle的有限刚体定义要求整个机器人占据区域避开障碍；完整几何连接要求从指定初态到指定终态的连续无碰撞路径，而不是找到一个无碰撞姿态。[作者书§4.3.1](https://lavalle.pl/planning/node160.html)、[平移时的完整轮廓障碍关系](https://lavalle.pl/planning/node161.html)

车还受滚动约束：简单车模型有`xdot=v*cos(yaw)`、`ydot=v*sin(yaw)`、`yawdot=v*tan(steer)/wheelbase`，转角受限意味着最小转弯半径；禁止倒车会改变可达集合。该模型忽略高速动力学，因此只能解释检查要素，不能替代本项目实际MuJoCo执行器。[LaValle作者书§13.1.2.1，式13.12–13.15及其限制](https://lavalle.pl/planning/node658.html)

据此作本项目的诊断区分：

| 证据 | 允许得出的结论 | 不能推出 |
|---|---|---|
| 单个完整车姿态无碰撞 | 该姿态可放置 | 能进入、能离开或能转向 |
| 有限局部姿态/转向扫描 | 哪些已查样本有净空/碰撞 | 未查状态无解或连续连接存在 |
| 实际原生控制从局部reset完整穿过 | 该局部起态与控制的动力学通行见证 | 原六米起点可连接或56秒内到达 |
| 原起点、原plant/场景/56秒、原.20米判据的成功 | 对应原任务真实完成 | 泛化、全开发稳定或全图可达 |

几何推导示例：矩形车身在平行直壁之间、相对通道轴偏航θ时，横向投影是`.54*abs(cosθ)+.86*abs(sinθ)`；这说明小偏航也会消耗净空。**它只适用于该矩形与平行直壁假设，不能代替本场景圆柱边界、轮胎和动态扫掠检查。**

## 原生API的正确用途

本机只读包元数据为MuJoCo **3.9.0**。官方`stable`文档已读目标段；版本化3.9.0文档URL未取到，另核对官方3.9.0源码，未复制或运行其代码。

- `mj_forward`重算派生量而不时间积分；`mj_step`才推进仿真。几何查询应在**复制的model/data**上同步派生变换，保护原快照、clock、控制与积分证据；forward不等于完成任何运动。[官方API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-forward)
- `mj_geomDistance`返回有符号最小距离并受`distmax`上界截断，可选返回两geom间的线段。应保存geom IDs、明确的distmax及截断标识，不能把恰等distmax当精确净空。[官方API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance)
- 3.9.0源码直接调用两geom的形状碰撞函数，没有替调用者执行整套mask/body过滤；不支持的形状对也返回distmax。故先建立原生可碰撞的robot/world对，区分可视geom、地板和障碍，确认形状支持，再解释距离。[官方3.9.0源码，mj_geomDistance](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_support.c#L519-L565)
- 官方指出legacy MPR距离查询在远处可能不正确，native GJK更适合距离；记录原nativeccd开关、ccd_tolerance、ccd_iterations，**不为获得“可过”而改原选项**。原生contact列表还受margin和过滤影响，和直接距离查询分列报告。[官方距离限制](https://mujoco.readthedocs.io/en/stable/computation/index.html#geom-distance)

现有[Stage依赖_collision实现](../../runtime/mk-mini/training/e2e_phase10_lidar_v1/phase10_scene_e2e_env.py)对已生成的非floor机器人/环境接触检查`contact.dist<=.002`；不能仅检查车身，或把“没有接触条目”等同于一张连续净空证明。最终仍以本次实际调用链的原生终止判据为准。

## 最小下一诊断建议（本笔记未执行）

1. 在独立有界协议内锁定原scene、完整碰撞geom/masks、实际plant/动作映射、零转向含义、速度与时标；只对瓶颈小范围作预先固定的姿态/轮角几何检查。扫描是离线测量，不生成全图、路点或部署控制器。若没有通过样本，报告有限样本失败；不能据此宣称无解。
2. **仅在一个完整整车局部起态通过静态检查时**，预注册一次局部直穿测试及一次完整精确重生：初态在瓶颈前、原生执行器和固定有界drive/零steer控制、从起始reset后只许mj_step推进，直到整车清出瓶颈或接触/预定时限停止。静态qpos平移、中心越线、动画插值均不能当通行成功。若中立直穿被轮胎否决，就保留失败，不临时扫转角或增加尝试。
3. 每个原物理步保留/核查原生contact、全机器人最小相关geom距离及发生位置，记录实际steering、速度、动作和完整状态；0.2秒决策端点净空不能冒称逐物理最小。穿过局部的证据仍是改变reset的诊断，不作为Actor示范、on-policy资料或原起点自主六米成功。离散物理步的见证也不是连续时间形式化安全证明。

只有另行从原六米起点实际执行并满足原56秒/.20米/无接触判据，才可声称对应原窄任务成功；目前资料**不保证完整原六米可解**。若整车局部通行都未建立，应先如实解决场景/车体可通过性合同，保留原场景和失败，不能悄悄缩车、屏蔽轮胎或改宽度来填“原场景通过”。

检索范围：2026-11-05定向阅读上述作者书目标小节、MuJoCo官方API/形状/碰撞目标段与3.9.0单函数源码；没有整本阅读、穷尽文献或引入外部规划实现。所有局部扫描/原生见证均由下一独立协议另行计费注册，本笔记只提出判别标准。
