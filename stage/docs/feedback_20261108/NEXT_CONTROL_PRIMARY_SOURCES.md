# T429 后续控制：三项一手依据与待证假设

2026-11-08；Stage only。本次只查文献、读既有证据并写本文；0 native、0 optimizer、无控制器或实验代码修改。下述是下一独立有界协议的依据，不是启动授权、成功证明或新算法主张。

## 已知问题

`FAILED_CLEARANCE_POSTHOC.json`：dev_task6_21 在22s相对同钟成功基线位置差0.065085m、yaw差4.4283°；接管速度0.11569m/s、参考0.15697m/s。反馈后约6–7cm横偏未消除，29.830s接触；后缀速度min/median/max为0.08137/0.11892/0.18004m/s，归一化正转向饱和占10%。这支持先查误差收敛/几何净空与执行器响应；不支持“已不可恢复”或“持续打满仍转不过来”。既有`reference.py`的反馈为`a_steer = a_ref + ky*left + kpsi*yaw_error`，`left`是当前车体坐标中的参考点偏移，并非已核验的前轴对路径法向距离。

## 1. Stanley：速度归一化横偏反馈，但稳定性证明有模型前提

[Hoffmann et al., 2007, Autonomous Automobile Trajectory Tracking for Off-Road Driving](https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf)，§II、§III-A、Eq.(4)–(7)。已读作者站点检索返回的这些正文/方程片段；PDF整文件重开超时，不声称通读全文。

控制律为`δ = ψ + atan(k*e/v)`并施加物理转角上下限；误差相对前轮附近参考路径，`δ`是转角。零横偏渐近稳定的证明针对运动学模型、正速度及给定转角范围。正文另以一阶转向响应`δ_dot=(δ_cmd-δ)/τ`描述执行器；Stanley上τ=0.4s是其车辆实验值，不能移植到Stage。

**本项目假设**：固定归一化动作域线性增益在0.08–0.18m/s间未保持同等横偏纠正率。标准Stanley提供待验证的机制差异；须先核实前轴/车体参考点、误差符号和角度映射。`v=0`时原式不定义；若加速度下限或softening，须明列为实现改动，不继承原证明。该论文不保证本车29.83s前恢复、不保证障碍净空。

## 2. Pure pursuit：几何曲率可纠偏，忽略执行器会延缓闭合

[Coulter, 1992, Implementation of the Pure Pursuit Path Tracking Algorithm, CMU-RI-TR-92-01](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)，§2、§3.3、§4.1、§4.3（PDF页9–14，报告页5–10）；这些相关正文已读。

算法在车辆局部坐标中选路径前方点，以圆弧曲率追踪；按前向x/左向y惯例重写为`κ=2*y/Ld²`，并非原文坐标字母。长lookahead使回归更缓、振荡较少，也可能跟不上弯曲路径。§4.3明确：方法不建模车辆/执行器能力，理想响应假设会忽略转向一阶滞后，从而不能按预期速度靠拢路径；另一限制是高速突变曲率导致侧滑。

**本项目假设**：低速不消除转向滞后或有限曲率限制。应读现有真实转向状态/参考曲率及误差变化，先区分“请求的纠正不足”和“请求后物理响应不足”。不能只缩短lookahead或放大增益就宣称解决；pure pursuit也不是避障保证。

## 3. DAgger：在学生实际访问状态取纠正标签，最终可部署单策略

[Ross, Gordon & Bagnell, 2011, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)，§3、Algorithm3.1、Theorems3.1–3.3；相关全文段落已读。

DAgger用当前策略访问状态，查询该状态的专家动作，聚合历史数据再监督训练；Algorithm3.1最后返回验证最好的单个学习策略。训练采集时可以混合专家，部署不要求混合专家。其保证依赖专家/损失、策略类、无遗憾及采样等条件，不能直接成为神经网络有限预算下的无碰撞保证。

**本项目约束**：成功原始轨迹上的动作不能直接充当当前偏离状态标签；恢复控制器须在该真实状态闭环成功并精确复核。仅保留已验证恢复用于后续训练是项目额外筛选，不等于完整原版DAgger；筛选后的分布保证须重新说明。学生训练输入维持120维因果history/真实carry，参考pose/path仅用于训练期控制与标签，最终仍单neural Actor。失败后缀不贴成功标签，混合行为不伪装on-policy PPO。

## 单位与下一证据合同

论文`δ`（rad）/`κ`（1/m）不等于`a_steer∈[-1,1]`。必须查实际运行的action→center command→Ackermann左右轮命令→测得joint angle链；真实joint角不必瞬时等于命令。Stage已核验依赖`stage9_runtime_fullplant_fix_v5_2.py:_apply_runtime_fullplant`正是从实际原始左右前轮命令读中心命令，注释明确不能猜`action[1]`。不要把旧wheelbase日志当实际FULLPLANT几何，不将论文完整转角控制律直接加到归一化参考动作上，亦须防止重复计算曲率前馈。

建议下一项先锁定既有成功参考、当前22s完整物理状态/clock/history/执行器，读出真实车身净空及前轴/后轴位置、命令与实际转向、误差导数，再择一个标准控制律与T429作固定预算对照。没有本项新增仿真；不扫gain/时刻/seed、不续T429、不自动训练或晋升。仅纠偏成功仍不等于旧275/宽54/fresh16/reload78完整验收通过。
