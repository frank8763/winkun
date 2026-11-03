# T327：独立宽缺口LEFT6人工课程

用户明确授权先易场景或放大缺口。T326只读原模型测得real_circle_obs_9与竖墙real_rect_obs_1净x间隔0.569999978m，chassis_collision本体宽0.54m/长0.86m；这不是全部轮胎包络或任意航向可达证明。只将circle9向+x移0.60m，间隔成为1.169999978m；不缩车、不改其它障碍。原地图文件/原快照/安全策略/所有旧结果保持。只固定一个宽度，不扫网格，跟T179缩圆半径不同，明确新任务scenehash。

位置变化用安装MuJoCo3.9.0的mj_setConst刷新静态BVH/界限，恢复原native积分后mj_forward再恢复积分；预检实际mj_ray新圆位置及独立MjData碰撞探针，0物理。雷达/碰撞/渲染共同使用同一新model。原4factory历史保留真实pose/goal/time/运动字段，仅inplace更新8lidar扇区以对应新几何；当前观测重读，新初始120单独记录。原native积分、goal、stepcap、2ms/hold100、56秒、0.2m半径和碰撞优先不变。宽场景GOAL只表示课程采集成功，不当原六米验收。

沿用T325已测的人控界面/请求去重/速度轮角遥测，最多3attempt，首次真GOAL后停止；每条完整12数组+积分原生精确重生，所有失败/行政删失原样保留。1零步预检+3记录初始化+3重生≤7全部命令，168k全部物理，main1800s末240重生收尾/外部timeout更早从严；预览复用首record不多开。own4MiB，3×512KiB raw+512KiB原子partial=2MiBpayload，1.5MiB metadata+.5buffer，global≤1GiB/free40GiB。0Actor/Adam/梯度/RL，不自动旧275/six54/fresh16/reload78/7或8。仅真实用户input推进，不由AI代驾驶/规划器/路点/运行时教师或路由。Phase全隔离。

输入SHA/真实0物理预检/源干净提交/远端精确根树完成后唯一启动。预检先保留原geometry待填protocol，固定+.60科学配置不变，实际宽场景初始hash计算后回填最终protocol，原注册副本保留；不重跑预检。active单writer不外写runs/改源/源码sync；结束全部父server/render退出后独立全部raw/history/reward/序号审计和RAM远端逐成员SHA。原场景失败和T325中止/碰撞保留。

文献核对2026-11-03：[MuJoCo官方Simulation文档](https://mujoco.readthedocs.io/en/3.6.0/programming/simulation.html)查询geom_pos/mj_setConst与静态bounds；安装3.9.0以实际ray/collision探针复核而非仅凭最新文档。[ACCEL ICML2022正式页](https://proceedings.mlr.press/v162/parker-holder22a.html)本次读摘要，说明环境难度课程有相关方法；不采用自动regret/evolution，不声称读完全文/复现。此前T179/T180障碍难度负结果限制泛化结论。此次是用户授权的固定易场景数据采集，不是新算法或原创贡献；训练后必须逐步回原几何并独立原任务验收，不保证迁移。

首次预检在刷新后witness对象匹配拒绝，1geometry/0物理/0Adam，原源/协议/错误保留preflight_failed_001。修复只保存并重新绑定已真实记录的history pose/goal/time到刷新后的frame，不编造历史。原3human/168k/1800s/4MiB保持；明确增加1个零步恢复geometry，预检2个+main最多6个=8全部命令，不将已做的预检藏掉。
