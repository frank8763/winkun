# T679固定尺度整回合参数探索：仅采集可用性，不训练

T678两目标各一次真实Actor更新均丢旧任务且auditPASS，完整407未执行，T501保留。多项确定性失败的随机数据surrogate为正，不能推出总体随机回报提高；这次转向检查探索分布，而非继续参考权重/步数/seed网格。历史T207低秩6系数参数探索LEFT失败仍保留，当前11906适配参数/64GRU策略不同，不称新算法或旧否定失效。

固定中心T501全部权重，只临时扰动11906现有适配参数：标准Gaussian epsilon、不按方向归一化，不给120输入加噪声，不改encoder/GRU/head，不增加planner/teacher/router/Q。8固定校准Gaussian方向独立于4固定评价Gaussian方向。对固定8任务各一条已存真实外部完整历史算网络参数JVP(自己的因果carry/冻结backbone特征)，按任务、时间、动作坐标均衡二阶矩设唯一全局s=.05/sqrt(mean(J_epsilon²))。数值差分epsilon1e-6、abs1e-8+rel1e-4；JVP是网络而非物理闭环导数，.05只参考旧IID每坐标rawsigma，不称命令方差/原生探索强度匹配。全部8实际评价variant固定历史raw-location RMS需[.025,.1]且finite，否则停止、不调整尺度/seed。

4评价方向antithetic±得到8冻结policy，全回合确定性seedNone。每variant8固定原起点任务：旧dev6_00/02/13/20，field0_offset3/1_offset1/7_offset1/7_offset2。所有64fresh完整保存+64独立原生exact，模型hash、各数组/积分SHA精确；原几何/目标/时限/20cm/场地适用边界不改。共享方向跨任务和正负相关，不能说64独立参数样本，exact不加样本；T672旧IID数据只描述对照，4/8原样分母，不当新匹配预算独立优越性试验。每ID报告GOAL/collision/timeout/boundary、回报Y=safeGOAL-collision、成对差、条件动作偏移和lag1相关，不以临时variant成功称中心已改善。

support门：校准PASS、64fresh齐、至少2old和2field的8个variant结果Y不全相同、exact/audit全部过，才PARAMETER_EXPLORATION_SUPPORT_PASS。门仅表数据机制可用，不保证梯度质量或后续训练；11906维仅4方向的估计可能极高方差。否则保科学否定停止这次固定机制，不扫幅度/seed/样本数量救。无Actor/Value/optimizer，无中心更新或candidate保留；8冻结variant原始模型/hash保留。

全部最坏物理16×sum(8task stepcap)，命令128+6=134，包含fresh/exact；3000main含300校准+300audit，root3600秒≤1h硬停，320MiB/global10GiB/free40GiB。64full4MiB+64exact32KiB+8models2MiB+metadata/partial槽有界。CPU8/10/12原生14校准/独立audit400%Nice19低优先4GiB无swap/CUDA关闭，资源重叠仅暂停自身cgroup，墙钟继续。先实际JVP/forward/参数恢复/序列化/realworker预检、ledgerworstRESERVE、完整GitHub精确树才唯一launch。main结束全科学PID退出后独立重建校准/8参数模型/所有轨迹，结果计费/失败保留，不与科学并行备份source。预注册阶段入口映射/待文献hash问题0物理0优化器原样记录，不是科学失败或旧任务重启。
