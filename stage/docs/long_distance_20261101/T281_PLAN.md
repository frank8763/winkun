# T281：实际任务梯度诊断，零物理/零优化器

只锁T280两臂各首16真实full120记录及相同安全T255初始权重，测量四组实际surrogate梯度/Gram/夹角、PCGrad精确投影和各参数层范数。T280系数159/262倍差异只是线索，本轮才测真实参数梯度，不能用旧T264课程诊断冒充。

固定600秒、两个batch、own1MiB/global1GiB/free40GiB、0原生/0优化器/0导航更新，不导出模型。Adam.step与native_setup入口替换为立即拒绝。记录full梯度SHA和双精度Gram，不占盘保存20MB向量；源/数据/模型哈希锁定，结束再核对。原生模型与旧数据只读，无Phase/地图/规划/路点/教师。

判读：有界代价的RIGHT/local真实梯度若均小于LEFT的0.1，支持进一步研究有限幅度任务平衡；若不弱，否定该优先假设，不硬套归一化。比例阈值只是本固定数据的开发筛选，非统计显著性或训练有效保证。PCGrad原实现需逐位一致，再用Float64收缩计算统计。检查soft100在未改父模型的梯度，另外固定旧dev_task6_24首/中/末三个状态，测确定性命令沿单位负PCGrad方向的一阶导数，不修改参数，不声称该线性量预测真实碰撞或Adam行为。

本轮核对[GradNorm正式摘要](https://proceedings.mlr.press/v80/chen18a.html)和[CAGrad正式摘要](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)，近邻/算法限制同T280_NEXT_DECISION。未移植作者实现，非算法原创，无自动新DRL/7m/8m。零梯度不能直接归一化放大，反向冲突与尺度不同。

当前GitHub认证401，T280与本轮新数据不称远端已备份。本轮代码/协议先本地提交，原始summary/详细JSON均保留。该只读诊断授权来自用户继续推进及已落地下一决定，不绕过正式导航训练的源同步/容量/验收条件。
