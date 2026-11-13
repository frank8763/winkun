# T536：精简状态的原生恢复验证（未注册/未启动）

T534已停止并保留错误，2旧窗口原生恢复通过；新目标因单snapshot文件cap未验证。T535仅零native序列化准备，通过两个已有快照的所有非日志状态比较，原错误完整保留，不能代替原生恢复。

下一真实工作是修复日志与动力状态耦合造成的文件膨胀。`t536_runtime.py`使用同4窗口、同对应源模型、同strict动作/观测/history/64carry、同精确判据。snapshot剔除rows/hist/hb/ha/proposal/owner/digests/tape历史记录，保留环境真实history/witness/完整integration/Python属性/clock/step/previous/hidden/previous_raw/location。恢复后的记录缓冲从空开始，物理clock/previous/history/hidden继续原状态；source匹配使用固定全回合索引，不能把记录缓冲长度当车辆时间。所有后缀由真实MuJoCo观测更新驱动。

拟锁全部物理73,000步、9命令，main480秒、独立零nativeaudit180秒、完整原件RAM逐SHA备份240秒、源码120秒及60秒余量；own1MiB，global1GiB/free40GiB，CPU14/15总1核/GPUoff。T534先前38,900物理不抹除；若新窗口获授权并执行，本方向累计111,900物理，必须分列。这是明确工程修复的新增测试预算，不冒称原T534剩余预算完成或新算法机制。

全部四窗逐位一致才恢复门过，任何失败止本轮，不参数搜索/Actor训练。即使过也只资格，不是新增导航成功。原生参数方向与真正RL配对仍须下一协议。

现有窗口17:15原生停止/17:30全部停止，当前余时不足上述完整最坏准入；因此没有注册protocol或launch。需要用户确认一个新的有界窗口后才注册、预检、精确源同步和唯一启动。现代码仅完成语法编译，不声称真实native预检通过。
