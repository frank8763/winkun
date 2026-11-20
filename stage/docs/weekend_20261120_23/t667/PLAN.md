# T667：完整因果历史局部响应可用性门槛

T666四实际遗忘任务基准成功/候选碰撞、8full+8exact与历史数组精确复现，独立auditPASS，所有科学PID已退出。此项只用其现有全历史，0物理、0原生命令、0优化器/Actor/Value更新，CPU14单线程、4GiB无swap，600秒计算＋600秒独立重构，总30分钟内结束，存储4MiB。周末额度单列入LEDGER。

严格依RETENTION_PRIMARY_SOURCES既定门槛：每pair从reset到首次XY差1cm的索引等距取8个完整prefix（去重最多32）。H是原120输入完整外部前缀，D为同时间候选历史减基准历史的原输入差；唯一一次归一化在q.sequence内部，不额外加sensor噪声、不拼hidden，不把插值前缀当物理可达状态。两个模型分别从reset重建自己的GRU carry，q.sequence输出最后实际clip命令，float64 autograd JVP与eps=.001中心差分比较；每个真实实际历史secant另存。

预注册numeric abs1e-8+rel1e-4，输出clip kink距离1e-6显式标记；至少16个两模型均非零/非kink前缀、跨至少3对；相对secant残差按非零点累计二范数error/signal，每对两个模型都<25%，至少3对。阈值为工程反驳标准，不是稳定性定理。报告原始每前缀JVP/差分/secant及所有失败，不选择好看子集。full-own q.sequence与原locations必须误差<1e-10且执行float32动作逐位相等。候选和基准同完整初始/目标/任务/积分由T666来源审计保证；不动原始。

计算和独立重构结果字典必须完全一致，模型hash不变、参数grad无遗留；审计PASS仅指正确执行，门槛科学失败独立报告。PASS才准下一一次nominal-vs-response单块PPO pilot；FAIL就停此机制，不扫权重/差分幅度/阶数/seed去救结果。研究问题是网络对已观察历史偏离的响应，不是闭环物理导数或因果保证。本任务不自动启动训练。
