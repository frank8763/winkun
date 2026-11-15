# T570R：零步启动隔离修复

T570在launch/worker/梯度/原生/优化器之前因sched_getaffinity未被AllowedCPUs自动改变而拒绝，旧protocol/source/entry_failure/journal完整保留，不重启旧目录。新T570R仅明确在systemd命令中加taskset -c 14,15；两子worker各14/15，科学两实际历史、24梯度评估/0更新/0原生、240秒/256KiB不变，不增加旧失败机制的科学预算。累计启动/worker命令上界5（旧失败1+新4），main失败入口约1秒另计，不影响日600/12300余额。原17:30全部截止不扩。

T570 PLAN及CONTINUAL_LEARNING_PRIMARY_REVIEW为方法依据；本轮为工程并发检查，不是新导航学习或原生/GPU基准。
