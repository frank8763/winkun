# T674：真实44任务冲突下的单次位移投影可行性

T673已完整独立重构auditPASS，全407名义保持+结果目标的首Adam提议真实伤害10个有结果信号的旧ID；原384仅44个本batch有非零结果梯度，另340无信号不能称保护。T670投影active空的无辨识结论保留，本项使用新全任务数据的真实冲突，不人为改权重制造冲突。

本项0物理0原生命令0Actor/Value/optimizer、无candidate导出。对T673已经保存的实际表示首Adam位移d0，44原旧结果梯度各按自身L2归一化成B，求最近Euclidean位移：min ||d-d0||²/2，B d>=0。用对偶NNLS min_lambda>=0 ||B.T lambda+d0||²求d=d0+B.Tlambda。NNLS是小型约束问题求解，不是Actor优化器更新；方法GEM类已有近邻，不称原创或安全保证。

预检覆盖真正active、inactive、秩缺失双向约束与真实完整history loader；科学执行核对所有primal/dual/stationarity/complementarity及实际表示位移，44任务与field24原分母不改。逐全407因果外部历史，原网络carry从reset重建，临时位移后复算样本surrogate与实际sigma latent条件KL，再精确恢复参数。仅一次固定投影，不扫步长/容差/权重/seed，不再做未经注册的线搜索救门。

必须同时通过：原提议确有旧冲突、投影位移确实改变(绝对范数>1e-8)、表示后44线性margin>=-1e-12、KKT误差<1e-10、field均值线性上升>1e-12、field固定数据非线性surrogate上升>1e-12、所有原旧固定数据非线性surrogate不下降(容差1e-12)、max task-average latentKL<=.01及finite。无信号旧任务即使surrogate为零也不是受保护证据。切线投影可能二阶失效，这是要检验的科学问题；失败保留并停止这项直接投影学习分支，不把auditPASS当科学通过、不调阈值救。

即使PROJECTION_FEASIBILITY_PASS，也仅支持进一步独立登记一个有限学习pilot，不自动获得闭环保留或上线资格。若失败，依据冲突/无信号与可重算进展机制另选一个主要变量；不空等用户，也不拼新组件。学习仍须samecandidate原384全保+新增+完整407/audit。

600main+600独立重构、1800秒根硬停(30min)，16MiB/global10GiB/free40GiB，CPU14/Nice19/4GiB无swap/CUDA关闭，资源冲突只等待本Stage、墙钟计费。数据与源码SHA冻结、先worstRESERVE、精确GitHub全树后唯一launch；main/audit所有数组、KKT与非线性结果精确。T673/T672所有科学PID退出后才本项，不修改旧源/协议或raw。
