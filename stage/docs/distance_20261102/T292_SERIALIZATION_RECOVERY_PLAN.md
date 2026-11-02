# T292最终JSON序列化恢复

原main在32轨迹/96梯度完成、evidence.npz/全部梯度SHA/密度/拆分写盘后，因NumPy布尔JSON序列化失败exit1；没有summary。全部父/inhibitor/timeout已退出，原status停GRADIENT_32及run.log错误逐SHA锁定，原源码/协议均不修改、不重新启动。

只读保存的Gram/预测/Gt重算标量指标并转换NumPy标量，新增summary明确state=FAILED/producer_failure，原status和error保留，不将原进程改称成功。后续独立审计正是原先注册1200秒/96全参数梯度重算，尚未运行过，不加重复次数/原生/Actor/优化器预算。独立读取全部原始输入重算8fold/32轨迹/96梯度/Gram/预测/拆分，与保存证据逐位/哈希核验。保留失败模型/证据、原own2MiB/global1GiB。先新恢复代码提交/远端精确同步，再后处理和唯一原注册审计。
