# Stage 后续实验加速默认（用户2026-11-17授权）

用户要求“充分利用cpu和gpu；加快速度推进实验”“以后实验都加速做，多思考分析”。新实验默认采用已验证GPU float64循环学习，原生MuJoCo用独立CPU worker并行；GPU不适合的零梯度/原生任务不占用GPU。以实际吞吐和数值合同决定并行度，不以利用率或开更多线程冒充提速。历史已结束协议保持原样。

当前已核GPU为RTX5060 Laptop8GB，PyTorch2.11+cu128/sm120实际训练梯度路径已测，CPU/GPU梯度在单批最大差4.4e-16，GPU重复逐位；完整GPU Adam学习/影子/独立重构已在T622通过。不要把单批倍数外推为全实验倍数。

重要兼容层：导入旧t493_runtime会改CUDA_VISIBLE_DEVICES及os.sched_setaffinity，并绑14。新GPU进程需在导入后恢复授权可见设备；CPU进程采用t624_native.restore_parent及实际worker路径恢复原系统接口，启动后读每个PID的真实affinity。已实测8/10/12三原生worker、parent8/10/12/14。旧CPU1/2/4计时受单核mask限制，不可用作多核扩展证明。

CPU总400%低优先、GPU单Stage进程为当前已核配置；Phase完全隔离，不能读取Phase数据或抢其GPU计算。GPU发现其他compute则让出，不杀其他任务；启动前检查资源，每新协议仍以明确预算/有效截止为准，授权硬件并不自动扩大物理/命令/学习次数/存储。

新科学模块必须唯一名字tNNN_*，避免learning/recover/run/replay等通用模块被深层历史import劫持。真实返回先保存后派生，跨行为策略摘要区分物理量与proposal/ownership。失败保原始、零步恢复不算新科学成功、不自动重启旧run。
