# T390R：零训练入口错误的受限恢复

T390首次主入口在multiprocessing序列化动态导入模块的worker函数时失败；实际commands=0/physics=0/两臂live=shadow=0，没有任何导航训练或worker就绪。原源码、协议、summary、failure和driver error全部保留，结束目录不重启。此前8个丢弃Adam预检有效，不重复克隆更新。

修复仅将传给spawn的入口改为当前脚本可序列化的顶层包装，它直接调用原N.native_worker。两臂dataset/fit/参数哈希/独立audit函数AST与原源码逐项完全一致，科学配置、1000live/1000shadow/1000audit每臂和模型初始化不变。新增真实两worker启动/ready/CPU亲和/标准流恢复/零退出预检，2geometry/0physics/0Adam，明确计入共同1000命令。主剩余max_commands=998，原physics上限30M不增加。

原T390+恢复T390R共同own12MiB、global1GiB/free40GiB；恢复reservation扣除原失败目录全部实际字节。学习、原生、全部截止分别取原T390 main_started/driver_started绝对时间，不从新编号重新计时；源/协议按实际绝对epoch锁定。main原相对上限继续与原绝对截止取更早值，driver外层timeout与每阶段也从严。没有足够剩余时间不启动；不是给失败科学机制续预算。

零原生工作与学习函数不变预检通过、干净提交及精确远端根树后唯一启动。原T390停止证据、T389原核验和T390R实际结果一起RAM远端完整读回逐成员SHA备份。所有主/worker退出后唯一独立2000历史Adam重构，既有8预检+本次0预检单列。结束仍只汇报自主单Actor真实结果，教师或工程PASS不是六米成功；不7/8，无运行时地图/规划器/路点/教师/路由，Stage/Phase隔离。
