# T311：为有限PPO对照预留空间的无损冷编码

固定范围：T263/T267/T280已停止轨迹中≥60000B、完整states/next_states逐行链一致、NumPy重新保存能精确生成原NPZ字节、非T310R注册输入的文件；以及T286完整远端逐成员读回中仍为gzip、≥3MiB、非当前输入/受保护T194/T255/T299/HD视频的旧checkpoint。按既有manifest验证原字节与完整远端收据。不是训练网格，也不选择/删除失败。

NPZ固定链消重+逐行字节XOR/转置+XZ preset1，必须逐字节恢复原NPZ（包括原成员顺序/压缩字节）。模型固定byteplane4+XZ1，全部模型/Adam/raw.pt及旧gzip两字节流精确保留，沿用已有完全远端备份。每项旧文件本地替换为.sxz，旧路径须用load_original显式RAM解码，或先核算容量后restore；不是symlink/移动绕限、不重启旧消费者。新输入包含所有现行T310R注册输入，不能被编码。

先只读编码/恢复/SHA核验、注册，至少预期释放30MiB才执行；600秒、metadata512KiB、0原生/0优化器、global1GiB/free40GiB不扩大。代码干净提交+精确远端根树后唯一launch，原生互斥锁排他；每文件先durable ledger，再rename现有gzip/NPZ后原地写入/truncate/fsync，不产生超额磁盘双份。写前完整原字节已在RAM和远端验证，崩溃时保留partial，根据既有完整远端member恢复，不能将partial称已本地恢复。

结束后独立audit重构每个原NPZ/raw模型/gzip及全部当前受保护输入SHA。新metadata/恢复账本完整备份，原二进制仍在旧完整远端收据，不能称新.sxz已上传。所有原始字节、失败/优化器/模型可恢复，不触碰Phase。实际空间达标后才另注册PPO/GAE主试验；本整理不启动导航。

首次只读容量拒绝：31714465B略低于原31MiB下限，0文件修改/0注册；源/错误在preflight_failed_001保留。后续主试验先按own43MiB上限重新核算，故注册前将纯存储释放下限设30MiB，保留metadata512KiB。不得将此改成新导航预算或声称已经释放。
