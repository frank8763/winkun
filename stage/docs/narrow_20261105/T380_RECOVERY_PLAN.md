# T380输出编码恢复：保留原失败，不重做producer拟合

2026-11-05。原T380在1.125374秒结束FAILED，0native/物理/Actor/Adam。3个模型的9个系数/归一化数组已完整写入models.npz；随后全部窗口预测NPZ为149992B，超出预注册单文件128KiB而拒绝，原failure/summary/源码协议保留不改。这个保存错误是工程失败，不是科学门槛失败或导航0%成功。

恢复使用原T380的**剩余256KiB合并输出额度、原all_work_deadline及原3次独立拟合审计额度**。original+recovery目录总逻辑<=256KiB，不增加native/优化器/拟合预算，不重启旧目录或修改旧源/协议。原T380模型/失败完整只读；新`stage/runs/t380_evidence_recovery`仅存恢复证据。

producer恢复不拟合：从已保存3组系数与归一化读取，重算固定窗口预测与描述统计。原目标pose本来就是SHA锁定原NPZ的next_poses按已存starts+offset选取；改为明确原数组引用（原NPZ路径/SHA、窗口索引、shape/dtype及恢复后原字节SHA），不重复存储9份目标副本。全部model/persistence预测及start索引仍完整保存<=128KiB；这是可精确恢复的表示变化，没有删去原始目标或失败记录。原model已经证明producer3拟合完成；原summary未知计数仍原样保留，不覆盖。

独立审计仅一次原预留3个解析拟合，调用原冻结compute，不用恢复系数代替拟合。逐位核对原3模型、所有预测与目标引用、描述统计及既定门槛；计数原producer3+恢复producer0+独立audit3，0native/Adam。独立结果另存，不假称原producer COMPLETE。

实现/源/输入/预算与简单静态检查后，干净提交/精确远端树，唯一恢复；每项最多300秒且总仍服从原T380launch+1200秒，不延长。恢复及审计完成、所有原进程退出后，两个目录一起RAM归档全远端二进制逐成员SHA，包含原错误/原协议、原模型及新完整证据。预算/时限不足则保留现状，不自动扩额。
