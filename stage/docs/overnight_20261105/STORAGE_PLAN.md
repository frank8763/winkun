# 通宵容量：五个旧冷检查点的可逆无损重编码

2026-11-05。0仿真/0优化器，不是新导航实验。固定T165 gamma099/gamma2s、T171、T170、T167五个已结束检查点；不动safeT255、T194、T299、人控、视频或当前S377的485输入及T379/T381/T382全部原件。原raw.pt已在既有完整远端归档中核验，T306本地raw/gzip/XZ独立审计通过。

先在RAM读取固定旧plain-XZ，验证raw、gzip6mtime0、旧XZpreset1三个原字节流逐SHA一致，再以固定XZpreset9EXTREME编码原raw。此编码是存储工程，不涉及模型张量或Adam改写。总净释放须≥18MiB，否则零修改拒绝。早先只读whole-file byteplane4探测使三文件膨胀，0修改，保留探测数值记录；不把失败方案落盘。

128KiB本项metadata、600秒写入/300秒独立audit、global1GiB/free40GiB。注册/零修改编码预检、源码提交与精确远端树后唯一启动。写前全RAM往返+原远端证明，journal fsync后rename旧文件到`.t383.rxz`并原位写入/截断，不产生超额度第二副本。写中断须保留partial/journal并以已核验原远端raw恢复，禁止称其已审计。全部原raw/gzip/旧lxz字节均可由新encoded和ledger逐位重建；新load_original/容量守卫恢复是后续读取入口，不能读不存在的旧裸路径或重启旧训练。

独立audit再次重建全部三个字节流、复核原备份与当前输入SHA后才将释放容量用于今晚新研究。新encoded二进制断网时remote PENDING，原raw远端备份仍保留；不删唯一研究数据，不默增额度。通宵链的容量必须在这一步完成后按实际字节重新admission。
