# T306固定冷checkpoint双字节流无损编码

只处理已经完整远端核验原始.pt并有本地gzip账本的旧已结束checkpoint：固定overnight_reserve/closed_resume/reverse_curriculum/smdp/cold_storage五份既有收据。原始Actor/Critic/replay/Adam全部保留，不删tensor、不挪目录/链接绕quota。T255/T299/T305输入、视频、轨迹和协议全部保护。

本轮0native/0优化器。固定旧resume/SMDP payload用XZ preset1；固定四个未选旧失败Critic用可逆4字节平面转置+XZ preset1。不是学习算法/压缩参数搜索。先在RAM核对原.pt SHA、旧.gz SHA、gzip6/mtime0再生成的原.gz逐字节完全相同，再验证新编码可逆；任一不符即不编码。

保留原始.pt及原.gz两种完整字节流的SHA、原始远端asset、既有完整核验收据、新编码SHA/字节数/算法和恢复工具。原始远端raw已验证，不声称新.lxz文件已上传；它可由原raw字节和固定配方重建。所有旧消费者以后要用本codec的load_original在RAM解码，或先核算完整空间再restore原路径，不能重启已结束作业。

因当前只余约5MiB、另存20–37MiB压缩副本会超quota，正式执行在完整RAM双重解码/远端原始收据核验后写durable per-file ledger，rename旧gzip到新.lxz路径，再原地覆盖缩短/fsync，不瞬时增加模型磁盘占用。若该原地写入中崩溃，该文件需要依据已验证旧远端raw恢复；ledger明确pending状态，部分文件不能称本地完整可恢复。这一中断风险和恢复途径写入协议，不静默删除唯一资料。无活动Stage进程时持独占锁、≤600秒、metadata≤512KiB，global≤1GiB/free≥40GiB。

编码前源/输入SHA锁定、零步预检、提交和精确远端树；结束确认main/timeout退出后独立逐模型重建原raw+原gzip、核对全部保护输入和先前远端收据。结果/账本/恢复代码进Git，原始模型仍在既有远端二进制归档，新metadata原始另备份。没有开始任何导航训练；释放空间后仍须新具体科学协议才能使用。
