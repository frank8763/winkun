# 2026-10-31追加归档与无损存储整理

- 新增T208–T223原文件480项，RAM生成44,101,657B压缩档上传Release，远端完整读回并逐成员SHA/大小核验；无磁盘临时归档。
- 本轮T225–T235已结束原文件308项（不含10个零字节锁），RAM归档19,708,483B，远端完整读回并逐成员核验；新源/失败/最终具名模型本地全部保留。
- 四T174/T175非选中失败Critic的原始模型/优化器等完整字节，历史原档完整远端SHA核验、各成员原字节核验后，本地以.pt.gz无损保留；不是重新torch.save、剥离优化器或删除失败证据，释放11,856,403B。
- 已结束且禁止续训的T170 checkpoint_latest.pt完整Actor/Critic/replay/优化器等原字节，同样经原档完整远端SHA与对应成员逐字节核验后，本地gzip保留，释放27,557,481B。
- 两项合计节省39,413,884B（约37.59MiB），全部原始payload仍在本地gzip内、并在原始远端归档中。当前使用的T159/T194/T221/T228/T231模型内容不修改。恢复旧原路径用stage/experiments/t224_archive/archive.py restore 原仓库相对路径，先检查1GiB逻辑容量和原SHA；也可在RAM解压读取，不产生新的磁盘副本。
- 本轮实验后全部newruns约1008.08MiB/1024MiB；相对开始净降低约15.80MiB。未静默增加1GiB额度、未创建散落home runs，未删除唯一数据/模型/失败，Phase和T145/T146只读区不动。

原路径/原字节SHA/压缩路径与SHA/远端原档收据：cold_storage_receipt.json、closed_resume_storage_receipt.json。新归档成员与Release收据：source_manifest.json、release_receipt.json、stability_source_manifest.json、stability_release_receipt.json。历史迁移和清理账本保留不覆写。


## 后续追加：完整历史payload无损冷存储

在不改原报告/旧删除账本的前提下：已结束T171完整断点原Release二进制完整读回、原成员每字节核验后，原始payload本地gzip保存，释放26,632,867B；已结束T167 learn同法释放16,168,026B。未torch重存或剥离Actor/Critic/replay/optimizer，当前T159/T194/T221/T231/T239/T240等保留策略不动。账本reverse_curriculum_storage_receipt.json、smdp_storage_receipt.json及archive.py restore的原SHA/1GiB守卫已保留。与前四失败Critic/T170的39,413,884B合计82,214,777B。

T236–T240654原文件与T242/T243/T245376原文件两Release均完整远端二进制读回及逐成员SHA/大小核验，本轮原失败/模型/数据不删除；归档全在RAM，不落磁盘临时大包。活动作业仍在stage/runs，总量守既有1GiB/free40GiB；历史数字不回改，准确当前大小以实时计算为准。Phase、原T145/T146未改动。


## 今夜有界预算预留（2026-10-31）

T165 gamma099/gamma2s和T169已结束全检查点经213930122B历史原始Release完整远端读回、原SHA与三个逐成员原字节匹配后，本地无损gzip保留所有Actor/Critic/replay/optimizer内容。释放173807286B（165.7555MiB）；没有删内容。原始路径/SHA/压缩路径/SHA与恢复守卫见overnight_reserve_storage_receipt.json及archive.py restore。T145/T146、T164暂停、具名最佳模型和Phase未改。全局newruns1GiB不变；T255新增峰值150MiB，启动前总量843.72MiB。
