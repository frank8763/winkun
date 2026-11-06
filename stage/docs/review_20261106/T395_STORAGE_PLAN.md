# T395前置容量准备：固定5旧冷模型无损重编码

2026-11-06。用户已授权56→55→54→53→52秒训练，明确先通过磁盘最坏预算。沿既有迁移整理授权，只处理固定T169一个和T174/T175四个已结束旧冷检查点的编码，不是新导航实验，0native／0backward／0optimizer。safeT255、T299、当前T393模型、人控、高清五米视频、当前所有输入和T393/T394原件不动。

复用已审计T306的plain或byteplane4变换与T383的XZ preset9 EXTREME无损编码。每个原始raw.pt、原gzip6mtime0、原XZpreset1三条字节流都须SHA逐位可恢复，旧完整远端raw附件证据先核验。原始旧XZ先在RAM解码与重编码、还原全三流后计算固定5总净收益；不足8MiB则0文件修改终止，不临时扩大集合或算法网格。

通过源码提交和精确远端根树后唯一写入：先逐文件durable ledger，再原位rename／编码／fsync／完整反解核验，不复制大包到磁盘、不删唯一字节或模型张量。若中断，保留partial和ledger，依据已有逐SHA远端raw恢复，不能称部分编码成功。之后独立审计全三原始流与当前保护输入；新metadata与恢复账本远端逐成员读回核验后再考虑启动训练。新encoded二进制本地保留、旧完整raw远端保留，不冒称新大编码包已上传。

本环节metadata≤128KiB、prepare≤600秒／mutation≤600秒／independent audit≤300秒；0物理或优化器。全newruns≤1GiB、free≥40GiB，重编码全过程也遵守，不因物理盘充足扩额度。用`storage.py load_original(path)`显式恢复旧raw/gzip/XZ到RAM；原旧裸.lxz路径不再存在，禁止假装它可直接读取；如需落盘须先做完整容量预留，不重启旧实验。

训练源码/科学协议另登记；本容量准备即使成功，也不代表T395导航已启动，更不代表六米通过。固定集合、原变换与字节流SHA写入storage_protocol，所有错误／尝试保留。

原初次prepare固定5编码可释放8761172B，未过草定12MiB门并在0文件修改时拒绝；原源／错误保留storage_preflight_failed_001。对应导航正式范围仅52–56秒恢复段，54/275/留出未安排，完整最坏own12MiB足够；故仅容量admission改为8MiB净释放，固定文件／编码／原字节和0native0Adam均不变，不增加冷集合或训练预算。
