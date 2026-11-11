# T483：替代本地无损日志存储准入，0训练/0原生

T481两次CLI读取与一次HTTP1批读取都在任何远端缓存删除前失败（EOF/SSL35）。原错误、开始marker、协议都保留，不再重试该远端删除机制。T481原生/Actor/Adam迄今为0。用户明确要求继续实验并延至今天20:30，本项只解决容量，不是新导航机制或扩大T481科学预算。

静态逐字节计划发现已结束且完整远端逐成员SHA收据齐全的T400–T480旧文本，可本地gzip约12.27MB；所有原字节仍在本地无损gzip和既有远端原档。仅选非当前输入、非模型/数组/摘要/协议/错误/失败/ledger等文件，不删除唯一资料。每条receipt+manifest+本地原SHA和结束证据须核验，落只读gzip、解码逐字节一致后才移除原裸文本。格式与既有t482_storage相同，独立新进程解码全部项；旧裸文本之后用t483_storage.text或受控restore。

注册计划后源精确同步；执行≤300s、独立审计≤180s、总600s，own1MiB/global1GiB/free40GiB。释放容量实际不足则停。不依赖新网络读回，因此不是对失败远端缓存准备的无期限延长。

审计全部原字节、T481所有注册输入SHA及完整22.5MiB根预留后，另写`T481_ALTERNATE_STORAGE_ADMISSION.json`和`cache_audit.json`：明确原远端cacheplan未执行、remote unlink=0、存储替代为本地无损编码。原协议科学配置/源SHA不修改，旧cache计划保留但不消费；不能伪称其缓存删除成功。之后仍必须原正式8丢弃Adam预检、成熟双Adam回退/worker零物理检查、精确源同步，才允许T481首次launch。原QUEUE_STOPPED不覆盖，首次训练另有真实launch。Phase隔离。
