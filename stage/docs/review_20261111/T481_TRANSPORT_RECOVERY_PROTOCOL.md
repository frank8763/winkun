# T481传输准备失败的有界恢复

2026-11-11 13:48，用户授权继续并安排至20:30。原T481队列在cache远端二进制CLI读回返回非0时停止，原错误/源/协议保留。此时尚无cache_remote_verified、cache_summary、preflight或launch；0native/0优化器、0缓存删除。99原日志本地可逆编码及独立audit已完成，实际释放3,686,336B。

反馈入口：同receipt首个asset，用原GitHub CLI二进制命令读回，比较返回码/字节数/SHA。原错误为`Existing GitHub CLI binary readback failed`；13:48实测返回0、524288B与SHA精确。首轮CLI查找误用PATH失败仅本地查找，未发送请求；之后使用仓库已登记gh路径。

三个可区分假设：①临时传输故障，重新读应可成功；②认证持续失效，应反复401；③原附件/清单损坏，应返回0但SHA/长度失败。目前首块完整读回支持①而不支持②③，全部归档仍须重新逐SHA才允许删除。

只调用已注册`t481_cache.evict(recovery=True)`：原始硬截止epoch1794376959.3351662不动、原选择清单不变，完整归档全部成员重新核验后才按既有账本移除缓存。只有一次恢复尝试，不重启原队列、不改科学源/协议/seed/epoch/1GiB、无新原生/优化器。失败即保留原始并停；成功仍须独立cache_audit和原正式8丢弃Adam预检，之后精确源码树同步才允许原T481首次launch。后续独立流程须写新的恢复记录，不覆盖原QUEUE_STOPPED。

## 13:54传输修正登记

原同CLI恢复再次EOF失败、未有任何cache_remote_verified/ledger/删除。进一步最小复核：T479/T478各首三块均SHA通过，T477块1/2的原CLI直接返回`Get ... EOF`，排除“仅凭第一次单块就认定全远端健康”的推断。新登记一次HTTP1 curl读取路径，仍在**同原epoch1794376959.3351662截止**，不加准备时间/物理/优化器。使用仓库既有curl配置pipe/已允许redirect主机，令原`t460_cache.archive(cli=False)`保持全部归档和成员SHA验证；仅取回传输方式改变。已验证原EOF块HTTP1读取成功才允许执行。全部组全成员核验完毕后才允许原选择清单删除，使用同一恢复marker/ledger格式，原所有失败原件不覆盖。独立cache_audit仍必须单独进程通过；再次失败则停止，不再恢复本准备机制。
