# T396启动前固定六个旧Actor的可逆容量整理

2026-11-06，0导航／0优化器。用户已授权继续两轮真实纠正＋RL，以及此前GitHub迁移与本地整理。只处理T228／T231／T239／T240／T243／T245各一个已结束的final_Actor.pt.gz；先把六个旧目录全部原始文件RAM上传并完整二进制读回逐成员SHA，再采用已审计T321的ZIP payload XOR＋字节平面XZ1。参考为不改动的T211 original_round0（沿用T352同家族参考）；原gzip和内部raw ZIP逐字节可恢复，旧远端原件保留。

只读预估safeT255参考释放1654473B不足，未修改任何文件；同固定六文件采用T211家族参考实际可释放4954685B，原两次只读结果在T396_STORAGE_FEASIBILITY.json。不扩大冷文件集合，不改模型权重或科学预算。safeT255／T299／当前T391–T395模型与人控／视频／原失败证据全部保护；不碰T194。新编码以显式cold/codec.py load_original(path)在RAM恢复原gzip，不能读取已不存在的旧裸路径，不写链接绕额度。

固定六文件、600秒整理／300秒独立字节审计，metadata≤256KiB、global≤1GiB／free≥40GiB；逐文件write-ahead账本、原子rename后原位写入更小编码并fsync，崩溃保留partial与账本，原始完整远端仍可恢复。科学源提交／精确远端根树后唯一launch；结束独立审计并备份新metadata，原始二进制不进Git。所有原生已停止，导航只在容量最坏预留通过后另行启动；本整理不等于新导航实验开始。
