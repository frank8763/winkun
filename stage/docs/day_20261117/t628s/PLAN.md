# T628S 原生内存估算修正，原授权剩余额度继续

T628/T628R两个producer均已明确失败并永久保留，都是原生采集前，0物理/0学习。第二次是真实初始化OOM，不能只归咎RSS重复计数。R故障由独立metadata-only收尾，无原生产summary冒充；cgroup已清空，完整远端成员SHA已过。

本项修正内部RAM容量估算2GiB→4GiB真实cgroup上限，MemorySwapMax=0，禁止占用交换空间。用户明确限定的物理/命令/更新/主管线/本地磁盘/远端总额、CPU400%低优先和22:30/23:00/23:30截止全不扩大。可用系统RAM约25GB；4GiB是受控RAM工作区，不是磁盘预算，不改Phase任务。

原两个故障按12命令/34.689961秒保守上界全部扣除，剩260命令/3565.3100386秒；本地旧两根505874B+新根4212718B=4.5MiB；远端旧269615B+新100393681B=96MiB。新原生最多14263788物理、Actor live/shadow/history各最多8，0Critic/额外优化器预检。

科学合同完全沿用已批准T628：同T608初始函数/SGD、四固定任务、共享种子、每轮各4新学生+4exact、两步SGD、33自身成功历史函数保持、真实full-history/GRUcarry、fixed(.2) vs anneal(.1→.05)，每组最多两轮。每轮完整40+53final+3close命令准入，预算不足不启后轮；最终完整49+4keyexact，原34全保且新增原起点才保留。只将共同首轮作匹配比较，轮数不同不可称等预算最终噪声优势。部署单Actor120直接动作，无运行规划/地图/路点/teacher/router/Critic动作选择；不宣布全局稳定或创新成立。

轻量驱动保留，真实唯一cgroup而非RSS和守内存上限。GPU/采样密度/hash/CPUloader/历史IO与回退零native零optimizer预检重核，不新增科学数据；所有源码先GitHub精确根树同步后唯一新root启动，旧根不重启。实际启动和实际资源读MEMORY_ACCOUNTING/launch/status。
