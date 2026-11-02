# 两段五米成功视频

两段均为1920×1080、25fps、H.264/yuv420p/MP4，含全景与车辆近景、时间/剩余距离/原生命令、目标和已走轨迹。动作由同一冻结T255 s2_reset神经策略产生，正常导航1倍速；开始0.8秒和成功后1.6秒为展示暂停。

|任务|原生到达时间|视频时长|剩余距离|无接触成功|文件大小|直接下载|
|---|---:|---:|---:|---|---:|---|
|左侧5米|44.434s|46.84s|0.19999165m|是|7.80MiB|[MP4](https://github.com/frank8763/winkun/releases/download/stage-five-metre-videos-2026-11-02/stage_5m_left_1080p.mp4)|
|右侧5米|43.636s|46.04s|0.19993449m|是|7.97MiB|[MP4](https://github.com/frank8763/winkun/releases/download/stage-five-metre-videos-2026-11-02/stage_5m_right_1080p.mp4)|

两段为已知五米中心任务的真实原生复现：LEFT为dev_task6_13、RIGHT为dev_task7_13；每段全部10数组（含120维观测/下一观测）和终态积分SHA均与原五米验收逐位相同。独立审核再次验证全部电影可解码和哈希一致；两MP4独立Release附件均完整下载读回验证原字节。摄像机、起点/目标球和已走轨迹仅用于渲染，不向策略提供地图、未来路径或教师动作。

视频使用当前已验证五米基线，不能作为六米成功或新算法优越性证据。当前六米仍0/54，未进入七米/八米。

模型policy SHA：31b694d1ecf92dbcf7909f6ba6e73a808bc9725aeeacfb7584d0aab8fb162d54。

## 下载与PPT使用

将MP4直接下载到本地，再通过PPT的“插入视频”选择文件；不需要依赖网页在线播放。PNG为1920×1080可直接插图，SVG为矢量图源；PPT_CONTENT.md包含六页文案与讲解句。

## MP4 SHA256

- LEFT: eb57d8825106a14baa37812fcbe0d230813643c7911293d8549bf4ced960772c
- RIGHT: fc7fe3ae7b58fedfda179ff7db918625404f61aa8dbf7bafa4ef607cda2d0a68
