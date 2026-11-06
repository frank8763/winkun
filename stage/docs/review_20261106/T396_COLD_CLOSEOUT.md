# T396固定六旧Actor容量整理收尾

固定六目录1028原始文件已完整远端逐成员SHA；六个gzip/raw ZIP原字节以T211不变参考本地可逆delta编码保存，释放4954685B，独立audit PASS、8份metadata全远端逐SHA。新编码大文件未重新上传，原完整二进制远端仍保留。旧路径显式使用stage/experiments/t396_iterative_recovery/cold/codec.py load_original(path)在RAM恢复，不直接读取已不存在gzip，不用链接绕容量。safeT255/T299/T194、当前全部输入、模型、人控、视频与失败不动。两次无修改编码参考只读容量结果／T228归档schema拒绝／未提交草稿启动guard均保留，不是导航失败或重复训练。全部cold进程已停止。
