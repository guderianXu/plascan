# 标准文件与 JSON I/O

`plascan_common_file` 提供不依赖 Qt、OpenCV 或业务算法的 C++20 文件操作。
`plascan_common_json_io` 在其上增加 header-only 的 `nlohmann::json` 读写。
Qt 应用可以在调用边界转换 `QString`，算法直接使用 `std::filesystem::path`。

| 接口 | 行为 |
|---|---|
| `pathFromUtf8` / `pathToUtf8` | UTF-8 元数据与平台原生路径转换；Windows 使用宽字符路径 |
| `absoluteNormalizedPath` | 绝对路径与词法规范化，不解析别名，不用于删除/覆盖安全判定 |
| `ensureDirectory` | 创建目录及父目录；目标为普通文件或路径为空时报告错误 |
| `readFile` | 完整二进制读取普通文件；空文件成功，失败清空输出 |
| `AtomicFile` | 流式写入同目录唯一临时文件，成功提交后替换目标，未提交时析构清理 |
| `writeFileAtomic` | 完整字节串的原子写入 |
| `readJson` | 解析 UTF-8 JSON，保留整数精度；失败保持原 JSON 输出并报告路径与原因 |
| `writeJsonAtomic` | 紧凑 UTF-8 JSON 原子写入；序列化失败不会覆盖原文件 |

原子写入保留现有普通文件的权限；现有文件符号链接解析到其目标，保留链接本身。
悬空或不可解析链接会明确失败。POSIX 使用同目录 rename，Windows 使用宽字符 MoveFileExW
替换现有文件；分支仅位于原子提交的平台边界。并发写入使用不同的临时文件，最终目标为一次完整提交。
失败不删除旧目标，不回退为直接截断写入。

提交前关闭流并同步临时文件（POSIX fsync / Windows FlushFileBuffers）；同步失败保持旧目标。
这里的原子性针对单个目标文件的可见替换，不承诺目录项的断电持久性，也不是多文件事务。
新模块不包含递归删除、项目 token 解析、目录交易、影像解码或点云格式。
现有 `common/io/PathIO` 的安全路径比较与 Qt 调用接口仍保持原职责。

测试使用 `build/tmp/aerial-standard-io-tests/` 的独立目录，并在每个测试结束后清理。
