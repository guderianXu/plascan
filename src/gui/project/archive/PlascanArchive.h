// =============================================================================
// 文件名: PlascanArchive.h
// 描述:   .plascan 格式归档（ZIP容器）的封装操作类。
//
//         .plascan 文件本质上是一个 ZIP 压缩包，包含以下标准条目：
//           manifest.json       <- 格式版本、类型标识
//           project.json        <- 旧版兼容条目（新版已迁移为以下两个）
//           project_files.json  <- 影像列表、流程结果数据
//           project_config.json <- 工作流设置、UI 配置
//
//         本类对 libzip C API 进行了轻量封装，隐藏 C API 细节，
//         提供读取、写入/替换单个条目等常用接口。
//
// 注意：
//   - 读操作（构造函数、readEntry、listEntries）以只读模式 (ZIP_RDONLY) 打开归档
//   - 写操作（writeEntry、createArchive）会重新以读写模式打开归档
//   - 若构建时未启用 libzip（无 HAVE_LIBZIP），所有操作均返回失败
//
// 线程安全：不线程安全，单个实例请在同一线程中使用
// =============================================================================
#pragma once

#include <QString>
#include <QVector>

// PlascanArchive: 轻量封装 .plascan（ZIP）容器的常用操作
// 目前实现：检查 libzip 可用性、列出条目、读取单个条目为 QByteArray
// 后续将实现打包/验证/原子替换等功能

class PlascanArchive
{
public:
    // 构造时以只读方式打开指定 .plascan 文件；isValid() 可检查是否打开成功
    explicit PlascanArchive(const QString &path);
    // 析构时关闭 libzip 句柄，释放资源
    ~PlascanArchive();

    // 返回是否打开（或 libzip 可用）
    bool isValid() const;

    // 列出容器内部所有路径（条目名列表）
    QVector<QString> listEntries();

    // 读取某个 entry 的原始字节（完全加载到内存，谨慎使用大文件）
    // entryPath: 归档内的相对路径（如 "project_files.json"）
    // err: 可选，失败时写入错误信息
    QByteArray readEntry(const QString &entryPath, QString *err = nullptr);

    // 静态方法：创建一个最小的 .plascan ZIP 容器，包含 manifest.json 与 project.json
    // path:         目标 .plascan 文件路径（已存在则截断重建）
    // manifestJson: manifest.json 的 JSON 字节内容
    // projectJson:  project.json / project_files.json 的 JSON 字节内容
    // err:          失败时写入错误原因
    // 返回 true 表示创建成功
    static bool createArchive(const QString &path,
                              const QByteArray &manifestJson,
                              const QByteArray &projectJson,
                              QString *err = nullptr);

    // 写入或替换归档内的单个条目（若条目已存在则先删除再加入，实现原子替换）
    // entryPath: 归档内的目标路径（如 "project_files.json"）
    // data:      要写入的字节内容
    // err:       失败时写入错误原因
    // 返回 true 表示写入成功
    bool writeEntry(const QString &entryPath,
                    const QByteArray &data,
                    QString *err = nullptr);

private:
    QString m_path;       // .plascan 文件的磁盘路径
    bool m_valid{false};  // 是否成功打开（libzip 句柄有效）
    void *m_impl{};       // 指向 zip_t* 的不透明指针，避免头文件暴露 libzip 类型
};
