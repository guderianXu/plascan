// =============================================================================
// 文件名: PlascanArchive.h
// 描述:   PlaScan 项目元数据归档（project.zip）的封装操作类。
//
//         .plascan 是轻量 XML 描述文件，实际 ZIP 位于同名
//         .files/project.zip。本类只接受当前分体工程描述文件。
//         project.zip 与各数字 Chunk 的 chunk.zip 都只保存 doc.json；
//         大文件以资源条目形式独立保存，不内嵌到 JSON 文档。
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
#include <QStringList>
#include <QPair>
#include <QVector>

enum class PlascanArchiveCompression
{
    Store,
    Deflate
};

enum class PlascanArchivePathType
{
    ProjectDescriptor,
    DirectArchive
};

// PlascanArchive: 轻量封装 project.zip 的常用操作
// 支持安全条目校验、流式文件写入/提取和批量原子更新。

class PlascanArchive
{
public:
    // 传入 .plascan 描述文件；构造时以只读方式打开对应 project.zip。
    explicit PlascanArchive(
        const QString &path,
        PlascanArchivePathType pathType =
            PlascanArchivePathType::ProjectDescriptor);
    // 析构时关闭 libzip 句柄，释放资源
    ~PlascanArchive();

    // 返回是否打开（或 libzip 可用）
    bool isValid() const;

    // 判断归档中是否存在指定安全条目。
    bool containsEntry(const QString &entryPath) const;

    // 列出容器内部所有路径（条目名列表）
    QVector<QString> listEntries();

    // 读取某个 entry 的原始字节（完全加载到内存，谨慎使用大文件）
    // entryPath: 归档内的相对路径（如 "doc.json"）
    // err: 可选，失败时写入错误信息
    QByteArray readEntry(const QString &entryPath, QString *err = nullptr);

    static bool createArchive(
        const QString &path,
        const QVector<QPair<QString, QByteArray>> &entries,
        QString *err = nullptr);

    // 写入或替换归档内的单个条目（若条目已存在则先删除再加入，实现原子替换）
    // entryPath: 归档内的目标路径（如 "doc.json"）
    // data:      要写入的字节内容
    // err:       失败时写入错误原因
    // 返回 true 表示写入成功
    bool writeEntry(const QString &entryPath,
                    const QByteArray &data,
                    QString *err = nullptr);
    bool writeEntries(const QVector<QPair<QString, QByteArray>> &entries,
                      QString *err = nullptr);

    // 从磁盘文件流式写入归档，不将完整文件载入内存。
    bool writeFileEntry(const QString &entryPath,
                        const QString &sourcePath,
                        PlascanArchiveCompression compression =
                            PlascanArchiveCompression::Store,
                        QString *err = nullptr);
    bool updateFileEntries(
        const QVector<QPair<QString, QString>> &fileEntries,
        const QStringList &entriesToDelete,
        PlascanArchiveCompression compression =
            PlascanArchiveCompression::Store,
        QString *err = nullptr);

    // 将归档条目流式提取到磁盘，使用 QSaveFile 原子替换目标文件。
    bool extractEntryToFile(const QString &entryPath,
                            const QString &destinationPath,
                            QString *err = nullptr);

private:
    void closeReadHandle();
    void reopenReadHandle();

    QString _path;       // 实际 project.zip 路径
    bool _valid{false};  // 是否成功打开（libzip 句柄有效）
    void *_impl{};       // 指向 zip_t* 的不透明指针，避免头文件暴露 libzip 类型
};
