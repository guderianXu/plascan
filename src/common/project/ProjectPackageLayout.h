#pragma once

#include <QString>

namespace xjw::common::project
{

// PlaScan projects follow the same split layout as Metashape projects:
//   project.plascan
//   project.files/project.zip
//   project.files/1/
//   project.files/2/
class ProjectPackageLayout
{
public:
    static constexpr const char *DescriptorVersion = "4.0.0";
    static constexpr const char *DescriptorType = "plascan_project";
    static constexpr const char *ArchiveRelativePath =
        "{projectname}.files/project.zip";

    static QString dataDirectory(const QString &projectPath);
    static QString metadataArchivePath(const QString &projectPath);
    static QString sharedDirectory(const QString &projectPath);
    static QString sharedImagesDirectory(const QString &projectPath);
    static QString chunkDirectory(const QString &projectPath,
                                  int directoryNumber);
    static QString chunkArchivePath(const QString &projectPath,
                                    int directoryNumber);
    // 清理旧版本提前创建的空目录；包含任何条目的目录保持不变。
    static bool pruneEmptyOptionalDirectories(
        const QString &projectPath,
        int directoryNumber,
        QString *errorMessage = nullptr);
    static bool isChunkDirectoryName(const QString &name);

    // 兼容旧调用：返回首个数字 Chunk 目录，不再返回根级 workspace/。
    static QString workspaceDirectory(const QString &projectPath);
    static QString resourcesDirectory(const QString &projectPath);

    static bool isDescriptor(const QString &projectPath,
                             QString *errorMessage = nullptr);
    static bool writeDescriptor(const QString &projectPath,
                                QString *errorMessage = nullptr);

    // 仅验证当前 Chunk 分体布局；任何旧工程格式均不迁移。
    static bool ensureSplitLayout(const QString &projectPath,
                                  QString *errorMessage = nullptr);

    // 仅解析当前版本描述文件；非描述文件和旧版单体工程返回错误。
    static QString resolveMetadataArchive(const QString &projectPath,
                                          QString *errorMessage = nullptr);

private:
    static bool parseDescriptor(const QString &projectPath,
                                QString *relativeArchivePath,
                                QString *errorMessage);
};

} // namespace xjw::common::project
