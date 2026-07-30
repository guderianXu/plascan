#pragma once

#include "project/ProjectChunkIndex.h"
#include "project/ProjectLock.h"

#include <QJsonObject>
#include <QMap>
#include <QString>

#include <memory>

namespace xjw::common::project
{

// 无界面工程会话：为 CLI 提供与 GUI 相同的 4.0 Chunk 创建、打开、
// URI 解析和结果写回能力，不依赖 QWidget 或 GUI 配置对象。
class ProjectSession
{
public:
    ProjectSession() = default;
    ~ProjectSession();

    ProjectSession(const ProjectSession &) = delete;
    ProjectSession &operator=(const ProjectSession &) = delete;

    bool create(const QString &projectPath,
                const QString &projectName,
                QString *errorMessage = nullptr);
    bool open(const QString &projectPath,
              QString *errorMessage = nullptr);
    bool openOrCreate(const QString &projectPath,
                      const QString &projectName,
                      QString *errorMessage = nullptr);
    bool selectChunk(const QString &chunkId,
                     const QString &chunkName = QString(),
                     QString *errorMessage = nullptr);
    void close();

    bool isOpen() const;
    QString projectPath() const;
    QString projectId() const;
    ProjectChunkRecord activeChunk() const;
    QString activeChunkRoot() const;

    QJsonObject projectFiles() const;
    QJsonObject projectResults() const;
    QJsonObject projectConfig() const;
    QJsonObject mergedMetadata() const;

    // 合并输入影像。已有影像及其相机字段会保留；新影像自动获得稳定 UUID。
    bool mergeImages(const QJsonArray &images,
                     QString *errorMessage = nullptr);
    bool updateImageCameras(
        const QMap<QString, QJsonObject> &cameraMetaByImage,
        int *updatedCount = nullptr,
        QString *errorMessage = nullptr);
    void appendResult(const QString &arrayKey,
                      const QJsonObject &record);
    void upsertResultByPath(const QString &arrayKey,
                            const QString &pathKey,
                            const QJsonObject &record);

    // 保存时将 Chunk 根目录内的产物路径转换为 plascan:/// URI，
    // 并同步写入当前 Chunk 的 resource_index。
    bool save(QString *errorMessage = nullptr);

private:
    bool load(QString *errorMessage);
    QString normalizedImagePath(const QString &path) const;

    QString _projectPath;
    QString _projectId;
    ProjectChunkRecord _activeChunk;
    QString _activeChunkRoot;
    QJsonObject _projectFiles;
    QJsonObject _projectResults;
    QJsonObject _projectConfig;
    bool _runtimeRegistered = false;
    std::unique_ptr<ProjectLock> _lock;
};

} // namespace xjw::common::project
