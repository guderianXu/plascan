// =============================================================================
// 文件名: ProjectData.cpp
// 描述:   ProjectData 数据层实现。
//
//         主要逻辑：
//           1. createProject  - 初始化 .plascan + .files 工程结构
//           2. openProject    - 优先从临时缓存恢复，再读归档
//           3. saveProject    - 将内存元数据写回当前 Chunk 的 chunk.zip
//           4. addImages      - 追加影像引用到 images[] 数组
//           5. setImageCameras- 批量写入相机参数到 images[*].camera
//           6. appendXxx      - 各类结果追加，统一写入 project_results.json
//           7. saveIpfindSettings - 更新可复现的工作流配置
//           8. saveUiSettings     - 更新独立的项目视图状态
//
//         持久化策略（双保险）：
//           - 运行时变更写 .plascan_tmp/ 做崩溃恢复，并通过防抖同步元数据
//           - 归档写失败时保留 .plascan_tmp/（防止数据丢失）
//           - 下次 openProject 时优先从 .plascan_tmp/ 恢复
// =============================================================================
#include "project/ProjectSessionModel.h"
#include "PlascanArchive.h"
#include "ProjectChunkStore.h"
#include "ProjectWorkspaceStore.h"
#include "ProjectUiConfigManager.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectChunkIndex.h"
#include "project/ProjectLock.h"
#include "project/ProjectSharedImageStore.h"
#include "project/PortableProjectFormat.h"
#include "project/ProjectIO.h"
#include "Logger.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QCoreApplication>
#include <QMetaObject>
#include <QPointer>
#include <QJsonDocument>
#include <QJsonArray>
#include <QSaveFile>
#include <QScopeGuard>
#include <QDateTime>
#include <QSet>
#include <QThreadPool>
#include <QTimer>
#include <QUuid>
#include <QtConcurrent/QtConcurrent>

using xjw::common::project::ProjectIO;
using xjw::common::project::ProjectChunkIndex;
using xjw::common::project::ProjectChunkRecord;
using xjw::common::project::ProjectPackageLayout;
using xjw::common::project::PortableProjectFormat;
using xjw::common::project::ProjectResourceIndex;
using xjw::common::project::ProjectLock;
using xjw::common::project::ProjectSharedImageStore;

struct ProjectData::PersistenceSnapshot
{
    PersistenceMode mode = PersistenceMode::TemporaryOnly;
    QString projectPath;
    QString chunkId;
    int chunkDirectory = 0;
    QString temporaryCorePath;
    QString temporaryResultsPath;
    QString temporaryConfigPath;
    QString temporaryUiStatePath;
    QJsonObject core;
    QJsonObject results;
    QJsonObject config;
    QJsonObject uiState;
    QJsonObject resourceIndex;
    bool resultsLoaded = false;
    bool writeCoreToArchive = false;
    bool writeResultsToArchive = false;
    bool writeConfigToArchive = false;
    bool writeUiStateToArchive = false;
    bool writeWorkspaceIndex = false;
    bool writeTemporary = false;
};

struct ProjectData::PersistenceResult
{
    PersistenceMode mode = PersistenceMode::TemporaryOnly;
    QString projectPath;
    QString errorMessage;
    bool success = false;
    bool archiveRequested = false;
    bool archiveSuccess = false;
    bool temporaryRequested = false;
    bool temporarySuccess = false;
    bool includedCore = false;
    bool includedResults = false;
    bool includedConfig = false;
    bool includedUiState = false;
    bool includedWorkspace = false;
    bool projectUriMetadata = false;
    QJsonObject projectUriCore;
    QJsonObject projectUriResults;
};

namespace {

QJsonObject versionedResultRecord(const QJsonObject &record)
{
    QJsonObject versioned = record;
    if (!versioned.contains(QStringLiteral("schema_version")))
    {
        versioned[QStringLiteral("schema_version")] = 1;
    }
    return versioned;
}

QString normalizedResultPath(const QString &projectPath, const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return QString();
    }
    const QString resolved =
        ProjectIO::resolveProjectResourcePath(projectPath, path.trimmed());
    QString normalized = QDir::cleanPath(QFileInfo(resolved).absoluteFilePath());
#ifdef Q_OS_WIN
    normalized = normalized.toCaseFolded();
#endif
    return normalized;
}

int chunkTiePointCount(const QJsonObject &projectResults)
{
    const QJsonArray results = projectResults.value(
        QStringLiteral("aerial_triangulation_results")).toArray();
    for (int index = results.size() - 1; index >= 0; --index)
    {
        const QJsonObject result = results.at(index).toObject();
        const QString sparsePath = result.value(QStringLiteral("files"))
                                       .toObject()
                                       .value(QStringLiteral(
                                           "sparse_cloud_xyz"))
                                       .toString()
                                       .trimmed();
        if (sparsePath.isEmpty())
        {
            continue;
        }

        int pointCount = result.value(
            QStringLiteral("sparse_point_count")).toInt(-1);
        if (pointCount < 0)
        {
            pointCount = result.value(
                QStringLiteral("point_count")).toInt(-1);
        }
        if (pointCount < 0)
        {
            pointCount = result.value(QStringLiteral("quality"))
                             .toObject()
                             .value(QStringLiteral("point_count"))
                             .toInt(-1);
        }
        return pointCount;
    }
    return -1;
}

bool writeFileAtomically(const QString &path,
                         const QByteArray &data,
                         QString *errorMessage)
{
    if (path.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("临时文件路径为空");
        }
        return false;
    }

    QDir().mkpath(QFileInfo(path).absolutePath());
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入临时文件: %1").arg(path);
        }
        return false;
    }
    if (file.write(data) != data.size() || !file.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("提交临时文件失败: %1").arg(path);
        }
        return false;
    }
    return true;
}

QJsonDocument parseJsonOrCompressedJson(const QByteArray &bytes)
{
    if (bytes.isEmpty())
    {
        return QJsonDocument();
    }

    if (static_cast<unsigned char>(bytes[0]) != '{')
    {
        const QByteArray uncompressed = qUncompress(bytes);
        if (!uncompressed.isEmpty())
        {
            const QJsonDocument doc = QJsonDocument::fromJson(uncompressed);
            if (!doc.isNull())
            {
                return doc;
            }
        }
    }

    return QJsonDocument::fromJson(bytes);
}

bool containsResultKeys(const QJsonObject &meta)
{
    for (auto it = meta.constBegin(); it != meta.constEnd(); ++it)
    {
        if (ProjectFilesManager::isResultKey(it.key()))
        {
            return true;
        }
    }
    return false;
}

QJsonObject defaultProjectUiState()
{
    ProjectUiConfigManager manager;
    manager.setData(ProjectUiConfigManager::defaultUiSettings());
    return QJsonObject{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("display_settings"), manager.data()}
    };
}

QJsonObject normalizedProjectUiState(const QJsonObject &state)
{
    QJsonObject normalized = state;
    ProjectUiConfigManager manager;
    manager.setData(ProjectUiConfigManager::defaultUiSettings());
    manager.applyPatch(
        state.value(QStringLiteral("display_settings")).toObject());
    normalized[QStringLiteral("schema_version")] = 1;
    normalized[QStringLiteral("display_settings")] = manager.data();
    return normalized;
}

QString normalizedProjectResourcePath(const QString &projectRoot, const QString &path)
{
    const QString cleanPath = QDir::cleanPath(path.trimmed());
    if (cleanPath.isEmpty())
    {
        return QString();
    }

    if (QFileInfo(cleanPath).isAbsolute())
    {
        return QDir::cleanPath(QFileInfo(cleanPath).absoluteFilePath());
    }

    if (projectRoot.trimmed().isEmpty())
    {
        return cleanPath;
    }

    return QDir::cleanPath(QDir(projectRoot).filePath(cleanPath));
}

QJsonObject readJsonObjectFile(const QString &path)
{
    if (path.isEmpty() || !QFileInfo::exists(path))
    {
        return QJsonObject();
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return QJsonObject();
    }

    const QJsonDocument doc = parseJsonOrCompressedJson(file.readAll());
    return doc.isObject() ? doc.object() : QJsonObject();
}

bool ensureImageUuids(QJsonObject *core)
{
    if (!core)
    {
        return false;
    }

    QJsonArray images = core->value(QStringLiteral("images")).toArray();
    QSet<QString> used_ids;
    bool changed = false;
    for (int index = 0; index < images.size(); ++index)
    {
        QJsonObject image = images[index].toObject();
        QString image_id = image.value(QStringLiteral("image_uuid")).toString().trimmed();
        if (image_id.isEmpty() || used_ids.contains(image_id))
        {
            do
            {
                image_id = QUuid::createUuid().toString(QUuid::WithoutBraces);
            }
            while (used_ids.contains(image_id));
            image[QStringLiteral("image_uuid")] = image_id;
            images[index] = image;
            changed = true;
        }
        used_ids.insert(image_id);
    }

    if (changed)
    {
        (*core)[QStringLiteral("images")] = images;
    }
    return changed;
}

} // namespace

ProjectData::ProjectData(QObject *parent)
    : QObject(parent)
{
    // 防抖归档写入定时器：将多次 appendIpfind/appendIpmatch/setImageCameras 合并为一次 ZIP 写入
    _archiveSyncTimer = new QTimer(this);
    _archiveSyncTimer->setSingleShot(true);
    connect(_archiveSyncTimer, &QTimer::timeout, this, &ProjectData::syncToArchive);

    _persistencePool = new QThreadPool(this);
    _persistencePool->setMaxThreadCount(1);
    _persistencePool->setExpiryTimeout(-1);
}

ProjectData::~ProjectData()
{
    _shuttingDown = true;
    if (_archiveSyncTimer)
    {
        _archiveSyncTimer->stop();
    }
    if (_persistencePool)
    {
        _persistencePool->clear();
        _persistencePool->waitForDone();
    }
    if (!_projectPath.isEmpty())
    {
        ProjectWorkspaceStore(_projectPath).releaseRuntime();
    }
}

void ProjectData::markDirtyIfRequested(bool markDirty)
{
    if (markDirty && !_isDirty)
    {
        _isDirty = true;
        emit dirtyStateChanged(true);
    }
}

void ProjectData::emitCurrentMetadataChanged()
{
    emit metadataChanged(_filesManager.data());
}

void ProjectData::scheduleArchiveSync(bool coreDirty,
                                      bool resultsDirty,
                                      bool writeTemporary,
                                      bool configDirty,
                                      bool uiStateDirty,
                                      bool workspaceDirty)
{
    if (coreDirty)
    {
        _coreFileDirtyForArchive = true;
    }
    if (resultsDirty)
    {
        _resultsDirtyForArchive = true;
    }
    if (configDirty)
    {
        _configDirtyForArchive = true;
    }
    if (uiStateDirty)
    {
        _uiStateDirtyForArchive = true;
    }
    if (workspaceDirty)
    {
        _workspaceDirtyForArchive = true;
    }

    if ((coreDirty || resultsDirty || configDirty || uiStateDirty || workspaceDirty)
        && _archiveSyncTimer
        && QCoreApplication::instance())
    {
        _archiveSyncTimer->start(2000);
    }

    if (writeTemporary)
    {
        scheduleTemporaryMetadataSave();
    }
}

bool ProjectData::createProject(const QString &plascanPath, const QString &projectName)
{
    const QString dataDirectory =
        ProjectPackageLayout::dataDirectory(plascanPath);
    const QString archivePath =
        ProjectPackageLayout::metadataArchivePath(plascanPath);
    if (QFileInfo::exists(plascanPath)
        || QFileInfo::exists(dataDirectory))
    {
        LOG_ERROR(QStringLiteral("项目文件或同名数据目录已存在: %1")
                      .arg(plascanPath));
        return false;
    }
    if (!QDir().mkpath(dataDirectory))
    {
        LOG_ERROR(QStringLiteral("无法创建项目数据目录: %1")
                      .arg(dataDirectory));
        return false;
    }
    auto newProjectLock = std::make_unique<ProjectLock>();
    QString lockError;
    if (!newProjectLock->acquire(plascanPath, &lockError))
    {
        LOG_ERROR(lockError);
        QDir(dataDirectory).removeRecursively();
        return false;
    }
    const auto cleanupCreatedProject = [&]()
    {
        if (newProjectLock)
        {
            newProjectLock->release();
            newProjectLock.reset();
        }
        QFile::remove(plascanPath);
        QDir(dataDirectory).removeRecursively();
    };

    const QString projectId = PortableProjectFormat::createProjectId();

    // 步骤1：构建 Chunk 核心元数据（空影像列表等）
    QJsonObject filesMeta = ProjectFilesManager::defaultFiles();

    // 步骤2：构建 Chunk 工作流配置，记录项目名称与创建时间
    QJsonObject configMeta = ProjectConfigManager::defaultConfig();
    configMeta["project_name"] = projectName;
    configMeta["created_at"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    configMeta["version"] = QString::fromLatin1(
        PortableProjectFormat::CurrentFormatVersion);
    configMeta["schema_version"] = 2;
    configMeta["project_id"] = projectId;

    // 步骤3：构建项目根文档和默认 Chunk 文档。
    const ProjectChunkIndex chunkIndex =
        ProjectChunkIndex::createInitial();
    const ProjectChunkRecord initialChunk = chunkIndex.defaultChunk();
    const QString chunkDirectory =
        ProjectPackageLayout::chunkDirectory(
            plascanPath, initialChunk.directory);
    const QString chunkArchivePath =
        ProjectPackageLayout::chunkArchivePath(
            plascanPath, initialChunk.directory);
    if (!QDir().mkpath(chunkDirectory))
    {
        LOG_ERROR(QStringLiteral("无法创建默认 Chunk 目录: %1")
                      .arg(chunkDirectory));
        cleanupCreatedProject();
        return false;
    }

    // 步骤4：project.zip 与 1/chunk.zip 各只保存一个 doc.json。
    QString err;
    const QJsonObject initialUiState = defaultProjectUiState();
    const QJsonObject projectDocument =
        PortableProjectFormat::createProjectDocument(
            projectId, chunkIndex, initialUiState);
    const QJsonObject chunkDocument =
        PortableProjectFormat::createChunkDocument(
            initialChunk,
            filesMeta,
            ProjectFilesManager::defaultResults(),
            configMeta,
            ProjectResourceIndex().toJson());
    if (!PlascanArchive::createArchive(
            chunkArchivePath,
            {
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::DocumentEntry),
                    QJsonDocument(chunkDocument)
                        .toJson(QJsonDocument::Compact))
            },
            &err)
        || !PlascanArchive::createArchive(
            archivePath,
            {
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::DocumentEntry),
                    QJsonDocument(projectDocument)
                        .toJson(QJsonDocument::Compact))
            },
            &err))
    {
        LOG_ERROR(QStringLiteral("初始化 Chunk 工程归档失败: %1").arg(err));
        cleanupCreatedProject();
        return false;
    }
    if (!ProjectPackageLayout::writeDescriptor(plascanPath, &err))
    {
        LOG_ERROR(QStringLiteral("创建项目描述文件失败: %1").arg(err));
        cleanupCreatedProject();
        return false;
    }

    ProjectWorkspaceStore workspace(
        plascanPath, initialChunk.directory);
    QString workspaceError;
    if (!workspace.validateProjectLayout(&workspaceError))
    {
        LOG_ERROR(QStringLiteral("完成项目布局初始化失败: %1")
                      .arg(workspaceError));
        cleanupCreatedProject();
        return false;
    }
    if (!workspace.initializeRuntime(nullptr, &workspaceError))
    {
        LOG_ERROR(QStringLiteral("初始化项目运行工作区失败: %1")
                      .arg(workspaceError));
        cleanupCreatedProject();
        return false;
    }

    // 目标项目已完整初始化后再结束旧会话。projectClosed 会使 GUI、异步刷新
    // 和待归档状态统一失效，防止旧项目资源进入新项目。
    if (hasProject())
    {
        closeProject();
    }

    // 步骤6：更新内存状态
    _projectLock = std::move(newProjectLock);
    _projectPath = plascanPath;
    _activeChunkId = initialChunk.id;
    _activeChunkName = initialChunk.name;
    _activeChunkDirectory = initialChunk.directory;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    _projectUiState = defaultProjectUiState();
    _uiStateDirtyForArchive = false;
    updateMetadata(filesMeta, false);   // false = 不标记为脏（刚创建，不需要保存）
    updateConfig(configMeta, false);
    scheduleTemporaryMetadataSave();

    LOG_INFO(QStringLiteral("项目创建成功: %1").arg(plascanPath));
    // 发出 projectOpened 信号（新建项目视同已打开）
    emit projectOpened(plascanPath);
    emit activeChunkChanged(
        _activeChunkId, _activeChunkName, _activeChunkDirectory);
    emit chunkListChanged(chunks(), _activeChunkId);

    return true;
}

bool ProjectData::openProject(const QString &plascanPath, QString *errorMsg)
{
    const ProjectOpenSnapshot snapshot = loadProjectOpenSnapshot(plascanPath);
    return openProjectFromSnapshot(snapshot, errorMsg);
}

QJsonArray ProjectData::chunks() const
{
    QJsonArray result;
    if (_projectPath.trimmed().isEmpty())
    {
        return result;
    }
    QString error;
    const ProjectChunkStore store(_projectPath);
    const QList<ProjectChunkRecord> records = store.chunks(&error);
    if (!error.isEmpty())
    {
        LOG_ERROR(QStringLiteral("读取 Chunk 列表失败: %1").arg(error));
        return result;
    }
    for (const ProjectChunkRecord &record : records)
    {
        QJsonObject chunk = record.toJson();
        QJsonObject document;
        QString documentError;
        if (store.readChunkDocument(
                record.directory, &document, &documentError))
        {
            const QJsonObject projectFiles = document.value(
                QString::fromLatin1(
                    PortableProjectFormat::ProjectFilesSection))
                                                 .toObject();
            const QJsonObject projectResults = document.value(
                QString::fromLatin1(
                    PortableProjectFormat::ProjectResultsSection))
                                                   .toObject();
            chunk[QStringLiteral("image_count")] = projectFiles.value(
                QStringLiteral("images")).toArray().size();
            const int tiePointCount =
                chunkTiePointCount(projectResults);
            if (tiePointCount >= 0)
            {
                chunk[QStringLiteral("tie_point_count")] =
                    tiePointCount;
            }
        }
        else
        {
            LOG_WARN(
                QStringLiteral("读取 Chunk 工作区摘要失败: %1")
                    .arg(documentError));
        }
        result.append(chunk);
    }
    return result;
}

bool ProjectData::createChunk(
    const QString &name,
    QString *createdChunkId,
    QString *errorMsg)
{
    if (_projectPath.trimmed().isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有打开的项目");
        }
        return false;
    }
    QString saveError;
    if (!saveProject(&saveError))
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("创建 Chunk 前保存当前 Chunk 失败: %1")
                    .arg(saveError);
        }
        return false;
    }

    QJsonObject config = _configManager.data();
    config[QStringLiteral("created_at")] =
        QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    ProjectChunkRecord created;
    ProjectChunkStore store(_projectPath);
    if (!store.createChunk(
            name,
            ProjectFilesManager::defaultFiles(),
            ProjectFilesManager::defaultResults(),
            config,
            ProjectResourceIndex().toJson(),
            &created,
            errorMsg))
    {
        return false;
    }
    if (createdChunkId)
    {
        *createdChunkId = created.id;
    }
    if (!switchChunk(created.id, errorMsg))
    {
        return false;
    }
    return true;
}

bool ProjectData::renameChunk(
    const QString &chunkId,
    const QString &name,
    QString *errorMsg)
{
    if (_projectPath.trimmed().isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有打开的项目");
        }
        return false;
    }
    ProjectChunkStore store(_projectPath);
    if (!store.renameChunk(chunkId, name, errorMsg))
    {
        return false;
    }
    if (_activeChunkId == chunkId)
    {
        ProjectChunkIndex index;
        if (!store.loadIndex(&index, errorMsg))
        {
            return false;
        }
        const ProjectChunkRecord renamed = index.chunk(chunkId);
        _activeChunkName = renamed.name;
        emit activeChunkChanged(
            renamed.id, renamed.name, renamed.directory);
    }
    emit chunkListChanged(chunks(), _activeChunkId);
    return true;
}

bool ProjectData::removeChunk(
    const QString &chunkId,
    QString *errorMsg)
{
    if (_projectPath.trimmed().isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有打开的项目");
        }
        return false;
    }
    QString saveError;
    if (!saveProject(&saveError))
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("删除 Chunk 前保存当前 Chunk 失败: %1")
                    .arg(saveError);
        }
        return false;
    }

    const bool removingActive = _activeChunkId == chunkId;
    ProjectChunkStore store(_projectPath);
    if (!store.removeChunk(chunkId, errorMsg))
    {
        return false;
    }
    if (removingActive)
    {
        const ProjectOpenSnapshot snapshot =
            loadProjectOpenSnapshot(_projectPath);
        if (!openProjectFromSnapshot(snapshot, errorMsg))
        {
            return false;
        }
    }
    else
    {
        emit chunkListChanged(chunks(), _activeChunkId);
    }
    return true;
}

bool ProjectData::switchChunk(
    const QString &chunkId,
    QString *errorMsg)
{
    if (_projectPath.trimmed().isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有打开的项目");
        }
        return false;
    }
    if (chunkId == _activeChunkId)
    {
        return true;
    }
    QString saveError;
    if (!saveProject(&saveError))
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("切换 Chunk 前保存当前 Chunk 失败: %1")
                    .arg(saveError);
        }
        return false;
    }

    ProjectChunkStore store(_projectPath);
    const QString previousChunkId = _activeChunkId;
    if (!store.setDefaultChunk(chunkId, errorMsg))
    {
        return false;
    }
    const ProjectOpenSnapshot snapshot =
        loadProjectOpenSnapshot(_projectPath);
    if (!openProjectFromSnapshot(snapshot, errorMsg))
    {
        QString rollbackError;
        store.setDefaultChunk(previousChunkId, &rollbackError);
        return false;
    }
    return true;
}

ProjectOpenSnapshot ProjectData::loadProjectOpenSnapshot(const QString &plascanPath)
{
    ProjectOpenSnapshot snapshot;
    snapshot.projectPath = QDir::cleanPath(QFileInfo(plascanPath).absoluteFilePath());

    QString layoutError;
    if (ProjectPackageLayout::resolveMetadataArchive(
            snapshot.projectPath, &layoutError).isEmpty())
    {
        snapshot.errorMessage = layoutError.isEmpty()
            ? QStringLiteral("无法解析项目数据目录")
            : layoutError;
        return snapshot;
    }

    ProjectChunkStore chunkStore(snapshot.projectPath);
    if (!chunkStore.ensureLayout(&snapshot.errorMessage))
    {
        return snapshot;
    }
    const ProjectChunkRecord chunk =
        chunkStore.defaultChunk(&snapshot.errorMessage);
    if (chunk.id.isEmpty())
    {
        return snapshot;
    }
    snapshot.chunkId = chunk.id;
    snapshot.chunkName = chunk.name;
    snapshot.chunkDirectory = chunk.directory;

    ProjectWorkspaceStore workspace(
        snapshot.projectPath, snapshot.chunkDirectory);
    if (!workspace.initializeRuntime(nullptr, &snapshot.errorMessage))
    {
        return snapshot;
    }
    const auto runtimeGuard = qScopeGuard(
        [&workspace]()
        {
            workspace.releaseRuntime();
        });

    QJsonObject projectDocument;
    QJsonObject chunkDocument;
    if (!chunkStore.loadProjectDocument(
            &projectDocument, &snapshot.errorMessage)
        || !chunkStore.readDefaultChunkDocument(
            &chunkDocument, &snapshot.errorMessage))
    {
        return snapshot;
    }

    snapshot.filesMeta = chunkDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectFilesSection)).toObject();
    snapshot.configMeta = chunkDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectConfigSection)).toObject();
    snapshot.uiState = projectDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectUiStateSection)).toObject();

    const QJsonObject runtimeFiles =
        readJsonObjectFile(ProjectIO::tempFilesPath(snapshot.projectPath));
    if (!runtimeFiles.isEmpty())
    {
        snapshot.filesMeta = runtimeFiles;
        snapshot.recoveredFromTemporary = true;
    }
    const QJsonObject runtimeConfig =
        readJsonObjectFile(ProjectIO::tempConfigPath(snapshot.projectPath));
    if (!runtimeConfig.isEmpty())
    {
        snapshot.configMeta = runtimeConfig;
        snapshot.recoveredFromTemporary = true;
    }
    const QJsonObject runtimeUiState =
        readJsonObjectFile(ProjectIO::tempUiStatePath(snapshot.projectPath));
    if (!runtimeUiState.isEmpty())
    {
        snapshot.uiState = runtimeUiState;
        snapshot.recoveredFromTemporary = true;
    }

    if (!snapshot.filesMeta.value(QStringLiteral("images")).isArray())
    {
        snapshot.errorMessage =
            QStringLiteral("Chunk doc.json 的 project_files 无效");
        return snapshot;
    }
    snapshot.configMeta =
        ProjectConfigManager::mergeWithDefaults(snapshot.configMeta);
    if (!snapshot.uiState.value(
            QStringLiteral("display_settings")).isObject())
    {
        snapshot.errorMessage =
            QStringLiteral("项目 doc.json 的 ui_state 无效");
        return snapshot;
    }
    snapshot.uiState = normalizedProjectUiState(snapshot.uiState);

    if (!workspace.materializeMetadata(
            &snapshot.filesMeta, &snapshot.errorMessage))
    {
        return snapshot;
    }

    const QString projectId =
        projectDocument.value(QStringLiteral("project_id")).toString();
    if (!projectId.isEmpty())
    {
        snapshot.configMeta[QStringLiteral("version")] =
            QString::fromLatin1(PortableProjectFormat::CurrentFormatVersion);
        snapshot.configMeta[QStringLiteral("schema_version")] = 2;
        snapshot.configMeta[QStringLiteral("project_id")] = projectId;
    }

    snapshot.resultsLoaded = containsResultKeys(snapshot.filesMeta);
    snapshot.success = true;
    workspace.releaseRuntime();
    return snapshot;
}

ProjectResultsSnapshot ProjectData::loadProjectResultsSnapshot(const QString &plascanPath)
{
    ProjectResultsSnapshot snapshot;
    snapshot.projectPath = QDir::cleanPath(QFileInfo(plascanPath).absoluteFilePath());

    ProjectChunkStore chunkStore(snapshot.projectPath);
    if (!chunkStore.ensureLayout(&snapshot.errorMessage))
    {
        return snapshot;
    }
    const ProjectChunkRecord chunk =
        chunkStore.defaultChunk(&snapshot.errorMessage);
    if (chunk.id.isEmpty())
    {
        return snapshot;
    }
    snapshot.chunkId = chunk.id;
    ProjectWorkspaceStore workspace(
        snapshot.projectPath, chunk.directory);
    if (!workspace.initializeRuntime(nullptr, &snapshot.errorMessage))
    {
        return snapshot;
    }
    const auto runtimeGuard = qScopeGuard(
        [&workspace]()
        {
            workspace.releaseRuntime();
        });

    snapshot.resultsMeta = readJsonObjectFile(ProjectIO::tempResultsPath(snapshot.projectPath));
    if (!snapshot.resultsMeta.isEmpty())
    {
        if (!workspace.materializeMetadata(
                &snapshot.resultsMeta, &snapshot.errorMessage))
        {
            return snapshot;
        }
        snapshot.success = true;
        snapshot.hasResults = true;
        return snapshot;
    }

    QString layoutError;
    if (ProjectPackageLayout::resolveMetadataArchive(
            snapshot.projectPath, &layoutError).isEmpty())
    {
        snapshot.errorMessage = layoutError.isEmpty()
            ? QStringLiteral("无法解析项目数据目录")
            : layoutError;
        return snapshot;
    }
    QJsonObject chunkDocument;
    if (!chunkStore.readChunkDocument(
            chunk.directory,
            &chunkDocument,
            &snapshot.errorMessage))
    {
        return snapshot;
    }

    snapshot.resultsMeta = chunkDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectResultsSection)).toObject();
    if (!workspace.materializeMetadata(
            &snapshot.resultsMeta, &snapshot.errorMessage))
    {
        snapshot.success = false;
        return snapshot;
    }
    snapshot.success = true;
    snapshot.hasResults = true;
    return snapshot;
}

bool ProjectData::openProjectFromSnapshot(const ProjectOpenSnapshot &snapshot, QString *errorMsg)
{
    if (!snapshot.success)
    {
        if (errorMsg)
        {
            *errorMsg = snapshot.errorMessage.isEmpty()
                ? QStringLiteral("项目快照加载失败")
                : snapshot.errorMessage;
        }
        return false;
    }

    if (snapshot.projectPath.trimmed().isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("项目路径为空");
        }
        return false;
    }

    const QString currentPath =
        QDir::cleanPath(QFileInfo(_projectPath).absoluteFilePath());
    const QString targetPath =
        QDir::cleanPath(QFileInfo(snapshot.projectPath).absoluteFilePath());
    const bool reuseLock =
        _projectLock && _projectLock->isLocked()
        && currentPath == targetPath;
    std::unique_ptr<ProjectLock> replacementLock;
    if (!reuseLock)
    {
        replacementLock = std::make_unique<ProjectLock>();
        QString lockError;
        if (!replacementLock->acquire(targetPath, &lockError))
        {
            if (errorMsg)
            {
                *errorMsg = lockError;
            }
            return false;
        }
    }

    ProjectWorkspaceStore workspace(
        snapshot.projectPath, snapshot.chunkDirectory);
    QString workspaceError;
    if (!workspace.initializeRuntime(nullptr, &workspaceError))
    {
        if (errorMsg)
        {
            *errorMsg = workspaceError;
        }
        if (replacementLock)
        {
            replacementLock->release();
        }
        return false;
    }

    if (!reuseLock && hasProject())
    {
        closeProject();
    }
    if (replacementLock)
    {
        _projectLock = std::move(replacementLock);
    }
    if (_archiveSyncTimer)
    {
        _archiveSyncTimer->stop();
    }

    _projectPath = snapshot.projectPath;
    _activeChunkId = snapshot.chunkId;
    _activeChunkName = snapshot.chunkName;
    _activeChunkDirectory = snapshot.chunkDirectory;
    _filesManager.setData(QJsonObject());
    _configManager.setData(QJsonObject());
    _resultsLoaded = false;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    _uiStateDirtyForArchive = false;
    _isDirty = false;

    const QJsonObject filesMeta = snapshot.filesMeta.isEmpty()
        ? ProjectFilesManager::defaultFiles()
        : snapshot.filesMeta;
    if (containsResultKeys(filesMeta))
    {
        _filesManager.setData(filesMeta);
        _resultsLoaded = true;
    }
    else
    {
        _filesManager.setCoreData(filesMeta);
    }

    QJsonObject core = _filesManager.coreData();
    const bool assignedImageUuids = ensureImageUuids(&core);
    if (assignedImageUuids)
    {
        _filesManager.setCoreData(core);
        // Missing UUIDs are a compatibility migration performed while loading,
        // not a user edit. Persist the stable identities in the background
        // without making an untouched project appear modified.
        scheduleArchiveSync(true, false, true);
    }

    const QJsonObject configMeta = snapshot.configMeta.isEmpty()
        ? ProjectConfigManager::defaultConfig()
        : ProjectConfigManager::mergeWithDefaults(snapshot.configMeta);
    updateConfig(configMeta, false);
    _projectUiState = snapshot.uiState.isEmpty()
        ? defaultProjectUiState()
        : normalizedProjectUiState(snapshot.uiState);

    emit dirtyStateChanged(_isDirty);
    emit projectOpened(_projectPath);
    emit activeChunkChanged(
        _activeChunkId, _activeChunkName, _activeChunkDirectory);
    emit chunkListChanged(chunks(), _activeChunkId);
    return true;
}

bool ProjectData::applyResultsSnapshot(const ProjectResultsSnapshot &snapshot, QString *errorMsg)
{
    if (!snapshot.success)
    {
        if (errorMsg)
        {
            *errorMsg = snapshot.errorMessage.isEmpty()
                ? QStringLiteral("项目结果数据加载失败")
                : snapshot.errorMessage;
        }
        return false;
    }

    if (QDir::cleanPath(snapshot.projectPath) != QDir::cleanPath(_projectPath))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("项目结果数据与当前项目不匹配");
        }
        return false;
    }
    if (!snapshot.chunkId.isEmpty()
        && snapshot.chunkId != _activeChunkId)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("项目结果数据与当前 Chunk 不匹配");
        }
        return false;
    }
    if (_resultsLoaded && _resultsDirtyForArchive)
    {
        LOG_INFO(QStringLiteral(
            "忽略已过期的异步项目结果快照：当前结果在快照加载期间已更新"));
        return true;
    }

    _resultsLoaded = true;
    _filesManager.setResultsData(snapshot.hasResults ? snapshot.resultsMeta : QJsonObject());
    emitCurrentMetadataChanged();
    return true;
}

ProjectData::PersistenceSnapshot ProjectData::createPersistenceSnapshot(
    PersistenceMode mode) const
{
    PersistenceSnapshot snapshot;
    snapshot.mode = mode;
    snapshot.projectPath = _projectPath;
    snapshot.chunkId = _activeChunkId;
    snapshot.chunkDirectory = _activeChunkDirectory;
    snapshot.temporaryCorePath = tempFilesPath();
    snapshot.temporaryResultsPath = tempResultsPath();
    snapshot.temporaryConfigPath = tempConfigPath();
    snapshot.temporaryUiStatePath = tempUiStatePath();
    snapshot.core = _filesManager.coreData();
    snapshot.resultsLoaded = _resultsLoaded;
    if (_resultsLoaded)
    {
        snapshot.results = _filesManager.resultsData();
    }
    snapshot.config = _configManager.data();
    snapshot.uiState = _projectUiState;

    if (mode == PersistenceMode::FullSave)
    {
        snapshot.writeCoreToArchive = true;
        snapshot.writeResultsToArchive = _resultsLoaded;
        snapshot.writeConfigToArchive = true;
        snapshot.writeUiStateToArchive = true;
        snapshot.writeWorkspaceIndex = true;
        snapshot.writeTemporary = true;
    }
    else if (mode == PersistenceMode::ArchiveSync)
    {
        snapshot.writeCoreToArchive = _coreFileDirtyForArchive;
        snapshot.writeResultsToArchive = _resultsDirtyForArchive && _resultsLoaded;
        snapshot.writeConfigToArchive = _configDirtyForArchive;
        snapshot.writeUiStateToArchive = _uiStateDirtyForArchive;
        snapshot.writeWorkspaceIndex = _workspaceDirtyForArchive;
        snapshot.writeTemporary = true;
    }
    else
    {
        snapshot.writeTemporary = true;
    }
    return snapshot;
}

ProjectData::PersistenceResult ProjectData::persistSnapshot(
    PersistenceSnapshot snapshot)
{
    PersistenceResult result;
    result.mode = snapshot.mode;
    result.projectPath = snapshot.projectPath;
    result.includedCore = snapshot.writeCoreToArchive;
    result.includedResults = snapshot.writeResultsToArchive;
    result.includedConfig = snapshot.writeConfigToArchive;
    result.includedUiState = snapshot.writeUiStateToArchive;
    result.includedWorkspace = snapshot.writeWorkspaceIndex;

    const bool workspaceIndexRequested =
        snapshot.writeCoreToArchive
        || snapshot.writeResultsToArchive
        || snapshot.writeWorkspaceIndex;
    if (workspaceIndexRequested)
    {
        ProjectWorkspaceStore workspace(
            snapshot.projectPath, snapshot.chunkDirectory);
        QJsonObject *results = snapshot.resultsLoaded
            ? &snapshot.results
            : nullptr;
        if (!workspace.prepareSplitMetadata(
                &snapshot.core,
                results,
                &snapshot.resourceIndex,
                &result.errorMessage))
        {
            result.archiveRequested = true;
            result.archiveSuccess = false;
            result.success = false;
            return result;
        }
        result.projectUriMetadata = true;
        result.projectUriCore = snapshot.core;
        if (snapshot.resultsLoaded)
        {
            result.projectUriResults = snapshot.results;
        }
    }

    QVector<QPair<QString, QJsonObject>> chunkSections;
    if (snapshot.writeCoreToArchive)
    {
        chunkSections.append(qMakePair(
            QString::fromLatin1(
                PortableProjectFormat::ProjectFilesSection),
            snapshot.core));
    }
    if (snapshot.writeResultsToArchive)
    {
        chunkSections.append(qMakePair(
            QString::fromLatin1(
                PortableProjectFormat::ProjectResultsSection),
            snapshot.results));
    }
    if (snapshot.writeConfigToArchive)
    {
        chunkSections.append(qMakePair(
            QString::fromLatin1(
                PortableProjectFormat::ProjectConfigSection),
            snapshot.config));
    }
    if (workspaceIndexRequested)
    {
        chunkSections.append(qMakePair(
            QString::fromLatin1(
                PortableProjectFormat::ResourceIndexSection),
            snapshot.resourceIndex));
    }

    result.archiveRequested =
        workspaceIndexRequested
        || !chunkSections.isEmpty()
        || snapshot.writeUiStateToArchive;
    result.archiveSuccess = true;
    ProjectChunkStore chunkStore(snapshot.projectPath);
    if (!chunkSections.isEmpty())
    {
        result.archiveSuccess = chunkStore.writeChunkSections(
            snapshot.chunkDirectory,
            chunkSections,
            &result.errorMessage);
    }
    if (result.archiveSuccess && snapshot.writeUiStateToArchive)
    {
        result.archiveSuccess = chunkStore.saveProjectUiState(
            snapshot.uiState, &result.errorMessage);
    }
    if (result.archiveSuccess && workspaceIndexRequested)
    {
        result.archiveSuccess =
            ProjectSharedImageStore(snapshot.projectPath)
                .pruneUnreferenced(&result.errorMessage);
    }
    if (result.archiveSuccess
        && snapshot.mode == PersistenceMode::FullSave)
    {
        result.archiveSuccess =
            ProjectPackageLayout::pruneEmptyOptionalDirectories(
                snapshot.projectPath,
                snapshot.chunkDirectory,
                &result.errorMessage);
    }

    if (result.archiveSuccess
        && snapshot.mode == PersistenceMode::FullSave)
    {
        ProjectWorkspaceStore workspace(
            snapshot.projectPath, snapshot.chunkDirectory);
        QString cleanupError;
        if (!workspace.validateProjectLayout(&cleanupError))
        {
            result.archiveSuccess = false;
            result.errorMessage = cleanupError;
        }
    }

    const bool shouldWriteTemporary =
        snapshot.writeTemporary
        && (snapshot.mode != PersistenceMode::FullSave || !result.archiveSuccess);
    result.temporaryRequested = shouldWriteTemporary;
    result.temporarySuccess = !shouldWriteTemporary;
    if (shouldWriteTemporary)
    {
        QString temporaryError;
        bool temporarySuccess = writeFileAtomically(
            snapshot.temporaryCorePath,
            QJsonDocument(snapshot.core).toJson(QJsonDocument::Compact),
            &temporaryError);
        if (temporarySuccess && snapshot.resultsLoaded)
        {
            const QByteArray resultsJson =
                QJsonDocument(snapshot.results).toJson(QJsonDocument::Compact);
            temporarySuccess = writeFileAtomically(
                snapshot.temporaryResultsPath,
                qCompress(resultsJson, 1),
                &temporaryError);
        }
        if (temporarySuccess)
        {
            temporarySuccess = writeFileAtomically(
                snapshot.temporaryConfigPath,
                QJsonDocument(snapshot.config).toJson(QJsonDocument::Compact),
                &temporaryError);
        }
        if (temporarySuccess)
        {
            temporarySuccess = writeFileAtomically(
                snapshot.temporaryUiStatePath,
                QJsonDocument(snapshot.uiState)
                    .toJson(QJsonDocument::Compact),
                &temporaryError);
        }
        result.temporarySuccess = temporarySuccess;
        if (!temporarySuccess)
        {
            if (!result.errorMessage.isEmpty())
            {
                result.errorMessage += QStringLiteral("; ");
            }
            result.errorMessage += temporaryError;
        }
    }

    if (snapshot.mode == PersistenceMode::TemporaryOnly)
    {
        result.success = result.temporarySuccess;
    }
    else
    {
        result.success = result.archiveSuccess
            && (!result.temporaryRequested || result.temporarySuccess);
    }
    return result;
}

void ProjectData::startNextPersistence()
{
    if (_shuttingDown || _persistenceRunning || !_persistencePool)
    {
        return;
    }

    PersistenceMode mode = PersistenceMode::TemporaryOnly;
    PersistenceSnapshot snapshot;
    bool hasDetachedSnapshot = false;
    if (_detachedPersistenceSnapshot)
    {
        snapshot = std::move(*_detachedPersistenceSnapshot);
        _detachedPersistenceSnapshot.reset();
        mode = snapshot.mode;
        hasDetachedSnapshot = true;
    }
    else if (_fullSavePending)
    {
        mode = PersistenceMode::FullSave;
        _fullSavePending = false;
        _archiveSyncPending = false;
        _temporarySavePending = false;
    }
    else if (_archiveSyncPending)
    {
        mode = PersistenceMode::ArchiveSync;
        _archiveSyncPending = false;
    }
    else if (_temporarySavePending)
    {
        mode = PersistenceMode::TemporaryOnly;
        _temporarySavePending = false;
    }
    else
    {
        return;
    }

    if (!hasDetachedSnapshot && _projectPath.trimmed().isEmpty())
    {
        if (mode == PersistenceMode::FullSave)
        {
            emit projectSaveCompleted(false, QStringLiteral("没有打开的项目"));
        }
        return;
    }

    if (!hasDetachedSnapshot)
    {
        snapshot = createPersistenceSnapshot(mode);
        if (mode == PersistenceMode::FullSave)
        {
            _coreFileDirtyForArchive = false;
            _resultsDirtyForArchive = false;
            _configDirtyForArchive = false;
            _uiStateDirtyForArchive = false;
            _workspaceDirtyForArchive = false;
        }
        else if (mode == PersistenceMode::ArchiveSync)
        {
            if (snapshot.writeCoreToArchive)
            {
                _coreFileDirtyForArchive = false;
            }
            if (snapshot.writeResultsToArchive)
            {
                _resultsDirtyForArchive = false;
            }
            if (snapshot.writeConfigToArchive)
            {
                _configDirtyForArchive = false;
            }
            if (snapshot.writeUiStateToArchive)
            {
                _uiStateDirtyForArchive = false;
            }
            if (snapshot.writeWorkspaceIndex)
            {
                _workspaceDirtyForArchive = false;
            }
        }
    }

    _persistenceRunning = true;
    QPointer<ProjectData> self(this);
    (void)QtConcurrent::run(
        _persistencePool,
        [self, snapshot = std::move(snapshot)]() mutable
        {
            PersistenceResult result =
                ProjectData::persistSnapshot(std::move(snapshot));
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(
                self.data(),
                [self, result = std::move(result)]() mutable
                {
                    if (self)
                    {
                        self->handlePersistenceFinished(std::move(result));
                    }
                },
                Qt::QueuedConnection);
        });
}

void ProjectData::handlePersistenceFinished(PersistenceResult result)
{
    _persistenceRunning = false;
    const bool sameProject =
        QDir::cleanPath(result.projectPath) == QDir::cleanPath(_projectPath);

    if (sameProject && result.archiveRequested && !result.archiveSuccess)
    {
        _coreFileDirtyForArchive =
            _coreFileDirtyForArchive || result.includedCore;
        _resultsDirtyForArchive =
            _resultsDirtyForArchive || result.includedResults;
        _configDirtyForArchive =
            _configDirtyForArchive || result.includedConfig;
        _uiStateDirtyForArchive =
            _uiStateDirtyForArchive || result.includedUiState;
        _workspaceDirtyForArchive =
            _workspaceDirtyForArchive || result.includedWorkspace;
        if (_archiveSyncTimer)
        {
            _archiveSyncTimer->start(5000);
        }
    }

    if (sameProject && result.archiveSuccess && result.projectUriMetadata)
    {
        ProjectWorkspaceStore workspace(
            _projectPath, _activeChunkDirectory);
        bool metadataChanged = false;
        if (result.includedCore && !_coreFileDirtyForArchive)
        {
            QJsonObject core = result.projectUriCore;
            QString materializeError;
            if (workspace.materializeMetadata(&core, &materializeError))
            {
                _filesManager.setCoreData(core);
                metadataChanged = true;
            }
            else
            {
                LOG_WARN(QStringLiteral("刷新工程资源缓存路径失败: %1")
                             .arg(materializeError));
            }
        }
        if (result.includedResults
            && !_resultsDirtyForArchive
            && _resultsLoaded)
        {
            QJsonObject results = result.projectUriResults;
            QString materializeError;
            if (workspace.materializeMetadata(&results, &materializeError))
            {
                _filesManager.setResultsData(results);
                metadataChanged = true;
            }
            else
            {
                LOG_WARN(QStringLiteral("刷新工程结果缓存路径失败: %1")
                             .arg(materializeError));
            }
        }
        if (metadataChanged)
        {
            emitCurrentMetadataChanged();
        }
    }

    if (result.mode == PersistenceMode::FullSave)
    {
        if (!result.archiveSuccess)
        {
            emit projectSaveCompleted(false, result.errorMessage);
        }
        else
        {
            if (sameProject)
            {
                const bool hasNewChanges =
                    _coreFileDirtyForArchive
                    || _resultsDirtyForArchive
                    || _configDirtyForArchive
                    || _uiStateDirtyForArchive
                    || _workspaceDirtyForArchive;
                if (!hasNewChanges)
                {
                    _isDirty = false;
                    emit dirtyStateChanged(false);
                    clearTemporaryMetadata();
                }
            }
            emit projectSaved(result.projectPath);
            emit projectSaveCompleted(true, QString());
            LOG_INFO(QStringLiteral("项目已后台保存: %1").arg(result.projectPath));
        }
    }
    else if (!result.success)
    {
        LOG_WARN(QStringLiteral("项目后台持久化失败: %1")
                     .arg(result.errorMessage));
    }

    startNextPersistence();
}

void ProjectData::saveProjectAsync()
{
    if (_projectPath.trimmed().isEmpty())
    {
        emit projectSaveCompleted(false, QStringLiteral("没有打开的项目"));
        return;
    }
    if (_archiveSyncTimer)
    {
        _archiveSyncTimer->stop();
    }
    _fullSavePending = true;
    startNextPersistence();
}

void ProjectData::scheduleTemporaryMetadataSave()
{
    if (_projectPath.trimmed().isEmpty())
    {
        return;
    }
    _temporarySavePending = true;
    startNextPersistence();
}

bool ProjectData::saveProject(QString *errorMsg)
{
    if (_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    // 完整保存前首先取消防抖定时器，避免重复写入
    if (_archiveSyncTimer) _archiveSyncTimer->stop();
    if (_persistencePool && _persistenceRunning)
    {
        _persistencePool->waitForDone();
    }

    QJsonObject projectUriCore = _filesManager.coreData();
    QJsonObject projectUriResults = _resultsLoaded
        ? _filesManager.resultsData()
        : QJsonObject();
    QJsonObject resourceIndex;
    ProjectWorkspaceStore workspace(
        _projectPath, _activeChunkDirectory);
    QString err;
    if (!workspace.prepareSplitMetadata(
            &projectUriCore,
            _resultsLoaded ? &projectUriResults : nullptr,
            &resourceIndex,
            &err))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("更新工程资源索引失败: %1").arg(err);
        }
        return false;
    }
    QVector<QPair<QString, QJsonObject>> chunkSections{
        qMakePair(
            QString::fromLatin1(
                PortableProjectFormat::ProjectFilesSection),
            projectUriCore),
        qMakePair(
            QString::fromLatin1(
                PortableProjectFormat::ProjectConfigSection),
            _configManager.data()),
        qMakePair(
            QString::fromLatin1(
                PortableProjectFormat::ResourceIndexSection),
            resourceIndex)
    };
    if (_resultsLoaded)
    {
        chunkSections.insert(
            1,
            qMakePair(
                QString::fromLatin1(
                    PortableProjectFormat::ProjectResultsSection),
                projectUriResults));
    }

    ProjectChunkStore chunkStore(_projectPath);
    if (!chunkStore.writeChunkSections(
            _activeChunkDirectory, chunkSections, &err)
        || !chunkStore.saveProjectUiState(_projectUiState, &err))
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("写入项目 doc.json 失败: %1").arg(err);
        }
        return false;
    }
    if (!ProjectSharedImageStore(_projectPath)
             .pruneUnreferenced(&err))
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("清理共享影像库失败: %1").arg(err);
        }
        return false;
    }
    if (!ProjectPackageLayout::pruneEmptyOptionalDirectories(
            _projectPath, _activeChunkDirectory, &err))
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("清理空的工作流目录失败: %1").arg(err);
        }
        return false;
    }

    if (!workspace.validateProjectLayout(&err))
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("项目保存后格式校验失败: %1")
                    .arg(err);
        }
        return false;
    }

    QString materializeError;
    if (workspace.materializeMetadata(&projectUriCore, &materializeError))
    {
        _filesManager.setCoreData(projectUriCore);
    }
    if (_resultsLoaded
        && workspace.materializeMetadata(
            &projectUriResults, &materializeError))
    {
        _filesManager.setResultsData(projectUriResults);
    }

    _isDirty = false;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    _uiStateDirtyForArchive = false;
    _workspaceDirtyForArchive = false;
    emit dirtyStateChanged(false);
    emit projectSaved(_projectPath);

    clearTemporaryMetadata();
    LOG_INFO(QStringLiteral("项目已保存: %1").arg(_projectPath));
    return true;
}

void ProjectData::closeProject()
{
    const QString closingProjectPath = _projectPath;
    if (_archiveSyncTimer)
    {
        _archiveSyncTimer->stop();
    }
    if (!_projectPath.trimmed().isEmpty()
        && (_isDirty
            || _coreFileDirtyForArchive
            || _resultsDirtyForArchive
            || _configDirtyForArchive
            || _uiStateDirtyForArchive
            || _workspaceDirtyForArchive
            || _temporarySavePending))
    {
        _detachedPersistenceSnapshot =
            std::make_unique<PersistenceSnapshot>(
                createPersistenceSnapshot(PersistenceMode::TemporaryOnly));
        _temporarySavePending = false;
        startNextPersistence();
    }
    _projectPath.clear();
    _activeChunkId.clear();
    _activeChunkName.clear();
    _activeChunkDirectory = 0;
    _filesManager.setData(QJsonObject());
    _configManager.setData(QJsonObject());
    _isDirty = false;
    _resultsLoaded = false;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    _projectUiState = QJsonObject();
    _uiStateDirtyForArchive = false;
    _workspaceDirtyForArchive = false;
    _fullSavePending = false;
    _archiveSyncPending = false;
    _temporarySavePending = false;
    if (!closingProjectPath.isEmpty())
    {
        ProjectWorkspaceStore(closingProjectPath).releaseRuntime();
    }
    if (_persistencePool && _persistenceRunning)
    {
        _persistencePool->waitForDone();
    }
    if (_projectLock)
    {
        _projectLock->release();
        _projectLock.reset();
    }
    emit projectClosed();
    emit chunkListChanged(QJsonArray(), QString());
}

// 防抖定时器回调：把最新快照交给串行持久化线程。
void ProjectData::syncToArchive()
{
    if (_projectPath.isEmpty()
        || (!_resultsDirtyForArchive
            && !_coreFileDirtyForArchive
            && !_configDirtyForArchive
            && !_uiStateDirtyForArchive
            && !_workspaceDirtyForArchive))
    {
        return;
    }

    _archiveSyncPending = true;
    startNextPersistence();
}

void ProjectData::updateMetadata(const QJsonObject &meta, bool markDirty)
{
    const bool hasResults = containsResultKeys(meta);
    const bool preserveLoadedResults = !hasResults && _resultsLoaded;
    const QJsonObject existingResults = preserveLoadedResults ? _filesManager.resultsData() : QJsonObject();

    _filesManager.setData(meta);

    QJsonObject core = _filesManager.coreData();
    if (ensureImageUuids(&core))
    {
        _filesManager.setCoreData(core);
    }

    if (hasResults) {
        _resultsLoaded = true;
    }
    else if (preserveLoadedResults) {
        _filesManager.setResultsData(existingResults);
    }

    markDirtyIfRequested(markDirty);
    if (markDirty) {
        scheduleArchiveSync(true, hasResults, false);
    }

    emitCurrentMetadataChanged();
}

// 惰性加载 project_results.json：仅在首次访问 results 时读归档
void ProjectData::ensureResultsLoaded() const
{
    if (_resultsLoaded) return;
    _resultsLoaded = true;   // 先置位，防止递归

    if (_projectPath.isEmpty()) return;

    // 尝试从 tmp 先加载（崩溃恢复）
    const QString tmpPath = tempResultsPath();
    if (!tmpPath.isEmpty() && QFile::exists(tmpPath)) {
        QFile f(tmpPath);
        if (f.open(QIODevice::ReadOnly)) {
            const QJsonDocument doc = parseJsonOrCompressedJson(f.readAll());
            if (!doc.isNull() && doc.isObject()) {
                QJsonObject results = doc.object();
                ProjectWorkspaceStore workspace(
                    _projectPath, _activeChunkDirectory);
                QString materializeError;
                if (!workspace.materializeMetadata(
                        &results, &materializeError))
                {
                    LOG_WARN(QStringLiteral("解析临时项目结果资源失败: %1")
                                 .arg(materializeError));
                    return;
                }
                _filesManager.setResultsData(results);
                LOG_INFO(QStringLiteral("从临时目录加载 results"));
                return;
            }
        }
    }

    QString err;
    QJsonObject document;
    if (!ProjectChunkStore(_projectPath).readChunkDocument(
            _activeChunkDirectory, &document, &err))
    {
        LOG_WARN(QStringLiteral("读取 Chunk doc.json 失败: %1").arg(err));
        return;
    }
    QJsonObject results = document.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectResultsSection)).toObject();
    ProjectWorkspaceStore workspace(
        _projectPath, _activeChunkDirectory);
    QString materializeError;
    if (!workspace.materializeMetadata(&results, &materializeError))
    {
        LOG_WARN(QStringLiteral("解析项目结果资源失败: %1")
                     .arg(materializeError));
        return;
    }
    _filesManager.setResultsData(results);
    LOG_INFO(QStringLiteral("从 Chunk doc.json 加载 project_results"));
}

void ProjectData::updateConfig(const QJsonObject &config, bool markDirty)
{
    _configManager.setData(config);

    if (markDirty)
    {
        markDirtyIfRequested(true);
        scheduleArchiveSync(false, false, true, true);
    }
}

void ProjectData::updateProjectUiState(const QJsonObject &state,
                                       bool markDirty)
{
    _projectUiState = state.isEmpty()
        ? defaultProjectUiState()
        : normalizedProjectUiState(state);
    if (markDirty)
    {
        markDirtyIfRequested(true);
        scheduleArchiveSync(false, false, true, false, true);
    }
}

bool ProjectData::loadTemporaryMetadata()
{
    bool loaded = false;

    // 尝试从 .plascan_tmp/project_files.json 恢复核心数据
    QString filesPath = tempFilesPath();
    if (!filesPath.isEmpty() && QFile::exists(filesPath)) {
        QFile file(filesPath);
        if (file.open(QIODevice::ReadOnly)) {
            QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
            if (!doc.isNull() && doc.isObject()) {
                _filesManager.setCoreData(doc.object());
                loaded = true;
            }
        }
    }

    // 尝试从 .plascan_tmp/project_results.json 恢复结果数据
    // （新版以 qCompress 压缩写入；通过首字节区分压缩/明文 JSON）
    QString resultsPath = tempResultsPath();
    if (!resultsPath.isEmpty() && QFile::exists(resultsPath)) {
        QFile file(resultsPath);
        if (file.open(QIODevice::ReadOnly)) {
            const QJsonDocument doc = parseJsonOrCompressedJson(file.readAll());
            if (!doc.isNull() && doc.isObject()) {
                _filesManager.setResultsData(doc.object());
                _resultsLoaded = true;
                loaded = true;
            }
        }
    }

    // 尝试从 .plascan_tmp/project_config.json 恢复配置数据
    QString configPath = tempConfigPath();
    if (!configPath.isEmpty() && QFile::exists(configPath)) {
        QFile file(configPath);
        if (file.open(QIODevice::ReadOnly)) {
            QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
            if (!doc.isNull() && doc.isObject()) {
                updateConfig(ProjectConfigManager::mergeWithDefaults(doc.object()), false);
                loaded = true;
            }
        }
    }

    const QString uiStatePath = tempUiStatePath();
    if (!uiStatePath.isEmpty() && QFile::exists(uiStatePath))
    {
        const QJsonObject state = readJsonObjectFile(uiStatePath);
        if (!state.isEmpty())
        {
            updateProjectUiState(state, false);
            loaded = true;
        }
    }

    return loaded;
}

bool ProjectData::saveTemporaryMetadata()
{
    QString filesPath = tempFilesPath();
    QString configPath = tempConfigPath();
    if (filesPath.isEmpty() || configPath.isEmpty())
        return false;

    QString error;
    if (!writeFileAtomically(
            filesPath,
            QJsonDocument(_filesManager.coreData()).toJson(QJsonDocument::Compact),
            &error))
    {
        return false;
    }

    // 写结果数据（qCompress 压缩的 Compact JSON，比原始 JSON 小 60-70%）
    if (_resultsLoaded) {
        QString resultsPath = tempResultsPath();
        if (!resultsPath.isEmpty()) {
            const QByteArray json =
                QJsonDocument(_filesManager.resultsData()).toJson(QJsonDocument::Compact);
            if (!writeFileAtomically(resultsPath, qCompress(json, 1), &error))
            {
                return false;
            }
        }
    }

    if (!writeFileAtomically(
        configPath,
        QJsonDocument(_configManager.data()).toJson(QJsonDocument::Compact),
        &error))
    {
        return false;
    }
    return writeFileAtomically(
        tempUiStatePath(),
        QJsonDocument(_projectUiState).toJson(QJsonDocument::Compact),
        &error);
}

void ProjectData::clearTemporaryMetadata()
{
    auto removeIfExists = [](const QString &p) {
        if (!p.isEmpty() && QFile::exists(p)) QFile::remove(p);
    };
    removeIfExists(tempFilesPath());
    removeIfExists(tempResultsPath());
    removeIfExists(tempConfigPath());
    removeIfExists(tempUiStatePath());
}

bool ProjectData::hasTemporaryMetadata() const
{
    auto exists = [](const QString &p) { return !p.isEmpty() && QFile::exists(p); };
    return exists(tempFilesPath())
        || exists(tempResultsPath())
        || exists(tempConfigPath())
        || exists(tempUiStatePath());
}

bool ProjectData::addImages(const QStringList &imagePaths, QString *errorMsg)
{
    if (_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    const QJsonArray images = _filesManager.coreData().value("images").toArray();
    QSet<QString> existingPaths;
    existingPaths.reserve(images.size());
    for (const QJsonValue &val : images) {
        const QString p = val.toObject().value("path").toString();
        if (!p.isEmpty())
            existingPaths.insert(p);
    }

    QStringList importedPaths;
    importedPaths.reserve(imagePaths.size());
    int skipped = 0;
    for (const QString &srcPath : imagePaths) {
        const QString absPath = QFileInfo(srcPath).absoluteFilePath();
        QString sharedUri;
        QString projectImagePath;
        if (!ProjectSharedImageStore(_projectPath).importImage(
                absPath,
                &sharedUri,
                &projectImagePath,
                errorMsg))
        {
            return false;
        }

        // 跳过已存在的重复图片
        if (existingPaths.contains(projectImagePath)) {
            ++skipped;
            continue;
        }
        existingPaths.insert(projectImagePath); // 防止同批次内重复
        importedPaths.append(projectImagePath);
    }

    return addImagesFromSharedStore(importedPaths, skipped, errorMsg);
}

bool ProjectData::addImagesFromSharedStore(const QStringList &projectImagePaths,
                                           int previouslySkipped,
                                           QString *errorMsg)
{
    if (_projectPath.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有打开的项目");
        }
        return false;
    }

    QJsonArray images = _filesManager.coreData().value("images").toArray();
    QSet<QString> existingPaths;
    existingPaths.reserve(images.size() + projectImagePaths.size());
    for (const QJsonValue &value : images)
    {
        const QString path = value.toObject().value("path").toString();
        if (!path.isEmpty())
        {
            existingPaths.insert(path);
        }
    }

    int skipped = std::max(0, previouslySkipped);
    for (const QString &projectImagePath : projectImagePaths)
    {
        const QString cleanPath = QDir::cleanPath(projectImagePath.trimmed());
        if (cleanPath.isEmpty() || existingPaths.contains(cleanPath))
        {
            ++skipped;
            continue;
        }
        if (!QFileInfo(cleanPath).isFile())
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("共享影像不存在: %1").arg(cleanPath);
            }
            return false;
        }

        existingPaths.insert(cleanPath);
        QJsonObject image;
        image["image_uuid"] = QUuid::createUuid().toString(QUuid::WithoutBraces);
        image["path"] = cleanPath;
        image["type"] = "shared";
        image["added_at"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        images.append(image);
    }

    if (skipped > 0 && errorMsg)
    {
        *errorMsg = QStringLiteral("已跳过 %1 张重复图片").arg(skipped);
    }

    // 更新元数据并写入临时缓存
    QJsonObject core = _filesManager.coreData();
    core["images"] = images;
    _filesManager.setCoreData(core);
    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(true, false, true);

    return true;
}

bool ProjectData::addImagesFromFolder(const QString &folderPath, QString *errorMsg)
{
    QDir dir(folderPath);
    if (!dir.exists()) {
        if (errorMsg) *errorMsg = QStringLiteral("文件夹不存在");
        return false;
    }

    // 支持的影像格式过滤器（包含大小写两种，兼容 Linux/Windows 文件系统）
    QStringList filters;
    filters << "*.tif" << "*.tiff" << "*.TIF" << "*.TIFF"
            << "*.png" << "*.PNG"
            << "*.jpg" << "*.jpeg" << "*.JPG" << "*.JPEG";
    
    // 递归只查找当前目录（QDir::Files，不含子目录）
    QStringList imagePaths;
    for (const QFileInfo &fi : dir.entryInfoList(filters, QDir::Files)) {
        imagePaths << fi.absoluteFilePath();
    }

    // 委托给 addImages 完成实际添加逻辑
    return addImages(imagePaths, errorMsg);
}

bool ProjectData::removeResource(const QString &resourcePath)
{
    return removeResources(QStringList() << resourcePath);
}

bool ProjectData::removeResources(const QStringList &resourcePaths)
{
    const QString projectRoot = ProjectIO::projectRootFromPlascan(_projectPath);
    const QString physicalRoot = ProjectIO::physicalProjectRoot(_projectPath);
    QStringList normalizedTargets;
    for (const QString &path : resourcePaths)
    {
        const QString normalized = normalizedProjectResourcePath(projectRoot, path);
        if (!normalized.isEmpty() && !normalizedTargets.contains(normalized))
        {
            normalizedTargets.append(normalized);
        }

        const QString absolute =
            QDir::cleanPath(QFileInfo(path).absoluteFilePath());
        QString physicalPrefix = QDir::cleanPath(physicalRoot);
        if (!physicalPrefix.endsWith(QLatin1Char('/')))
        {
            physicalPrefix += QLatin1Char('/');
        }
#if defined(Q_OS_WIN)
        const bool usesPhysicalRoot =
            absolute.startsWith(physicalPrefix, Qt::CaseInsensitive);
#else
        const bool usesPhysicalRoot = absolute.startsWith(physicalPrefix);
#endif
        if (usesPhysicalRoot)
        {
            const QString relative =
                QDir(physicalRoot).relativeFilePath(absolute);
            const QString runtimeAlias =
                QDir::cleanPath(QDir(projectRoot).filePath(relative));
            if (!normalizedTargets.contains(runtimeAlias))
            {
                normalizedTargets.append(runtimeAlias);
            }
        }
    }

    QJsonArray images = _filesManager.coreData().value("images").toArray();
    QJsonArray newImages;
    QStringList removedProjectFiles;

    for (const QJsonValue &val : images) {
        QJsonObject obj = val.toObject();
        const QString storedPath = normalizedProjectResourcePath(projectRoot, obj.value("path").toString());
        if (!normalizedTargets.contains(storedPath))
        {
            newImages.append(val);
        }
        else
        {
            removedProjectFiles.append(storedPath);
        }
    }

    const QString importedRoot = QDir(projectRoot).filePath(
        QStringLiteral("assets/imported"));
    QString importedPrefix = QDir::cleanPath(importedRoot);
    if (!importedPrefix.endsWith(QLatin1Char('/')))
    {
        importedPrefix += QLatin1Char('/');
    }
    for (const QString &path : removedProjectFiles)
    {
        const QString cleanPath = QDir::cleanPath(path);
#if defined(Q_OS_WIN)
        const bool isImported =
            cleanPath.startsWith(importedPrefix, Qt::CaseInsensitive);
#else
        const bool isImported = cleanPath.startsWith(importedPrefix);
#endif
        if (!isImported || !QFileInfo(cleanPath).isFile())
        {
            continue;
        }
        QFile::remove(cleanPath);

        QDir parent(QFileInfo(cleanPath).absolutePath());
        while (parent.absolutePath().size() > importedRoot.size()
               && parent.entryList(
                      QDir::AllEntries | QDir::NoDotAndDotDot)
                      .isEmpty())
        {
            const QString directory = parent.absolutePath();
            parent.cdUp();
            QDir().rmdir(directory);
        }
    }

    QJsonObject core = _filesManager.coreData();
    core["images"] = newImages;
    _filesManager.setCoreData(core);
    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(true, false, true);
    return true;
}

bool ProjectData::setImageCamera(const QString &imagePath, const QJsonObject &cameraMeta, QString *errorMsg)
{
    QMap<QString, QJsonObject> one;
    one.insert(imagePath, cameraMeta);
    return setImageCameras(one, nullptr, errorMsg);
}

bool ProjectData::setImageCameras(const QMap<QString, QJsonObject> &cameraMetaByImage,
                                  int *updatedCount,
                                  QString *errorMsg)
{
    if (updatedCount) *updatedCount = 0;

    if (_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    if (cameraMetaByImage.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有可写入的相机元数据");
        return false;
    }

    // 预处理：将输入的所有键路径规范化为 cleanPath + absoluteFilePath，
    // 避免不同表示方式（相对路径 vs 绝对路径、双斜线等）导致匹配失败
    QMap<QString, QJsonObject> normalizedMap;
    for (auto it = cameraMetaByImage.constBegin(); it != cameraMetaByImage.constEnd(); ++it) {
        normalizedMap.insert(QDir::cleanPath(QFileInfo(it.key()).absoluteFilePath()), it.value());
    }

    // 遍历影像数组，对匹配的条目写入 camera 字段
    QJsonObject core = _filesManager.coreData();
    QJsonArray images = core.value("images").toArray();
    int changed = 0;

    for (int i = 0; i < images.size(); ++i) {
        if (!images[i].isObject()) continue;
        QJsonObject imgObj = images[i].toObject();
        // 同样规范化影像路径后再查找
        const QString imgPath = QDir::cleanPath(QFileInfo(imgObj.value("path").toString()).absoluteFilePath());
        auto it = normalizedMap.constFind(imgPath);
        if (it == normalizedMap.constEnd()) continue;

        // 写入相机参数到影像对象的 camera 字段
        imgObj.remove(QStringLiteral("camera_file"));
        imgObj["camera"] = it.value();
        images[i] = imgObj;
        ++changed;
    }

    if (changed <= 0) {
        if (errorMsg) *errorMsg = QStringLiteral("未找到可匹配的影像记录");
        return false;
    }

    core["images"] = images;
    _filesManager.setCoreData(core);
    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(true, false, true);

    if (updatedCount) *updatedCount = changed;
    return true;
}

bool ProjectData::replaceImageCameras(const QStringList &targetImagePaths,
                                      const QMap<QString, QJsonObject> &cameraMetaByImage,
                                      int *updatedCount,
                                      int *clearedCount,
                                      QString *errorMsg)
{
    if (updatedCount) *updatedCount = 0;
    if (clearedCount) *clearedCount = 0;

    if (_projectPath.isEmpty())
    {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }
    if (targetImagePaths.isEmpty())
    {
        if (errorMsg) *errorMsg = QStringLiteral("没有指定要替换相机结果的影像");
        return false;
    }

    QSet<QString> normalizedTargets;
    for (const QString &path : targetImagePaths)
    {
        normalizedTargets.insert(QDir::cleanPath(QFileInfo(path).absoluteFilePath()));
    }

    QMap<QString, QJsonObject> normalizedCameras;
    for (auto it = cameraMetaByImage.constBegin(); it != cameraMetaByImage.constEnd(); ++it)
    {
        normalizedCameras.insert(QDir::cleanPath(QFileInfo(it.key()).absoluteFilePath()), it.value());
    }

    QJsonObject core = _filesManager.coreData();
    QJsonArray images = core.value(QStringLiteral("images")).toArray();
    int foundTargets = 0;
    int updated = 0;
    int cleared = 0;
    for (int index = 0; index < images.size(); ++index)
    {
        if (!images[index].isObject())
        {
            continue;
        }

        QJsonObject image = images[index].toObject();
        const QString imagePath = QDir::cleanPath(
            QFileInfo(image.value(QStringLiteral("path")).toString()).absoluteFilePath());
        if (!normalizedTargets.contains(imagePath))
        {
            continue;
        }
        ++foundTargets;

        const auto cameraIt = normalizedCameras.constFind(imagePath);
        if (cameraIt != normalizedCameras.constEnd())
        {
            image.remove(QStringLiteral("camera_file"));
            image[QStringLiteral("camera")] = cameraIt.value();
            ++updated;
        }
        else if (image.contains(QStringLiteral("camera")) ||
                 image.contains(QStringLiteral("camera_file")))
        {
            image.remove(QStringLiteral("camera"));
            image.remove(QStringLiteral("camera_file"));
            ++cleared;
        }
        images[index] = image;
    }

    if (foundTargets <= 0)
    {
        if (errorMsg) *errorMsg = QStringLiteral("未找到可匹配的影像记录");
        return false;
    }

    // 一次提交并只发出一次 metadataChanged，避免界面短暂显示新旧位姿混合状态。
    core[QStringLiteral("images")] = images;
    _filesManager.setCoreData(core);
    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(true, false, true);

    if (updatedCount) *updatedCount = updated;
    if (clearedCount) *clearedCount = cleared;
    return true;
}

// ---------- 清除相机参数 ----------

bool ProjectData::clearImageCameras(const QStringList &imagePaths,
                                    int *clearedCount,
                                    QString *errorMsg)
{
    if (clearedCount) *clearedCount = 0;

    if (_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    if (imagePaths.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有指定要清除的影像");
        return false;
    }

    // 规范化路径
    QSet<QString> normalizedSet;
    for (const QString &p : imagePaths)
        normalizedSet.insert(QDir::cleanPath(QFileInfo(p).absoluteFilePath()));

    QJsonObject core = _filesManager.coreData();
    QJsonArray images = core.value("images").toArray();
    int cleared = 0;

    for (int i = 0; i < images.size(); ++i) {
        if (!images[i].isObject()) continue;
        QJsonObject imgObj = images[i].toObject();
        const QString imgPath = QDir::cleanPath(QFileInfo(imgObj.value("path").toString()).absoluteFilePath());
        if (!normalizedSet.contains(imgPath)) continue;

        imgObj.remove(QStringLiteral("camera"));
        images[i] = imgObj;
        ++cleared;
    }

    if (cleared <= 0) {
        if (errorMsg) *errorMsg = QStringLiteral("未找到可匹配的影像记录");
        return false;
    }

    core["images"] = images;
    _filesManager.setCoreData(core);
    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(true, false, true);

    if (clearedCount) *clearedCount = cleared;
    return true;
}

bool ProjectData::appendIntersectionResult(const QJsonObject &result, QString *errorMsg)
{
    if (!appendResultRecord(QStringLiteral("intersection_results"), result, true)) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    return true;
}

QJsonArray ProjectData::getIntersectionResults() const
{
    ensureResultsLoaded();
    return _filesManager.resultsData().value(QLatin1String("intersection_results")).toArray();
}

bool ProjectData::appendBundleAdjustResult(const QJsonObject &result, QString *errorMsg)
{
    if (!appendResultRecord(QStringLiteral("bundle_adjust_results"), result, true)) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    return true;
}

QJsonArray ProjectData::getBundleAdjustResults() const
{
    ensureResultsLoaded();
    return _filesManager.resultsData().value(QLatin1String("bundle_adjust_results")).toArray();
}

bool ProjectData::appendResultRecord(const QString &arrayKey,
                                     const QJsonObject &record,
                                     bool markDirty)
{
    if (_projectPath.isEmpty())
    {
        return false;
    }

    ensureResultsLoaded();
    QJsonObject results = _filesManager.resultsData();
    QJsonArray array = results.value(arrayKey).toArray();
    array.append(versionedResultRecord(record));
    results[arrayKey] = array;
    _filesManager.setResultsData(results);

    markDirtyIfRequested(markDirty);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, true);
    return true;
}

bool ProjectData::upsertResultRecordByPath(const QString &arrayKey,
                                           const QString &pathKey,
                                           const QJsonObject &record,
                                           bool markDirty)
{
    if (_projectPath.isEmpty())
    {
        return false;
    }

    ensureResultsLoaded();
    QJsonObject results = _filesManager.resultsData();
    const QString targetPath = record.value(pathKey).toString();
    const QString normalizedTargetPath =
        normalizedResultPath(_projectPath, targetPath);
    const QJsonArray source = results.value(arrayKey).toArray();
    QJsonArray deduped;
    for (const QJsonValue &value : source)
    {
        const QString existingPath = value.toObject().value(pathKey).toString();
        if (!normalizedTargetPath.isEmpty()
            && normalizedResultPath(_projectPath, existingPath)
                == normalizedTargetPath)
        {
            continue;
        }
        deduped.append(value);
    }
    deduped.append(versionedResultRecord(record));
    results[arrayKey] = deduped;
    _filesManager.setResultsData(results);

    markDirtyIfRequested(markDirty);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, true);
    return true;
}

bool ProjectData::upsertResultRecordByIndex(const QString &arrayKey,
                                            const QJsonObject &record,
                                            int replaceIndex,
                                            bool markDirty)
{
    if (_projectPath.isEmpty())
    {
        return false;
    }

    ensureResultsLoaded();
    QJsonObject results = _filesManager.resultsData();
    QJsonArray array = results.value(arrayKey).toArray();
    if (replaceIndex >= 0 && replaceIndex < array.size())
    {
        array[replaceIndex] = versionedResultRecord(record);
    }
    else
    {
        array.append(versionedResultRecord(record));
    }
    results[arrayKey] = array;
    _filesManager.setResultsData(results);

    markDirtyIfRequested(markDirty);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, true);
    return true;
}

bool ProjectData::replaceResultRecordWithLatest(const QString &arrayKey,
                                                const QJsonObject &record,
                                                bool markDirty)
{
    if (_projectPath.isEmpty())
    {
        return false;
    }

    ensureResultsLoaded();
    QJsonObject results = _filesManager.resultsData();
    results[arrayKey] =
        QJsonArray{versionedResultRecord(record)};
    _filesManager.setResultsData(results);

    markDirtyIfRequested(markDirty);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, true);
    return true;
}

bool ProjectData::packResource(const QString &resourcePath, QString *errorMsg)
{
    if (_projectPath.isEmpty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("没有打开的项目");
        }
        return false;
    }

    const QFileInfo resource(resourcePath);
    if (!resource.isFile() && !resource.isDir())
    {
        if (errorMsg)
        {
            *errorMsg =
                QStringLiteral("待打包资源不存在: %1").arg(resourcePath);
        }
        return false;
    }

    QString stagedPath;
    if (!ProjectWorkspaceStore(_projectPath, _activeChunkDirectory)
             .stagePackedResource(resource.absoluteFilePath(),
                                  &stagedPath,
                                  errorMsg))
    {
        return false;
    }

    QJsonObject core = _filesManager.coreData();
    QJsonArray packed = core.value(QStringLiteral("packed_resources")).toArray();
    for (const QJsonValue &value : packed)
    {
        if (QDir::cleanPath(
                value.toObject().value(QStringLiteral("path")).toString())
            == stagedPath)
        {
            return true;
        }
    }
    packed.append(QJsonObject{
        {QStringLiteral("name"), resource.fileName()},
        {QStringLiteral("path"), stagedPath},
        {QStringLiteral("resource_type"),
         resource.isDir() ? QStringLiteral("directory")
                          : QStringLiteral("file")}
    });
    core[QStringLiteral("packed_resources")] = packed;
    _filesManager.setCoreData(core);
    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(true, false, true);
    return true;
}

QStringList ProjectData::getAllImages() const
{
    return _filesManager.getAllImages();
}

QStringList ProjectData::getImagesByCategory(const QString &category) const
{
    return _filesManager.getImagesByCategory(category);
}

QMap<QString, QString> ProjectData::getImageMatchOutputMap() const
{
    return _filesManager.getImageMatchOutputMap();
}

QString ProjectData::findMatchFile(const QString &imgA, const QString &imgB) const
{
    return _filesManager.findMatchFile(imgA, imgB);
}

void ProjectData::saveImageMatchingSettings(const QJsonObject &settings)
{
    _configManager.setWorkflowSettings("image_matching", settings);
    updateConfig(_configManager.data());
}

QJsonObject ProjectData::loadImageMatchingSettings() const
{
    return _configManager.workflowSettings("image_matching");
}

void ProjectData::saveUiSettings(const QJsonObject &settings)
{
    if (!hasProject())
    {
        return;
    }

    QJsonObject state = _projectUiState.isEmpty()
        ? defaultProjectUiState()
        : _projectUiState;
    ProjectUiConfigManager manager;
    manager.setData(
        state.value(QStringLiteral("display_settings")).toObject());
    manager.applyPatch(settings);
    state[QStringLiteral("display_settings")] = manager.data();
    if (normalizedProjectUiState(state) == normalizedProjectUiState(_projectUiState))
    {
        return;
    }
    // Project-scoped view state is persisted automatically, but changing a
    // panel, dock layout, or active view is not a project content edit and
    // must not trigger an unsaved-project prompt.
    updateProjectUiState(state, false);
    scheduleArchiveSync(false, false, true, false, true);
}

void ProjectData::markWorkspaceDirty()
{
    if (_projectPath.trimmed().isEmpty())
    {
        return;
    }
    markDirtyIfRequested(true);
    scheduleArchiveSync(false, false, true, false, false, true);
}

QJsonObject ProjectData::loadUiSettings() const
{
    const QJsonObject settings =
        _projectUiState.value(QStringLiteral("display_settings")).toObject();
    return settings.isEmpty()
        ? ProjectUiConfigManager::defaultUiSettings()
        : settings;
}

void ProjectData::appendImageMatchResult(const ProjectImageMatchResultRecord &record)
{
    ensureResultsLoaded();
    _filesManager.appendImageMatchResult(record);

    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, false);
}

void ProjectData::appendImageMatchResults(
    const QVector<ProjectImageMatchResultRecord> &records)
{
    if (records.isEmpty())
    {
        return;
    }

    ensureResultsLoaded();
    _filesManager.appendImageMatchResults(records);

    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, false);
}

// 辅助方法实现 — 通过 ProjectIO 解耦路径计算逻辑

// 返回项目根目录（即 .plascan 文件所在目录）
QString ProjectData::projectDir() const
{
    return ProjectIO::projectRootFromPlascan(_projectPath);
}

// 返回临时 project_files.json 的完整磁盘路径（.plascan_tmp/project_files.json）
QString ProjectData::tempFilesPath() const
{
    return ProjectIO::tempFilesPath(_projectPath);
}

// 返回临时 project_config.json 的完整磁盘路径（.plascan_tmp/project_config.json）
QString ProjectData::tempConfigPath() const
{
    return ProjectIO::tempConfigPath(_projectPath);
}

// 返回临时 project_results.json 的完整磁盘路径（.plascan_tmp/project_results.json）
QString ProjectData::tempResultsPath() const
{
    return ProjectIO::tempResultsPath(_projectPath);
}

QString ProjectData::tempUiStatePath() const
{
    return ProjectIO::tempUiStatePath(_projectPath);
}

// 返回 assets/ 目录路径
QString ProjectData::assetsDir() const
{
    return ProjectIO::projectAssetsDir(_projectPath);
}

// 返回 assets/images/ 目录路径
QString ProjectData::imagesDir() const
{
    return ProjectIO::projectImagesDir(_projectPath);
}
