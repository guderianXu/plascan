// =============================================================================
// 文件名: ProjectData.cpp
// 描述:   ProjectData 数据层实现。
//
//         主要逻辑：
//           1. createProject  - 初始化 .plascan ZIP 归档 + 目录结构
//           2. openProject    - 优先从临时缓存恢复，再读归档
//           3. saveProject    - 将内存元数据写回 .plascan 归档
//           4. addImages      - 追加影像引用到 images[] 数组
//           5. setImageCameras- 批量写入相机参数到 images[*].camera
//           6. appendXxx      - 各类结果追加，统一写入 project_results.json
//           7. saveIpfindSettings / saveUiSettings - 更新项目配置
//
//         持久化策略（双保险）：
//           - 运行时变更写 .plascan_tmp/ 做崩溃恢复，并通过防抖同步到 .plascan 归档
//           - 归档写失败时保留 .plascan_tmp/（防止数据丢失）
//           - 下次 openProject 时优先从 .plascan_tmp/ 恢复
// =============================================================================
#include "ProjectData.h"
#include "PlascanArchive.h"
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
#include <QDateTime>
#include <QSet>
#include <QThreadPool>
#include <QTimer>
#include <QUuid>
#include <QtConcurrent/QtConcurrent>

using xjw::common::project::ProjectIO;

struct ProjectData::PersistenceSnapshot
{
    PersistenceMode mode = PersistenceMode::TemporaryOnly;
    QString projectPath;
    QString temporaryCorePath;
    QString temporaryResultsPath;
    QString temporaryConfigPath;
    QJsonObject core;
    QJsonObject results;
    QJsonObject config;
    bool resultsLoaded = false;
    bool writeCoreToArchive = false;
    bool writeResultsToArchive = false;
    bool writeConfigToArchive = false;
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
};

namespace {

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

QJsonObject readJsonObjectEntry(PlascanArchive &archive, const QString &entryName, QString *error)
{
    const QByteArray bytes = archive.readEntry(entryName, error);
    if (bytes.isEmpty())
    {
        return QJsonObject();
    }

    const QJsonDocument doc = parseJsonOrCompressedJson(bytes);
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
                                      bool configDirty)
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

    if ((coreDirty || resultsDirty || configDirty)
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
    // 步骤1：构建 project_files.json 的默认 JSON 结构（空影像列表等）
    QJsonObject filesMeta = ProjectFilesManager::defaultFiles();

    // 步骤2：构建 project_config.json 的默认结构，记录项目名称与创建时间
    QJsonObject configMeta = ProjectConfigManager::defaultConfig();
    configMeta["project_name"] = projectName;
    configMeta["created_at"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    configMeta["version"] = "1.0";

    // 步骤3：构建 manifest.json —— ZIP 容器的元数据头，标明格式版本与类型
    QJsonObject manifest;
    manifest["format_version"] = "1.0";
    manifest["type"] = "plascan_project";

    // 步骤4：调用 PlascanArchive::createArchive 创建 .plascan 文件（ZIP）
    //         createArchive 会同时写入 manifest.json 和 project.json（兼容旧版）
    QString err;
    QJsonDocument filesDoc(filesMeta);
    QJsonDocument manifestDoc(manifest);

    if (!PlascanArchive::createArchive(plascanPath,
                                      manifestDoc.toJson(QJsonDocument::Compact),
                                      filesDoc.toJson(QJsonDocument::Compact),
                                      &err)) {
        LOG_ERROR(QStringLiteral("创建项目失败: %1").arg(err));
        return false;
    }

    // 步骤5：向归档追加 project_files.json 和 project_config.json（新版格式）
    PlascanArchive archive(plascanPath);
    if (archive.isValid()) {
        QJsonDocument configDoc(configMeta);
        archive.writeEntries(
            {
                qMakePair(
                    QStringLiteral("project_files.json"),
                    filesDoc.toJson(QJsonDocument::Compact)),
                qMakePair(
                    QStringLiteral("project_config.json"),
                    configDoc.toJson(QJsonDocument::Compact))
            },
            &err);
    }

    // 步骤6：更新内存状态
    _projectPath = plascanPath;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    updateMetadata(filesMeta, false);   // false = 不标记为脏（刚创建，不需要保存）
    updateConfig(configMeta, false);
    scheduleTemporaryMetadataSave();

    // 步骤7：在磁盘上创建项目所需的子目录结构（assets/images、assets/ip、assets/matches）
    QDir projectRoot(ProjectIO::projectRootFromPlascan(_projectPath));
    projectRoot.mkpath(QDir(projectRoot.path()).relativeFilePath(ProjectIO::projectImagesDir(_projectPath)));
    projectRoot.mkpath(QDir(projectRoot.path()).relativeFilePath(ProjectIO::ipfindOutputDir(_projectPath)));
    projectRoot.mkpath(QDir(projectRoot.path()).relativeFilePath(ProjectIO::ipmatchOutputDir(_projectPath)));

    LOG_INFO(QStringLiteral("项目创建成功: %1").arg(plascanPath));
    // 发出 projectOpened 信号（新建项目视同已打开）
    emit projectOpened(plascanPath);

    return true;
}

bool ProjectData::openProject(const QString &plascanPath, QString *errorMsg)
{
    // 先验证归档是否可访问
    PlascanArchive archive(plascanPath);
    if (!archive.isValid()) {
        if (errorMsg) *errorMsg = QStringLiteral("无法打开项目文件");
        return false;
    }

    // 记录路径并清空旧数据
    _projectPath = plascanPath;
    _filesManager.setData(QJsonObject());
    _configManager.setData(QJsonObject());
    _resultsLoaded = false;    // results 延迟加载：不在此时读取
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    _isDirty = false;

    // 优先从 .plascan_tmp/ 临时目录恢复（处理上次异常退出后的未保存数据）
    if (loadTemporaryMetadata()) {
        LOG_INFO(QStringLiteral("从临时目录加载项目数据"));
    }

    // 若临时目录中没有 core 数据，则从归档读取（仅读核心数据）
    QString err;
    if (_filesManager.coreData().isEmpty()) {
        // 优先尝试新版文件名（project_files.json），回退到旧版（project.json）
        QByteArray filesData = archive.readEntry("project_files.json", &err);
        if (filesData.isEmpty()) {
            filesData = archive.readEntry("project.json", &err);
        }

        if (filesData.isEmpty()) {
            _filesManager.setData(ProjectFilesManager::defaultFiles());
        } else {
            QJsonDocument doc = QJsonDocument::fromJson(filesData);
            QJsonObject obj = doc.object();
            // 旧格式：project_files.json 可能包含 results 键
            // 若包含，则通过 setData() 拆分，并标记 results 已加载（旧格式一次性读入）
            const bool hasLegacyResults = containsResultKeys(obj);
            _filesManager.setData(obj);    // setData() 自动拆分到 coreFiles + resultFiles
            if (hasLegacyResults) {
                _resultsLoaded = true;     // 旧格式数据已全部和 core 一起读入
                LOG_INFO(QStringLiteral("检测到旧格式项目，已将内嵌 results 拆分到独立存储"));
            }
        }
    }

    // 若临时目录中没有 config 数据，则从归档读取
    if (_configManager.data().isEmpty()) {
        QByteArray configData = archive.readEntry("project_config.json", &err);
        if (configData.isEmpty()) {
            updateConfig(ProjectConfigManager::defaultConfig(), false);
        } else {
            QJsonDocument doc = QJsonDocument::fromJson(configData);
            updateConfig(ProjectConfigManager::mergeWithDefaults(doc.object()), false);
        }
    }

    QJsonObject core = _filesManager.coreData();
    if (ensureImageUuids(&core))
    {
        _filesManager.setCoreData(core);
        markDirtyIfRequested(true);
        scheduleArchiveSync(true, false, true);
        LOG_INFO(QStringLiteral("已为旧工程影像补齐稳定 UUID"));
    }

    LOG_INFO(QStringLiteral("项目核心数据加载完成: %1").arg(plascanPath));
    emit projectOpened(plascanPath);

    return true;
}

ProjectOpenSnapshot ProjectData::loadProjectOpenSnapshot(const QString &plascanPath)
{
    ProjectOpenSnapshot snapshot;
    snapshot.projectPath = QDir::cleanPath(QFileInfo(plascanPath).absoluteFilePath());

    PlascanArchive archive(snapshot.projectPath);
    if (!archive.isValid())
    {
        snapshot.errorMessage = QStringLiteral("无法打开项目文件");
        return snapshot;
    }

    snapshot.filesMeta = readJsonObjectFile(ProjectIO::tempFilesPath(snapshot.projectPath));
    if (!snapshot.filesMeta.isEmpty())
    {
        snapshot.recoveredFromTemporary = true;
    }

    snapshot.configMeta = readJsonObjectFile(ProjectIO::tempConfigPath(snapshot.projectPath));
    if (!snapshot.configMeta.isEmpty())
    {
        snapshot.recoveredFromTemporary = true;
    }

    QString err;
    if (snapshot.filesMeta.isEmpty())
    {
        snapshot.filesMeta = readJsonObjectEntry(archive, ProjectFilesManager::kArchiveCoreFile, &err);
        if (snapshot.filesMeta.isEmpty())
        {
            snapshot.filesMeta = readJsonObjectEntry(archive, QStringLiteral("project.json"), &err);
        }
        if (snapshot.filesMeta.isEmpty())
        {
            snapshot.filesMeta = ProjectFilesManager::defaultFiles();
        }
    }

    if (snapshot.configMeta.isEmpty())
    {
        snapshot.configMeta = readJsonObjectEntry(archive, QStringLiteral("project_config.json"), &err);
        if (snapshot.configMeta.isEmpty())
        {
            snapshot.configMeta = ProjectConfigManager::defaultConfig();
        }
        else
        {
            snapshot.configMeta = ProjectConfigManager::mergeWithDefaults(snapshot.configMeta);
        }
    }
    else
    {
        snapshot.configMeta = ProjectConfigManager::mergeWithDefaults(snapshot.configMeta);
    }

    snapshot.resultsLoaded = containsResultKeys(snapshot.filesMeta);
    snapshot.success = true;
    return snapshot;
}

ProjectResultsSnapshot ProjectData::loadProjectResultsSnapshot(const QString &plascanPath)
{
    ProjectResultsSnapshot snapshot;
    snapshot.projectPath = QDir::cleanPath(QFileInfo(plascanPath).absoluteFilePath());

    snapshot.resultsMeta = readJsonObjectFile(ProjectIO::tempResultsPath(snapshot.projectPath));
    if (!snapshot.resultsMeta.isEmpty())
    {
        snapshot.success = true;
        snapshot.hasResults = true;
        return snapshot;
    }

    PlascanArchive archive(snapshot.projectPath);
    if (!archive.isValid())
    {
        snapshot.errorMessage = QStringLiteral("无法打开项目文件");
        return snapshot;
    }

    QString err;
    const QByteArray bytes = archive.readEntry(ProjectFilesManager::kArchiveResultsFile, &err);
    if (bytes.isEmpty())
    {
        snapshot.success = true;
        snapshot.hasResults = false;
        return snapshot;
    }

    const QJsonDocument doc = parseJsonOrCompressedJson(bytes);
    if (!doc.isObject())
    {
        snapshot.errorMessage = QStringLiteral("解析项目结果数据失败");
        return snapshot;
    }

    snapshot.resultsMeta = doc.object();
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

    if (_archiveSyncTimer)
    {
        _archiveSyncTimer->stop();
    }

    _projectPath = snapshot.projectPath;
    _filesManager.setData(QJsonObject());
    _configManager.setData(QJsonObject());
    _resultsLoaded = false;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
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
        markDirtyIfRequested(true);
        scheduleArchiveSync(true, false, true);
    }

    const QJsonObject configMeta = snapshot.configMeta.isEmpty()
        ? ProjectConfigManager::defaultConfig()
        : ProjectConfigManager::mergeWithDefaults(snapshot.configMeta);
    updateConfig(configMeta, false);

    emit dirtyStateChanged(_isDirty);
    emit projectOpened(_projectPath);
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
    snapshot.temporaryCorePath = tempFilesPath();
    snapshot.temporaryResultsPath = tempResultsPath();
    snapshot.temporaryConfigPath = tempConfigPath();
    snapshot.core = _filesManager.coreData();
    snapshot.resultsLoaded = _resultsLoaded;
    if (_resultsLoaded)
    {
        snapshot.results = _filesManager.resultsData();
    }
    snapshot.config = _configManager.data();

    if (mode == PersistenceMode::FullSave)
    {
        snapshot.writeCoreToArchive = true;
        snapshot.writeResultsToArchive = _resultsLoaded;
        snapshot.writeConfigToArchive = true;
        snapshot.writeTemporary = true;
    }
    else if (mode == PersistenceMode::ArchiveSync)
    {
        snapshot.writeCoreToArchive = _coreFileDirtyForArchive;
        snapshot.writeResultsToArchive = _resultsDirtyForArchive && _resultsLoaded;
        snapshot.writeConfigToArchive = _configDirtyForArchive;
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

    QVector<QPair<QString, QByteArray>> entries;
    if (snapshot.writeCoreToArchive)
    {
        entries.append(qMakePair(
            QString::fromLatin1(ProjectFilesManager::kArchiveCoreFile),
            QJsonDocument(snapshot.core).toJson(QJsonDocument::Compact)));
    }
    if (snapshot.writeResultsToArchive)
    {
        entries.append(qMakePair(
            QString::fromLatin1(ProjectFilesManager::kArchiveResultsFile),
            QJsonDocument(snapshot.results).toJson(QJsonDocument::Compact)));
    }
    if (snapshot.writeConfigToArchive)
    {
        entries.append(qMakePair(
            QStringLiteral("project_config.json"),
            QJsonDocument(snapshot.config).toJson(QJsonDocument::Compact)));
    }

    result.archiveRequested = !entries.isEmpty();
    result.archiveSuccess = !result.archiveRequested;
    if (result.archiveRequested)
    {
        PlascanArchive archive(snapshot.projectPath);
        if (!archive.isValid())
        {
            result.errorMessage = QStringLiteral("无法打开项目文件");
        }
        else
        {
            result.archiveSuccess = archive.writeEntries(entries, &result.errorMessage);
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
        if (_archiveSyncTimer)
        {
            _archiveSyncTimer->start(5000);
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
                    || _configDirtyForArchive;
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

    PlascanArchive archive(_projectPath);
    if (!archive.isValid()) {
        if (errorMsg) *errorMsg = QStringLiteral("无法打开项目文件");
        return false;
    }

    QVector<QPair<QString, QByteArray>> entries{
        qMakePair(
            QString::fromLatin1(ProjectFilesManager::kArchiveCoreFile),
            QJsonDocument(_filesManager.coreData()).toJson(QJsonDocument::Compact)),
        qMakePair(
            QStringLiteral("project_config.json"),
            QJsonDocument(_configManager.data()).toJson(QJsonDocument::Compact))
    };
    if (_resultsLoaded)
    {
        entries.insert(
            1,
            qMakePair(
                QString::fromLatin1(ProjectFilesManager::kArchiveResultsFile),
                QJsonDocument(_filesManager.resultsData()).toJson(QJsonDocument::Compact)));
    }

    QString err;
    if (!archive.writeEntries(entries, &err))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("写入项目归档失败: %1").arg(err);
        }
        return false;
    }

    _isDirty = false;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    emit dirtyStateChanged(false);
    emit projectSaved(_projectPath);

    clearTemporaryMetadata();
    LOG_INFO(QStringLiteral("项目已保存: %1").arg(_projectPath));
    return true;
}

void ProjectData::closeProject()
{
    if (_archiveSyncTimer)
    {
        _archiveSyncTimer->stop();
    }
    if (!_projectPath.trimmed().isEmpty()
        && (_isDirty
            || _coreFileDirtyForArchive
            || _resultsDirtyForArchive
            || _configDirtyForArchive
            || _temporarySavePending))
    {
        _detachedPersistenceSnapshot =
            std::make_unique<PersistenceSnapshot>(
                createPersistenceSnapshot(PersistenceMode::TemporaryOnly));
        _temporarySavePending = false;
        startNextPersistence();
    }
    _projectPath.clear();
    _filesManager.setData(QJsonObject());
    _configManager.setData(QJsonObject());
    _isDirty = false;
    _resultsLoaded = false;
    _resultsDirtyForArchive = false;
    _coreFileDirtyForArchive = false;
    _configDirtyForArchive = false;
    _fullSavePending = false;
    _archiveSyncPending = false;
    _temporarySavePending = false;
    emit projectClosed();
}

// 防抖定时器回调：把最新快照交给串行持久化线程。
void ProjectData::syncToArchive()
{
    if (_projectPath.isEmpty()
        || (!_resultsDirtyForArchive
            && !_coreFileDirtyForArchive
            && !_configDirtyForArchive))
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
                _filesManager.setResultsData(doc.object());
                LOG_INFO(QStringLiteral("从临时目录加载 results"));
                return;
            }
        }
    }

    // 从归档读取
    PlascanArchive archive(_projectPath);
    if (!archive.isValid()) return;

    QString err;
    // 从归档加载 results（Compact JSON，zip 内层已压缩）
    QByteArray data = archive.readEntry(ProjectFilesManager::kArchiveResultsFile, &err);
    if (data.isEmpty()) {
        // 归档无 results 条目——不是错误
        return;
    }
    const QJsonDocument doc = parseJsonOrCompressedJson(data);
    if (doc.isObject()) {
        _filesManager.setResultsData(doc.object());
        LOG_INFO(
            QStringLiteral("从归档惰性加载 results（%1 字节）").arg(data.size()));
    }
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

    return writeFileAtomically(
        configPath,
        QJsonDocument(_configManager.data()).toJson(QJsonDocument::Compact),
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
}

bool ProjectData::hasTemporaryMetadata() const
{
    auto exists = [](const QString &p) { return !p.isEmpty() && QFile::exists(p); };
    return exists(tempFilesPath()) || exists(tempResultsPath()) || exists(tempConfigPath());
}

bool ProjectData::addImages(const QStringList &imagePaths, QString *errorMsg)
{
    if (_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    // 获取现有影像列表，准备追加
    QJsonArray images = _filesManager.coreData().value("images").toArray();

    // 构建已有路径集合用于去重
    QSet<QString> existingPaths;
    existingPaths.reserve(images.size());
    for (const QJsonValue &val : images) {
        const QString p = val.toObject().value("path").toString();
        if (!p.isEmpty())
            existingPaths.insert(p);
    }

    int skipped = 0;
    for (const QString &srcPath : imagePaths) {
        const QString absPath = QFileInfo(srcPath).absoluteFilePath();

        // 跳过已存在的重复图片
        if (existingPaths.contains(absPath)) {
            ++skipped;
            continue;
        }
        existingPaths.insert(absPath); // 防止同批次内重复

        // 构建影像条目 JSON 对象：
        //   path:     影像绝对路径（引用型，不复制原始文件）
        //   type:     "reference" 表示外部引用（不打包到归档）
        //   added_at: UTC 时间戳，方便排序和溯源
        QJsonObject imgObj;
        imgObj["image_uuid"] = QUuid::createUuid().toString(QUuid::WithoutBraces);
        imgObj["path"] = absPath;
        imgObj["type"] = "reference";
        imgObj["added_at"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        images.append(imgObj);
    }

    if (skipped > 0 && errorMsg) {
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
    const QString projectRoot = QFileInfo(_projectPath).absolutePath();
    QStringList normalizedTargets;
    for (const QString &path : resourcePaths)
    {
        const QString normalized = normalizedProjectResourcePath(projectRoot, path);
        if (!normalized.isEmpty() && !normalizedTargets.contains(normalized))
        {
            normalizedTargets.append(normalized);
        }
    }

    QJsonArray images = _filesManager.coreData().value("images").toArray();
    QJsonArray newImages;

    for (const QJsonValue &val : images) {
        QJsonObject obj = val.toObject();
        const QString storedPath = normalizedProjectResourcePath(projectRoot, obj.value("path").toString());
        if (!normalizedTargets.contains(storedPath))
            newImages.append(val);
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
    array.append(record);
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
    const QString targetBaseName = QFileInfo(targetPath).fileName();
    const QJsonArray source = results.value(arrayKey).toArray();
    QJsonArray deduped;
    for (const QJsonValue &value : source)
    {
        const QString existingPath = value.toObject().value(pathKey).toString();
        if (existingPath == targetPath)
        {
            continue;
        }
        if (!targetBaseName.isEmpty() && QFileInfo(existingPath).fileName() == targetBaseName)
        {
            continue;
        }
        deduped.append(value);
    }
    deduped.append(record);
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
        array[replaceIndex] = record;
    }
    else
    {
        array.append(record);
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
    results[arrayKey] = QJsonArray{record};
    _filesManager.setResultsData(results);

    markDirtyIfRequested(markDirty);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, true);
    return true;
}

bool ProjectData::packResource(const QString &resourcePath, QString *errorMsg)
{
    // TODO: 实现打包外部资源
    if (errorMsg) *errorMsg = QStringLiteral("打包功能未实现");
    return false;
}

QStringList ProjectData::getAllImages() const
{
    return _filesManager.getAllImages();
}

QStringList ProjectData::getImagesByCategory(const QString &category) const
{
    return _filesManager.getImagesByCategory(category);
}

QMap<QString, QString> ProjectData::getIpfindOutputMap() const
{
    return _filesManager.getIpfindOutputMap();
}

QString ProjectData::findMatchFile(const QString &imgA, const QString &imgB) const
{
    return _filesManager.findMatchFile(imgA, imgB);
}

void ProjectData::saveIpfindSettings(const QJsonObject &settings)
{
    _configManager.setWorkflowSettings("ipfind", settings);
    updateConfig(_configManager.data());

    // 立即写入项目配置：如果存在已打开项目，则直接将 project_config.json 写入归档，
    // 以避免只写入临时元数据的行为。若未打开项目，则回落为保存到临时目录。
    if (!_projectPath.isEmpty())
    {
        PlascanArchive archive(_projectPath);
        if (archive.isValid()) 
        {
            QString err;
            QJsonDocument configDoc(_configManager.data());
            if (!archive.writeEntry("project_config.json", configDoc.toJson(QJsonDocument::Compact), &err)) 
            {
                LOG_WARN(QStringLiteral("保存 ipfind 设置到项目归档失败: %1").arg(err));
                // 写入失败：不回落到临时元数据以避免在无项目状态下产生文件
            }
        } 
        else 
        {
            // 无效归档：记录警告，但不写入临时元数据
            LOG_WARN(QStringLiteral("无法打开项目归档以保存 ipfind 设置"));
        }
    } 
    else 
    {
        // 没有打开项目：按照要求不写入任何东西，使用默认配置即可
        LOG_INFO(QStringLiteral("未保存 ipfind 设置：没有打开的项目，保留默认设置"));
    }
}

QJsonObject ProjectData::loadIpfindSettings() const
{
    return _configManager.workflowSettings("ipfind");
}

void ProjectData::setIpmatchSettings(const QJsonObject &settings)
{
    _configManager.setWorkflowSettings("ipmatch", settings);
    updateConfig(_configManager.data(), false);
}

QJsonObject ProjectData::getIpmatchSettings() const
{
    return _configManager.workflowSettings("ipmatch");
}

void ProjectData::saveUiSettings(const QJsonObject &settings)
{
    _configManager.setUiSettings(settings);
    updateConfig(_configManager.data());
}

QJsonObject ProjectData::loadUiSettings() const
{
    return _configManager.uiSettings();
}

void ProjectData::appendIpfindResult(const QString &input, const QString &output, const QJsonObject &settings)
{
    ensureResultsLoaded();
    _filesManager.appendIpfindResult(input, output, settings);

    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, true);
}

void ProjectData::appendIpfindResults(const QVector<ProjectIpfindResultRecord> &records)
{
    if (records.isEmpty())
    {
        return;
    }

    ensureResultsLoaded();
    _filesManager.appendIpfindResults(records);

    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, true);
}

void ProjectData::appendIpmatchResult(const QStringList &outputs, const QJsonObject &settings)
{
    ensureResultsLoaded();
    _filesManager.appendIpmatchResult(outputs, settings);

    markDirtyIfRequested(true);
    emitCurrentMetadataChanged();
    scheduleArchiveSync(false, true, false);
}

void ProjectData::appendIpmatchResults(const QVector<ProjectIpmatchResultRecord> &records)
{
    if (records.isEmpty())
    {
        return;
    }

    ensureResultsLoaded();
    _filesManager.appendIpmatchResults(records);

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
