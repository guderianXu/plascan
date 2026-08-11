// =============================================================================
// 文件名: ProjectData.h
// 描述:   项目数据层核心类声明。
//         ProjectData 是项目信息的唯一数据来源（Single Source of Truth），
//         封装 .plascan + .files 工程读写、运行时元数据的内存管理，
//         以及应用缓存中运行工作区和崩溃恢复数据的持久化策略。
//
// 层次关系:
//   UI层(ProjectManager) --> 数据层(ProjectData) --> 存储层(PlascanArchive)
//
// 设计原则:
//   - ProjectData 不持有 QWidget、不弹对话框；持久化快照在专用串行线程执行
//   - 项目元数据写操作触发 metadataChanged / dirtyStateChanged 信号
//   - 临时保存（saveTemporaryMetadata）用于崩溃恢复
// =============================================================================
#pragma once

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QMap>
#include <QMutex>
#include <QStringList>
#include <QTimer>
#include <QtGlobal>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>

#include "project/ProjectDocumentModel.h"
#include "project/ProjectConfigManager.h"

class QThreadPool;
namespace xjw::common::project
{
class ProjectLock;

// Serializes the final filesystem commit of persistence snapshots. A cleanup
// advances the generation before committing its metadata, so an older worker
// can no longer overwrite the cleanup snapshot after the artifacts are gone.
class ProjectPersistenceCommitCoordinator final
{
public:
    quint64 currentGeneration() const;
    quint64 advanceGeneration();
    // Resource cleanup must not start moving artifacts until a filesystem
    // commit that already passed the generation check has left the lock.
    quint64 advanceGenerationAfterCurrentCommit();
    bool runIfCurrent(quint64 generation,
                      const std::function<void()> &commit);

private:
    QMutex _commitMutex;
    std::atomic<quint64> _generation{1};
};
}

struct ProjectOpenSnapshot
{
    bool success = false;
    QString projectPath;
    QString errorMessage;
    QJsonObject filesMeta;
    QJsonObject configMeta;
    QJsonObject uiState;
    QString chunkId;
    QString chunkName;
    int chunkDirectory = 0;
    bool recoveredFromTemporary = false;
    bool resultsLoaded = false;
};

struct ProjectResultsSnapshot
{
    bool success = false;
    QString projectPath;
    QString errorMessage;
    QJsonObject resultsMeta;
    QString chunkId;
    bool hasResults = false;
};

class ProjectResourceCleanupPersistence final
{
public:
    bool isValid() const;
    bool commitUpdated(QString *errorMsg = nullptr,
                       bool *archiveCommitted = nullptr) const;
    bool commitOriginal(QString *errorMsg = nullptr,
                        bool *archiveCommitted = nullptr) const;

private:
    friend class ProjectData;

    quint64 _generation = 0;
    QString _projectPath;
    QString _chunkId;
    int _chunkDirectory = 0;
    QJsonObject _originalMetadata;
    QJsonObject _updatedMetadata;
    bool _wasDirty = false;
    std::function<bool(QString *, bool *)> _commitUpdated;
    std::function<bool(QString *, bool *)> _commitOriginal;
};

// ProjectData: 轻量级项目数据管理类
// 职责:
// 1. 项目文件(.plascan)的读写
// 2. 元数据管理：core/results 写入 Chunk doc.json 的独立字段
// 3. 资源路径的查询
// 不负责: UI交互、对话框、业务逻辑执行
class ProjectData : public QObject
{
    Q_OBJECT

public:
    using ProjectOpenPreflight =
        std::function<bool(const QString &, QString *)>;

    explicit ProjectData(QObject *parent = nullptr);
    ~ProjectData() override;

    // Storage workflows can install a path-only recovery step that runs
    // before resource-index validation in loadProjectOpenSnapshot().
    static void installProjectOpenPreflight(ProjectOpenPreflight preflight);

    // === 项目基本信息 ===
    QString currentProjectPath() const { return _projectPath; }
    QString activeChunkId() const { return _activeChunkId; }
    QString activeChunkName() const { return _activeChunkName; }
    int activeChunkDirectory() const { return _activeChunkDirectory; }
    QJsonArray chunks() const;
    bool hasProject() const { return !_projectPath.isEmpty(); }
    bool isDirty() const { return _isDirty; }

    // === 项目生命周期 ===
    // 创建新项目
    bool createProject(const QString &plascanPath, const QString &projectName);
    // 打开已存在项目
    bool openProject(const QString &plascanPath, QString *errorMsg = nullptr);
    static ProjectOpenSnapshot loadProjectOpenSnapshot(const QString &plascanPath);
    static ProjectResultsSnapshot loadProjectResultsSnapshot(const QString &plascanPath);
    bool openProjectFromSnapshot(const ProjectOpenSnapshot &snapshot, QString *errorMsg = nullptr);
    bool applyResultsSnapshot(const ProjectResultsSnapshot &snapshot, QString *errorMsg = nullptr);
    // 保存项目(将运行时元数据写回.plascan归档)
    bool saveProject(QString *errorMsg = nullptr);
    // GUI 使用的异步保存入口；完成后发出 projectSaveCompleted。
    void saveProjectAsync();
    // 关闭当前项目；仅在最新状态已同步到归档或临时恢复快照后释放项目锁。
    bool closeProject(QString *errorMsg = nullptr);

    // === Chunk 管理 ===
    bool createChunk(const QString &name,
                     QString *createdChunkId = nullptr,
                     QString *errorMsg = nullptr);
    bool renameChunk(const QString &chunkId,
                     const QString &name,
                     QString *errorMsg = nullptr);
    bool removeChunk(const QString &chunkId,
                     QString *errorMsg = nullptr);
    bool switchChunk(const QString &chunkId,
                     QString *errorMsg = nullptr);

    // === 元数据访问 ===
    // 获取当前已加载的运行时元数据；getter 不执行文件 IO。
    QJsonObject metadata() const { return _filesManager.data(); }
    /// 返回包含惰性 results 字段的完整当前 Chunk 元数据。
    QJsonObject metadataIncludingResults() const;
    // 获取核心数据（仅 project_files 字段，不触发惰性加载，快速）
    QJsonObject coreFilesMeta() const { return _filesManager.coreData(); }
    // 更新运行时元数据
    void updateMetadata(const QJsonObject &meta, bool markDirty = true);
    // 更新配置数据
    void updateConfig(const QJsonObject &config, bool markDirty = true);
    std::optional<ProjectCameraModelPolicy> cameraModelPolicy() const;
    void setCameraModelPolicy(ProjectCameraModelPolicy policy);
    QJsonObject projectUiState() const { return _projectUiState; }
    void updateProjectUiState(const QJsonObject &state,
                              bool markDirty = true);
    // 从临时目录加载元数据(.plascan_tmp)
    bool loadTemporaryMetadata();
    // 保存运行时元数据到临时目录
    bool saveTemporaryMetadata();
    // 将资源清理后的元数据作为新的持久化代次同步提交到归档和恢复快照。
    // 旧后台 worker 在提交边界会被代次检查拒绝，返回成功后调用方才可清空事务区。
    bool commitResourceCleanupMetadata(const QJsonObject &metadata,
                                       QString *errorMsg = nullptr,
                                       bool *archiveCommitted = nullptr);
    ProjectResourceCleanupPersistence prepareResourceCleanupPersistence(
        const QJsonObject &updatedMetadata);
    bool finalizeResourceCleanupPersistence(
        const ProjectResourceCleanupPersistence &persistence,
        bool keepUpdatedMetadata,
        bool metadataStateCommitted);
    // 合并并异步写入临时恢复数据。
    void scheduleTemporaryMetadataSave();
    // 删除临时元数据
    void clearTemporaryMetadata();
    // 检查是否存在临时元数据
    bool hasTemporaryMetadata() const;

    // === 资源管理 ===
    // 添加影像到项目；立即复制到工程级共享影像库并跨 Chunk 去重。
    bool addImages(const QStringList &imagePaths, QString *errorMsg = nullptr);
    // 提交已经复制到工程级共享影像库的路径；仅更新当前 Chunk 元数据，不执行影像 IO。
    // GUI 异步导入流程在后台完成哈希与复制后调用此接口。
    bool addImagesFromSharedStore(const QStringList &projectImagePaths,
                                  int previouslySkipped = 0,
                                  QString *errorMsg = nullptr);
    // 添加文件夹中的影像
    bool addImagesFromFolder(const QString &folderPath, QString *errorMsg = nullptr);
    // 移除资源引用；共享影像仅在全部 Chunk 都解除引用后删除。
    bool removeResource(const QString &resourcePath);
    bool removeResources(const QStringList &resourcePaths);
    // 为指定影像写入相机元数据（写入 images[*].camera）
    bool setImageCamera(const QString &imagePath, const QJsonObject &cameraMeta, QString *errorMsg = nullptr);
    // 批量写入相机元数据，键为影像绝对路径，值为 camera 元数据
    bool setImageCameras(const QMap<QString, QJsonObject> &cameraMetaByImage,
                         int *updatedCount = nullptr,
                         QString *errorMsg = nullptr);
    /// 用一轮 SfM 的结果原子替换目标影像的相机集合；未出现在结果中的目标影像会清除旧相机。
    bool replaceImageCameras(const QStringList &targetImagePaths,
                             const QMap<QString, QJsonObject> &cameraMetaByImage,
                             int *updatedCount = nullptr,
                             int *clearedCount = nullptr,
                             QString *errorMsg = nullptr);
    /// 清除指定影像的相机参数（从元数据中移除 camera 字段）
    bool clearImageCameras(const QStringList &imagePaths,
                           int *clearedCount = nullptr,
                           QString *errorMsg = nullptr);
    bool appendIntersectionResult(const QJsonObject &result, QString *errorMsg = nullptr);
    QJsonArray getIntersectionResults() const;
    bool appendBundleAdjustResult(const QJsonObject &result, QString *errorMsg = nullptr);
    QJsonArray getBundleAdjustResults() const;
    bool appendResultRecord(const QString &arrayKey,
                            const QJsonObject &record,
                            bool markDirty = true);
    bool upsertResultRecordByPath(const QString &arrayKey,
                                  const QString &pathKey,
                                  const QJsonObject &record,
                                  bool markDirty = true);
    bool upsertResultRecordByIndex(const QString &arrayKey,
                                   const QJsonObject &record,
                                   int replaceIndex,
                                   bool markDirty = true);
    bool replaceResultRecordWithLatest(const QString &arrayKey,
                                       const QJsonObject &record,
                                       bool markDirty = true);
    // 打包外部资源到项目内部
    bool packResource(const QString &resourcePath, QString *errorMsg = nullptr);

    // === 查询接口 ===
    // 获取所有影像路径
    QStringList getAllImages() const;
    // 按类别获取影像
    QStringList getImagesByCategory(const QString &category) const;
    // 获取逐影像匹配分片映射(image -> .pimatch)
    QMap<QString, QString> getImageMatchOutputMap() const;
    // 查找匹配文件
    QString findMatchFile(const QString &imgA, const QString &imgB) const;

    // === 设置管理 ===
    // 保存/加载统一影像匹配设置。
    void saveImageMatchingSettings(const QJsonObject &settings);
    QJsonObject loadImageMatchingSettings() const;
    // 保存/加载UI设置
    void saveUiSettings(const QJsonObject &settings);
    QJsonObject loadUiSettings() const;
    // 运行工作区中的独立文件发生变化（例如 project_dialog.json）。
    void markWorkspaceDirty();

    // === 结果追加 ===
    // 追加逐影像匹配分片索引到元数据。
    void appendImageMatchResult(const ProjectImageMatchResultRecord &record);
    void appendImageMatchResults(const QVector<ProjectImageMatchResultRecord> &records);

signals:
    // 项目状态变化
    void projectOpened(const QString &plascanPath);
    void projectSaved(const QString &plascanPath);
    void projectSaveCompleted(bool success, const QString &errorMessage);
    void projectClosed();
    void chunkListChanged(const QJsonArray &chunks,
                          const QString &activeChunkId);
    void activeChunkChanged(const QString &chunkId,
                            const QString &chunkName,
                             int chunkDirectory);

    // 元数据变化
    void metadataChanged(const QJsonObject &meta);
    void dirtyStateChanged(bool dirty);

private Q_SLOTS:
    /// 防抖定时器触发：将项目数据批量写入 .plascan 归档（一次 zip_open/close）
    void syncToArchive();

private:
    enum class PersistenceMode
    {
        FullSave,
        ArchiveSync,
        TemporaryOnly,
        CleanupCommit
    };
    struct PersistenceSnapshot;
    struct PersistenceResult;

    QString _projectPath;                          // 当前.plascan路径
    QString _activeChunkId;
    QString _activeChunkName;
    int _activeChunkDirectory = 0;
    mutable ProjectFilesManager _filesManager;     // Chunk core/results 字段管理（mutable 用于惰性加载）
    ProjectConfigManager _configManager;           // Chunk project_config 字段管理
    bool _isDirty = false;                         // 是否有未保存更改
    mutable bool _resultsLoaded = false;           // project_results 字段是否已载入内存
    mutable bool _resultsLoading = false;          // 防止惰性加载重入，不把失败误标为已加载

    // 防抖归档写入：每次 appendIpfind/appendIpmatch/setImageCameras 不再单次打开 ZIP，
    // 而是启动 2s 单射定时器，到期一次性批量写入
    QTimer *_archiveSyncTimer{};                   // 单射，2s防抖
    bool _resultsDirtyForArchive{false};           // project_results 字段需同步
    bool _coreFileDirtyForArchive{false};          // project_files 字段需同步
    bool _configDirtyForArchive{false};            // project_config 字段需同步
    QJsonObject _projectUiState;                   // 根 doc.json 的 ui_state 字段
    bool _uiStateDirtyForArchive{false};
    bool _workspaceDirtyForArchive{false};          // workspace/ 资源索引需更新
    QThreadPool *_persistencePool{};
    bool _persistenceRunning{false};
    quint64 _runningPersistenceGeneration{0};
    quint64 _runningPersistenceSessionGeneration{0};
    quint64 _sessionGeneration{0};
    bool _fullSavePending{false};
    bool _archiveSyncPending{false};
    bool _temporarySavePending{false};
    quint64 _resourceCleanupPersistenceGeneration{0};
    bool _shuttingDown{false};
    std::shared_ptr<
        xjw::common::project::ProjectPersistenceCommitCoordinator>
        _persistenceCommitCoordinator;
    std::unique_ptr<xjw::common::project::ProjectLock> _projectLock;

    // 惰性加载 results：仅在首次访问时读取 project_results.json
    bool ensureResultsLoaded() const;
    void markDirtyIfRequested(bool markDirty);
    void emitCurrentMetadataChanged();
    void scheduleArchiveSync(bool coreDirty,
                             bool resultsDirty,
                             bool writeTemporary,
                             bool configDirty = false,
                             bool uiStateDirty = false,
                             bool workspaceDirty = false);
    PersistenceSnapshot createPersistenceSnapshot(PersistenceMode mode) const;
    static PersistenceResult persistSnapshot(PersistenceSnapshot snapshot);
    void startNextPersistence();
    void handlePersistenceFinished(PersistenceResult result);
    bool drainPersistenceForClose(QString *errorMessage);

    // 辅助方法
    QString projectDir() const;        // 返回项目目录(.plascan文件所在目录)
    QString tempFilesPath() const;     // 返回临时 project_files.json 路径
    QString tempResultsPath() const;   // 返回临时 project_results.json 路径
    QString tempConfigPath() const;    // 返回临时 project_config.json 路径
    QString tempUiStatePath() const;
    QString assetsDir() const;         // 返回assets目录
    QString imagesDir() const;         // 返回assets/images目录
};
