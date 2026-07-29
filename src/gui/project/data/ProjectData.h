// =============================================================================
// 文件名: ProjectData.h
// 描述:   项目数据层核心类声明。
//         ProjectData 是项目信息的唯一数据来源（Single Source of Truth），
//         封装了 .plascan 归档（ZIP格式）的读写、运行时元数据的内存管理，
//         以及临时缓存（.plascan_tmp/）的持久化策略。
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
#include <QStringList>
#include <QTimer>

#include <memory>

#include "ProjectFilesManager.h"
#include "ProjectConfigManager.h"

class QThreadPool;

struct ProjectOpenSnapshot
{
    bool success = false;
    QString projectPath;
    QString errorMessage;
    QJsonObject filesMeta;
    QJsonObject configMeta;
    bool recoveredFromTemporary = false;
    bool resultsLoaded = false;
};

struct ProjectResultsSnapshot
{
    bool success = false;
    QString projectPath;
    QString errorMessage;
    QJsonObject resultsMeta;
    bool hasResults = false;
};

// ProjectData: 轻量级项目数据管理类
// 职责:
// 1. 项目文件(.plascan)的读写
// 2. 元数据(JSON)的管理：core 写 project_files.json，workflow results 写 project_results.json
// 3. 资源路径的查询
// 不负责: UI交互、对话框、业务逻辑执行
class ProjectData : public QObject
{
    Q_OBJECT

public:
    explicit ProjectData(QObject *parent = nullptr);
    ~ProjectData() override;

    // === 项目基本信息 ===
    QString currentProjectPath() const { return _projectPath; }
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
    // 关闭当前项目
    void closeProject();

    // === 元数据访问 ===
    // 获取当前已加载的运行时元数据；getter 不执行文件 IO。
    QJsonObject metadata() const { return _filesManager.data(); }
    // 获取核心数据（仅 project_files.json，不触发惰性加载，快速）
    QJsonObject coreFilesMeta() const { return _filesManager.coreData(); }
    // 更新运行时元数据
    void updateMetadata(const QJsonObject &meta, bool markDirty = true);
    // 更新配置数据
    void updateConfig(const QJsonObject &config, bool markDirty = true);
    // 从临时目录加载元数据(.plascan_tmp)
    bool loadTemporaryMetadata();
    // 保存运行时元数据到临时目录
    bool saveTemporaryMetadata();
    // 合并并异步写入临时恢复数据。
    void scheduleTemporaryMetadataSave();
    // 删除临时元数据
    void clearTemporaryMetadata();
    // 检查是否存在临时元数据
    bool hasTemporaryMetadata() const;

    // === 资源管理 ===
    // 添加影像到项目（默认保存外部文件绝对路径引用，不复制原始文件）
    bool addImages(const QStringList &imagePaths, QString *errorMsg = nullptr);
    // 添加文件夹中的影像
    bool addImagesFromFolder(const QString &folderPath, QString *errorMsg = nullptr);
    // 移除资源引用(仅从元数据中删除)
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
    // 获取ipfind结果映射(input -> output)
    QMap<QString, QString> getIpfindOutputMap() const;
    // 查找匹配文件
    QString findMatchFile(const QString &imgA, const QString &imgB) const;
    
    // === 设置管理 ===
    // 保存/加载ipfind设置
    void saveIpfindSettings(const QJsonObject &settings);
    QJsonObject loadIpfindSettings() const;
    
    // 保存/加载ipmatch设置
    void setIpmatchSettings(const QJsonObject &settings);
    QJsonObject getIpmatchSettings() const;
    // 保存/加载UI设置
    void saveUiSettings(const QJsonObject &settings);
    QJsonObject loadUiSettings() const;

    // === 结果追加 ===
    // 追加ipfind结果到元数据
    void appendIpfindResult(const QString &input, const QString &output, const QJsonObject &settings);
    void appendIpfindResults(const QVector<ProjectIpfindResultRecord> &records);
    // 追加ipmatch结果到元数据
    void appendIpmatchResult(const QStringList &outputs, const QJsonObject &settings);
    void appendIpmatchResults(const QVector<ProjectIpmatchResultRecord> &records);

signals:
    // 项目状态变化
    void projectOpened(const QString &plascanPath);
    void projectSaved(const QString &plascanPath);
    void projectSaveCompleted(bool success, const QString &errorMessage);
    void projectClosed();
    
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
        TemporaryOnly
    };
    struct PersistenceSnapshot;
    struct PersistenceResult;

    QString _projectPath;                          // 当前.plascan路径
    mutable ProjectFilesManager _filesManager;     // project_files.json / project_results.json 管理（mutable 用于惰性加载）
    ProjectConfigManager _configManager;           // project_config.json 管理
    bool _isDirty = false;                         // 是否有未保存更改
    mutable bool _resultsLoaded = false;           // project_results.json 是否已载入内存

    // 防抖归档写入：每次 appendIpfind/appendIpmatch/setImageCameras 不再单次打开 ZIP，
    // 而是启动 2s 单射定时器，到期一次性批量写入
    QTimer *_archiveSyncTimer{};                   // 单射，2s防抖
    bool _resultsDirtyForArchive{false};           // project_results.json 需同步到归档
    bool _coreFileDirtyForArchive{false};          // project_files.json 需同步到归档
    bool _configDirtyForArchive{false};             // project_config.json 需同步到归档
    QThreadPool *_persistencePool{};
    bool _persistenceRunning{false};
    bool _fullSavePending{false};
    bool _archiveSyncPending{false};
    bool _temporarySavePending{false};
    bool _shuttingDown{false};
    std::unique_ptr<PersistenceSnapshot> _detachedPersistenceSnapshot;

    // 惰性加载 results：仅在首次访问时读取 project_results.json
    void ensureResultsLoaded() const;
    void markDirtyIfRequested(bool markDirty);
    void emitCurrentMetadataChanged();
    void scheduleArchiveSync(bool coreDirty,
                             bool resultsDirty,
                             bool writeTemporary,
                             bool configDirty = false);
    PersistenceSnapshot createPersistenceSnapshot(PersistenceMode mode) const;
    static PersistenceResult persistSnapshot(PersistenceSnapshot snapshot);
    void startNextPersistence();
    void handlePersistenceFinished(PersistenceResult result);

    // 辅助方法
    QString projectDir() const;        // 返回项目目录(.plascan文件所在目录)
    QString tempFilesPath() const;     // 返回临时 project_files.json 路径
    QString tempResultsPath() const;   // 返回临时 project_results.json 路径
    QString tempConfigPath() const;    // 返回临时 project_config.json 路径
    QString assetsDir() const;         // 返回assets目录
    QString imagesDir() const;         // 返回assets/images目录
};
