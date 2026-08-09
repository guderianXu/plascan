// =============================================================================
// 文件名: ProjectManager.h
// 描述:   项目操作的 UI 协调层核心类声明。
//
//         ProjectManager 是 UI 层与数据层之间的桥梁：
//           - 接收来自菜单/工具栏的用户操作（创建/打开/保存/关闭项目）
//           - 通过 QFileDialog 等对话框与用户交互，获取文件路径
//           - 将实际数据操作委托给 ProjectData（数据层）
//           - 将 ProjectData 的信号转发给更上层的 UI 组件
//
// 信号-槽流程示意：
//   MenuBar::actionNewProject()
//     --> ProjectManager::createNewProject()（显示保存对话框）
//         --> ProjectData::createProject()（写 .plascan 归档）
//             --> ProjectData::projectOpened 信号
//                 --> ProjectManager::projectCreated 信号（转发）
//                     --> MainWindow - 更新标题栏等
//
// 线程安全：ProjectManager 本身运行在主线程（GUI线程）；
//           后台任务由各工作流控制器或服务层启动，
//           结果通过 Qt::QueuedConnection 回调到主线程。
// =============================================================================
#pragma once

#include "Camera.h"
#include "project/ProjectDocumentModel.h"
#include "ProjectSessionContext.h"
#include "ProjectTerrainRequests.h"

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QJsonArray>
#include <QMap>
#include <QPointer>
#include <QStringList>

#include <atomic>
#include <memory>

class QWidget;
class GenerateMaskDialog;
class ProjectData;
class ProjectSparseReconstructionManager;
class ProjectPointCloudWorkflowController;
class ProjectLifecycleController;
class ProjectMaskWorkflowController;
class ProjectModelManager;
class ProjectTerrainProductsManager;
class ProjectCameraSetupManager;
class ProjectUiCommands;

// ProjectManager: 简化版 - 只负责UI对话框协调
// 职责:
// 1. 触发项目创建/打开/保存的文件对话框
// 2. 触发添加图片/文件夹的对话框
// 3. 协调 Dialog 与 ProjectData / 各服务层之间的通信
// 不负责: 数据管理(ProjectData)、具体后台任务执行
class ProjectManager : public QObject
{
    Q_OBJECT

public:
    // 构造时注入依赖：数据层(ProjectData)、父窗口(parent)
    explicit ProjectManager(ProjectData *projectData, QWidget *parent = nullptr);
    ~ProjectManager() override;

signals:
    // === 项目状态变化信号（转发自 ProjectData） ===
    // 新项目创建成功后发出（注意: projectOpened 也会同时发出）
    void projectCreated(const QString &plascanPath);
    // 项目打开成功后发出（新建/打开 均触发此信号）
    void projectOpened(const QString &plascanPath);
    // 项目保存成功后发出
    void projectSaved(const QString &plascanPath);
    // 项目关闭后发出（数据已清空）
    void projectClosed();
    // 项目打开、关闭或活动块切换后发出，用于作废旧会话的异步 UI 状态。
    void projectSessionChanged();
    void projectOpenStarted(const QString &plascanPath);
    void projectOpenProgressChanged(const QString &message, int percent);
    void projectOpenFinished(bool success, const QString &message);
    void chunkListChanged(const QJsonArray &chunks,
                          const QString &activeChunkId);
    
    // === 元数据变化信号（转发自 ProjectData） ===
    // 运行时项目元数据（project_files + project_results 合并视图）变化时发出
    void projectMetadataChanged(const QJsonObject &meta);
    // 元数据被更新并已持久化到临时缓存时发出（包含 plascanPath 便于接收方定位文件）
    void projectMetadataUpdated(const QString &plascanPath);
    // 脏状态变化：dirty=true 表示有未保存更改
    void metadataDirtyChanged(bool dirty);
    // 逐影像匹配分片写入项目后发出，用于刷新当前影像的匹配观测显示。
    void imageMatchResultAppended(const QString &imagePath);
    // 照片蒙版生成进度（主窗口状态栏右下角显示）
    void maskGenerationProgressChanged(const QString &stage, int done, int total);
    // 照片蒙版生成结束（success=true 表示全部目标影像生成成功）
    void maskGenerationFinished(bool success);
    // 影像导入进度；total=0 表示正在扫描，尚不能确定总数。
    void imageImportProgressChanged(const QString &stage, int done, int total);
    void imageImportFinished(bool success, const QString &message);
    // 照片蒙版生成完成后发出，供当前影像视图刷新轮廓覆盖层。
    void masksGenerated(const QStringList &imagePaths);
    /// 每对影像匹配完成时实时发出（在主线程中）
    /// img0/img1: 影像绝对路径; matchFilePath: owner `.pimatch` 分片; numMatches: 内点数
    void matchPairReady(const QString &img0, const QString &img1,
                        const QString &matchFilePath, int numMatches);
    // 光束法平差（Bundle Adjust）单次运行完成后发出预览结果
    // 此时结果尚未写回项目相机，用户需确认后才通过 acceptBundleAdjustPreview() 提交
    void bundleAdjustPreviewReady(const QJsonObject &preview);
    
    // === 保存操作状态信号 ===
    // saveProject() 开始执行时发出（可用于显示进度条）
    void saveStarted();
    // saveProject() 执行完毕时发出；success=false 时已弹出错误对话框
    void saveFinished(bool success);

    // 网格重建进度（主窗口状态栏显示）
    void meshProgressChanged(const QString &stage, int percent);
    // 网格重建结束
    void meshProgressFinished(bool success);

    // 深度图估计与稠密点云融合进度；与网格重建分开，避免两个任务共用状态。
    void pointCloudProgressChanged(const QString &stage, int percent);
    void pointCloudProgressFinished(bool success);
    void pointCloudResultReady(const QString &path, int pointCount);

    // 空中三角测量（AT/SFM）进度（主窗口状态栏显示）
    void atProgressChanged(const QString &stage, int percent);
    // AT 流程结束（success=true 表示正常完成）
    void atProgressFinished(bool success);

    // 支持并发实例的后台任务进度，供主窗口聚合到系统任务栏。
    void backgroundTaskProgressChanged(const QString &taskId, int value, int maximum);
    void backgroundTaskFinished(const QString &taskId);
    // DEM 流水线进度
    void demPipelineProgressChanged(const QString &stage, int percent);
    void demPipelineFinished(bool success, const QString &message);
    // 正射影像流水线使用独立信号，避免与 DEM 任务状态串线。
    void orthoPipelineStarted();
    void orthoPipelineProgressChanged(const QString &stage, int percent);
    void orthoPipelineFinished(bool success,
                               const QString &message,
                               const QJsonObject &result);

public slots:
    // === 项目操作（内部会显示文件对话框） ===
    // 显示"另存为"对话框，然后创建新 .plascan 项目
    void createNewProject();
    // 显示"打开"对话框，然后打开所选 .plascan 项目
    void openProject();
    // 直接从指定路径打开项目（不显示对话框，适合最近文件列表等场景）
    void openProjectFromPath(const QString &plascanPath);
    // 保存当前项目（将内存元数据写回 .plascan 归档）
    void saveProject();
    // 关闭当前项目（清空数据层状态，触发 projectClosed 信号）
    void closeProject();
    void createChunk();
    void renameChunk(const QString &chunkId);
    void removeChunk(const QString &chunkId);
    void switchChunk(const QString &chunkId);
    
    // === 资源管理（内部会显示文件对话框） ===
    // 弹出多选图片对话框，将选中图片引用添加到项目元数据
    void addPhoto();
    // 弹出文件夹选择对话框，将文件夹中所有支持格式的图片添加到项目
    void addFolder();
    // 导入 Metashape/通用 OBJ、PLY、XYZ 点云并登记为稠密点云成果。
    void importPointCloud();
    // 导入 Metashape/通用 OBJ、PLY 模型，并复制 OBJ 的材质与纹理依赖。
    void importModel();
    // 为单张影像导入 .tsai 相机文件（弹出单文件对话框）
    bool importCameraForImage(const QString &imagePath);
    // 批量按文件名匹配：选择一个含多个 .tsai 文件的目录，
    // 自动将同名影像与对应相机文件关联（baseName匹配，区分大小写不敏感，有歧义则跳过）
    bool importCamerasByFilenameBatch();
    // 按 EXIF/默认焦距为影像写入相机初值（内参 + 默认外参）
    bool initializeCamerasFromExifOrDefault(const QJsonObject &settings);
    // 按手工输入内参为影像写入相机初值（内参 + 默认外参）
    bool initializeCamerasFromIntrinsics(const QJsonObject &settings);
    // 使用相对定向 / 增量 SFM 为缺少相机文件或仅有内参的影像恢复初始化位姿
    bool initializeCameraPosesWithSFM(const QJsonObject &settings);
    // 批量移除资源引用
    void removeResources(const QStringList &resourcePaths);
    // 以外部引用方式导入 DEM/LiDAR/点云参考数据，默认用于精度检查。
    void importReferenceDataset();
    // 打开控制点/检查点/比例尺管理窗口，支持导入 CSV 并写入项目元数据。
    void openSurveyControlDialog();
    // 记录当前照片视图里的活跃影像，供“生成蒙版/当前照片”使用。
    void setActiveImagePath(const QString &imagePath);
    // 生成照片蒙版，并将 mask_path 写入对应影像元数据。
    void openGenerateMaskDialog();
    // 为照片面板当前选择生成蒙版；对话框默认作用于传入照片。
    void openGenerateMaskDialogForImages(const QStringList &selectedImages);
    // 取消正在运行的照片蒙版生成任务。
    void cancelMaskGeneration();
    // 生成参考 DEM/LiDAR 与当前项目成果的精度检查准备报告。
    void runReferenceQualityCheck();
    // 生成参考地形软约束 BA 前置检查报告；真正 BA 只在检查通过后进入后续流程。
    void prepareReferenceTerrainBundleAdjust();
    // 将参考数据记录写入 project_results/reference_datasets；按 path 去重。
    bool registerReferenceDataset(const QString &path,
                                  const QString &type = QString(),
                                  const QString &role = QStringLiteral("validation"),
                                  QString *errorMsg = nullptr);
    // 删除非照片生成数据（删除元数据记录及关联生成文件）
    void deleteGeneratedData(const QString &section, const QStringList &resourcePaths);
    // 将外部资源复制进同名 .files 数据目录
    void packResource(const QString &resourcePath);
    
    // === 设置管理（委托给 ProjectData） ===
    // 加载和保存根 doc.json 中独立的 ui_state。
    QJsonObject loadUiSettings() const;
    void saveUiSettings(const QJsonObject &settings);
    void markWorkspaceDirty();
    
    // 取消正在运行的模型生成任务
    void cancelModelGeneration();
    // 取消正在运行的深度图估计或点云融合任务。
    void cancelPointCloudGeneration();
    // 取消正在运行的正射影像生成任务。
    void cancelMapProject();

    // 取消正在运行的 AT/SFM 任务
    void cancelAt();

    /// 设置 AT/SFM 取消标志（供 MenuWorkflowController 在启动后台任务时调用）
    void setAtCancelFlag(const std::shared_ptr<std::atomic<bool>> &flag)
    {
        _atCancelFlag = flag;
    }

    bool hasActiveAtTask() const
    {
        return _atCancelFlag != nullptr;
    }

    bool ownsAtCancelFlag(const std::shared_ptr<std::atomic<bool>> &flag) const
    {
        return _atCancelFlag == flag;
    }

    void clearAtCancelFlag(const std::shared_ptr<std::atomic<bool>> &flag)
    {
        if (_atCancelFlag == flag)
        {
            _atCancelFlag.reset();
        }
    }
    
    // === 结果追加接口（供后台任务通过 Qt::QueuedConnection 调用） ===
    // 追加逐影像 `.pimatch` 分片索引，并立即持久化到归档。
    void appendImageMatchResults(const QVector<ProjectImageMatchResultRecord> &records);
    // 将影像-相机参数映射批量写回项目 ProjectData（供 AT 服务结果写回时调用）
    // 参数: cameras       - 影像绝对路径 → 相机 JSON 的映射
    //       updatedCount  - 可选，输出实际更新的影像数量
    //       errorMsg      - 可选，失败时的错误描述
    // 返回值: true 表示写入成功，false 表示部分或全部失败
    bool setImageCameras(const QMap<QString, QJsonObject> &cameras,
                         int *updatedCount = nullptr,
                         QString *errorMsg  = nullptr);
    /// 原子替换本轮 SfM 的对齐相机集合，并清除本轮未注册影像的旧位姿。
    bool replaceImageCameras(const QStringList &targetImagePaths,
                             const QMap<QString, QJsonObject> &cameras,
                             int *updatedCount = nullptr,
                             int *clearedCount = nullptr,
                             QString *errorMsg = nullptr);
    /// 清除指定影像的相机参数
    bool clearImageCameras(const QStringList &imagePaths,
                         int    *updatedCount = nullptr,
                         QString *errorMsg    = nullptr);
    // 异步启动光束法平差（Bundle Adjustment）：
    //   images: 参与平差的影像列表
    //   outputDir: 中间结果输出目录
    //   threads: 并行线程数
    //   dryRun: 仅验证输入不执行计算
    //   extraSettings: 附加求解参数（收敛阈值、迭代次数等）
    void startBundleAdjustAsync(const QStringList &images,
                                const QString &outputDir,
                                int threads,
                                bool dryRun,
                                const QJsonObject &extraSettings);
    void startDemFromPointCloudAsync(
        const xjw::gui::project::DemGenerationRequest &request);
    // 异步启动正射影像制作（settings 包含影像、DEM、输出路径与核心选项）。
    void startMapProjectAsync(
        const xjw::gui::project::OrthoGenerationRequest &request);
    void startGenerateModelAsync(const QJsonObject &settings);
    // 使用正式空三相机和稀疏云，异步估计/复用深度图并融合稠密点云。
    void startCreatePointCloudAsync(const QJsonObject &settings);
    // 异步执行网格重建。
    void startMeshReconstructionAsync(const QJsonObject &settings);
    // 异步执行纹理映射（从最近一次网格生成 OBJ+MTL+PNG）
    void startTextureMappingAsync(const QJsonObject &settings);

    // 异步执行初始稀疏点云三角化，并将结果注册到 aerial_triangulation_results。
    void startTriangulationAsync(const QJsonObject &settings);

    // 异步执行稀疏点云离群点分层剔除，并将新结果注册到 aerial_triangulation_results。
    // settings 由调用方提供稀疏点云离群点剔除参数。
    void startSparseCloudOutlierRemovalAsync(const QJsonObject &settings);

    // 异步执行稀疏点云空间清理，并将新结果注册到 aerial_triangulation_results。
    // settings 由调用方提供稀疏点云局部优化参数。
    void startSparseCloudLocalOptimAsync(const QJsonObject &settings);

    // 异步执行稀疏点云精修，并将新结果注册到 aerial_triangulation_results。
    // settings 由调用方提供稀疏点云后处理参数。
    void startSparseCloudRefineAsync(const QJsonObject &settings);

    // 接受 BA 预览结果，将待写入相机参数写回项目（此后预览缓存清空）
    bool acceptBundleAdjustPreview(QString *errorMsg = nullptr);
    // 丢弃 BA 预览结果（不修改任何相机参数）
    void discardBundleAdjustPreview();
    // 用最新空三（SFM）结果替换当前连接点，并刷新 DataTree
    bool replaceTiePointResult(const QString &sparseCloudPath,
                               int sparsePointCount,
                               const QStringList &selectedImages,
                               const QString &outputDir,
                               const QJsonObject &extraRecord = {});

    // 追加前方交会结果到项目元数据
    bool appendIntersectionResult(const QJsonObject &result, QString *errorMsg = nullptr);
    // 获取所有前方交会结果
    QJsonArray intersectionResults() const;
    
    // === 查询接口（委托给 ProjectData） ===
    bool isDirty() const;                               // 是否有未保存更改
    QString currentProjectPath() const;                 // 当前 .plascan 路径
    xjw::gui::project::ProjectSessionContext currentSessionContext() const;
    bool isCurrentSession(const xjw::gui::project::ProjectSessionContext &context) const;
    QJsonObject currentMeta() const;                    // 当前运行时元数据快照（含 results）
    QJsonObject coreProjectMeta() const;                // 仅核心数据（images+camera），无需惰性加载，速度极快
    QStringList getImagesByCategory(const QString &category) const; // 按类别获取影像
    QStringList getAllImages() const;                    // 获取所有影像路径
    QString findMatchFileForPair(const QString &imgA, const QString &imgB) const; // 查找匹配文件
    const ProjectData *projectData() const { return _projectData; } // 获取底层数据对象
    void discardTemporaryMeta();                        // 删除临时缓存文件
    // 刷新重建质量报告，并将报告注册到工作区目录树。
    void refreshReconstructionQualityReport();
    // 为指定影像列表中每张影像加载对应的 xjw::Camera 对象；
    // hasCamerasForAll 出参为 true 表示列表中每张影像均有有效相机参数。
    // 返回值： 影像路径 → Camera 的映射（已成功解析的影像）
    QMap<QString, xjw::Camera> getCamerasForImages(
            const QStringList &images,
            bool *hasCamerasForAll = nullptr) const;
    
    // 注入文件对话框状态管理器（记住上次使用的目录等）
    void setFileDialogStateManager(class FileDialogStateManager *manager);
    // 辅助：从 FileDialogStateManager 或默认路径获取上次使用的目录
    QString getLastUsedDir(const QString &key) const;
    // 辅助：将当前选择的目录保存到 FileDialogStateManager
    void saveLastUsedDir(const QString &key, const QString &dir);
private:
    QWidget *_parent = nullptr;                        // 父窗口指针（用于对话框父窗口）
    ProjectData *_projectData = nullptr;               // 数据层：负责所有数据读写
    FileDialogStateManager *_fileDialogState = nullptr; // 文件对话框状态（记住上次路径）
    ProjectSparseReconstructionManager *_sparseReconstructionManager = nullptr;
    ProjectPointCloudWorkflowController *_pointCloudWorkflowController = nullptr;
    ProjectModelManager *_modelManager = nullptr;
    ProjectTerrainProductsManager *_terrainProductsManager = nullptr;
    ProjectCameraSetupManager *_cameraSetupManager = nullptr;
    ProjectUiCommands *_uiCommands = nullptr;
    ProjectLifecycleController *_lifecycleController = nullptr;
    ProjectMaskWorkflowController *_maskWorkflowController = nullptr;
    quint64 _projectSessionGeneration = 0;
    QJsonObject _pendingAutomaticModelSettings;
    bool _automaticModelDepthPreparationActive = false;
    bool _imageImportActive = false;

    // AT/SFM 取消标志（跨线程共享）
    std::shared_ptr<std::atomic<bool>> _atCancelFlag;

    // BA 预览缓存：BA 运行后先缓存到此处，
    // 用户点击"保留"后再通过 acceptBundleAdjustPreview() 写回项目相机参数。
    // 键: 影像绝对路径，值: 更新后的相机元数据 JSON
    QMap<QString, QJsonObject> _pendingBaCameraMeta;
    QMap<QString, QJsonObject> _pendingBaBeforeCameraMeta;
    QJsonObject _pendingBaResult;    // BA 完整结果 JSON（含统计信息）
    bool _hasPendingBaPreview = false; // 是否有待确认的 BA 预览结果

    // 辅助：统一提示框（默认标题“提示”）
    void showWarning(const QString &message,
                     const QString &title = QStringLiteral("提示")) const;
    // 展示当前 BA 关键指标，并要求用户明确保留或丢弃待提交结果。
    void presentBundleAdjustPreview();
    void startImageImport(const QStringList &imagePaths,
                          const QString &sourceLabel);
    void importProjectAsset(bool modelAsset);
};
