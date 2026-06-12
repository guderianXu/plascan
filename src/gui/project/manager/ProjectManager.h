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

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QJsonArray>
#include <QMap>
#include <QPointer>

#include <atomic>
#include <memory>

class QWidget;
class ProjectData;
class ProjectReconstructionManager;
class ProjectTerrainProductsManager;
class ProjectCameraSetupManager;
class ProjectTaskDispatcher;
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
    
    // === 元数据变化信号（转发自 ProjectData） ===
    // 运行时项目元数据（project_files.json + project_results.json 合并视图）变化时发出
    void projectMetadataChanged(const QJsonObject &meta);
    // 元数据被更新并已持久化到临时缓存时发出（包含 plascanPath 便于接收方定位文件）
    void projectMetadataUpdated(const QString &plascanPath);
    // 脏状态变化：dirty=true 表示有未保存更改
    void metadataDirtyChanged(bool dirty);
    // 单个 ipfind 结果追加到元数据后发出，参数为对应的输入影像绝对路径
    void ipfindResultAppended(const QString &imagePath, const QString &suffix = QString());
    /// 每对影像匹配完成时实时发出（在主线程中）
    /// img0/img1: 影像绝对路径; matchFilePath: .match 文件路径; numMatches: 内点数
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

    // MVS 稠密重建进度（主窗口状态栏显示）
    // stage: 当前阶段描述；percent: 0~100
    void mvsProgressChanged(const QString &stage, int percent);
    // MVS 流程结束（success=true 表示正常完成）
    void mvsProgressFinished(bool success);

    // 网格重建进度（主窗口状态栏显示）
    void meshProgressChanged(const QString &stage, int percent);
    // 网格重建结束
    void meshProgressFinished(bool success);

    // 空中三角测量（AT/SFM）进度（主窗口状态栏显示）
    void atProgressChanged(const QString &stage, int percent);
    // AT 流程结束（success=true 表示正常完成）
    void atProgressFinished(bool success);

    // DEM 流水线进度
    void demPipelineProgressChanged(const QString &stage, int percent);
    void demPipelineFinished(bool success, const QString &message);

    // 观测网络构建进度（主窗口状态栏显示）
    void obsNetProgressChanged(const QString &stage, int percent);
    // 观测网络构建结束（success=true 正常完成）
    void obsNetProgressFinished(bool success);

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
    
    // === 资源管理（内部会显示文件对话框） ===
    // 弹出多选图片对话框，将选中图片引用添加到项目元数据
    void addPhoto();
    // 弹出文件夹选择对话框，将文件夹中所有支持格式的图片添加到项目
    void addFolder();
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
    // 从项目元数据中移除单个资源引用（不删除磁盘文件）
    void removeResource(const QString &resourcePath);
    // 批量移除资源引用
    void removeResources(const QStringList &resourcePaths);
    // 删除非照片生成数据（删除元数据记录及关联生成文件）
    void deleteGeneratedData(const QString &section, const QStringList &resourcePaths);
    // 将外部资源打包进 .plascan 归档（功能待完善）
    void packResource(const QString &resourcePath);
    
    // === 设置管理（委托给 ProjectData） ===
    // 加载旧版 UI 配置（project_config.json），仅供向后兼容迁移使用
    QJsonObject loadUiSettings() const;
    
    // 取消正在运行的 MVS 任务
    void cancelMvs();

    // 取消正在运行的 AT/SFM 任务
    void cancelAt();

    /// 设置 AT/SFM 取消标志（供 MenuWorkflowController 在启动后台任务时调用）
    void setAtCancelFlag(const std::shared_ptr<std::atomic<bool>> &flag)
    {
        m_atCancelFlag = flag;
    }
    
    // === 结果追加接口（供后台任务通过 Qt::QueuedConnection 调用） ===
    // 追加一条 ipfind 结果到项目元数据，并立即持久化到归档
    void appendIpfindResult(const QString &input, const QString &output, const QJsonObject &settings);
    // 追加批量 ipmatch 结果到项目元数据，并立即持久化到归档
    void appendIpmatchResult(const QStringList &outputs, const QJsonObject &settings);
    // 将影像-相机参数映射批量写回项目 ProjectData（供 AT 服务结果写回时调用）
    // 参数: cameras       - 影像绝对路径 → 相机 JSON 的映射
    //       updatedCount  - 可选，输出实际更新的影像数量
    //       errorMsg      - 可选，失败时的错误描述
    // 返回值: true 表示写入成功，false 表示部分或全部失败
    bool setImageCameras(const QMap<QString, QJsonObject> &cameras,
                         int *updatedCount = nullptr,
                         QString *errorMsg  = nullptr);
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
    // 异步启动立体视觉与点云/DEM生成全流程：
    //   images: 影像列表（通常为2张立体像对）
    //   outputDir: 输出目录
    //   threads: 并行线程数
    //   genPointCloud: 是否生成密集点云
    //   demResolution: DEM 分辨率（米/像素）
    //   demType: DEM 类型标识（"DTM" / "DSM"）
    //   t_srs: 目标空间参考系（WKT 或 EPSG 编码）
    void startStereoAndPoint2DemAsync(const QStringList &images,
                                      const QString &outputDir,
                                      int threads,
                                      bool genPointCloud,
                                      double demResolution,
                                      const QString &demType,
                                      const QString &t_srs);

    // 异步启动完整 DEM 流水线（自动模式）：
    //   images: 影像列表（2张立体像对）
    //   outputDir: DEM 输出目录
    //   pipelineSettings: 流水线配置（包含相机文件路径、DEM参数等）
    void startFullDemPipelineAsync(const QStringList &images,
                                   const QString &outputDir,
                                   const QJsonObject &pipelineSettings);

    // 异步从密集点云生成 DEM（手动模式）：
    //   denseCloudPath: 密集点云路径（为空则自动查找最新）
    //   outputDir: DEM 输出目录
    //   demResolution: DEM 分辨率
    //   demType: DEM 数据类型
    void startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                     const QString &outputDir,
                                     double demResolution,
                                     const QString &demType);
    // 异步启动正射影像制作（将影像投影到 DEM 上）
    void startMapProjectAsync(const QStringList &images,
                              const QString &demPath,
                              const QString &outputPath,
                              double resolution);
    // 异步启动模型生成（密集重建 → 网格 → 纹理 全流程）
    void startGenerateModelAsync();
    // 异步执行网格重建（从最近一次密集点云生成三角网格）
    void startMeshReconstructionAsync(const QJsonObject &settings);
    // 异步执行纹理映射（从最近一次网格生成 OBJ+MTL+PNG）
    void startTextureMappingAsync(const QJsonObject &settings);

    // 异步执行密集立体匹配（DenseMatchService）。
    // settings 支持的字段见 DenseMatchDialog::collectSettings()。
    void startDenseMatchAsync(const QJsonObject &settings);
    void startDenseMatchAsyncWithProgress(const QJsonObject &settings,
                                          std::shared_ptr<std::atomic<int>> progress,
                                          std::shared_ptr<std::atomic<bool>> cancelFlag = nullptr);

    // 异步执行初始稀疏点云三角化，并将结果注册到 aerial_triangulation_results。
    void startTriangulationAsync(const QJsonObject &settings);

    // 异步执行稀疏点云离群点分层剔除，并将新结果注册到 aerial_triangulation_results。
    // settings 支持的字段见 SparseCloudPostProcessDialog::collectSettings()。
    void startSparseCloudOutlierRemovalAsync(const QJsonObject &settings);

    // 异步执行稀疏点云空间清理，并将新结果注册到 aerial_triangulation_results。
    // settings 支持的字段见 SparseCloudPostProcessDialog::collectSettings()。
    void startSparseCloudLocalOptimAsync(const QJsonObject &settings);

    // 异步执行稀疏点云精修，并将新结果注册到 aerial_triangulation_results。
    // settings 支持的字段见 SparseCloudPostProcessDialog::collectSettings()。
    void startSparseCloudRefineAsync(const QJsonObject &settings);

    // 获取所有空三结果的摘要列表（供 DenseCloudDialog 选择AT结果）
    // 每项包含: index, created_at, image0, image1, output_dir, sparse_cloud_xyz
    QJsonArray getAvailableAtResults() const;
    // 仅执行深度图估计，并保存可复用的原始深度/置信图。
    void startEstimateDepthMapsAsync(const QJsonObject &settings);
    // 基于最近一次深度图估计结果执行融合，只生成密集点云。
    void startFuseDepthMapsAsync(const QJsonObject &settings);
    // 异步启动MVS稠密点云生成（接受来自DenseCloudDialog的完整配置JSON）
    void startGenerateDenseCloudAsync(const QJsonObject &settings);

    // 异步对已生成的密集点云执行后处理（SOR/体素下采样/法向量估计）
    // settings 支持的字段见 DenseCloudRefineDialog::collectSettings()
    void startDenseCloudRefineAsync(const QJsonObject &settings);

    // 接受 BA 预览结果，将待写入相机参数写回项目（此后预览缓存清空）
    bool acceptBundleAdjustPreview(QString *errorMsg = nullptr);
    // 丢弃 BA 预览结果（不修改任何相机参数）
    void discardBundleAdjustPreview();
    // 空三流程中自动应用 BA 结果：无弹窗、自动写回相机参数并发 atProgressFinished
    void applyBundleAdjustForAt(const QString &assetsDir,
                                const QStringList &images,
                                const QString &outputDir,
                                const QMap<QString, QJsonObject> &beforeCameras);
    // 追加空三（SFM）结果到 aerial_triangulation_results，并刷新 DataTree
    void appendAtResult(const QString &sparseCloudPath,
                        int sparsePointCount,
                        const QStringList &selectedImages,
                        const QString &outputDir,
                        const QJsonObject &extraRecord = {},
                        int replaceIndex = -1);

    /// 追加观测网络构建结果到 observation_network_results，并刷新 DataTree
    void appendObsNetResult(int nodeCount, int edgeCount,
                            const QString &algorithmName,
                            const QJsonObject &extraInfo = {});

    // 追加前方交会结果到项目元数据
    bool appendIntersectionResult(const QJsonObject &result, QString *errorMsg = nullptr);
    // 获取所有前方交会结果
    QJsonArray intersectionResults() const;
    
    // === 查询接口（委托给 ProjectData） ===
    bool isDirty() const;                               // 是否有未保存更改
    QString currentProjectPath() const;                 // 当前 .plascan 路径
    QJsonObject currentMeta() const;                    // 当前运行时元数据快照（含 results）
    QJsonObject coreProjectMeta() const;                // 仅核心数据（images+camera），无需惰性加载，速度极快
    QStringList getImagesByCategory(const QString &category) const; // 按类别获取影像
    QStringList getAllImages() const;                    // 获取所有影像路径
    QString findMatchFileForPair(const QString &imgA, const QString &imgB) const; // 查找匹配文件
    const ProjectData *projectData() const { return m_projectData; } // 获取底层数据对象
    bool hasTemporaryMeta() const;                      // 是否存在临时缓存
    void discardTemporaryMeta();                        // 删除临时缓存文件
    // 更新内存元数据并异步写入临时缓存（线程安全，适合在后台任务完成时调用）
    void writeMetadataToTempAsync(const QJsonObject &meta, bool markDirty = true);
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
    QWidget *m_parent = nullptr;                        // 父窗口指针（用于对话框父窗口）
    ProjectData *m_projectData = nullptr;               // 数据层：负责所有数据读写
    FileDialogStateManager *m_fileDialogState = nullptr; // 文件对话框状态（记住上次路径）
    ProjectReconstructionManager *m_reconstructionManager = nullptr;
    ProjectTerrainProductsManager *m_terrainProductsManager = nullptr;
    ProjectCameraSetupManager *m_cameraSetupManager = nullptr;
    ProjectTaskDispatcher *m_taskDispatcher = nullptr;
    ProjectUiCommands *m_uiCommands = nullptr;

    // AT/SFM 取消标志（跨线程共享）
    std::shared_ptr<std::atomic<bool>> m_atCancelFlag;

    // BA 预览缓存：BA 运行后先缓存到此处，
    // 用户点击"保留"后再通过 acceptBundleAdjustPreview() 写回项目相机参数。
    // 键: 影像绝对路径，值: 更新后的相机元数据 JSON
    QMap<QString, QJsonObject> m_pendingBaCameraMeta;
    QMap<QString, QJsonObject> m_pendingBaBeforeCameraMeta;
    QJsonObject m_pendingBaResult;    // BA 完整结果 JSON（含统计信息）
    bool m_hasPendingBaPreview = false; // 是否有待确认的 BA 预览结果

    // 辅助：统一提示框（默认标题“提示”）
    void showWarning(const QString &message,
                     const QString &title = QStringLiteral("提示")) const;
    // 辅助：校验项目是否已打开；失败时统一弹窗提示并返回 false
    bool ensureProjectOpen(const QString &message = QStringLiteral("请先打开项目"),
                           const QString &title = QStringLiteral("提示")) const;
};
