#pragma once

/**
 * @file AerialTriangulationOptions.h
 * @brief “对齐照片/空中三角测量”工作流的外部参数和已准备输入契约。
 *
 * AerialTriangulationOptions 面向 GUI/CLI，包含连接点前端和 SfM/BA 两阶段参数；
 * Workflow::resolveConfig 会把它拆成 MatchPhotosOptions 与
 * PreparedAerialTriangulationInput。SfM 管线不得重新解释特征/匹配参数。
 */

#include "FramePinholeCamera.h"
#include "common/SfmTypes.h"

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>
#include <memory>

namespace xjw::aerial_triangulation
{

struct PreparedTiePointGraph;

// 面向 GUI/CLI 的“对齐照片”参数。连接点前端参数在 Workflow 中转换为
// MatchPhotosOptions，不允许继续传入纯 SfM 重建管线。
struct AerialTriangulationOptions
{
    // 当前处理集合和工程上下文。
    QStringList images; ///< 参与本次空三的影像绝对路径，顺序定义 ImageId。
    QStringList cameraPaths; ///< 可选外部相机文件；完整时必须与 images 一一对应。
    QString projectPath; ///< .plascan 工程路径，用于标记 sidecar 和项目根目录。
    QString outputDir; ///< 空三资产根目录，管线会在其下创建 sfm_sparse。
    QJsonObject projectMeta; ///< 调用时工程元数据快照。

    // Metashape 风格常用参数。
    QString quality = QStringLiteral("high"); ///< highest/high/medium/low/lowest 质量预设。
    bool genericPreselection = true; ///< 使用外观检索等通用方法缩小匹配候选对。
    bool referencePreselection = false; ///< 使用外部相机或照片序列先验规划候选对。
    QString referenceMode = QStringLiteral("source_code"); ///< source_code/estimated_pose/sequence。
    bool resetAlignment = true; ///< 丢弃旧 SfM 位姿，但不必删除匹配/连接点。
    // 重置相机对齐与重建连接点是两个独立动作。默认保留兼容的特征、匹配和连接点缓存，
    // 让用户可以在相同观测网络上重新尝试 SfM/BA。
    bool reuseExistingMatches = true; ///< 复用兼容缓存；false 才强制清理并重建连接点。
    bool saveAfterEachStep = false; ///< 由上层工程服务决定是否在阶段边界持久化。

    // 连接点前端配置。
    int keypointLimit = 40000; ///< 每幅影像检测关键点上限，0 表示前端约定的不限。
    int tiepointLimit = 4000; ///< 每幅影像最终连接点配额，按网格均匀化。
    QString maskApplyMode = QStringLiteral("none"); ///< none/keypoints/tie_points 等前端模式。
    bool excludeFixedTiePoints = true; ///< 删除跨帧像素位置近乎不动的伪连接点。
    bool guidedImageMatching = false; ///< 在已有几何/位姿时对弱 pair 执行引导补匹配。
    bool adaptiveCameraModelFitting = true; ///< 无可信内参时允许焦距粗搜后联合自标定。
    // Benchmark/reference-camera workflows can request exact external poses.
    // Normal GUI alignment keeps the existing soft-prior refinement behavior.
    bool lockInputCameraPoses = false; ///< 精确参考工作流中固定输入外参，不做软先验细化。
    bool useInitialPairHint = false; ///< 优先尝试指定初始对，仍需通过几何门控。
    ImageId initialImageId1 = kInvalidImageId; ///< 初始对第一幅影像 ID。
    ImageId initialImageId2 = kInvalidImageId; ///< 初始对第二幅影像 ID。

    // 算法、设备和缓存位置。
    QString matchingAlgorithmId = QStringLiteral("auto_sift"); ///< 统一算法注册标识。
    QString lightGlueTensorRtEnginePath; ///< 可选本机 TensorRT engine；留空时自动查找。
    QString lomaRTensorRtPackagePath; ///< LoMa-R 双 engine JSON 清单；留空时自动查找。
    int lomaRKeypointBudget = 0; ///< 0=显存感知自动选择，或 1024/2048/3840。
    QString device = QStringLiteral("auto"); ///< auto/cpu/cuda。
    int threads = 0; ///< 整体 CPU 线程预算；0 表示自动使用当前机器的逻辑核心数。
    int cudaDevice = 0; ///< CUDA 设备序号；SIFT 与 LightGlue 必须使用同一设备。
    int featureMaxImageDim = 0; ///< 特征输入最长边；0 使用质量预设。
    int cudaParallelPairs = 0; ///< 并行 GPU pair 请求值；0 由前端按显存决定。
    int featurePrefetchDepth = 2; ///< CUDA SIFT 前端并行预读影像数，范围由前端收敛。
    float matchThreshold = 0.15f; ///< LightGlue 最低匹配置信度。
    float siftMaximumRatio = 0.98f; ///< SIFT ratio/歧义门限的用户上限。
    float siftMinimumAdaptiveRatio = 0.78f; ///< 候选充足时自适应收紧的下限。
    bool adaptiveSiftRatio = true;
    double geometryReprojThreshold = 1.5; ///< USAC 几何内点的像素残差门限。
    int geometryMinInliers = 20; ///< 一个像对通过几何质量门控所需的最少内点数。
    double geometryMinInlierRatio = 0.18;
    double geometryMinGridCoverage = 0.12;
    int geometryMaxIterations = 10000; ///< USAC 最大采样迭代次数。
    int tiePointGridColumns = 8; ///< 连接点空间均匀化网格列数。
    int tiePointGridRows = 8; ///< 连接点空间均匀化网格行数。
    int maxTiePointsPerGridCell = 0; ///< 单网格连接点上限；0 按影像总限额自动计算。
    float stationaryTiePointMaxPixelMotion = 1.0f; ///< 固定连接点判定的最大跨影像像素位移。
    bool autoGenerateMissingMatches = false; ///< 复用已有结果时补齐缺失 pair。
    bool restrictPairs = false; ///< 仅匹配 allowedPairs；保留字段供调用方显式表达。
    QStringList allowedPairs; ///< 规范 pair key 列表；非空时使用 ManualOnly。
    QString assetsDir; ///< 可覆盖工程 assets 根目录。
    QString matchDir; ///< 可覆盖逐影像 `.pimatch` 分片目录。
    QMap<QString, QString> maskPaths; ///< 影像规范路径到蒙版路径。
    QMap<QString, FramePinholeCamera> referenceCameras; ///< 影像路径到可信参考相机。
    float featureGrayscaleMin = 5.0f / 255.0f; ///< 特征前端灰度有效下限。
    float featureGrayscaleMax = 1.0f; ///< 特征前端灰度有效上限。

    std::shared_ptr<std::atomic<bool>> cancelFlag; ///< 跨连接点/SfM 阶段共享取消标志。
    /// 阶段文本和整体百分比回调；由 Workflow 把连接点映射到 0-35%、SfM 映射到 35-100%。
    std::function<void(const QString &stage, int percent)> progressFn;
    /// 每个 pair 落盘后通知上层更新工程索引；不得在回调中阻塞匹配 worker。
    std::function<void(const QString &img0,
                       const QString &img1,
                       const QString &matchPath,
                       int numMatches)> pairMatchedFn;
};

// 已完成连接点准备后的 SfM/BA 输入。该类型刻意不包含任何特征提取和匹配参数。
struct PreparedAerialTriangulationInput
{
    QStringList images; ///< 稳定 ImageId 顺序。
    QStringList cameraPaths; ///< 可选一一对应外部相机文件。
    QString projectPath; ///< 工程路径，仅供标记/结果回写上下文。
    QString markerSetPath; ///< 完整标记系统 sidecar。
    QString tiePointPath; ///< matchphototask 生成的多视连接点 JSON。
    /// 同一次焦距搜索中由所有候选共享的只读连接点图。
    std::shared_ptr<const PreparedTiePointGraph> preparedTiePointGraph;
    QString outputDir; ///< 正式稀疏结果目录。
    QJsonObject projectMeta; ///< 解析相机内参和旧状态的只读快照。

    int quality = 2; ///< 已解析 SfM 质量级别，数值越大越保守/精细。
    int threads = 0; ///< CPU 线程预算；0 在运行时解析为当前机器逻辑线程数。
    QString device = QStringLiteral("auto"); ///< BA 等下游设备偏好。
    bool useProjectCameraIntrinsics = true; ///< 只使用来源可信的工程内参。
    bool useProjectCameraPoses = false; ///< false 时旧对齐外参不得作为当前解。
    bool adaptiveCameraModelFitting = true; ///< 最佳焦距种子后是否释放共享焦距细化。
    bool lockInputCameraPoses = false; ///< 已知位姿路径是否固定外参。
    bool useSequencePoseRecovery = false; ///< 使用相邻序号位姿为 PnP 提供插值/外推初值和缺口恢复。
    bool enforceSequencePoseConsistency = false; ///< 可选序列相邻距离门控，默认关闭。
    bool sequenceLoopClosure = false; ///< 候选排序时评估首尾闭环连续性。
    bool useInitialPairHint = false; ///< 是否携带显式初始对提示。
    ImageId initialImageId1 = kInvalidImageId; ///< 初始对第一 ID。
    ImageId initialImageId2 = kInvalidImageId; ///< 初始对第二 ID。
    // 无标定相机的初始焦距，以“焦距像素 / 影像最长边”表示。
    double estimatedFocalScale = 1.2;
    bool hasTrustedFocalPrior = false; ///< EXIF/固定镜头目录已提供可复现焦距，不再做无标定粗搜索。
    QString focalPriorSource; ///< 诊断用来源，例如 exif_focal_length_35mm。
    int focalPriorSampleCount = 0; ///< 批次中通过一致性检查的元数据样本数。

    // 以下字段只由 AerialTriangulationPipeline 为焦距候选试算设置，GUI/CLI 不直接暴露。
    // 大工程不能让每个候选都完整注册全部影像，否则一次焦距粗搜会重复执行十余次全量 SfM。
    bool coarseFocalEvaluation = false; ///< 使用低成本候选配置，不作为最终生产结果写出。
    int maxRegisteredImages = 0; ///< 候选最多注册影像数；0 表示不限制。

    std::shared_ptr<std::atomic<bool>> cancelFlag; ///< worker 间共享，只读取/置位。
    std::function<void(const QString &stage, int percent)> progressFn;
};

} // namespace xjw::aerial_triangulation
