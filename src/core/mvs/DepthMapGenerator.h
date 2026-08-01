#pragma once
// =============================================================================
// 文件: DepthMapGenerator.h
// 模块: MVS - Qt 封装的深度图生成 + 融合 + 点云生成
// 说明:
//   管理完整 MVS 流程：
//     1. 对每个参考帧调用 PatchMatchCUDA::estimate
//     2. DepthMapFusion::fuse (COLMAP BFS) → 直接输出 3D FusedPoint
//     3. 通过 Qt 信号将中间结果与最终点云发送到 UI 层
// =============================================================================

#include "MvsTypes.h"
#include "PatchMatchCUDA.h"
#include "DepthMapFusion.h"
#include "DepthPyramidEstimator.h"
#include "DepthFrameQualityGate.h"
#include "DepthCompletenessMetrics.h"
#include "MvsQualityReport.h"
#include "MvsSceneClassifier.h"
#include "DenseCloudBuilder.h"
#include "DensePointCloudCUDA.h"
#include "MvsSourcePlanner.h"
#include "MvsViewSelection.h"
#include "MvsWorkspaceManifest.h"
#include "SparseCloudPreprocessor.h"

#include <QObject>
#include <QFuture>
#include <QString>
#include <QSharedPointer>
#include <QMetaType>
#include <QJsonObject>
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <cstdint>

namespace xjw
{
namespace mvs
{

/// 单帧深度估计结果
struct DepthFrameResult
{
    int refViewIdx = -1;  ///< 参考帧在 views 数组中的下标
    bool depthFlippedZ = false;
    Camera cameraModel;  ///< 与输出深度栅格严格对应的正深度、零畸变工作相机
    std::vector<int> sourceViewIndices;  ///< PatchMatch 实际使用的源视图下标，用于限制一致性检查范围
    std::vector<MvsSourcePlanEntry> sourceViewPlan; ///< 实际源视图的可审计几何选择依据
    int requestedSourceViewCount = 0; ///< 选择阶段请求的源视图数，不随缓存释放丢失
    int sourceViewShortfall = 0; ///< 请求数与实际可用源视图数之差
    std::string sourceViewShortfallReason; ///< 源视图不足的主要原因
    QSharedPointer<cv::Mat> depthMap;    ///< 深度图 (CV_32F)
    QSharedPointer<cv::Mat> confidence;  ///< 置信图 (CV_32F)
    QSharedPointer<cv::Mat> normalMap;   ///< 最终层法线图 (CV_32FC3)，可为空
    QSharedPointer<cv::Mat> supportCount; ///< 最终层多视支持计数 (CV_16U)
    QSharedPointer<cv::Mat> geometrySupportCount; ///< 参考帧+跨视几何确认数 (CV_16U)
    QSharedPointer<cv::Mat> geometrySourceMask; ///< bit N 对应 sourceViewIndices 的第 N 个来源 (CV_16U)
    QSharedPointer<cv::Mat> inverseDepthMean; ///< 几何确认观测的逆深度均值 (CV_32F)
    QSharedPointer<cv::Mat> inverseDepthRelativeSpread; ///< 逆深度相对标准差 (CV_32F)
    QSharedPointer<cv::Mat> adaptiveGeometrySupportWeight; ///< 连续跨视几何支持权重 (CV_32F)
    QSharedPointer<cv::Mat> adaptiveGeometryEffectiveViewCount; ///< 连续证据有效视图数 (CV_32F)
    QSharedPointer<cv::Mat> adaptiveGeometryConflictRatio; ///< 可观测证据中的冲突比例 [0, 1] (CV_32F)
    QSharedPointer<cv::Mat> crossViewRepairedMask; ///< 跨视图补回像素；不参与帧准入评分 (CV_8U)
    QSharedPointer<cv::Mat> validMask;   ///< 最终输出空间的权威有效蒙版 (CV_8U)
    QSharedPointer<cv::Mat> supportRegionMask; ///< 项目/内容允许参与重建的区域，不含深度孔洞
    DepthPostProcessStats depthPostprocess; ///< 融合前深度图后处理统计
    DepthCompletenessDiagnostics depthCompleteness; ///< 蒙版内覆盖和逐阶段损失诊断
    QJsonObject crossViewRepairDiagnostics; ///< 跨视补回和锚定插值的逐原因统计
    std::vector<DepthLevelSummary> pyramidLevels; ///< 三级深度估计逐层摘要
    std::vector<DepthLevelResult> intermediatePyramidLevels; ///< 可选的 L3/L2 调试结果
    DepthMapQualityMetrics qualityMetrics; ///< 帧级覆盖、连通性与搜索边界统计
    DepthFrameQualityDecision qualityDecision; ///< 是否允许进入多视融合
    std::string maskSource;                  ///< project/content/full_image
    float maskCoverage = 1.0f;               ///< 参考影像中允许参与 MVS 的像素比例
    int selectedLevel = 0;                   ///< 实际采用的深度金字塔层级
    std::string fallbackReason;              ///< 层级减少或细层失败原因
    int pyramidRequestedLevelCount = 3;
    int pyramidActiveLevelCount = 0;
    int pyramidMinimumShortSide = 0;
    std::string pyramidDegradedReason;
    float effectivePatchMatchConfidenceThreshold = 0.0f;
    bool depthPostprocessApplied = false;   ///< true 表示 depthMap/confidence 已应用上述后处理
    bool success = false;
    double elapsedMs = 0.0;               ///< 单帧深度估计耗时，不含异步写盘
    std::string device;                   ///< 实际调度设备：GPU/CPU
    std::string errorMsg;

    bool eligibleForFusion() const
    {
        return success && qualityDecision.acceptance == DepthFrameAcceptance::Accepted;
    }

    bool eligibleForConsistencyCheck() const
    {
        return success && qualityDecision.acceptance != DepthFrameAcceptance::Rejected;
    }

    bool eligibleAsConsistencySource() const
    {
        return eligibleForConsistencyCheck();
    }

    void releasePixelStorage()
    {
        depthMap.clear();
        confidence.clear();
        normalMap.clear();
        supportCount.clear();
        geometrySupportCount.clear();
        geometrySourceMask.clear();
        inverseDepthMean.clear();
        inverseDepthRelativeSpread.clear();
        adaptiveGeometrySupportWeight.clear();
        adaptiveGeometryEffectiveViewCount.clear();
        adaptiveGeometryConflictRatio.clear();
        crossViewRepairedMask.clear();
        validMask.clear();
        supportRegionMask.clear();
        intermediatePyramidLevels.clear();
    }
};

class DepthMapGenerator : public QObject
{
    Q_OBJECT

public:
    explicit DepthMapGenerator(QObject *parent = nullptr);

    ~DepthMapGenerator() override;

    /// 设置输入数据
    void setViews(const std::vector<CameraView> &views);
    void setSparseCloud(const SparseCloud &sparse);
    void setConfig(const DepthGenConfig &config);
    void setSkippedFrameIndices(const std::vector<int> &indices);

    const std::vector<CameraView> &views() const
    {
        return _views;
    }

    const SparseCloud &sparse() const
    {
        return _sparse;
    }

    const DepthGenConfig &config() const
    {
        return _config;
    }

    /// 异步启动
    Q_INVOKABLE void start();

    /// 设置输出目录（深度图 PNG 保存位置）
    void setOutputDir(const std::string &dir)
    {
        _outputDir = dir;
    }

    /// 请求取消
    void cancel()
    {
        _cancelled = true;
    }

    void requestCancel()
    {
        _cancelled = true;
    }

    bool isCancelled() const
    {
        return _cancelled.load();
    }

    /// 基于稀疏点投影生成深度过滤支撑区，返回 CV_8U 掩码 (255=保留)；覆盖率不可靠时返回空 Mat
    static cv::Mat buildSparseSupportMask(const std::vector<CameraView> &views,
                                          const SparseCloud &sparse,
                                          int refIdx,
                                          int W,
                                          int H,
                                          const std::vector<int> &sourceIndices = {});

    /// 将同一帧可见稀疏点投影一次，供 hint 与支撑掩码在不同工作分辨率复用
    static std::vector<ProjectedSparseDepthSample> collectProjectedSparseDepthSamples(
        const SparseCloud &sparse,
        const Camera &camera,
        int imageWidth,
        int imageHeight,
        const std::vector<size_t> &visiblePointIndices);

    /// 基于已投影样本生成 PatchMatch hint 深度图
    static cv::Mat buildHintDepthFromProjectedSamples(
        int refIdx,
        int W,
        int H,
        const std::vector<ProjectedSparseDepthSample> &samples);

    /// 基于已投影样本仅生成局部种子深度，不做距离传播；用于精细层覆盖粗层 hint
    static cv::Mat buildSparseSeedDepthFromProjectedSamples(
        int refIdx,
        int W,
        int H,
        const std::vector<ProjectedSparseDepthSample> &samples,
        int seedRadius = 3);

    /// 基于已投影样本生成稀疏支撑掩码，避免重复投影同一批稀疏点
    static cv::Mat buildSparseSupportMaskFromProjectedSamples(
        int refIdx,
        int W,
        int H,
        const std::vector<ProjectedSparseDepthSample> &samples);

    /// 基于原始灰度图生成内容区域掩码；近似全图有效时返回空 Mat 表示跳过过滤
    static cv::Mat buildContentMask(const cv::Mat &gray,
                                    float *coverage = nullptr,
                                    double *otsuThreshold = nullptr,
                                    int *adaptiveThreshold = nullptr);

    /// 将项目蒙版转换为 MVS 有效区：项目蒙版非零=排除，返回值 255=有效
    static cv::Mat projectMaskToValidMask(const cv::Mat &projectMask,
                                          cv::Size targetSize);

    /// 暗背景环拍物体可用内容亮度挖出项目蒙版内部开口；保护外轮廓并限制最大移除比例
    static cv::Mat refineOrbitalProjectValidMask(const cv::Mat &gray,
                                                 const cv::Mat &projectValidMask,
                                                 bool *refined = nullptr,
                                                 float *retainedRatio = nullptr);

    /// 稀疏点支撑只作为置信度软先验，不直接删除 PatchMatch 深度像素
    static void applySparseSupportPrior(cv::Mat &depthMap,
                                        cv::Mat &confidenceMap,
                                        const cv::Mat &supportMask,
                                        int refIdx);

    /// 融合前基于局部中值剔除孤立深度突刺，并同步清零对应置信度
    static int removeLocalDepthOutliers(cv::Mat &depthMap,
                                        cv::Mat &confidenceMap,
                                        int kernelSize,
                                        float relDepthThreshold,
                                        float maxRemovalRatio,
                                        int refIdx);

    /// 融合前剔除孤立小连通域深度斑点，并同步清零对应置信度
    static int removeSmallDepthComponents(cv::Mat &depthMap,
                                          cv::Mat &confidenceMap,
                                          int minComponentArea,
                                          float maxRemovalRatio,
                                          int refIdx);

    /// 融合前统一后处理深度图，并返回置信度/局部离群过滤统计
    static DepthPostProcessStats postprocessFusionDepthMap(cv::Mat &depthMap,
                                                           cv::Mat &confidenceMap,
                                                           const FusionConfig &config,
                                                           int refIdx,
                                                           int viewCount);

    /// CUDA PatchMatch 显存不足后的下一次重试配置
    static PatchMatchConfig nextCudaRetryPatchMatchConfig(const PatchMatchConfig &config,
                                                          int imageWidth,
                                                          int imageHeight);

signals:
    /// 每估计完一帧就发出
    void depthMapReady(DepthFrameResult result);
    /// 每帧深度图保存为 PNG 后发出（path, width, height, refImagePath）
    void depthMapSaved(QString pngPath, int width, int height, QString refImagePath);
    /// 每帧深度图全部产物保存后发出结构化元数据，供项目树增量刷新
    void depthMapArtifactSaved(QJsonObject artifact);
    /// 点云生成完毕
    void pointCloudReady(std::vector<DensePoint> cloud);
    /// 进度更新
    void progressChanged(QString stage, float ratio);
    /// 出错
    void errorOccurred(QString msg);
    /// 整个流程完成
    void finished(bool success);

private:
    struct FrameMvsCache
    {
        std::vector<size_t> visiblePointIndices;
        std::vector<size_t> sourceSharedPointIndices;
        std::vector<int> sourceViewIndices;
        std::vector<MvsSourcePlanEntry> sourceViewScores;
        int requestedSourceViewCount = 0;
        int sourceViewShortfall = 0;
        std::string sourceViewShortfallReason;
    };

    /// 在 QtConcurrent 线程中运行的主函数
    void runInBackground();

    /// 计算单帧深度图
    DepthFrameResult computeDepthForView(int refIdx, const DepthGenConfig *configOverride = nullptr);

    /// 预计算 MVS 可见性与源视图候选，避免每帧重复全量扫描稀疏点
    void prepareFrameCaches();
    void clearFrameCaches();
    std::vector<int> sourceViewIndicesForFrame(int refIdx, int maxSources) const;
    std::vector<size_t> visibleSparsePointIndicesForFrame(int refIdx,
                                                          const std::vector<int> &sourceIndices,
                                                          int minSourceViews) const;
    bool isSparsePointVisibleInFrame(int viewIdx, size_t pointIndex) const;

    /// 将 DepthFrameResult 组装为 FusionFrameInput
    FusionFrameInput buildFusionFrame(const DepthFrameResult &res) const;

    /// 估计参考帧的深度范围
    bool estimateDepthRange(int refIdx,
                            float &zNear,
                            float &zFar,
                            const std::vector<int> &sourceIndices = {}) const;
    bool estimateDepthRangeFromVisiblePoints(int refIdx,
                                             const std::vector<size_t> &visiblePointIndices,
                                             float &zNear,
                                             float &zFar) const;

    /// 从稀疏点云生成提示深度图
    cv::Mat buildHintDepth(int refIdx,
                           int W,
                           int H,
                           const std::vector<int> &sourceIndices = {}) const;
    cv::Mat buildHintDepthFromVisiblePoints(int refIdx,
                                            int W,
                                            int H,
                                            const std::vector<size_t> &visiblePointIndices) const;
    cv::Mat buildHintDepthForCamera(int refIdx,
                                    const Camera &camera,
                                    int W,
                                    int H,
                                    const std::vector<size_t> &visiblePointIndices) const;

    cv::Mat buildSparseSupportMaskFromVisiblePoints(int refIdx,
                                                    int W,
                                                    int H,
                                                    const std::vector<size_t> &visiblePointIndices) const;
    cv::Mat buildSparseSupportMaskForCamera(int refIdx,
                                            const Camera &camera,
                                            int W,
                                            int H,
                                            const std::vector<size_t> &visiblePointIndices) const;

    /// 双视图深度图左右一致性检查（剔除互不一致的深度像素）
    void crossCheckDepthConsistency();
    bool crossCheckDepthConsistencyStreaming();

    /// 保存单帧深度图预览、原始深度和置信图，并通知 GUI 更新项目结果树
    bool saveDepthFrameArtifacts(int frameIndex,
                                 const DepthFrameResult &result,
                                 const QString &stageLabel);
    void initializeWorkspaceManifest();
    void markManifestFrameRunning(int frameIndex);
    void markManifestFrameFailed(int frameIndex, const QString &error);
    bool persistWorkspaceManifest(QString *errorMsg = nullptr);

    /// 预加载所有图像到内存，避免逐帧重复磁盘读取
    void preloadImages();
    void refreshViewImageDimensionsFromCache();

    std::vector<CameraView> _views;
    SparseCloud _sparse;
    DepthGenConfig _config;
    MvsSceneClassification _sceneClassification;
    MvsSceneProfile _effectiveSceneProfile = MvsSceneProfile::OrbitalObject;
    DepthFilterMode _effectiveDepthFilterMode = DepthFilterMode::Mild;
    std::atomic<bool> _cancelled{false};
    QFuture<void> _backgroundFuture;
    std::string _outputDir;
    std::string _consistencyDepthDirectory;
    bool _streamConsistencyStorageEnabled = false;

    /// 缓存已估计的深度帧
    std::vector<DepthFrameResult> _depthFrames;
    std::vector<uint8_t> _skipFrameMask;

    /// 图像缓存（灰度图，预加载一次复用多次）
    std::vector<cv::Mat> _grayCache;

    /// 最终有效区域掩码（CV_8U，255=有效）。优先使用项目蒙版，无项目蒙版时回退内容检测。
    std::vector<cv::Mat> _validRegionMasks;
    std::vector<uint8_t> _projectMaskLoaded;

    /// MVS 稀疏点可见性与源视图缓存；runInBackground 中预计算一次，帧 worker 仅读取
    std::vector<FrameMvsCache> _frameCaches;
    std::vector<uint64_t> _visibilityBits;
    std::vector<int> _pairCommonCounts;
    size_t _visibilityWordCount = 0;
    bool _frameCachesReady = false;

    QString _workspaceManifestPath;
    QString _depthConfigHash;
    MvsWorkspaceManifest _workspaceManifest;
    std::mutex _workspaceManifestMutex;

public:
    /// 融合完可获取每帧一致性过滤的深度图（返回副本，线程安全）
    std::vector<cv::Mat> filteredDepths() const
    {
        std::lock_guard<std::mutex> lock(_filteredDepthsMutex);
        return _filteredDepths;
    }

private:
    std::vector<cv::Mat> _filteredDepths;
    mutable std::mutex   _filteredDepthsMutex;
};

} // namespace mvs
} // namespace xjw

Q_DECLARE_METATYPE(xjw::mvs::DepthFrameResult)
Q_DECLARE_METATYPE(QSharedPointer<cv::Mat>)
Q_DECLARE_METATYPE(std::vector<xjw::mvs::DensePoint>)
