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
#include "DenseCloudBuilder.h"
#include "DensePointCloudCUDA.h"
#include "SparseCloudPreprocessor.h"

#include <QObject>
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
    int imageIndex = -1;  ///< 向后兼容别名 = refViewIdx
    bool depthFlippedZ = false;
    std::vector<int> sourceViewIndices;  ///< PatchMatch 实际使用的源视图下标，用于限制一致性检查范围
    QSharedPointer<cv::Mat> depthMap;    ///< 深度图 (CV_32F)
    QSharedPointer<cv::Mat> confidence;  ///< 置信图 (CV_32F)
    bool success = false;
    double elapsedMs = 0.0;               ///< 单帧深度估计耗时，不含异步写盘
    std::string device;                   ///< 实际调度设备：GPU/CPU
    std::string errorMsg;

    void releasePixelStorage()
    {
        depthMap.clear();
        confidence.clear();
    }
};

class DepthMapGenerator : public QObject
{
    Q_OBJECT

public:
    explicit DepthMapGenerator(QObject *parent = nullptr);

    /// 向后兼容构造函数
    DepthMapGenerator(const std::vector<CameraView> &views,
                      const PreprocessResult        &ppResult,
                      const DepthGenConfig          &config,
                      QObject                       *parent = nullptr);
    ~DepthMapGenerator() override;

    /// 设置输入数据
    void setViews(const std::vector<CameraView> &views);
    void setSparseCloud(const SparseCloud &sparse);
    void setConfig(const DepthGenConfig &config);
    void setSkippedFrameIndices(const std::vector<int> &indices);

    const std::vector<CameraView> &views() const
    {
        return m_views;
    }

    const SparseCloud &sparse() const
    {
        return m_sparse;
    }

    const DepthGenConfig &config() const
    {
        return m_config;
    }

    /// 异步启动
    Q_INVOKABLE void start();

    /// 设置输出目录（深度图 PNG 保存位置）
    void setOutputDir(const std::string &dir)
    {
        m_outputDir = dir;
    }

    /// 请求取消
    void cancel()
    {
        m_cancelled = true;
    }

    void requestCancel()
    {
        m_cancelled = true;
    }

    bool isCancelled() const
    {
        return m_cancelled.load();
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
        const PositiveDepthCameraModel &camera,
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
                                    const PositiveDepthCameraModel &camera,
                                    int W,
                                    int H,
                                    const std::vector<size_t> &visiblePointIndices) const;

    cv::Mat buildSparseSupportMaskFromVisiblePoints(int refIdx,
                                                    int W,
                                                    int H,
                                                    const std::vector<size_t> &visiblePointIndices) const;
    cv::Mat buildSparseSupportMaskForCamera(int refIdx,
                                            const PositiveDepthCameraModel &camera,
                                            int W,
                                            int H,
                                            const std::vector<size_t> &visiblePointIndices) const;

    /// 双视图深度图左右一致性检查（剔除互不一致的深度像素）
    void crossCheckDepthConsistency();

    /// 保存单帧深度图预览、原始深度和置信图，并通知 GUI 更新项目结果树
    bool saveDepthFrameArtifacts(int frameIndex,
                                 const DepthFrameResult &result,
                                 const QString &stageLabel);

    /// 预加载所有图像到内存，避免逐帧重复磁盘读取
    void preloadImages();
    void refreshViewImageDimensionsFromCache();

    std::vector<CameraView> m_views;
    SparseCloud m_sparse;
    DepthGenConfig m_config;
    std::atomic<bool> m_cancelled{false};
    std::string m_outputDir;

    /// 缓存已估计的深度帧
    std::vector<DepthFrameResult> m_depthFrames;
    std::vector<uint8_t> m_skipFrameMask;

    /// 图像缓存（灰度图，预加载一次复用多次）
    std::vector<cv::Mat> m_grayCache;

    /// 内容区域掩码（在 CLAHE 增强前基于原始图像计算，CV_8U 0/255）
    /// gamma/CLAHE 会将黑边像素 (gray≈3) 提升到 37+，使暗区掩码失效
    /// 因此必须在增强前计算真正的内容/黑边分界
    std::vector<cv::Mat> m_contentMasks;

    /// MVS 稀疏点可见性与源视图缓存；runInBackground 中预计算一次，帧 worker 仅读取
    std::vector<FrameMvsCache> m_frameCaches;
    std::vector<uint64_t> m_visibilityBits;
    std::vector<int> m_pairCommonCounts;
    size_t m_visibilityWordCount = 0;
    bool m_frameCachesReady = false;

public:
    /// 融合完可获取每帧一致性过滤的深度图（返回副本，线程安全）
    std::vector<cv::Mat> filteredDepths() const
    {
        std::lock_guard<std::mutex> lock(m_filteredDepthsMutex);
        return m_filteredDepths;
    }

private:
    std::vector<cv::Mat> m_filteredDepths;
    mutable std::mutex   m_filteredDepthsMutex;
};

} // namespace mvs
} // namespace xjw

Q_DECLARE_METATYPE(xjw::mvs::DepthFrameResult)
Q_DECLARE_METATYPE(QSharedPointer<cv::Mat>)
Q_DECLARE_METATYPE(std::vector<xjw::mvs::DensePoint>)
