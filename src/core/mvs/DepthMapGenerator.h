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
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>

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
    std::string errorMsg;
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

    /// 基于原始灰度图生成内容区域掩码；近似全图有效时返回空 Mat 表示跳过过滤
    static cv::Mat buildContentMask(const cv::Mat &gray,
                                    float *coverage = nullptr,
                                    double *otsuThreshold = nullptr,
                                    int *adaptiveThreshold = nullptr);

    /// CUDA PatchMatch 显存不足后的下一次重试配置
    static PatchMatchConfig nextCudaRetryPatchMatchConfig(const PatchMatchConfig &config,
                                                          int imageWidth,
                                                          int imageHeight);

signals:
    /// 每估计完一帧就发出
    void depthMapReady(DepthFrameResult result);
    /// 每帧深度图保存为 PNG 后发出（path, width, height, refImagePath）
    void depthMapSaved(QString pngPath, int width, int height, QString refImagePath);
    /// 点云生成完毕
    void pointCloudReady(std::vector<DensePoint> cloud);
    /// 进度更新
    void progressChanged(QString stage, float ratio);
    /// 出错
    void errorOccurred(QString msg);
    /// 整个流程完成
    void finished(bool success);

private:
    /// 在 QtConcurrent 线程中运行的主函数
    void runInBackground();

    /// 计算单帧深度图
    DepthFrameResult computeDepthForView(int refIdx, const DepthGenConfig *configOverride = nullptr);

    /// 将 DepthFrameResult 组装为 FusionFrameInput
    FusionFrameInput buildFusionFrame(const DepthFrameResult &res) const;

    /// 估计参考帧的深度范围
    bool estimateDepthRange(int refIdx,
                            float &zNear,
                            float &zFar,
                            const std::vector<int> &sourceIndices = {}) const;

    /// 从稀疏点云生成提示深度图
    cv::Mat buildHintDepth(int refIdx,
                           int W,
                           int H,
                           const std::vector<int> &sourceIndices = {}) const;

    /// 双视图深度图左右一致性检查（剔除互不一致的深度像素）
    void crossCheckDepthConsistency();

    /// 预加载所有图像到内存，避免逐帧重复磁盘读取
    void preloadImages();

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
