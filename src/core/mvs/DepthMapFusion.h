#pragma once
// =============================================================================
// 文件: DepthMapFusion.h
// 模块: MVS - COLMAP 风格多视深度图融合
// 说明:
//   采用 COLMAP StereoFusion 思路：
//     1. 对每个参考帧的每个有效深度像素，BFS 遍历重叠视图
//     2. 检查重投影误差、相对深度误差、法线夹角
//     3. 满足最小一致视图数的像素聚合为 3D 点（中位数聚合）
//   生成全局一致的稠密点云。
// =============================================================================

#include "MvsTypes.h"
#include <opencv2/core.hpp>
#include <atomic>
#include <string>
#include <functional>
#include <memory>
#include <cstdint>
#include <vector>

namespace xjw
{
namespace mvs
{

// =============================================================================
// 融合配置（COLMAP StereoFusionOptions 风格）
// =============================================================================
struct StereoFusionConfig
{
    int   minNumPixels       = 3;      ///< 融合到一个 3D 点的最少像素数
    int   maxNumPixels       = 10000;  ///< 防止过度融合（退化情况保护）
    float maxReprojError     = 2.0f;   ///< 最大重投影误差（像素）
    float maxDepthError      = 0.01f;  ///< 最大相对深度误差（|d_meas - d_expect| / d_expect）
    float maxNormalError     = 10.0f;  ///< 最大法线角度差（度）
    int   checkNumImages     = 50;     ///< 传递检查的重叠图像数目
    int   workerCount        = 0;      ///< CPU 融合线程数；0 表示按硬件自动选择
    bool  useColor           = true;   ///< 是否按原图给融合点赋色
    int   colorCacheCapacity = 4;      ///< 原图懒加载 LRU 缓存容量
    bool  requireValidMask   = false;  ///< 要求每帧提供尺寸一致的权威有效蒙版
    int   minSupportViews    = 0;      ///< PatchMatch 支持数下限；0 表示不检查
    float maxLocalDepthGradient = 0.20f; ///< 局部相对深度突变上限；<=0 表示不检查
    bool  fuseOnlyFirstFrame = false;  ///< 流式窗口模式：只从窗口首帧产生点
    bool  enableLowYieldFallback = false; ///< 显式预览/低产出模式才允许降到双视一致
    float lowYieldFallbackMinRatio = 0.01f; ///< 严格融合点数 / 首帧有效深度低于该比例时触发 fallback
    int   lowYieldFallbackMinNumPixels = 2; ///< fallback 的最少一致观测数；默认仍要求至少双视一致
    bool  useBoundingBox     = false;  ///< 是否裁剪到指定包围盒
    float bboxMin[3]         = {-1e9f, -1e9f, -1e9f};
    float bboxMax[3]         = { 1e9f,  1e9f,  1e9f};
    std::shared_ptr<std::atomic_bool> cancelFlag; ///< 外部取消标志；置位后融合尽快返回 false
};

struct FusionRejectionStats
{
    std::uint64_t maskRejected = 0;
    std::uint64_t supportRejected = 0;
    std::uint64_t depthGradientRejected = 0;
    std::uint64_t reprojectionRejected = 0;
    std::uint64_t depthConsistencyRejected = 0;
    std::uint64_t normalRejected = 0;
    std::uint64_t insufficientObservations = 0;
};

// =============================================================================
// 融合输出点（含法线）
// =============================================================================
struct FusedPoint
{
    float   x = 0.f, y = 0.f, z = 0.f;
    float   nx = 0.f, ny = 0.f, nz = 0.f;
    uint8_t r = 0, g = 0, b = 0;
};

// =============================================================================
// COLMAP 风格深度图融合器
// =============================================================================
class DepthMapFusion
{
public:
    explicit DepthMapFusion(const StereoFusionConfig &config = StereoFusionConfig{});

    void setConfig(const StereoFusionConfig &c)
    {
        _config = c;
    }

    const StereoFusionConfig &config() const
    {
        return _config;
    }

    /// 主接口：将多帧深度图融合为稠密点云
    /// @param frames          各相机帧（深度图 + 法线图 + 相机参数）
    /// @param fusedPoints     输出：融合后的 3D 点云
    /// @param progressCb      进度回调
    /// @param errorMsg        出错时填充
    /// @return 成功返回 true
    bool fuse(const std::vector<FusionFrameInput> &frames,
              std::vector<FusedPoint>             &fusedPoints,
              MvsProgressCallback                  progressCb = nullptr,
              std::string                         *errorMsg   = nullptr);

    /// 辅助：获取每帧一致性过滤后的深度图（通过 fuse() 后可用）
    const std::vector<cv::Mat> &filteredDepths() const
    {
        return _filteredDepths;
    }

    FusionRejectionStats rejectionStats() const;

private:
    StereoFusionConfig _config;

    /// 每帧的投影/逆投影矩阵（预计算）
    struct FrameGeometry
    {
        float P[12];       ///< 3×4 投影矩阵 K * [R|T]
        float invP[12];    ///< 3×4 逆投影矩阵
        float invR[9];     ///< R_cw^T = R_wc
        FramePinholeCamera cameraModel;
        int W = 0;
        int H = 0;
    };

    /// 预计算每帧的投影几何
    void prepareGeometry(const std::vector<FusionFrameInput> &frames,
                         std::vector<FrameGeometry> &geom);

    /// 计算重叠图像列表
    void computeOverlappingImages(const std::vector<FusionFrameInput> &frames,
                                  const std::vector<FrameGeometry> &geom,
                                  std::vector<std::vector<int>> &overlapping);

    /// 对单个像素执行 BFS 深度融合
    bool fusePixel(int imageIdx, int row, int col,
                   const std::vector<FusionFrameInput> &frames,
                   const std::vector<FrameGeometry> &geom,
                   const std::vector<std::vector<int>> &overlapping,
                   const std::function<cv::Mat(int)> &colorProvider,
                   std::vector<std::vector<char>> &fusedMask,
                   FusedPoint &outPoint);

    bool isPixelEligible(const FusionFrameInput &frame, int row, int col);
    void resetRejectionStats();

    /// 两视图 minNumPixels<=1 场景的并行快速融合路径
    bool fuseTwoViewSingleObservationFast(
        const std::vector<FusionFrameInput> &frames,
        const std::vector<FrameGeometry> &geom,
        const std::function<cv::Mat(int)> &colorProvider,
        std::vector<FusedPoint> &fusedPoints,
        MvsProgressCallback progressCb);

    /// 流式窗口模式：深度图已在估计阶段完成一致性过滤，直接并行反投影首帧有效像素
    bool fuseFirstFrameObservationsFast(
        const std::vector<FusionFrameInput> &frames,
        const std::vector<FrameGeometry> &geom,
        const std::function<cv::Mat(int)> &colorProvider,
        std::vector<FusedPoint> &fusedPoints,
        MvsProgressCallback progressCb);

    /// 取中位数
    static float median(std::vector<float> &v);

    std::vector<cv::Mat> _filteredDepths;
    std::atomic<std::uint64_t> _maskRejected{0};
    std::atomic<std::uint64_t> _supportRejected{0};
    std::atomic<std::uint64_t> _depthGradientRejected{0};
    std::atomic<std::uint64_t> _reprojectionRejected{0};
    std::atomic<std::uint64_t> _depthConsistencyRejected{0};
    std::atomic<std::uint64_t> _normalRejected{0};
    std::atomic<std::uint64_t> _insufficientObservations{0};
};

} // namespace mvs
} // namespace xjw
