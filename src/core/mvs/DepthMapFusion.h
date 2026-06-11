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
#include <string>
#include <functional>
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
    bool  useBoundingBox     = false;  ///< 是否裁剪到指定包围盒
    float bboxMin[3]         = {-1e9f, -1e9f, -1e9f};
    float bboxMax[3]         = { 1e9f,  1e9f,  1e9f};
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
        m_config = c;
    }

    const StereoFusionConfig &config() const
    {
        return m_config;
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

    /// 兼容旧接口：输出 DensePoint（无法线）
    bool fuse(const std::vector<FusionFrameInput> &frames,
              std::vector<DensePoint>             &densePoints,
              MvsProgressCallback                  progressCb = nullptr,
              std::string                         *errorMsg   = nullptr);

    /// 辅助：获取每帧一致性过滤后的深度图（通过 fuse() 后可用）
    const std::vector<cv::Mat> &filteredDepths() const
    {
        return m_filteredDepths;
    }

private:
    StereoFusionConfig m_config;

    /// 每帧的投影/逆投影矩阵（预计算）
    struct FrameGeometry
    {
        float P[12];       ///< 3×4 投影矩阵 K * [R|T]
        float invP[12];    ///< 3×4 逆投影矩阵
        float invR[9];     ///< R_cw^T = R_wc
        PositiveDepthCameraModel cameraModel;
        int W = 0;
        int H = 0;
    };

    /// BFS 队列项
    struct FusionData
    {
        int imageIdx;
        int row;
        int col;
        int traversalDepth;
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
                   const std::vector<cv::Mat> &colorImages,
                   std::vector<std::vector<char>> &fusedMask,
                   FusedPoint &outPoint);

    /// 两视图 minNumPixels<=1 场景的并行快速融合路径
    bool fuseTwoViewSingleObservationFast(
        const std::vector<FusionFrameInput> &frames,
        const std::vector<FrameGeometry> &geom,
        const std::vector<cv::Mat> &colorImages,
        std::vector<FusedPoint> &fusedPoints,
        MvsProgressCallback progressCb);

    /// 取中位数
    static float median(std::vector<float> &v);

    std::vector<cv::Mat> m_filteredDepths;
};

} // namespace mvs
} // namespace xjw
