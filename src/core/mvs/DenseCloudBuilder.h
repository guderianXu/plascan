#pragma once
// =============================================================================
// 文件: DenseCloudBuilder.h
// 模块: MVS - 稠密点云构建
// 说明:
//   将融合后深度图 + 有效掩码反投影为世界坐标系点云（带颜色）。
// =============================================================================

#include "MvsTypes.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <functional>

namespace xjw
{
namespace mvs
{

struct DenseCloudOptions
{
    bool  useGPU     = true;   ///< 是否优先使用 GPU 反投影（DensePointCloudCUDA）
    bool  clipAABB   = false;  ///< 是否裁剪 AABB 包围盒
    float minX=-1e9f, maxX=1e9f;
    float minY=-1e9f, maxY=1e9f;
    float minZ=-1e9f, maxZ=1e9f;
    float minDepth = 0.01f;   ///< 忽略深度过小的像素
    float maxDepth = 1e6f;    ///< 忽略深度过大的像素
    int   subsample = 1;      ///< 像素降采样（1=全分辨率，2=每2格取1）
    int   cudaBlockW = 16;    ///< GPU unproject kernel 线程块宽度（必须为 2 的幂）
    int   cudaBlockH = 16;    ///< GPU unproject kernel 线程块高度（必须为 2 的幂）
};

/// CPU 实现的稠密点云构建器
class DenseCloudBuilder
{
public:
    /// 将单帧深度图反投影为点云
    /// @param depth     融合后深度图 (CV_32F)
    /// @param mask      有效像素掩码 (CV_8U, 255=有效)
    /// @param cameraModel 正深度相机模型
    /// @param colorImg  对应的彩色图像 (CV_8UC3 BGR) 或灰度 (CV_8UC1)，可为空
    /// @param options   选项
    /// @return 点云列表
    static std::vector<DensePoint> unproject(
        const cv::Mat            &depth,
        const cv::Mat            &mask,
        const PositiveDepthCameraModel &cameraModel,
        const cv::Mat            &colorImg,
        const DenseCloudOptions &options = DenseCloudOptions{});

    /// 将多帧点云合并（直接拼接）
    static std::vector<DensePoint> merge(
        const std::vector<std::vector<DensePoint>> &clouds);

    /// 可选：PLY 格式保存
    static bool savePLY(const std::string &path,
                        const std::vector<DensePoint> &cloud,
                        std::string *errorMsg = nullptr);

    /// 统计离群点过滤 (Statistical Outlier Removal)
    /// 对每个点计算 k 近邻平均距离，移除偏离均值过多的点
    /// @param cloud       输入点云
    /// @param kNeighbors  近邻数量（默认 30）
    /// @param stdRatio    标准差倍数阈值（默认 1.5，越小越严格）
    /// @return 过滤后的干净点云
    static std::vector<DensePoint> statisticalOutlierRemoval(
        const std::vector<DensePoint> &cloud,
        int   kNeighbors = 30,
        float stdRatio   = 1.5f);

    /// 半径离群点过滤
    /// 移除在指定半径内邻居数少于阈值的点
    /// @param cloud       输入点云
    /// @param radius      搜索半径
    /// @param minNeighbors 最少邻居数（默认 6）
    /// @return 过滤后的干净点云
    static std::vector<DensePoint> radiusOutlierRemoval(
        const std::vector<DensePoint> &cloud,
        float radius,
        int   minNeighbors = 6);
};

} // namespace mvs
} // namespace xjw
